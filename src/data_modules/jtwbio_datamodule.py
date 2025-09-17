
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from einops import rearrange


class SingleSequenceDataset(Dataset): 
    def __init__(self, data_array, num_patches_Nt, patch_size_t, stride_between_sequences_raw):
        super().__init__()
        self.data_array = torch.from_numpy(data_array).float() # (Num_orig_samples, channel=2, Original_Time_Length)
        
        self.num_patches_Nt = num_patches_Nt
        self.patch_size_t = patch_size_t
        self.one_sample_raw_length = num_patches_Nt * patch_size_t 

        self.sequences = [] # sample tensors of shape (N_t, channel=2, t_patch)
        num_orig_recordings, _, original_time_length_of_recording = self.data_array.shape

        if self.one_sample_raw_length > original_time_length_of_recording:
            print(f"Warning: Required sequence length ({self.one_sample_raw_length}) is greater than "
                  f"original recording length ({original_time_length_of_recording}).")
            return

        for i in range(num_orig_recordings):
            original_recording_sample = self.data_array[i] # shape: (2, Original_Time_Length)
            start_indices_raw = np.arange(
                0, 
                original_time_length_of_recording - self.one_sample_raw_length + 1, 
                stride_between_sequences_raw 
            )

            for start_raw in start_indices_raw:
                raw_segment_for_one_sample = original_recording_sample[:, start_raw : start_raw + self.one_sample_raw_length]
                if raw_segment_for_one_sample.shape[1] == self.one_sample_raw_length:
                    model_ready_sample = rearrange(
                        raw_segment_for_one_sample, 
                        'c (nt t_patch) -> nt c t_patch', 
                        nt=self.num_patches_Nt, 
                        t_patch=self.patch_size_t
                    )
                    self.sequences.append(model_ready_sample)
        
        if not self.sequences:
            print(f"Warning: No sequences were created from data with original shape {data_array.shape} ")
        else:
            print(f"Total samples of shape ({num_patches_Nt}, 2, {patch_size_t}) created: {len(self.sequences)}")


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class ECGPPGDataModule(L.LightningDataModule):
    def __init__(self,
                 train_npy_file: str = None,
                 val_npy_file: str = None,
                 test_npy_file: str = None,
                 batch_size: int = 64,
                 num_workers: int = 10,
                 pin_memory: bool = True,
                 model_input_num_patches_Nt: int = 10,  # N_t
                 model_input_patch_size_t: int = 128,   # t_patch
                 stride_for_sequences_raw: int = 128 * 5, 
                 val_split_ratio: float = 0.1,
                 test_split_ratio: float = 0.1,
                 split_from_train_ratios: list = None, 
                 seed: int = 42, 
                 ):
        super().__init__()
        self.save_hyperparameters() 
        
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        

    def setup(self, stage: str):
        
        train_data = np.load(self.hparams.train_npy_file)
        
        def _create_seq_dataset(data_np):
            if data_np is None or len(data_np) == 0:
                return None
            return SingleSequenceDataset(
                data_array=data_np,
                num_patches_Nt=self.hparams.model_input_num_patches_Nt,
                patch_size_t=self.hparams.model_input_patch_size_t,
                stride_between_sequences_raw=self.hparams.stride_for_sequences_raw 
            )

        if self.hparams.split_from_train_ratios and \
           not self.hparams.val_npy_file and \
           not self.hparams.test_npy_file:
            
            print("Setup: Splitting train, validation, and test sets from `train_npy_files`.")
            if self.dataset_train is not None and self.dataset_val is not None and self.dataset_test is not None and stage != "fit":
                print("Datasets already split. Skipping.")
                return

            ratios = self.hparams.split_from_train_ratios

            full_dataset = _create_seq_dataset(train_data)
            if full_dataset is None:
                raise ValueError("Primary dataset from `train_npy_files` is empty after processing.")

            num_total_samples = len(full_dataset)
            train_s = int(ratios[0] * num_total_samples)
            val_s = int(ratios[1] * num_total_samples)
            test_s = num_total_samples - train_s - val_s # Ensure sum is correct

            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                full_dataset, [train_s, val_s, test_s],
                generator=torch.Generator().manual_seed(self.hparams.seed)
            )
                
        else:
            print("Setup: Loading datasets from potentially separate train, validation, or test files.")
            if stage == "fit" or stage is None:

                dataset_from_train_files = _create_seq_dataset(train_data)
                
                if self.hparams.val_npy_file: 
                    self.dataset_train = dataset_from_train_files
                    val_data_np = np.load(self.hparams.val_npy_file)
                    if val_data_np is None:
                        print("Warning: No validation data loaded from `val_npy_file`. Validation dataloader will be empty.")
                        self.dataset_val = None
                    else:
                        self.dataset_val = _create_seq_dataset(val_data_np)

                else: # Split validation from the training data
                    num_total_train_samples = len(dataset_from_train_files)
                    val_size = int(self.hparams.val_split_ratio * num_total_train_samples)
                    train_size = num_total_train_samples - val_size
                    
                    if val_size > 0:
                        self.dataset_train, self.dataset_val = random_split(
                            dataset_from_train_files, 
                            [train_size, val_size],
                            generator=torch.Generator().manual_seed(self.hparams.seed)
                        )
                    else:
                        self.dataset_train = dataset_from_train_files
                        self.dataset_val = None
                        print("Warning: Validation set is empty after splitting.")

                print(f"Training samples: {len(self.dataset_train) if self.dataset_train else 0}")
                print(f"Validation samples: {len(self.dataset_val) if self.dataset_val else 0}")

            if stage == "test" or stage is None:
                if self.hparams.test_npy_file:
                    test_data_np = np.load(self.hparams.test_npy_file)
                    if test_data_np is None:
                        print("Warning: No test data loaded from `test_npy_file`")
                        self.dataset_test = None
                    else:
                        self.dataset_test = _create_seq_dataset(test_data_np)
                else: 
                    print("Warning: No `test_npy_file` provided, use validation set for testing")
                    if not self.dataset_val and (stage is None or stage == "fit"): 
                        self.setup("fit")
                    self.dataset_test = self.dataset_val 

                print(f"Test samples: {len(self.dataset_test) if self.dataset_test else 0}")
            
        if self.dataset_train: print(f"Training samples: {len(self.dataset_train)}")
        else: print("Training dataset Not initialized")
        if self.dataset_val: print(f"Validation samples: {len(self.dataset_val)}")
        else: print("Validation dataset Not initialized")
        if self.dataset_test: print(f"Test samples: {len(self.dataset_test)}")
        else: print("Test dataset not initialized")

    def train_dataloader(self):
        if not self.dataset_train:
            raise RuntimeError("Training dataset not initialized")
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True 
        )

    def val_dataloader(self):
        if not self.dataset_val:
            return None 
        return DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False
        )

    def test_dataloader(self):
        if not self.dataset_test:
            return None
        return DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False
        )


# Refer to the following code:
# https://github.com/mims-harvard/TFC-pretraining/blob/main/code/TFC/

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

import numpy as np
import os
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ..SimCLR.Resnet1d import ResNet1D 

import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra
from omegaconf import OmegaConf, DictConfig

# --- Time-Domain Augmentations ---

def jitter(x, sigma=0.03):
    return x + torch.randn_like(x) * sigma

def scale(x, sigma=0.1):
    scaling_factor = torch.normal(mean=1.0, std=sigma, size=(x.shape[0], 1)).to(x.device)
    return x * scaling_factor

def time_flip(x):
    return torch.flip(x, dims=[1])

def time_segment_shuffle(x, num_segments=4):
    seq_len = x.shape[1]
    if seq_len < num_segments:
        return x
    segment_len = seq_len // num_segments

    segments = torch.split(x[:, :segment_len * num_segments], segment_len, dim=1)
    
    shuffled_indices = torch.randperm(len(segments))
    shuffled_segments = [segments[i] for i in shuffled_indices]
    
    shuffled_x = torch.cat(shuffled_segments, dim=1)
    
    if x.shape[1] > segment_len * num_segments:
        tail = x[:, segment_len * num_segments:]
        shuffled_x = torch.cat([shuffled_x, tail], dim=1)
        
    return shuffled_x

def cutout(x, num_segments=1, max_mask_ratio=0.2):

    _, seq_len = x.shape
    masked_x = x.clone()
    
    for _ in range(num_segments):
        mask_len = int(seq_len * random.uniform(0, max_mask_ratio))
        start = random.randint(0, seq_len - mask_len)
        masked_x[:, start : start + mask_len] = 0.0
        
    return masked_x

def random_crop(x, min_length_ratio=0.8, max_length_ratio=1.0):
    _, seq_len = x.shape
    crop_len = int(seq_len * random.uniform(min_length_ratio, max_length_ratio))
    start = random.randint(0, seq_len - crop_len)
    cropped_x = x[:, start : start + crop_len]
    resizer = torch.nn.functional.interpolate
    resized_x = resizer(cropped_x.unsqueeze(1), size=seq_len, mode='linear', align_corners=False)
    
    return resized_x.squeeze(1)

def apply_augmentations_td(x):
    augmentations = [
        (jitter, 0.5),
        (scale, 0.5),
        (time_flip, 0.3),
        (time_segment_shuffle, 0.7),
        (cutout, 0.3),
        (random_crop, 0.3)
    ]
    
    for aug_func, prob in augmentations:
        if random.random() < prob:
            x = aug_func(x)
    return x

# --- Frequency-Domain Augmentations ---

def remove_frequency(x, pertub_ratio=0.1):
    mask = torch.FloatTensor(x.shape).uniform_().to(x.device) > pertub_ratio
    return x * mask

def add_frequency(x, pertub_ratio=0.1):
    mask = torch.FloatTensor(x.shape).uniform_().to(x.device) > (1 - pertub_ratio)
    max_amplitude = x.max() if x.numel() > 0 else 1.0
    random_am = torch.rand(mask.shape).to(x.device) * (max_amplitude * 0.1)
    pertub_matrix = mask * random_am
    return x + pertub_matrix

def apply_augmentations_fd(sample, pertub_ratio=0.1):
    aug_1 = remove_frequency(sample, pertub_ratio=pertub_ratio)
    aug_2 = add_frequency(sample, pertub_ratio=pertub_ratio)
    aug_F = aug_1 + aug_2
    return aug_F

class TFC_Dataset(Dataset):
    def __init__(self, data_subset, fd_pertub_ratio=0.1):
        super().__init__()
        self.data_subset = data_subset
        self.fd_pertub_ratio = fd_pertub_ratio
        self.transform = T.Compose([
            T.RandomApply([lambda x: jitter(x, sigma=0.05)], p=0.8),
            T.RandomApply([lambda x: scale(x, sigma=0.2)], p=0.8),
            T.RandomApply([time_flip], p=0.5), 
            T.RandomApply([lambda x: time_segment_shuffle(x, num_segments=8)], p=0.7),
            T.RandomApply([lambda x: cutout(x, num_segments=2, max_mask_ratio=0.15)], p=0.5),
            T.RandomApply([lambda x: random_crop(x, min_length_ratio=0.8)], p=0.3)
        ])
    
    def __len__(self):
        return len(self.data_subset)
    
    def __getitem__(self, idx):
        x = self.data_subset[idx][0, :].unsqueeze(0)  

        x_t1 = self.transform(x)
        x_t2 = self.transform(x)

        # Transform to frequency domain 
        # FFT is applied on the last dimension (the sequence length L)
        x_f1_raw = torch.fft.fft(x_t1, dim=-1).abs()
        x_f2_raw = torch.fft.fft(x_t2, dim=-1).abs()

        x_f1 = apply_augmentations_fd(x_f1_raw, pertub_ratio=self.fd_pertub_ratio)
        x_f2 = apply_augmentations_fd(x_f2_raw, pertub_ratio=self.fd_pertub_ratio)

        return x_t1, x_t2, x_f1, x_f2

def nt_xent_loss(features, device, temperature=0.07):

    batch_size = features.shape[0] // 2
    
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    features = F.normalize(features, dim=1, eps=1e-8)
    similarity_matrix = torch.matmul(features, features.T)
    
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    loss_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, loss_labels

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 base_filters: int,
                 kernel_size: int,
                 stride: int,
                 n_block: int,
                 hidden_dim: int, 
                 projection_dim: int): 
        super().__init__()
        
        self.backbone = ResNet1D(
            in_channels=in_channels,
            base_filters=base_filters,
            kernel_size=kernel_size,
            stride=stride,
            groups=1,
            n_block=n_block,
            n_classes=hidden_dim
        ) 
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        feature = self.backbone(x)
        projection = self.projection_head(feature)
        return projection

class TFC_Model(nn.Module):

    def __init__(self, 
                 in_channels: int,
                 base_filters: int,
                 kernel_size: int,
                 stride: int,
                 n_block: int,
                 hidden_dim: int, 
                 projection_dim: int):
        super().__init__()
        
        encoder_args = {
            "in_channels": in_channels, "base_filters": base_filters,
            "kernel_size": kernel_size, "stride": stride, "n_block": n_block,
            "hidden_dim": hidden_dim, "projection_dim": projection_dim
        }
        
        self.time_encoder = Encoder(**encoder_args)
        self.freq_encoder = Encoder(**encoder_args)
        
    def forward(self, x_t1, x_t2, x_f1, x_f2):
        z_t1 = self.time_encoder(x_t1)
        z_t2 = self.time_encoder(x_t2)
        
        z_f1 = self.freq_encoder(x_f1)
        z_f2 = self.freq_encoder(x_f2)
        
        return z_t1, z_t2, z_f1, z_f2

def set_seed(seed, rank):
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup():
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def train_one_epoch(epoch, model, dataloader, optimizer, scheduler, writer, global_step, rank, device, scaler, cfg):
    model.train()
    total_loss_meter = torch.tensor(0.0).to(device)

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False)
    else:
        pbar = dataloader

    for batch in pbar:
        x_t1, x_t2, x_f1, x_f2 = [item.to(device, non_blocking=True) for item in batch]
        
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            z_t1, z_t2, z_f1, z_f2 = model(x_t1, x_t2, x_f1, x_f2)

            features_t = torch.cat([z_t1, z_t2], dim=0)
            logits_t, labels_t = nt_xent_loss(features_t, device, temperature=cfg.training.temperature)
            loss_t = F.cross_entropy(logits_t, labels_t)
            
            features_f = torch.cat([z_f1, z_f2], dim=0)
            logits_f, labels_f = nt_xent_loss(features_f, device, temperature=cfg.training.temperature)
            loss_f = F.cross_entropy(logits_f, labels_f)
            
            features_tf1 = torch.cat([z_t1, z_f1], dim=0)
            logits_tf1, labels_tf1 = nt_xent_loss(features_tf1, device, temperature=cfg.training.temperature)
            loss_tf1 = F.cross_entropy(logits_tf1, labels_tf1)

            features_tf2 = torch.cat([z_t2, z_f2], dim=0)
            logits_tf2, labels_tf2 = nt_xent_loss(features_tf2, device, temperature=cfg.training.temperature)
            loss_tf2 = F.cross_entropy(logits_tf2, labels_tf2)
            
            loss_tf = (loss_tf1 + loss_tf2) / 2.0
            
            total_loss = loss_t + loss_f + cfg.training.loss.alpha * loss_tf
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        total_loss_meter += total_loss.detach()

        if rank == 0:
            step_loss = total_loss.item()
            pbar.set_postfix({"loss": step_loss, "lr": optimizer.param_groups[0]['lr']})
            writer.add_scalar('Loss/train_step', step_loss, global_step)
            writer.add_scalar('LR/step', optimizer.param_groups[0]['lr'], global_step)
            # Log individual losses
            writer.add_scalar('Loss/time_intra_modal', loss_t.item(), global_step)
            writer.add_scalar('Loss/freq_intra_modal', loss_f.item(), global_step)
            writer.add_scalar('Loss/cross_modal', loss_tf.item(), global_step)
    
        global_step += 1
    
    dist.all_reduce(total_loss_meter, op=dist.ReduceOp.SUM)
    avg_loss = total_loss_meter.item() / (len(dataloader) * dist.get_world_size())

    return avg_loss, global_step

def validate_one_epoch(model, dataloader, rank, device, cfg):
    model.eval()
    total_loss_meter = torch.tensor(0.0).to(device)

    if rank == 0:
        pbar = tqdm(dataloader, desc='Validating', leave=False)
    else:
        pbar = dataloader

    with torch.no_grad():
        for batch in pbar:
            x_t1, x_t2, x_f1, x_f2 = [item.to(device, non_blocking=True) for item in batch]
            
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                z_t1, z_t2, z_f1, z_f2 = model(x_t1, x_t2, x_f1, x_f2)

                # Loss calculation is the same as in training
                features_t = torch.cat([z_t1, z_t2], dim=0)
                logits_t, labels_t = nt_xent_loss(features_t, device, temperature=cfg.training.temperature)
                loss_t = F.cross_entropy(logits_t, labels_t)
                
                features_f = torch.cat([z_f1, z_f2], dim=0)
                logits_f, labels_f = nt_xent_loss(features_f, device, temperature=cfg.training.temperature)
                loss_f = F.cross_entropy(logits_f, labels_f)
                
                features_tf1 = torch.cat([z_t1, z_f1], dim=0)
                logits_tf1, labels_tf1 = nt_xent_loss(features_tf1, device, temperature=cfg.training.temperature)
                loss_tf1 = F.cross_entropy(logits_tf1, labels_tf1)

                features_tf2 = torch.cat([z_t2, z_f2], dim=0)
                logits_tf2, labels_tf2 = nt_xent_loss(features_tf2, device, temperature=cfg.training.temperature)
                loss_tf2 = F.cross_entropy(logits_tf2, labels_tf2)
                
                loss_tf = (loss_tf1 + loss_tf2) / 2.0
                total_loss = loss_t + loss_f + cfg.training.loss.alpha * loss_tf
            
            total_loss_meter += total_loss.detach()
            
    dist.all_reduce(total_loss_meter, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_loss = total_loss_meter.item() / (len(dataloader) * dist.get_world_size())
        return avg_loss
    
    return None

@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):

    rank, local_rank, world_size = setup()
    device = torch.device("cuda", local_rank)
    set_seed(cfg.seed, rank)
    torch.set_float32_matmul_precision('high')

    writer = None
    checkpoints_dir = cfg.training.checkpoints_dir
    if rank == 0:
        writer = SummaryWriter(cfg.training.logs_dir)
        os.makedirs(checkpoints_dir, exist_ok=True)
        # print(f"Hydra Config:\n{OmegaConf.to_yaml(cfg)}")
        print(f'Using {world_size} gpus for training')

    data_path = cfg.data.path
    data = np.load(data_path)
    data_tensor = torch.from_numpy(data).float()

    dataset_size = len(data_tensor)
    train_size = int(cfg.data.split_ratios[0] * dataset_size)
    val_size = int(cfg.data.split_ratios[1] * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_subset, val_subset, test_subset = random_split(
        data_tensor, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(cfg.data.split_seed)
    )

    train_dataset = TFC_Dataset(train_subset, fd_pertub_ratio=cfg.data.fd_pertub_ratio)
    val_dataset = TFC_Dataset(val_subset, fd_pertub_ratio=cfg.data.fd_pertub_ratio)
    test_dataset = TFC_Dataset(test_subset, fd_pertub_ratio=cfg.data.fd_pertub_ratio)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, shuffle=False, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, shuffle=False, sampler=test_sampler)

    model = TFC_Model(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,   
        kernel_size=cfg.model.kernel_size,
        stride=cfg.model.stride,
        n_block=cfg.model.n_block,        
        hidden_dim=cfg.model.hidden_dim,    
        projection_dim=cfg.model.projection_dim
    )
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Manual Total Parameters: {total_params}")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*cfg.training.epochs)
    scaler = torch.amp.GradScaler()

    if rank == 0:
        patience = cfg.training.early_stopping.patience
        early_stopping_counter = 0
        best_val_loss = float('inf')

    global_step = 0
    for epoch_num in range(1, cfg.training.epochs + 1):
        train_sampler.set_epoch(epoch_num)
        val_sampler.set_epoch(epoch_num)

        train_loss, global_step = train_one_epoch(epoch_num, ddp_model, train_loader, optimizer, scheduler, writer, global_step, rank, device, scaler, cfg)
        val_loss = validate_one_epoch(ddp_model, val_loader, rank, device, cfg)
        
        early_stop_signal = torch.tensor(0.0).to(device)
        if rank == 0:
            print(f"Epoch {epoch_num}/{cfg.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            writer.add_scalar('Loss/train_epoch', train_loss, epoch_num)
            writer.add_scalar('Loss/val_epoch', val_loss, epoch_num)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(checkpoints_dir, 'best_model.pth')
                torch.save(ddp_model.module.state_dict(), save_path)
                print(f"Loss improved, Model saved to {save_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"Loss not improved: {early_stopping_counter}/{patience}")

            if early_stopping_counter >= patience:
                print("Early stopping")
                early_stop_signal.fill_(1.0)
                
        dist.broadcast(early_stop_signal, src=0)
        if early_stop_signal.item() == 1.0:
            break

    dist.barrier()
    if rank == 0:
        writer.close()
        print("Training finished. Testing on best model...")
    
    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    test_sampler.set_epoch(0)
    test_loss = validate_one_epoch(ddp_model, test_loader, rank, device, cfg)

    if rank == 0:
        print(f"Test Loss on best model: {test_loss:.4f}")

    cleanup()

if __name__ == "__main__":
    main()
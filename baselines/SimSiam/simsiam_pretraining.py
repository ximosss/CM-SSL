
# Refer to the following code:
# SimCLR: https://github.com/sthalles/SimCLR/blob/master/models/resnet_simclr.py
# ResNet1D: https://github.com/hsd1503/resnet1d/blob/master/resnet1d.py
# SimSiam: https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
# DDP Tutorial: https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

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

def jitter(x, sigma=0.03):
    return x + torch.randn_like(x) * sigma

def scale(x, sigma=0.1):
    scaling_factor = torch.normal(mean=1.0, std=sigma, size=(x.shape[0], 1)).to(x.device)
    return x * scaling_factor

def time_flip(x):
    return torch.flip(x, dims=[1])

def time_segment_shuffle(x, num_segments=4):    
    seq_len = x.shape[1]
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

def apply_augmentations(x):
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

class EP_Dataset(Dataset):
    def __init__(self, data_subset):
        super().__init__()
        self.data_subset = data_subset
    
    def __len__(self):
        return len(self.data_subset)
    
    def __getitem__(self, idx):
        
        ep = self.data_subset[idx][0, :].unsqueeze(0)
        ep1 = apply_augmentations(ep)
        ep2 = apply_augmentations(ep)
        return ep1, ep2

def simsiam_loss(p1, p2, z1, z2):
    loss = -(F.cosine_similarity(p1, z2, dim=-1).mean() + F.cosine_similarity(p2, z1, dim=-1).mean()) * 0.5
    return loss

class SimSiam_TS(nn.Module):
    def __init__(self,
                 in_channels: int,
                 base_filters: int,
                 kernel_size: int,
                 stride: int,
                 n_block: int,
                 projection_dim: int = 2048,
                 prediction_dim: int = 512):
        super().__init__()
        
        self.encoder = ResNet1D(in_channels=in_channels,
                                base_filters=base_filters,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=1,
                                n_block=n_block,
                                n_classes=projection_dim,
                                use_do=False)
        
        prev_dim = self.encoder.dense.in_features
        self.encoder.dense = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(prev_dim, projection_dim), 
            nn.BatchNorm1d(projection_dim, affine=False) 
        )

        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, prediction_dim, bias=False),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(prediction_dim, projection_dim) 
        )

    def forward(self, x1, x2):
        # Compute features for one view
        z1 = self.encoder(x1)             
        z1 = z1.reshape(z1.shape[0], -1) 
        z2 = self.encoder(x2)             
        z2 = z2.reshape(z2.shape[0], -1)  

        # Compute predictions
        p1 = self.predictor(z1)   
        p2 = self.predictor(z2)   

        # Return with stop-gradient on the targets
        return p1, p2, z1.detach(), z2.detach()

def train_one_epoch(epoch, model, dataloader, optimizer, scheduler, writer, global_step, rank, device, scaler):
    model.train()
    total_loss = torch.tensor(0.0).to(device)

    dataloader.sampler.set_epoch(epoch)

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False)
    else:
        pbar = dataloader

    for x1, x2 in pbar:
        x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            p1, p2, z1, z2 = model(x1, x2)
            loss = simsiam_loss(p1, p2, z1, z2)

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.detach()

        if rank == 0:
            step_loss = loss.item()
            pbar.set_postfix({"loss": step_loss, "lr": optimizer.param_groups[0]['lr']})
            writer.add_scalar('Loss/train_step', step_loss, global_step)
            writer.add_scalar('LR/step', optimizer.param_groups[0]['lr'], global_step)
    
        global_step += 1
    
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / (len(dataloader) * dist.get_world_size())

    return avg_loss, global_step

def validate_one_epoch(model, dataloader, rank, device):
    model.eval()
    total_loss = torch.tensor(0.0).to(device)
    
    dataloader.sampler.set_epoch(0)

    if rank == 0:
        pbar = tqdm(dataloader, desc='Validating', leave=False)
    else:
        pbar = dataloader

    with torch.no_grad():
        for x1, x2 in pbar:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            # with torch.autocast(device_type=device.type, dtype=torch.float16):
            p1, p2, z1, z2 = model(x1, x2)
            loss = simsiam_loss(p1, p2, z1, z2)
            
            total_loss += loss.detach()
            
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

    avg_loss = total_loss.item() / (len(dataloader) * dist.get_world_size())
    return avg_loss

@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    # torch.autograd.set_detect_anomaly(True)   # debug
    rank, local_rank, world_size = setup()
    device = torch.device("cuda", local_rank)

    set_seed(42, rank)
    torch.set_float32_matmul_precision('high')   # hardware-specific operation

    writer = None
    checkpoints_dir = cfg.training.checkpoints_dir
    if rank == 0:
        writer = SummaryWriter(cfg.training.logs_dir)
        os.makedirs(checkpoints_dir, exist_ok=True)
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

    train_dataset = EP_Dataset(train_subset)
    val_dataset = EP_Dataset(val_subset)
    test_dataset = EP_Dataset(test_subset)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, shuffle=False, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, shuffle=False, sampler=test_sampler)

    model = SimSiam_TS(
        in_channels=cfg.model.in_channels, 
        base_filters=cfg.model.base_filters,   
        kernel_size=cfg.model.kernel_size,
        stride=cfg.model.stride,
        n_block=cfg.model.n_block,         
        projection_dim=cfg.model.hidden_dim,  
        prediction_dim=cfg.model.projection_dim   
    ).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Manual Total Parameters: {total_params}")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)   # important! make multi-gpu training possible 

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)

    warmup_epochs = cfg.training.get('warmup_epochs', 0)
    main_scheduler_iters = len(train_loader) * (cfg.training.epochs - warmup_epochs)

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=main_scheduler_iters
    )

    if warmup_epochs > 0:
        warmup_iters = len(train_loader) * warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1e-5, 
            end_factor=1.0, 
            total_iters=warmup_iters
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_iters]
        )
    else:
        scheduler = main_scheduler

    scaler = torch.GradScaler()

    if rank == 0:
        patience = cfg.training.early_stopping.patience
        early_stopping_counter = 0
        best_val_loss = float('inf')

    global_step = 0
    for epoch_num in range(1,  cfg.training.epochs + 1):
        train_loss, global_step = train_one_epoch(epoch_num, ddp_model, train_loader, optimizer, scheduler, writer, global_step, rank, device, scaler)
        val_loss = validate_one_epoch(ddp_model, val_loader, rank, device)
        
        early_stop_signal = torch.tensor(0.0).to(device)
        if rank == 0:
            print(f"Epoch {epoch_num}/{cfg.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            writer.add_scalar('Loss/train_epoch', train_loss, epoch_num)
            writer.add_scalar('Loss/val_epoch', val_loss, epoch_num)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(checkpoints_dir, 'best_model.pth')
                torch.save(ddp_model.module.state_dict(), save_path)
                print(f"Validation loss improved to {val_loss:.4f}. Model saved to {save_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"Validation loss did not improve: {early_stopping_counter}/{patience}")

            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                early_stop_signal.fill_(1.0)
                
        dist.broadcast(early_stop_signal, src=0)
        if early_stop_signal.item() == 1.0:
            break

    dist.barrier()

    if rank == 0:
        writer.close()
        print("\nTraining finished.")
        print("Testing on best model...")
        
    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    test_sampler.set_epoch(0)
    test_loss = validate_one_epoch(ddp_model, test_loader, rank, device)
    if rank == 0:
        print(f"Final Test Loss on best model: {test_loss:.4f}")

    cleanup()

if __name__ == "__main__":
    main()


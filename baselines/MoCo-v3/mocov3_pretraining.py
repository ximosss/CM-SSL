
# MoCo v3 Reference: https://github.com/facebookresearch/moco-v3
# ViT for 1D Reference: Using timm library with custom PatchEmbed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

import numpy as np
import os
import random
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra
from omegaconf import OmegaConf, DictConfig

import timm.optim.lars as lars  
from timm.models.vision_transformer import VisionTransformer
from timm.layers import trunc_normal_

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

def jitter(x, sigma=0.05): return x + torch.randn_like(x) * sigma
def scale(x, sigma=0.2): return x * torch.normal(mean=1.0, std=sigma, size=(x.shape[0], 1)).to(x.device)
def time_flip(x): return torch.flip(x, dims=[1])
def time_segment_shuffle(x, num_segments=8):    
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

class EP_Dataset(Dataset):
    def __init__(self, data_subset):
        super().__init__()
        self.data_subset = data_subset
        self.transform = T.Compose([
            T.RandomApply([lambda x: jitter(x, sigma=0.05)], p=0.8),
            T.RandomApply([lambda x: scale(x, sigma=0.2)], p=0.8),
            T.RandomApply([time_flip], p=0.5), 
            T.RandomApply([lambda x: time_segment_shuffle(x, num_segments=8)], p=0.7),
            T.RandomApply([lambda x: cutout(x, num_segments=2, max_mask_ratio=0.15)], p=0.5),
        ])
    
    def __len__(self):
        return len(self.data_subset)
    
    def __getitem__(self, idx):
        ep = self.data_subset[idx][1, :].unsqueeze(0)
        ep1 = self.transform(ep)
        ep2 = self.transform(ep)
        return ep1, ep2


class PatchEmbed1D_Linear(nn.Module):
    def __init__(self, seq_len=5000, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        assert seq_len % patch_size == 0, f"Sequence length ({seq_len}) must be divisible by patch size ({patch_size})."
        
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(in_chans * patch_size, embed_dim)

    def forward(self, x):
        B, C, L = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.proj(x)
        return x

def vit_1d(seq_len, in_chans, patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., **kwargs):

    model = VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        qkv_bias=True, 
        img_size=224,
        **kwargs
    )
    model.patch_embed = PatchEmbed1D_Linear(
        seq_len=seq_len, patch_size=patch_size, in_chans=in_chans, embed_dim=model.embed_dim
    )
    
    num_patches = model.patch_embed.num_patches
    model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    trunc_normal_(model.pos_embed, std=.02)
    
    return model

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

class MoCo_ViT_TS(nn.Module):
    def __init__(self, vit_backbone, dim=256, mlp_dim=4096, T=0.2):
        super().__init__()
        self.T = T
        self.base_encoder = vit_backbone(num_classes=mlp_dim)
        self.momentum_encoder = vit_backbone(num_classes=mlp_dim)
        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2)); mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        labels = (torch.arange(logits.shape[0], dtype=torch.long) + logits.shape[0] * dist.get_rank()).cuda()
        return F.cross_entropy(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m, update_momentum=True):
        q1, q2 = self.predictor(self.base_encoder(x1)), self.predictor(self.base_encoder(x2))
        with torch.no_grad():
            if update_momentum:
                self._update_momentum_encoder(m)
            k1, k2 = self.momentum_encoder(x1), self.momentum_encoder(x2)
        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


def train_one_epoch(epoch, model, dataloader, optimizer, scheduler, moco_m_schedule, writer, global_step, rank, device, scaler, cfg):
    model.train()
    total_loss = torch.tensor(0.0).to(device)

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False)
    else:
        pbar = dataloader
    
    for i, (x1, x2) in enumerate(pbar):
        current_iter = epoch * len(dataloader) + i
        total_warmup_iters = cfg.training.warmup_epochs * len(dataloader)
        if current_iter < total_warmup_iters:
            # Linear warmup
            lr_scale = current_iter / total_warmup_iters
            for g in optimizer.param_groups:
                g['lr'] = cfg.training.optimizer.lr * lr_scale
        else:
            # Cosine decay after warmup
            scheduler.step()

        x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
        moco_m = moco_m_schedule[global_step]
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            loss = model(x1, x2, m=moco_m, update_momentum=True)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach()
        if rank == 0:
            step_loss = loss.item()
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": step_loss, "lr": f"{lr:.6f}", "m": f"{moco_m:.4f}"})
            writer.add_scalar('Loss/train_step', step_loss, global_step)
            writer.add_scalar('LR/step', lr, global_step)
            writer.add_scalar('MoCo/momentum', moco_m, global_step)
        global_step += 1
    
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / (len(dataloader) * dist.get_world_size())
    return avg_loss, global_step

def validate_one_epoch(model, dataloader, rank, device):
    model.eval()
    total_loss = torch.tensor(0.0).to(device)

    if rank == 0:
        pbar = tqdm(dataloader, desc='Validating', leave=False)
    else:
        pbar = dataloader

    with torch.no_grad():
        for x1, x2 in pbar:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            loss = model(x1, x2, m=1.0, update_momentum=False)
            total_loss += loss.detach()
            
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / (len(dataloader) * dist.get_world_size())
    return avg_loss


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):
    rank, local_rank, world_size = setup()
    device = torch.device("cuda", local_rank)
    set_seed(cfg.training.seed, rank)
    torch.set_float32_matmul_precision('high')

    writer = None
    checkpoints_dir = cfg.training.checkpoints_dir
    if rank == 0:
        writer = SummaryWriter(cfg.training.logs_dir)
        os.makedirs(checkpoints_dir, exist_ok=True)
        print(f'Using {world_size} GPUs for training')
        print("Config:\n", OmegaConf.to_yaml(cfg))

    data = torch.from_numpy(np.load(cfg.data.path)).float()
    train_size = int(cfg.data.split_ratios[0] * len(data))
    val_size = int(cfg.data.split_ratios[1] * len(data))
    train_subset, val_subset, test_subset = random_split(
        data, [train_size, val_size, len(data) - train_size - val_size],
        generator=torch.Generator().manual_seed(cfg.data.split_seed)
    )
    train_dataset, val_dataset, test_dataset = EP_Dataset(train_subset), EP_Dataset(val_subset), EP_Dataset(test_subset)
    train_sampler, val_sampler, test_sampler = DistributedSampler(train_dataset), DistributedSampler(val_dataset, shuffle=False), DistributedSampler(test_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, sampler=test_sampler)

    model = MoCo_ViT_TS(
        vit_backbone=lambda num_classes: vit_1d(
            seq_len=cfg.model.seq_len, in_chans=cfg.model.in_channels,
            patch_size=cfg.model.vit.patch_size, embed_dim=cfg.model.vit.embed_dim,
            depth=cfg.model.vit.depth, num_heads=cfg.model.vit.num_heads,
            mlp_ratio=cfg.model.vit.mlp_ratio, num_classes=num_classes
        ),
        dim=cfg.model.moco.dim, mlp_dim=cfg.model.moco.mlp_dim, T=cfg.model.moco.T
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {total_params / 1e6:.2f}M")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(model, device_ids=[local_rank])
    # optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)
    def get_param_groups(model):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Avoid weight decay on bias and normalization layers
            if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': decay, 'weight_decay': cfg.training.optimizer.weight_decay},
            {'params': no_decay, 'weight_decay': 0.}
        ]
    param_groups = get_param_groups(ddp_model)
    optimizer = lars.Lars(
        param_groups,
        lr=cfg.training.optimizer.lr, 
        momentum=0.9
    
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * (cfg.training.epochs - cfg.training.warmup_epochs))
    scaler = torch.GradScaler()
    total_steps = len(train_loader) * cfg.training.epochs
    m_initial, m_final = cfg.model.moco.m_initial, cfg.model.moco.m_final
    moco_m_schedule = [m_final - (m_final - m_initial) * (math.cos(math.pi * i / total_steps) + 1) / 2 for i in range(total_steps)]

    if rank == 0:
        patience, early_stopping_counter, best_val_loss = cfg.training.early_stopping.patience, 0, float('inf')
    global_step = 0
    for epoch_num in range(1, cfg.training.epochs + 1):
        train_sampler.set_epoch(epoch_num)
        train_loss, global_step = train_one_epoch(
            epoch_num, ddp_model, train_loader, optimizer, scheduler, moco_m_schedule, writer, global_step, rank, device, scaler, cfg
        )
        val_sampler.set_epoch(epoch_num)
        val_loss = validate_one_epoch(ddp_model, val_loader, rank, device)
        early_stop_signal = torch.tensor(0.0).to(device)
        if rank == 0:
            print(f"Epoch {epoch_num}/{cfg.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            writer.add_scalar('Loss/train_epoch', train_loss, epoch_num)
            writer.add_scalar('Loss/val_epoch', val_loss, epoch_num)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ddp_model.module.state_dict(), os.path.join(checkpoints_dir, 'best_model.pth'))
                print("Loss improved, Model saved.")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"Loss not improved: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print("Early stopping")
                early_stop_signal.fill_(1.0)
        dist.broadcast(early_stop_signal, src=0)
        if early_stop_signal.item() == 1.0: break
    dist.barrier()

    if rank == 0:
        writer.close()
        print("Training finished. Testing on best model...")
    state_dict = torch.load(os.path.join(checkpoints_dir, 'best_model.pth'), map_location='cpu')
    ddp_model.module.load_state_dict(state_dict)
    test_sampler.set_epoch(0)
    test_loss = validate_one_epoch(ddp_model, test_loader, rank, device)
    if rank == 0:
        print(f"Test Loss on best model: {test_loss:.4f}")

    cleanup()

if __name__ == "__main__":
    main()
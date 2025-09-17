import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np
import os
import logging
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra
from omegaconf import OmegaConf, DictConfig

from baselines.SimCLR.run_simclr_finetuning import FineTuneDataset, FineTuningModel, get_metrics, train_one_epoch, \
evaluate, test_and_save_predictions, set_seed, setup, cleanup

from timm.models.vision_transformer import VisionTransformer
from timm.layers import trunc_normal_

log = logging.getLogger(__name__)

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
    """ function to create a ViT for 1D signals. """
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
    if not dist.is_initialized():
        return tensor
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
        for param_m in self.momentum_encoder.parameters():
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

@hydra.main(config_path='../../conf', config_name='config.yaml', version_base=None)
def main(cfg: DictConfig):
    rank, local_rank, world_size = setup()
    device = torch.device("cuda", local_rank)
    set_seed(cfg.data.split_seed, rank)

    log_dir = 'tensorboard_logs'
    checkpoints_dir = 'checkpoints'
    writer = None

    if rank == 0:
        log.info("Starting MoCo-v3 ViT fine-tuning for task: %s", cfg.experiment.name)
        log.info("Hydra output directory: %s", os.getcwd())
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

    log.info("Loading pre-split datasets from config paths...")
    paths = cfg.experiment.data_paths

    train_features = torch.from_numpy(np.load(paths.train_features)).float()
    train_labels = torch.from_numpy(np.load(paths.train_labels))
    val_features = torch.from_numpy(np.load(paths.val_features)).float()
    val_labels = torch.from_numpy(np.load(paths.val_labels))
    test_features = torch.from_numpy(np.load(paths.test_features)).float()
    test_labels = torch.from_numpy(np.load(paths.test_labels))
    
    if cfg.experiment.type == 'classification' or cfg.experiment.type == 'multi-label classification':
        train_labels, val_labels, test_labels = train_labels.long(), val_labels.long(), test_labels.long()
    else:
        train_labels, val_labels, test_labels = train_labels.float(), val_labels.float(), test_labels.float()
    
    if 'target_column_idx' in cfg.experiment and cfg.experiment.target_column_idx is not None:
        col_idx = cfg.experiment.target_column_idx
        log.info(f"Selecting target label from column index: {col_idx}")
    
        train_labels = train_labels[:, col_idx]
        val_labels = val_labels[:, col_idx]
        test_labels = test_labels[:, col_idx]

    log.info("Normalizing features based ON TRAINING SET statistics...")
    feature_mean = train_features.mean(dim=0, keepdim=True)
    feature_std = train_features.std(dim=0, keepdim=True)
    feature_std[feature_std == 0] = 1.0  
    train_features = (train_features - feature_mean) / feature_std
    val_features = (val_features - feature_mean) / feature_std
    test_features = (test_features - feature_mean) / feature_std
    label_mean, label_std = None, None

    if cfg.experiment.type == 'regression':
        log.info("Normalizing labels for regression task based ON TRAINING SET statistics...")
        label_mean = train_labels.mean(dim=0, keepdim=True)
        label_std = train_labels.std(dim=0, keepdim=True)
        label_std[label_std == 0] = 1.0
        train_labels = (train_labels - label_mean) / label_std
        val_labels = (val_labels - label_mean) / label_std
        test_labels = (test_labels - label_mean) / label_std
        
    train_dataset = FineTuneDataset(train_features, train_labels, 
                                    target_length=cfg.data.target_signal_length, 
                                    target_channels=cfg.model.in_channels)
    val_dataset = FineTuneDataset(val_features, val_labels,
                                  target_length=cfg.data.target_signal_length, 
                                  target_channels=cfg.model.in_channels)
    test_dataset = FineTuneDataset(test_features, test_labels,
                                   target_length=cfg.data.target_signal_length, 
                                   target_channels=cfg.model.in_channels)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, sampler=test_sampler)


    if cfg.experiment.type == 'classification' or cfg.experiment.type == 'regression' or cfg.experiment.type == 'multi-label classification':
        num_outputs = cfg.experiment.num_outputs
    else:
        raise ValueError(f"Unknown task type in config: {cfg.experiment.type}")

    full_moco_model = MoCo_ViT_TS(
        vit_backbone=lambda num_classes: vit_1d(
            seq_len=cfg.model.seq_len, 
            in_chans=cfg.model.in_channels,
            patch_size=cfg.model.vit.patch_size, 
            embed_dim=cfg.model.vit.embed_dim,
            depth=cfg.model.vit.depth, 
            num_heads=cfg.model.vit.num_heads,
            mlp_ratio=cfg.model.vit.mlp_ratio, 
            num_classes=num_classes
        ),
        dim=cfg.model.moco.dim, 
        mlp_dim=cfg.model.moco.mlp_dim, 
        T=cfg.model.moco.T
    )

    if rank == 0:
        log.info(f"Loading pretrained MoCo v3 weights from: {cfg.model.pretrained_model_path}")
    
    pretrained_dict = torch.load(cfg.model.pretrained_model_path, map_location='cpu')
    full_moco_model.load_state_dict(pretrained_dict, strict=False) 
    
    if rank == 0:
        log.info("MoCo v3 ViT weights loaded successfully.")
    
    backbone = full_moco_model.base_encoder
    feature_dim = backbone.embed_dim
    backbone.head = nn.Identity()
    model = FineTuningModel(backbone, feature_dim, num_outputs).to(device)
    
    if cfg.training.finetune_mode == 'linear_probing':
        if rank == 0: log.info("Mode: Linear Probing. Freezing backbone.")
        for param in model.backbone.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)
    elif cfg.training.finetune_mode == 'full_finetuning':
        if rank == 0: log.info("Mode: Full Fine-tuning. Training all parameters.")
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': cfg.training.optimizer.backbone_lr},
            {'params': model.head.parameters(), 'lr': cfg.training.optimizer.lr}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.training.optimizer.weight_decay)

    scheduler = None
    if cfg.training.scheduler.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * cfg.training.epochs)
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if cfg.experiment.type == 'classification':
        loss_fn = nn.CrossEntropyLoss()
        if num_outputs == 1: 
             model.module.head = nn.Sequential(model.module.head, nn.Flatten(0))
    elif cfg.experiment.type == 'multi-label classification':
        loss_fn = nn.BCEWithLogitsLoss()
    elif cfg.experiment.type == 'regression':
        loss_fn = nn.L1Loss() 
        if num_outputs == 1:
            model.module.head = nn.Sequential(model.module.head, nn.Flatten(0))

    train_metrics = get_metrics(cfg.experiment.type, num_classes=cfg.experiment.num_outputs, device=device, num_outputs=cfg.experiment.num_outputs)
    val_metrics = get_metrics(cfg.experiment.type, num_classes=cfg.experiment.num_outputs, device=device, num_outputs=cfg.experiment.num_outputs)
    
    if rank == 0:
        best_val_metric = -float('inf') if cfg.experiment.type in ['classification', 'multi-label classification'] else float('inf')
        early_stopping_counter = 0

    global_step = 0
    
    for epoch in range(1, cfg.training.epochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        train_loss, train_epoch_metrics, global_step = train_one_epoch(
            epoch, model, train_loader, optimizer, loss_fn, train_metrics, writer, global_step, rank, device, 
            task_type=cfg.experiment.type, num_outputs=cfg.experiment.num_outputs
        )

        if scheduler:
            scheduler.step()

        val_loss, val_epoch_metrics = evaluate(model, val_loader, loss_fn, val_metrics, device, rank, 
                                               task_type=cfg.experiment.type, num_outputs=cfg.experiment.num_outputs,
                                               label_mean=label_mean,label_std=label_std)
        
        early_stop_signal = torch.tensor(0.0).to(device)
        
        if rank == 0:
            log.info(f"--- Epoch {epoch}/{cfg.training.epochs} Summary ---")
            log.info(f"Train Loss: {train_loss:.4f}; Val Loss: {val_loss:.4f}")
            for name, value in val_epoch_metrics.items(): log.info(f"  Val {name}: {value.item():.4f}")
            if writer:
                writer.add_scalar('Loss/train_epoch', train_loss, epoch)
                writer.add_scalar('Loss/val_epoch', val_loss, epoch)
                for name, value in val_epoch_metrics.items(): writer.add_scalar(f'Metrics/Val_{name}', value.item(), epoch)
            
            if cfg.experiment.type == 'classification':
                current_metric = val_epoch_metrics.get('BalancedAccuracy', val_loss)
                is_better = current_metric > best_val_metric
            elif cfg.experiment.type == 'multi-label classification':
                current_metric = val_epoch_metrics.get('Accuracy', val_loss)
                is_better = current_metric > best_val_metric
            else: 
                current_metric = val_epoch_metrics.get('MAE', val_loss)
                is_better = current_metric < best_val_metric

            if is_better:
                best_val_metric = current_metric
                save_path = os.path.join(checkpoints_dir, 'best_finetuned_model.pth')
                torch.save(model.module.state_dict(), save_path)
                log.info(f"Validation metric improved to {best_val_metric:.4f}. Model saved to {save_path}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                log.info(f"Validation metric did not improve. Counter: {early_stopping_counter}/{cfg.training.early_stopping.patience}")

            if early_stopping_counter >= cfg.training.early_stopping.patience:
                log.info("Early stopping triggered.")
                early_stop_signal.fill_(1.0)

        dist.broadcast(early_stop_signal, src=0)
        if early_stop_signal.item() == 1.0:
            break

    dist.barrier()
    
    if rank == 0:
        log.info("\n--- Final Testing on the Best Model ---")
        best_model_path = os.path.join(checkpoints_dir, 'best_finetuned_model.pth')
        if os.path.exists(best_model_path):
            model.module.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            log.warning("No best model found to test.")
    
    dist.barrier()
    
    test_metrics = get_metrics(cfg.experiment.type, num_classes=cfg.experiment.num_outputs, device=device, num_outputs=cfg.experiment.num_outputs)

    test_sampler.set_epoch(0)
    test_and_save_predictions(
        model=model, dataloader=test_loader, metrics=test_metrics,
        device=device, rank=rank, world_size=world_size, writer=writer,
        cfg=cfg, label_mean=label_mean, label_std=label_std
    )
    if rank == 0:
        writer.close()
        log.info("Fine-tuning finished.")

    cleanup()

if __name__ == "__main__":
    main()
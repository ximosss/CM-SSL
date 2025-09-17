import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra
from omegaconf import OmegaConf, DictConfig
import logging

from .jtwbio_encoder import JTwBio_Encoder
from ..utils.core import train_one_epoch, evaluate, test_and_save_predictions, get_metrics, set_seed, setup, cleanup

log = logging.getLogger(__name__)

class FineTuneDataset(Dataset):
    def __init__(self, data, labels, num_patches, patch_size, target_channels: int = 2):
        self.data = data
        self.labels = labels
        self.num_patches = num_patches     
        self.patch_size = patch_size       
        self.target_length = num_patches * patch_size 
        self.target_channel = target_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].clone().detach().float()
        y = self.labels[idx].clone().detach()

        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        current_length = x.shape[1]
        if current_length != self.target_length:
            x = x.unsqueeze(0)  
            x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False) 
            x = x.squeeze(0) 
            
        if x.shape[0] == 12:
            x = x[1, :].unsqueeze(0)

        x_patched = x.reshape(self.num_patches, self.patch_size)
  
        return x_patched, y
 
class FineTuningModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_outputs):
        super().__init__()
        self.backbone = backbone
        # self.avg_pooling = nn.AdaptiveAvgPool1d(1) 
        # self.max_pooling = nn.AdaptiveMaxPool1d(1)
        # self.head = nn.Linear(feature_dim * 2, num_outputs) 
        self.pooling = nn.AdaptiveAvgPool1d(1) 
        # self.head = nn.Linear(feature_dim, num_outputs)
        
        # nn.Sequential(nn.Linear(feature_dim, 2 * feature_dim),
        #                           nn.GELU(),
        #                           nn.Linear(2* feature_dim, num_outputs))

        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            # nn.BatchNorm1d(feature_dim // 2), 
            # nn.Dropout(0.25),
            nn.Linear(feature_dim // 2, num_outputs)
        )

    def forward(self, x):
        # x shape from dataloader: (B, n_t, t)
        features_seq = self.backbone(x)

        features_seq_permuted = features_seq.permute(0, 2, 1) # -> (B, E_embed, n_t)
        # pooled_features = self.pooling(features_seq_permuted).squeeze(-1) # -> (B, E_embed)

        # avg_pooled = self.avg_pooling(features_seq_permuted).squeeze(-1) # -> (B, E_embed)
        # max_pooled = self.max_pooling(features_seq_permuted).squeeze(-1) # -> (B, E_embed)
        

        # pooled_features = torch.cat([avg_pooled, max_pooled], dim=1) # -> (B, E_embed * 2)

        pooled_features = self.pooling(features_seq_permuted).squeeze(-1)

        output = self.head(pooled_features)
        return output

@hydra.main(config_path='../../conf', config_name='jtwbio_encoder_finetuning', version_base=None)
def main(cfg: DictConfig):

    rank, local_rank, world_size = setup()
    device = torch.device("cuda", local_rank)
    set_seed(cfg.data.split_seed, rank)

    log_dir = 'tensorboard_logs' 
    checkpoints_dir = 'checkpoints'

    writer = None

    if rank == 0:
        log.info("Starting fine-tuning for task: %s", cfg.experiment.name) 
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

    train_dataset = FineTuneDataset(data=train_features, labels=train_labels, 
                                    num_patches=cfg.data.num_patches, patch_size=cfg.data.patch_size ,
                                    target_channels=cfg.model.in_channels)
    val_dataset = FineTuneDataset(data=val_features, labels=val_labels, 
                                    num_patches=cfg.data.num_patches, patch_size=cfg.data.patch_size ,
                                    target_channels=cfg.model.in_channels)
    test_dataset = FineTuneDataset(data=test_features, labels=test_labels, 
                                    num_patches=cfg.data.num_patches, patch_size=cfg.data.patch_size ,
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
    
    if rank == 0:
        log.info(f"Loading pretrained model from: {cfg.model.pretrained_model_path}")

    jtwbio_model = JTwBio_Encoder.load_from_checkpoint(
        cfg.model.pretrained_model_path, 
        map_location='cpu',
        strict=False
    )

    if rank == 0:
        log.info("JTwBio_Encoder loaded successfully.")

    backbone = jtwbio_model.encoder
    
    feature_dim = jtwbio_model.hparams.E_embed
    
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
        loss_fn = nn.MSELoss() 
        if num_outputs == 1:
            model.module.head = nn.Sequential(model.module.head, nn.Flatten(0))

    train_metrics = get_metrics(cfg.experiment.type, num_classes=cfg.experiment.num_outputs, device=device, num_outputs=cfg.experiment.num_outputs)
    val_metrics = get_metrics(cfg.experiment.type, num_classes=cfg.experiment.num_outputs, device=device, num_outputs=cfg.experiment.num_outputs)
    
    if rank == 0:
        best_val_metric = -float('inf') if cfg.experiment.type == 'classification' else float('inf')
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
        model=model,
        dataloader=test_loader,
        metrics=test_metrics,
        device=device,
        rank=rank,
        world_size=world_size,
        writer=writer,
        cfg=cfg,
        label_mean=label_mean,
        label_std=label_std
    )
    if rank == 0:
        writer.close()
        log.info("Fine-tuning finished.")

    cleanup()

if __name__ == "__main__":
    main()

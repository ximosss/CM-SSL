import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import csv
import os
import random
from tqdm import tqdm

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra
from omegaconf import OmegaConf, DictConfig
import logging

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score, MulticlassAveragePrecision, \
MultilabelAUROC, MultilabelAccuracy, MultilabelF1Score, MultilabelAveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

from baselines.SimCLR.Resnet1d import ResNet1D 

log = logging.getLogger(__name__)

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

class SimCLR_TS(nn.Module):
    def __init__(self, in_channels, base_filters, kernel_size, stride, n_block, hidden_dim, projection_dim): 
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

class FineTuneDataset(Dataset):
    def __init__(self, data, labels, target_length=1250, target_channels=1):
        self.data = data
        self.labels = labels
        self.target_length = target_length
        self.target_channels = target_channels

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
        # current_channels = x.shape[0]
        # if current_channels < self.target_channels:
        #     padding_channels = self.target_channels - current_channels
        #     padding = torch.zeros(padding_channels, self.target_length, dtype=x.dtype, device=x.device)
        #     x = torch.cat([x, padding], dim=0)

        return x, y

class FineTuningModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_outputs):
        super().__init__()
        self.backbone = backbone
        # self.head = nn.Linear(feature_dim, num_outputs)
        self.head = nn.Sequential(nn.Linear(feature_dim, 2 * feature_dim),
                                  nn.GELU(),
                                  nn.Linear(2* feature_dim, num_outputs))

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

def get_metrics(task_type, num_classes, device, num_outputs=1):
    if task_type == 'classification':
        metrics = MetricCollection({
            'AUROC': MulticlassAUROC(num_classes=num_classes, average="macro"),
            'AUC_PR': MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
            'BalancedAccuracy': MulticlassAccuracy(num_classes=num_classes, average="macro"),
            'CohenKappa': MulticlassCohenKappa(num_classes=num_classes),
            'F1_weighted': MulticlassF1Score(num_classes=num_classes, average='weighted'),
        }).to(device)
    elif task_type == 'multi-label classification':
        metrics = MetricCollection({
            'AUROC': MultilabelAUROC(num_labels=num_classes, average="macro"),
            'AUC_PR': MultilabelAveragePrecision(num_labels=num_classes, average="macro"),
            'Accuracy': MultilabelAccuracy(num_labels=num_classes, average="macro"),
            'F1_weighted': MultilabelF1Score(num_labels=num_classes, average='weighted'),
        }).to(device)
    elif task_type == 'regression':
            metrics = MetricCollection({
                'MAE': MeanAbsoluteError(), 
                'RMSE': MeanSquaredError(squared=False),
                'R2': R2Score()
            }).to(device)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return metrics

def train_one_epoch(epoch, model, dataloader, optimizer, loss_fn, metrics, writer, global_step, rank, device, task_type='classification', num_outputs=1):
    model.train()
    metrics.reset()
    total_loss = torch.tensor(0.0).to(device)
    total_samples_in_epoch = torch.tensor(0.0).to(device)

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(rank != 0))

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        y_origin = y
        if task_type == 'multi-label classification':
            y = y.float() 

        optimizer.zero_grad()
        outputs = model(x)
        if y.ndim > 1 and outputs.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        # print(outputs.shape, y.shape)
        loss = loss_fn(outputs, y)
        total_loss += loss.detach() * x.size(0)
        total_samples_in_epoch += x.size(0)
        loss.backward()
        optimizer.step()

        preds_detached = outputs.detach()
        y_detached = y.detach()
        y_detached_for_metrics = y_origin.detach()

        if task_type == 'regression':
            metrics.update(preds_detached, y_detached)
        else: 
            metrics.update(preds_detached, y_detached_for_metrics)

            # metrics.update(preds_detached, y_detached)
        
        # total_loss += loss.detach()
        if rank == 0:
            pbar.set_postfix({"loss": loss.item()})
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('LR/step', optimizer.param_groups[0]['lr'], global_step)
            
        global_step += 1
    
    epoch_metrics = metrics.compute()
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_in_epoch, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / total_samples_in_epoch.item()
    # avg_loss = total_loss.item() / len(dataloader)
    return avg_loss, epoch_metrics, global_step

def evaluate(model, dataloader, loss_fn, metrics, device, rank, task_type='classification', num_outputs=1, label_mean=None, label_std=None):
    model.eval()
    metrics.reset()
    total_loss = torch.tensor(0.0).to(device)
    total_samples_in_epoch = torch.tensor(0.0).to(device)
    pbar = tqdm(dataloader, desc='Evaluating', disable=(rank != 0))
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            y_orign = y
            if task_type == 'multi-label classification':
                y = y.float() 

            outputs = model(x)
            if y.ndim > 1 and outputs.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze(1)
            loss = loss_fn(outputs, y)
            total_loss += loss.detach() * x.size(0)
            total_samples_in_epoch += x.size(0)
            # total_loss += loss.detach()

            if task_type == 'regression':
                mean = label_mean.to(device, non_blocking=True)
                std = label_std.to(device, non_blocking=True)
                outputs = outputs * std + mean
                y = y * std + mean

            y_detached_for_metrics = y_orign.detach()

            if task_type == 'regression':
                metrics.update(outputs, y)
            else:
                metrics.update(outputs, y_detached_for_metrics)
            # metrics.update(outputs, y)
    
    epoch_metrics = metrics.compute()
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_in_epoch, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / total_samples_in_epoch.item()
    # avg_loss = total_loss.item() / len(dataloader)
    return avg_loss, epoch_metrics

def get_all_possible_metric_names():

    classification_metrics = ['AUROC', 'AUC_PR', 'BalancedAccuracy', 'CohenKappa', 'F1_weighted', 'Accuracy']
    single_regression_metrics = ['MAE', 'RMSE', 'R2']

    all_metrics = set(classification_metrics + single_regression_metrics)
 
    header = ['model_name', 'task_name'] + sorted(list(all_metrics))
    return header

def test_and_save_predictions(model, dataloader, metrics, device, rank, world_size, writer, cfg, label_mean=None, label_std=None):
    model.eval()
    
    metrics.reset()

    local_predictions = []
    local_labels = []

    pbar = tqdm(dataloader, desc='Final Testing & Saving Predictions', disable=(rank != 0))

    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            y_origin = y
            if cfg.experiment.type == 'multi-label classification':
                y = y.float()

            outputs = model(x)
            if cfg.experiment.type == 'regression':
                mean = label_mean.to(device, non_blocking=True)
                std = label_std.to(device, non_blocking=True)
                outputs = outputs * std + mean
                y = y * std + mean

            if cfg.experiment.type == 'regression':
                if y.ndim > 1 and outputs.ndim == 2 and y.shape[1] == 1:
                    y_squeezed = y.squeeze(1)
                    metrics.update(outputs, y_squeezed)
                else:
                    metrics.update(outputs, y)
            else: 
                if y.ndim > 1 and outputs.ndim == 2 and y.shape[1] == 1:
                    y_squeezed = y_origin.squeeze(1)
                    metrics.update(outputs, y_squeezed)
                else:
                    metrics.update(outputs, y_origin)

            local_predictions.append(outputs.cpu())
            local_labels.append(y.cpu())

    final_metrics = metrics.compute()

    local_predictions = torch.cat(local_predictions)
    local_labels = torch.cat(local_labels)
    
    gathered_predictions = [None for _ in range(world_size)]
    gathered_labels = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_predictions, local_predictions)
    dist.all_gather_object(gathered_labels, local_labels)

    if rank == 0:
        all_predictions = torch.cat(gathered_predictions).float().numpy()
        all_labels = torch.cat(gathered_labels).numpy()

        log.info("\n--- Final Test Results ---")
        log.info(f"Total predictions gathered: {all_predictions.shape}")
        
        for name, value in final_metrics.items():
            log.info(f" Test {name}: {value.item():.4f}")
            if writer: 
                writer.add_scalar(f'Metrics/Test_{name}', value.item(), cfg.training.epochs)

        output_path = cfg.experiment.prediction_output_path
        if not output_path.endswith('.npz'):
            output_path += '.npz'
            
        np.savez(
            output_path, 
            labels=all_labels, 
            predictions=all_predictions
        )
        log.info(f"Predictions and labels saved to {os.path.abspath(output_path)}")

    if rank == 0:        
        summary_file = cfg.results.summary_file_path
        fieldnames = get_all_possible_metric_names() 

        results_row = {
            'model_name': cfg.model.name,  
            'task_name': cfg.experiment.name   
        }
        for name, value in final_metrics.items():
            results_row[name] = f"{value.item():.4f}"

        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        file_exists = os.path.isfile(summary_file)
        
        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(results_row)

        log.info(f"Aggregated results for model '{cfg.model.name}' on task '{cfg.experiment.name}' have been saved to {summary_file}")

    dist.barrier()
    
    return final_metrics

@hydra.main(config_path='../../conf', config_name='config.yaml', version_base=None)
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
    
    simclr_model = SimCLR_TS(cfg.model.in_channels, cfg.model.base_filters, cfg.model.kernel_size, cfg.model.stride, cfg.model.n_block, cfg.model.hidden_dim, cfg.model.projection_dim)
    
    if rank == 0: 
         log.info(f"Loading pretrained weights from: {cfg.model.pretrained_model_path}")
    pretrained_dict = torch.load(cfg.model.pretrained_model_path, map_location='cpu', weights_only=False)
    simclr_model.load_state_dict(pretrained_dict, strict=False)
    
    backbone = simclr_model.backbone

    feature_dim = backbone.dense.in_features
    backbone.dense = nn.Identity()
        
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



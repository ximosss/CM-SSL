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

from baselines.SimCLR.run_simclr_finetuning import get_metrics, get_all_possible_metric_names, test_and_save_predictions, set_seed, setup, cleanup

from .modeling_pretrain import ST_ECGFormer

log = logging.getLogger(__name__)

class HeartLangFineTuneDataset(Dataset):
    def __init__(self, sentences, in_chans, in_times, labels):
        self.sentences = sentences
        self.in_chans = in_chans
        self.in_times = in_times
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.sentences[idx].clone().detach().float(),
            self.in_chans[idx].clone().detach().long(), 
            self.in_times[idx].clone().detach().long(),
            self.labels[idx].clone().detach()
        )

class HeartLangFineTuningModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_outputs):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_outputs)

    def forward(self, sentence, in_chans, in_times):
        # The backbone returns all tokens including [CLS]
        # We need the [CLS] token which is at index 0
        all_tokens = self.backbone(sentence, in_chans, in_times, return_all_tokens=True)
        cls_feature = all_tokens[:, 0]
        output = self.head(cls_feature)
        return output

def train_one_epoch(epoch, model, dataloader, optimizer, loss_fn, metrics, writer, global_step, rank, device, task_type='classification', num_outputs=1):
    model.train()
    metrics.reset()
    total_loss = torch.tensor(0.0).to(device)
    total_samples_in_epoch = torch.tensor(0.0).to(device)

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(rank != 0))

    for sentence, in_chans, in_times, y in pbar:

        sentence = sentence.to(device, non_blocking=True)
        in_chans = in_chans.to(device, non_blocking=True)
        in_times = in_times.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
       
        y_origin = y
        if task_type == 'multi-label classification':
            y = y.float() 

        optimizer.zero_grad()
        outputs = model(sentence, in_chans, in_times)

        # print(f"\n[DEBUG] Before loss:")
        # print(f"  - outputs.shape: {outputs.shape}, outputs.dtype: {outputs.dtype}")
        # print(f"  - y.shape: {y.shape}, y.dtype: {y.dtype}")

        if y.ndim > 1 and outputs.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        # print(outputs.shape, y.shape)

        # print(f"[DEBUG] After squeeze:")
        # print(f"  - y.shape: {y.shape}, y.dtype: {y.dtype}")

        loss = loss_fn(outputs, y)

        total_loss += loss.detach() * sentence.size(0)
        total_samples_in_epoch += sentence.size(0)
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
        for sentence, in_chans, in_times, y in pbar:

            sentence = sentence.to(device, non_blocking=True)
            in_chans = in_chans.to(device, non_blocking=True)
            in_times = in_times.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_orign = y
            if task_type == 'multi-label classification':
                y = y.float() 

            outputs = model(sentence, in_chans, in_times)
            if y.ndim > 1 and outputs.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze(1)
            loss = loss_fn(outputs, y)
            total_loss += loss.detach() * sentence.size(0)
            total_samples_in_epoch += sentence.size(0)
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

def test_and_save_predictions(model, dataloader, metrics, device, rank, world_size, writer, cfg, label_mean=None, label_std=None):
    model.eval()
    
    metrics.reset()

    local_predictions = []
    local_labels = []

    pbar = tqdm(dataloader, desc='Final Testing & Saving Predictions', disable=(rank != 0))

    with torch.no_grad():
        for sentence, in_chans, in_times, y in pbar:
            sentence = sentence.to(device, non_blocking=True)
            in_chans = in_chans.to(device, non_blocking=True)
            in_times = in_times.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_origin = y
            if cfg.experiment.type == 'multi-label classification':
                y = y.float()
            
            outputs = model(sentence, in_chans, in_times)

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
        all_predictions = torch.cat(gathered_predictions).numpy()
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
    
    log.info("Loading pre-tokenized HeartLang datasets...")
    paths = cfg.experiment.heartlang_data_paths
    train_sentences = torch.from_numpy(np.load(paths.train_sentences)).float()
    train_in_chans = torch.from_numpy(np.load(paths.train_in_chans))
    train_in_times = torch.from_numpy(np.load(paths.train_in_times))
    train_labels = torch.from_numpy(np.load(paths.train_labels))

    val_sentences = torch.from_numpy(np.load(paths.val_sentences)).float()
    val_in_chans = torch.from_numpy(np.load(paths.val_in_chans))
    val_in_times = torch.from_numpy(np.load(paths.val_in_times))
    val_labels = torch.from_numpy(np.load(paths.val_labels))

    test_sentences = torch.from_numpy(np.load(paths.test_sentences)).float()
    test_in_chans = torch.from_numpy(np.load(paths.test_in_chans))
    test_in_times = torch.from_numpy(np.load(paths.test_in_times))
    test_labels = torch.from_numpy(np.load(paths.test_labels))

    train_labels, val_labels, test_labels = train_labels.long(), val_labels.long(), test_labels.long()
    
    if 'target_column_idx' in cfg.experiment and cfg.experiment.target_column_idx is not None:
        col_idx = cfg.experiment.target_column_idx
        log.info(f"Selecting target label from column index: {col_idx}")
    
        train_labels = train_labels[:, col_idx]
        val_labels = val_labels[:, col_idx]
        test_labels = test_labels[:, col_idx]

    train_dataset = HeartLangFineTuneDataset(train_sentences, train_in_chans, train_in_times, train_labels)
    val_dataset = HeartLangFineTuneDataset(val_sentences, val_in_chans, val_in_times, val_labels)
    test_dataset = HeartLangFineTuneDataset(test_sentences, test_in_chans, test_in_times, test_labels)
    
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
    
    log.info("Creating HeartLang ST_ECGFormer backbone...")
    backbone = ST_ECGFormer(
        seq_len=cfg.model.seq_len,
        time_window=cfg.model.time_window,
        depth=cfg.model.depth,
        embed_dim=cfg.model.embed_dim,
        heads=cfg.model.heads,
        mlp_dim=cfg.model.mlp_dim
    )
    
    if rank == 0:
        log.info(f"Loading pretrained weights from: {cfg.model.pretrained_model_path}")
    pretrained_dict = torch.load(cfg.model.pretrained_model_path, map_location='cpu', weights_only=False)['model']
    # Filter out keys that don't belong (e.g., lm_head, mask_token from pre-training)
    backbone_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict() and "lm_head" not in k and "mask_token" not in k}
    backbone.load_state_dict(backbone_dict, strict=False)
    
    model = HeartLangFineTuningModel(
        backbone=backbone,
        feature_dim=cfg.model.embed_dim,
        num_outputs=cfg.experiment.num_outputs
    ).to(device)

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
                                               )
        
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
    )
    if rank == 0:
        writer.close()
        log.info("Fine-tuning finished.")

    cleanup()

if __name__ == "__main__":
    main()

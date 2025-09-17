import torch

import numpy as np
import csv
from tqdm import tqdm
import random
import os

import torch.distributed as dist

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score, MulticlassAveragePrecision, \
MultilabelAUROC, MultilabelAccuracy, MultilabelF1Score, MultilabelAveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

import logging

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
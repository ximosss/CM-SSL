

# Refer to the following code:
# https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py

import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ..SimCLR.Resnet1d import ResNet1D

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra

# helper functions
def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu") 

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False, sync_batchnorm = None):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, sync_batchnorm = self.sync_batchnorm)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation.float())
        projection = projector(representation.float())
        return projection, representation

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        sync_batchnorm = None
    ):
        super().__init__()
        self.net = net

        self.augment1 = default(augment_fn, lambda x: x) 
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer = hidden_layer,
            use_simsiam_mlp = not use_momentum,
            sync_batchnorm = sync_batchnorm
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        device = get_module_device(net)
        self.to(device)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):

        if return_embedding:
            if isinstance(self.online_encoder, DDP):
                 return self.online_encoder.module(x, return_projection = return_projection)
            return self.online_encoder(x, return_projection = return_projection)

        view_one, view_two = self.augment1(x), self.augment2(x)

        online_proj_one, _ = self.online_encoder(view_one)
        online_proj_two, _ = self.online_encoder(view_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)


        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder

            target_proj_one, _ = target_encoder(view_one)
            target_proj_two, _ = target_encoder(view_two)
            
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two)
        loss_two = loss_fn(online_pred_two, target_proj_one)

        loss = loss_one + loss_two
        return loss.mean()

def set_seed(seed, rank):
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ddp setting
def setup():
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

# data augmentation
def jitter(x, sigma=0.03):
    return x + torch.randn_like(x) * sigma

def scale(x, sigma=0.1):
    scaling_factor = torch.normal(mean=1.0, std=sigma, size=(x.shape[0], 1, 1)).to(x.device)
    return x * scaling_factor

def time_flip(x):
    return torch.flip(x, dims=[2]) 

def time_segment_shuffle(x, num_segments=4):    
    seq_len = x.shape[2] 
    segment_len = seq_len // num_segments

    if segment_len == 0: return x 

    segments = torch.split(x[:, :, :segment_len * num_segments], segment_len, dim=2)
    shuffled_indices = torch.randperm(len(segments))
    shuffled_segments = [segments[i] for i in shuffled_indices]
    
    shuffled_x = torch.cat(shuffled_segments, dim=2)
    if x.shape[2] > segment_len * num_segments:
        tail = x[:, :, segment_len * num_segments:]
        shuffled_x = torch.cat([shuffled_x, tail], dim=2)
        
    return shuffled_x

def cutout(x, num_segments=1, max_mask_ratio=0.2):

    _, _, seq_len = x.shape
    masked_x = x.clone()
    
    for _ in range(num_segments):
        mask_len = int(seq_len * random.uniform(0, max_mask_ratio))
        start = random.randint(0, seq_len - mask_len)
        masked_x[:, :, start : start + mask_len] = 0.0
        
    return masked_x

def random_crop(x, min_length_ratio=0.8, max_length_ratio=1.0):
    _, _, seq_len = x.shape
    crop_len = int(seq_len * random.uniform(min_length_ratio, max_length_ratio))
    start = random.randint(0, seq_len - crop_len)
    cropped_x = x[:, :, start : start + crop_len]
    resizer = torch.nn.functional.interpolate
    resized_x = resizer(cropped_x, size=seq_len, mode='linear', align_corners=False)
    
    return resized_x

def apply_augmentations(x):
    augmentations = [
        (jitter, 0.5),
        (scale, 0.5),
        (time_flip, 0.3),
        (time_segment_shuffle, 0.7),
        # (cutout, 0.3),
        # (random_crop, 0.3)
    ]
    
    augmented_batch = []
    for sample in x:
        sample_aug = sample.clone()
        for aug_func, prob in augmentations:
            if random.random() < prob:
                sample_aug = aug_func(sample_aug.unsqueeze(0)).squeeze(0)
        augmented_batch.append(sample_aug)

    return torch.stack(augmented_batch)
    
class EP_Dataset(Dataset):
    def __init__(self, data_subset):
        super().__init__()
        self.data_subset = data_subset
    
    def __len__(self):
        return len(self.data_subset)
    
    def __getitem__(self, idx):
        ep = self.data_subset[idx][1, :].unsqueeze(0) 
        return ep

def train_one_epoch(epoch, model, dataloader, optimizer, scheduler, writer, global_step, rank, device, scaler):
    model.train()
    total_loss = torch.tensor(0.0).to(device)

    dataloader.sampler.set_epoch(epoch)

    pbar = dataloader
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False)

    for x in pbar:
        x = x.to(device, non_blocking=True)
        
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            loss = model(x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()
        model.module.update_moving_average()

        if scheduler:
            scheduler.step()

        total_loss += loss.detach()

        if rank == 0:
            step_loss = loss.item()
            pbar.set_postfix({"loss": step_loss, "lr": optimizer.param_groups[0]['lr']})
            if writer:
                writer.add_scalar('Loss/train_step', step_loss, global_step)
                writer.add_scalar('LR/step', optimizer.param_groups[0]['lr'], global_step)
    
        global_step += 1
    
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss_per_gpu = total_loss.item() / len(dataloader) 
    avg_loss = avg_loss_per_gpu / dist.get_world_size()

    return avg_loss, global_step

def validate_one_epoch(model, dataloader, rank, device):
    model.eval()
    total_loss = torch.tensor(0.0).to(device)
    
    dataloader.sampler.set_epoch(0) 

    pbar = dataloader
    if rank == 0:
        pbar = tqdm(dataloader, desc='Validating', leave=False)

    with torch.no_grad():
        for x in pbar:
            x = x.to(device, non_blocking=True)
            loss = model(x)
            total_loss += loss.detach()
            
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss_per_gpu = total_loss.item() / len(dataloader)
    avg_loss = avg_loss_per_gpu / dist.get_world_size()

    return avg_loss

@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg):

    torch.autograd.set_detect_anomaly(True)
    rank, local_rank, world_size = setup()
    device = torch.device("cuda", local_rank)

    set_seed(42, rank)

    torch.set_float32_matmul_precision('high')

    writer = None
    if rank == 0:
        writer = SummaryWriter(cfg.training.logs_dir)
        checkpoints_dir = cfg.training.checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)
        print(f'Using {world_size} gpus for training with BYOL')

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
    
    backbone = ResNet1D(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        kernel_size=cfg.model.kernel_size,
        stride=cfg.model.stride,
        groups=1,
        n_block=cfg.model.n_block,
        n_classes=cfg.model.hidden_dim
    )
    
    model = BYOL(
        net=backbone,
        hidden_layer=-1, 
        projection_size=512, 
        projection_hidden_size=4096, 
        augment_fn=apply_augmentations, 
        moving_average_decay=0.99, 
        sync_batchnorm=True 
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Manual Total Parameters: {total_params}")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)   

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*cfg.training.epochs)

    scaler = torch.GradScaler()

    if rank == 0:
        patience = cfg.training.early_stopping.patience
        early_stopping_counter = 0
        best_val_loss = float('inf')

    global_step = 0

    for epoch_num in range(1, cfg.training.epochs + 1):
        train_loss, global_step = train_one_epoch(epoch_num, ddp_model, train_loader, optimizer, scheduler, writer, global_step, rank, device, scaler)
        val_loss = validate_one_epoch(ddp_model, val_loader, rank, device)
        
        early_stop_signal = torch.tensor(0.0).to(device)
        if rank == 0:
            print(f"Epoch {epoch_num}/{cfg.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if writer:
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

    if rank == 0 and writer:
        writer.close()
        print("Training finished")

    cleanup()

if __name__ == "__main__":
    main()

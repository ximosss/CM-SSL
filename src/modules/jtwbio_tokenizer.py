import torch
import torch.nn as nn
import torch.nn.functional as F 
import lightning as L
import math
from einops import rearrange, repeat
from .vector_quantization import NormEMAVectorQuantizer
from timm.scheduler import CosineLRScheduler
from ..utils.optim import LayerDecayValueAssigner, get_parameter_groups_custom


class Temporal_Encoder(nn.Module):
    def __init__(self, patch_size, D_embed, num_patches=10, used_channel=[0, 1]):
        super().__init__()
        self.patch_size = patch_size
        self.D_embed = D_embed
        self.num_patches = num_patches
        self.used_channel = used_channel
        num_channel = len(self.used_channel)
        
        self.linear = nn.Linear(in_features= num_channel * patch_size, out_features=D_embed)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = x[:, :, self.used_channel, :] 
        x = rearrange(x, 'b n_t c t -> (b n_t) (c t)')
        x = self.linear(x)
        x = self.gelu(x)
        x = rearrange(x, '(b n_t) d_embed -> b n_t d_embed', n_t=self.num_patches)
        return x  # output_shape: (B, N_t, D_embed)


class Positioning_Encoding(nn.Module):
    def __init__(self, D_embed, max_seq_length: int = 1250):
        super().__init__()
        self.D_embed = D_embed
        self.max_seq_length = max_seq_length

        pe = torch.zeros(max_seq_length, D_embed)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_embed, 2).float() * (-math.log(10000.0) / D_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) 
                            
        self.register_buffer('pe', pe) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1)] 


class Joint_Encoder(nn.Module):
    def __init__(self, D_embed, nhead=8, num_layers=4, drop_rate=0.1):
        super().__init__()
        self.D_embed =D_embed

        transformerlayer = nn.TransformerEncoderLayer(
            d_model=D_embed, 
            nhead=nhead,
            dim_feedforward=D_embed * 4,
            dropout=drop_rate,
            activation=nn.GELU(),
            batch_first=True)

        self.encoder = nn.TransformerEncoder(transformerlayer, num_layers)
        
    def forward(self, x):
        return self.encoder(x)
    

class Decoder(nn.Module):
    def __init__(self, D_q_embed, num_channels, patch_size, nhead=8, num_layers=4, drop_rate=0.1):
        super().__init__()
        self.D_q_embed = D_q_embed
        self.patch_size = patch_size
        self.num_channels = num_channels

        transformerlayer = nn.TransformerEncoderLayer(
            d_model=D_q_embed, 
            nhead=nhead,
            dim_feedforward=D_q_embed * 4,
            dropout=drop_rate,
            activation=nn.GELU(),
            batch_first=True)
        self.decoder = nn.TransformerEncoder(transformerlayer, num_layers)
        self.mlp =nn.Sequential(
            nn.Linear(D_q_embed, 4 * D_q_embed),
            nn.GELU(),
            nn.Linear(4 * D_q_embed, 4 * D_q_embed),
            nn.GELU(),
            nn.Linear(4 * D_q_embed, self.num_channels * patch_size)
        )

    def forward(self ,x_quantized):
        x = self.decoder(x_quantized) 
        x = self.mlp(x) 
        x_re = rearrange(x, 'b N_t (c t) -> b N_t c t', c=self.num_channels, t=self.patch_size)
        return x_re


class JTwBio_Tokenzier(L.LightningModule):
    def __init__(self, 
                 D_embed,
                 num_patches, 
                 patch_size,
                 max_seq_length,
                 num_encoder_transformer_heads,
                 num_decoder_transformer_heads,
                 num_transformer_encoder_layers,
                 num_transformer_decoder_layers,
                 quantizer_num_embeddings,
                 quantizer_embedding_dim,
                 quantizer_commitment_beta,
                 quantizer_kmeans_init: bool = False,
                 quantizer_codebook_init_path: str = '',
                 used_channel: list = [0, 1],
                 ecg_weight: float = 0.0,
                 ppg_weight: float = 3.0,
                 learning_rate: float= 1e-4,
                 weight_decay: float = 1e-5,
                 lr_scheduler_type: str = 'cosine',
                 warmup_epochs: int = 10,
                 min_lr: float = 1e-6,
                 total_epochs: int = 100,
                 step_size: int = 10,
                 gamma: float = 0.1):
        super().__init__()

        self.save_hyperparameters()
        self.num_used_channels = len(self.hparams.used_channel)

        self.te = Temporal_Encoder(patch_size=patch_size, D_embed=D_embed, num_patches=num_patches, used_channel=used_channel)
        self.pe = Positioning_Encoding(D_embed=D_embed, max_seq_length=max_seq_length)
        self.je = Joint_Encoder(D_embed=D_embed, nhead=num_encoder_transformer_heads, 
                                num_layers=num_transformer_encoder_layers)
        self.quantizer = NormEMAVectorQuantizer(num_embeddings=quantizer_num_embeddings,
                                                embedding_dim=quantizer_embedding_dim, 
                                                commitment_beta=quantizer_commitment_beta,
                                                kmeans_init=quantizer_kmeans_init,
                                                codebook_init_path=quantizer_codebook_init_path)
        self.de = Decoder(D_q_embed=quantizer_embedding_dim, num_channels=self.num_used_channels, patch_size=patch_size, 
                          nhead=num_decoder_transformer_heads, num_layers=num_transformer_decoder_layers)
        
        self.proj = nn.Linear(D_embed, quantizer_embedding_dim)

        # self.proj = nn.Linear()
        self.ecg_loss = nn.L1Loss()
        self.ppg_loss = nn.L1Loss()

    def forward(self, x):
        x = self.te(x)
        x = self.pe(x)
        x = self.je(x)
        x = self.proj(x)
        x_q, quant_loss, embedding_indices = self.quantizer(x)
        re_x = self.de(x_q)
        return re_x, quant_loss, embedding_indices
    
    def _shared_step(self, batch):
        original_x = batch 
        re_x, quant_loss, _ = self.forward(original_x) 

        original_x_used_channels = batch[:, :, self.hparams.used_channel, :]

        original_x_flat = rearrange(original_x_used_channels, 'b n_t c t -> b c (n_t t)')
        re_x_flat = rearrange(re_x, 'b n_t c t -> b c (n_t t)')


        total_loss = quant_loss
        recon_ecg_loss = torch.tensor(0.0, device=self.device)
        recon_ppg_loss = torch.tensor(0.0, device=self.device)

        for i, channel_idx in enumerate(self.hparams.used_channel):
            original_signal = original_x_flat[:, i, :]
            re_signal = re_x_flat[:, i, :]

            if channel_idx == 0:  # ECG channel
                loss = self.ecg_loss(re_signal, original_signal)
                total_loss += self.hparams.ecg_weight * loss
                recon_ecg_loss = loss
            elif channel_idx == 1:  # PPG channel
                loss = self.ppg_loss(re_signal, original_signal)
                total_loss += self.hparams.ppg_weight * loss
                recon_ppg_loss = loss


        # original_x_flat = rearrange(original_x, 'b n_t c t -> b c (n_t t)')
        # re_x_flat = rearrange(re_x, 'b n_t c t -> b c (n_t t)')

        # ecg = original_x_flat[:, 0, :]
        # ppg = original_x_flat[:, 1, :]
        # re_ecg = re_x_flat[:, 0, :]
        # re_ppg = re_x_flat[:, 1, :]

        # recon_ecg_loss = self.ecg_loss(re_ecg, ecg)
        # recon_ppg_loss = self.ppg_loss(re_ppg, ppg)
        
        # total_loss = (self.hparams.ecg_weight * recon_ecg_loss) + (self.hparams.ppg_weight * recon_ppg_loss) + quant_loss 
        
        return {
            'total_loss': total_loss,
            'recon_ecg_loss': recon_ecg_loss,
            'recon_ppg_loss': recon_ppg_loss,
            'quant_loss': quant_loss
        }

    def training_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)
        
        self.log('train_recon_ecg_loss', metrics['recon_ecg_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_recon_ppg_loss', metrics['recon_ppg_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_quant_loss', metrics['quant_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_total_loss', metrics['total_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return metrics['total_loss']
    
    def on_train_epoch_end(self):

        if hasattr(self.quantizer, 'embedding') and hasattr(self.quantizer.embedding, 'cluster_size'):
            codebook_cluster_size = self.quantizer.embedding.cluster_size
            total_embeddings = self.hparams.quantizer_num_embeddings

            zero_count = (codebook_cluster_size == 0).sum().item()

            self.log('train/quantizer_unused_codes', float(zero_count), on_epoch=True, prog_bar=False, logger=True)
            self.log('train/quantizer_unused_codes_ratio', zero_count / total_embeddings, on_epoch=True, prog_bar=False, logger=True)
            
            self.log('train/quantizer_min_usage', codebook_cluster_size.min().item(), on_epoch=True, prog_bar=False, logger=True)
            self.log('train/quantizer_max_usage', codebook_cluster_size.max().item(), on_epoch=True, prog_bar=False, logger=True)
            self.log('train/quantizer_mean_usage', codebook_cluster_size.mean().item(), on_epoch=True, prog_bar=False, logger=True)


    def configure_optimizers(self):

        # Layer-wise LR Decay
        num_encoder_layers = self.hparams.num_transformer_encoder_layers # same
        num_decoder_layers = self.hparams.num_transformer_decoder_layers 
        
        max_logical_layer_id = num_encoder_layers + num_decoder_layers + 1 # TE/PE(0) + JE_layers + DE_layers + Quantizer/MLP(last)

        layer_decay_rate = 0.75 
        layer_decay_values = [
            (layer_decay_rate ** (max_logical_layer_id - i)) 
            for i in range(max_logical_layer_id + 1)
        ]
        
        layer_decay_assigner = LayerDecayValueAssigner(values=layer_decay_values, hparams=self.hparams)

        parameters_with_group = get_parameter_groups_custom(
            self, 
            weight_decay=self.hparams.weight_decay,
            skip_list=['bias', 'norm'], 
            get_num_layer=layer_decay_assigner.get_layer_id,
            get_layer_scale=layer_decay_assigner.get_scale
        )
        
        optimizer = torch.optim.AdamW(parameters_with_group, lr=self.hparams.learning_rate)
        
        # learning_rate scheduler

        if self.hparams.lr_scheduler_type == 'cosine':
            if self.trainer is not None and self.trainer.estimated_stepping_batches > 0:
                total_training_steps = self.trainer.estimated_stepping_batches
            else:
                print("Warning: trainer.estimated_stepping_batches not available. Estimating total_training_steps for CosineLRScheduler.")
                estimated_steps_per_epoch = 1000
                total_training_steps = self.hparams.total_epochs * estimated_steps_per_epoch

            warmup_steps = self.hparams.warmup_epochs * (total_training_steps // self.hparams.total_epochs if self.hparams.total_epochs > 0 else estimated_steps_per_epoch)

            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=total_training_steps,
                lr_min=self.hparams.min_lr,
                warmup_t=warmup_steps,
                warmup_lr_init=self.hparams.learning_rate * 0.01, 
                cycle_limit=1, 
                t_in_epochs=False, 
            )

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",  
                "frequency": 1,     
                "name": "lr_scheduler",
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        
        elif self.hparams.lr_scheduler_type == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.hparams.step_size, 
                gamma=self.hparams.gamma
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        elif self.hparams.lr_scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,  
                mode='min', 
                factor=0.5, 
                patience=5,
                min_lr=self.hparams.min_lr
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",  
                "frequency": 1,
                "monitor": "valid_total_loss", 
                "name": "lr_scheduler",
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        
        else:
            return optimizer 
        
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
 
        # scheduler.step(self.trainer.global_step)
        idx_to_use = optimizer_idx if optimizer_idx is not None else 0
        current_lr = self.trainer.optimizers[idx_to_use].param_groups[0]['lr']

        if self.trainer.logger: 
            self.logger.experiment.add_scalar("lr", current_lr, self.trainer.global_step)


    def validation_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)
        
        self.log('valid_recon_ecg_loss', metrics['recon_ecg_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_recon_ppg_loss', metrics['recon_ppg_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_quant_loss', metrics['quant_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_total_loss', metrics['total_loss'], on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
         if hasattr(self.quantizer, 'embedding') and hasattr(self.quantizer.embedding, 'cluster_size'):
            codebook_cluster_size = self.quantizer.embedding.cluster_size
            total_embeddings = self.hparams.quantizer_num_embeddings

            zero_count = (codebook_cluster_size == 0).sum().item()

            self.log('valid/quantizer_unused_codes', float(zero_count), on_epoch=True, prog_bar=False, logger=True)
            self.log('valid/quantizer_unused_codes_ratio', zero_count / total_embeddings, on_epoch=True, prog_bar=False, logger=True)
            self.log('valid/quantizer_min_usage', codebook_cluster_size.min().item(), on_epoch=True, prog_bar=False, logger=True)
            self.log('valid/quantizer_max_usage', codebook_cluster_size.max().item(), on_epoch=True, prog_bar=False, logger=True)
            self.log('valid/quantizer_mean_usage', codebook_cluster_size.mean().item(), on_epoch=True, prog_bar=False, logger=True)


    def test_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)
        self.log('test_total_loss', metrics['total_loss'], on_epoch=True)
        self.log('test_recon_ecg_loss', metrics['recon_ecg_loss'], on_epoch=True)
        self.log('test_recon_ppg_loss', metrics['recon_ppg_loss'], on_epoch=True)
        self.log('test_quant_loss', metrics['quant_loss'], on_epoch=True)
    
    def on_train_end(self):

        if self.trainer.is_global_zero:
            print("\n--- Calculating final codebook usage on validation set (on_train_end hook) ---")

        val_dataloader = self.trainer.datamodule.val_dataloader()
        self.calculate_current_codebook_usage(val_dataloader)
    
    @torch.no_grad()
    def calculate_current_codebook_usage(self, data_loader): # exact codebook_counts per_epoch 

        self.eval()
        num_embeddings = self.hparams.quantizer_num_embeddings
        codebook_counts = torch.zeros(num_embeddings, dtype=torch.long, device=self.device)

        print("Calculating exact codebook usage...")
        for i, batch in enumerate(data_loader):
            EEG = batch.float().to(self.device, non_blocking=True) / 100
    
            _, _, embedding_indices = self.forward(EEG)  
            
            current_batch_indices = embedding_indices.view(-1)
            batch_bincount = torch.bincount(current_batch_indices, minlength=num_embeddings)
        
            codebook_counts += batch_bincount
            
            if i % 100 == 0:
                print(f"Processed {i} batches...")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(codebook_counts, op=torch.distributed.ReduceOp.SUM)
        
        zero_cnt = (codebook_counts == 0).sum().item()
        print(f"Total tokens used: {codebook_counts.sum().item()}")
        print(f"STAT: {zero_cnt} tokens ({(zero_cnt / num_embeddings) * 100:.2f}%) never used in this dataset pass.")
        self.train() 
        return codebook_counts


        

        


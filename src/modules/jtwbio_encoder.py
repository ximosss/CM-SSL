import torch
import torch.nn as nn
import torch.nn.functional as F 
import lightning as L
from einops import rearrange

from .jtwbio_tokenizer import Temporal_Encoder, Positioning_Encoding, Joint_Encoder
from .vector_quantization import NormEMAVectorQuantizer

from transformers import get_cosine_schedule_with_warmup
from torchvision.ops import StochasticDepth

class Decoderremoved_Tokenzier(nn.Module):
    def __init__(self, 
                 D_embed,
                 num_patches, 
                 patch_size,
                 max_seq_length,
                 num_transformer_heads,
                 num_transformer_encoder_layers,
                 quantizer_num_embeddings,
                 quantizer_embedding_dim,
                 quantizer_commitment_beta,
                 quantizer_kmeans_init: bool = False,
                 quantizer_codebook_init_path: str = '',
                 used_channel: list = [0, 1]):
        super().__init__()

        self.te = Temporal_Encoder(patch_size=patch_size, D_embed=D_embed, num_patches=num_patches, used_channel=used_channel)
        self.pe = Positioning_Encoding(D_embed=D_embed, max_seq_length=max_seq_length)
        self.je = Joint_Encoder(D_embed=D_embed, nhead=num_transformer_heads, 
                                num_layers=num_transformer_encoder_layers)
        self.quantizer = NormEMAVectorQuantizer(num_embeddings=quantizer_num_embeddings,
                                                embedding_dim=quantizer_embedding_dim, 
                                                commitment_beta=quantizer_commitment_beta,
                                                kmeans_init=quantizer_kmeans_init,
                                                codebook_init_path=quantizer_codebook_init_path)
        
    def forward(self, x):
        x = self.te(x)
        x = self.pe(x)
        x = self.je(x)
        x_q, quant_loss, embedding_indices = self.quantizer(x)
        return x_q, quant_loss, embedding_indices

class Encoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 E_embed, 
                 max_seq_length, 
                 nhead, 
                 drop_rate, 
                 num_layers,
                 stochastic_depth_prob: float = 0.1):
        super().__init__()
        
        self.linear_projection = nn.Linear(in_dim, E_embed)
        self.pe = Positioning_Encoding(D_embed=E_embed, max_seq_length=max_seq_length)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=E_embed, 
                nhead=nhead,
                dim_feedforward=E_embed * 4,
                dropout=drop_rate,
                activation=nn.GELU(),
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # self.num_layers = num_layers

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        # transformerlayer = nn.TransformerEncoderLayer(
        #     d_model=E_embed, 
        #     nhead=nhead,
        #     dim_feedforward=E_embed * 4,
        #     dropout=drop_rate,
        #     activation=nn.GELU(),
        #     batch_first=True)
        # self.encoder = nn.TransformerEncoder(transformerlayer, num_layers)

    def forward(self, x):
        proj = self.linear_projection(x)   # proj.shape: (B, n_t, E_embed)
        embed = self.pe(proj)
        out = embed 
        for layer in self.layers:
            out = self.stochastic_depth(layer(out))
        return out
    
class JTwBio_Encoder(L.LightningModule):
    def __init__(self,
                 E_in_dim, 
                 E_embed,
                 E_nhead, 
                 E_drop_rate, 
                 E_num_layers,
                 D_embed,
                 num_patches, 
                 patch_size,
                 max_seq_length,
                 num_transformer_heads,
                 num_transformer_encoder_layers,
                 quantizer_num_embeddings,
                 quantizer_embedding_dim,
                 quantizer_commitment_beta,
                 quantizer_kmeans_init: bool = False,
                 quantizer_codebook_init_path: str = '',
                 tokenizer_path: str = '',
                 learning_rate: float = 1e-4,
                 stochastic_depth_prob: float = 0.1,
                 tokenizer_used_channel: list = [0, 1],
                 encoder_input_channel: int = 0):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = Decoderremoved_Tokenzier(D_embed=D_embed, 
                                                  num_patches=num_patches,
                                                  patch_size=patch_size,
                                                  max_seq_length=max_seq_length,
                                                  num_transformer_heads=num_transformer_heads,
                                                  num_transformer_encoder_layers=num_transformer_encoder_layers,
                                                  quantizer_num_embeddings=quantizer_num_embeddings,
                                                  quantizer_embedding_dim=quantizer_embedding_dim,
                                                  quantizer_commitment_beta=quantizer_commitment_beta,
                                                  quantizer_kmeans_init=quantizer_kmeans_init,
                                                  quantizer_codebook_init_path=quantizer_codebook_init_path,
                                                  used_channel=self.hparams.tokenizer_used_channel
                                                  )        

        checkpoint = torch.load(tokenizer_path, map_location='cpu')
        para = checkpoint['state_dict'] 
        self.tokenizer.load_state_dict(para, strict=False)

        self.tokenizer.eval()  
        for param in self.tokenizer.parameters():
            param.requires_grad = False  

        self.encoder = Encoder(E_embed=E_embed,
                               in_dim=E_in_dim,
                               max_seq_length=max_seq_length,
                               nhead=E_nhead,
                               drop_rate=E_drop_rate,
                               num_layers=E_num_layers,
                               stochastic_depth_prob=stochastic_depth_prob)
        
        self.prediction_head = nn.Linear(E_embed, quantizer_num_embeddings)

    def forward(self, x):
        with torch.no_grad():
            _, _, target_indices = self.tokenizer(x) # x.shape: (B, n_t, c, t)

        # x_ppg = x[:, :, 1, :]
        # x_ecg = x[:, :, 0, :]
        x_uni = x[:, :, self.hparams.encoder_input_channel, :]
        pred_features = self.encoder(x_uni)
        pred_indices = self.prediction_head(pred_features)
        return target_indices, pred_indices
    
    def training_step(self, batch, batch_idx):
        x = batch
        target_indices, pred_indices = self.forward(x)
        loss_input = rearrange(pred_indices, 'b n c -> (b n) c')
        loss_target = target_indices
        # print(loss_input.shape)
        # print(loss_target.shape)
        train_loss = F.cross_entropy(loss_input, loss_target)

        self.log('train_loss', train_loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

        return train_loss
        
    def validation_step(self, batch, batch_idx):
        x = batch
        target_indices, pred_indices = self.forward(x)
        loss_input = rearrange(pred_indices, 'b n c -> (b n) c')
        loss_target = target_indices

        val_loss = F.cross_entropy(loss_input, loss_target)

        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

        return val_loss
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.1, 
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,  
            },
        ]
    
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        if self.trainer.max_steps:
            num_training_steps = self.trainer.max_steps
        else:
            num_training_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs

        num_warmup_steps = int(num_training_steps * 0.10) 

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  
                "frequency": 1,
            },
        }


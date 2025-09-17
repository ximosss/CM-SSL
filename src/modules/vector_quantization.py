# This code snippet is referenced from github repository of LaBram
# https://github.com/935963004/LaBraM/blob/main/norm_ema_quantizer.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


#  ---- helper function  ----
def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device
    
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim = -1)

        buckets = dists.max(dim = -1).indices
        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


# --- EmbeddingEMA (Codebook management) ---
class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps 

        if codebook_init_path == '':   
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_tokens, codebook_dim) 
            self.register_buffer('initted', torch.tensor([not kmeans_init])) 
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.tensor([True])) 
            
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True 
 

    @torch.jit.ignore 
    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing KMeans init for codebook...")
        # KMeans returns (means, cluster_sizes)
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim = True)
        self.weight.data.copy_(embed) 
        self.cluster_size.data.copy_(cluster_size) 
        self.initted.data.copy_(torch.tensor([True]))

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        ema_inplace(self.cluster_size, new_cluster_size, self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        ema_inplace(self.embed_avg, new_embed_avg, self.decay)

    def weight_update(self): 
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_tokens * self.eps) * n 
            )
        embed_normalized = l2norm(self.embed_avg / smoothed_cluster_size.unsqueeze(1)) 
        self.weight.data.copy_(embed_normalized)   

# def norm_ema_inplace(moving_avg, new, decay):
#     """In-place EMA update followed by L2 normalization."""
#     moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))
#     moving_avg.data.copy_(l2norm(moving_avg.data))


# --- NormEMAVectorQuantizer (Main Quantizer) ---
class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_beta, decay=0.99, eps=1e-5, 
                statistic_code_usage=True, kmeans_init=False, codebook_init_path=''): 
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = num_embeddings 
        self.beta = commitment_beta
        self.decay = decay
        self.eps = eps 

        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)
        
        if torch.distributed.is_available() and torch.distributed.is_initialized(): 
            print("DDP is enabled, using distributed.all_reduce to sync statistics for each GPU!")
            self.all_reduce_fn = torch.distributed.all_reduce
        else:
            self.all_reduce_fn = lambda x: x 
    
    def forward(self, z):
        
        z = l2norm(z) # z.shape: (B, N_t, D_embed)
        z_flattened = z.reshape(-1, self.codebook_dim)
        
        self.embedding.init_embed_(z_flattened)
        
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) 
        
        encoding_indices = torch.argmin(d, dim=1) # shape: (B * N_t,)

        z_q = self.embedding(encoding_indices)
        z_q = z_q.view(z.shape) 
        
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     
        
        # EMA update of codebook and cluster sizes
        if self.training and self.embedding.update: 
            bins = encodings.sum(0) 
            self.all_reduce_fn(bins) 

            # self.embedding.cluster_size_ema_update(bins) # ?
            # zero_mask = (bins == 0) 
            # bins_clamped = bins.masked_fill(zero_mask, 1.) 

            embed_sum = z_flattened.t() @ encodings 
            self.all_reduce_fn(embed_sum) 
            
            self.embedding.cluster_size_ema_update(bins)
            self.embedding.embed_avg_ema_update(embed_sum.t())
            self.embedding.weight_update()
            # embed_normalized = (embed_sum / bins_clamped.unsqueeze(0)).t() 
            # embed_normalized = l2norm(embed_normalized) 
            # embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight, embed_normalized)
            # norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        loss = self.beta * F.mse_loss(z_q.detach(), z) 
        
        z_q = z + (z_q - z).detach()
    
        return z_q, loss, encoding_indices 

import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange
from einops import einsum

def exists(val):
    return val is not None

def divisible_by(num, den):
    return (num % den) == 0

def extract_patches(image, patch_size):
    bs, c, h, w = image.shape
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(bs, -1, c * patch_size * patch_size)
    return patches

def reconstruct_image(patches, image_shape, patch_size):
    bs, c, h, w = image_shape
    num_patches = (h // patch_size) * (w // patch_size)
    patches = patches.reshape(bs, h // patch_size, w // patch_size, c, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).reshape(bs, c, h, w)
    return patches


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        normed = self.ln(x)
        return normed * (self.gamma + 1)

class ConditionalNorm2d(torch.nn.Module):
    def __init__(self, hidden_size, num_features):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.fcw = torch.nn.Linear(num_features, hidden_size)
        self.fcb = torch.nn.Linear(num_features, hidden_size)

    def forward(self, x, features):
        bs, s, l = x.shape
        out = self.norm(x)
        w = self.fcw(features).reshape(bs, 1, -1)
        b = self.fcb(features).reshape(bs, 1, -1)
        return w * out + b


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        cross_attend=False,
        reuse_attention=False
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.reuse_attention = reuse_attention
        self.cross_attend = cross_attend

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.norm = LayerNorm(dim) if not reuse_attention else nn.Identity()
        self.norm_context = LayerNorm(dim) if cross_attend else nn.Identity()

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False) if not reuse_attention else None
        self.to_k = nn.Linear(dim, inner_dim, bias=False) if not reuse_attention else None
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        return_qk_sim=False,
        qk_sim=None
    ):
        x = self.norm(x)

        assert not (exists(context) ^ self.cross_attend)

        if self.cross_attend:
            context = self.norm_context(context)
        else:
            context = x

        v = self.to_v(context)
        v = self.split_heads(v)

        if not self.reuse_attention:
            qk = (self.to_q(x), self.to_k(context))
            q, k = tuple(self.split_heads(t) for t in qk)

            q = q * self.scale
            qk_sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        else:
            assert exists(qk_sim), 'qk sim matrix must be passed in for reusing previous attention'

        attn = self.attend(qk_sim)
        attn = self.dropout(attn)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = self.to_out(out)

        if not return_qk_sim:
            return out

        return out, qk_sim





class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, num_features=128):
        super(TransformerBlock, self).__init__()
        
        dim_head = hidden_size // num_heads  
    
        self.self_attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=dim_head,
            dropout=0.0,
            cross_attend=False,
            reuse_attention=False
        )
        
        self.cross_attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=dim_head,
            dropout=0.0,
            cross_attend=True,
            reuse_attention=False
        )
        
        self.con_norm = ConditionalNorm2d(hidden_size, num_features)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
                
    def forward(self, x, highres_x, features):
        x = self.self_attn(x) + x
        x = self.cross_attn(x, context=highres_x) + x
        norm_x = self.con_norm(x, features)
        x = self.mlp(norm_x) + x
        return x


class DiT(torch.nn.Module):
    def __init__(self, image_size=28, channels_in=1, patch_size=4, highres_patch_size=2, 
                 hidden_size=260, num_features=128, num_layers=10, num_heads=10):
        super().__init__()
        
        assert divisible_by(image_size, patch_size), "image_size must be divisible by patch_size"
        assert divisible_by(image_size, highres_patch_size), "image_size must be divisible by highres_patch_size"
        assert highres_patch_size < patch_size, "highres_patch_size must be smaller than patch_size"
        
        self.patch_size = patch_size
        self.highres_patch_size = highres_patch_size
        self.image_size = image_size
        self.channels_in = channels_in
        
        self.time_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(num_features),
            torch.nn.Linear(num_features, 2 * num_features),
            torch.nn.GELU(),
            torch.nn.Linear(2 * num_features, num_features),
            torch.nn.GELU()
        )
        
        self.fc_in = torch.nn.Linear(channels_in * patch_size * patch_size, hidden_size) 
        self.fc_in_highres = torch.nn.Linear(channels_in * highres_patch_size * highres_patch_size, hidden_size) 
        
        seq_length = (image_size // patch_size) ** 2  
        self.pos_embedding = torch.nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))
        
        highres_seq_length = (image_size // highres_patch_size) ** 2  
        self.highres_pos_embedding = torch.nn.Parameter(
            torch.empty(1, highres_seq_length, hidden_size).normal_(std=0.02)
        )
        
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_features) for _ in range(num_layers)
        ])
        
        self.fc_out = torch.nn.Linear(hidden_size, channels_in * patch_size * patch_size)  

    def forward(self, image_in, index):
        index_features = self.time_mlp(index)
        
        patch_seq = extract_patches(image_in, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)
        embs = patch_emb + self.pos_embedding
        
        highres_patch_seq = extract_patches(image_in, patch_size=self.highres_patch_size)
        highres_patch_emb = self.fc_in_highres(highres_patch_seq)
        highres_embs = highres_patch_emb + self.highres_pos_embedding
        
        for block in self.blocks:
            embs = block(embs, highres_embs, index_features)
        
        image_out = self.fc_out(embs)
        return reconstruct_image(image_out, image_in.shape, patch_size=self.patch_size)
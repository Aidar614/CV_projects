import torch.nn as nn
import math
import torch
from pynvml import *

def extract_patches(image_tensor, patch_size=8):

    bs, c, h, w = image_tensor.size()
    
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded = unfold(image_tensor)
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)
    
    return unfolded


class SinusoidalPosEmb(nn.Module):
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
    
    
class Decoder(nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
        super(Decoder, self).__init__()
        

        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.embedding.weight.data = 0.001 * self.embedding.weight.data

        self.pos_emb = SinusoidalPosEmb(hidden_size)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, 
                                                   dim_feedforward=hidden_size * 4, dropout=0.0,
                                                   batch_first=True)
        
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
                
    
        self.fc_out = nn.Linear(hidden_size, num_emb)
        
    def forward(self, input_seq, encoder_output, input_padding_mask=None, 
                encoder_padding_mask=None):
   
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape
        seq_indx = torch.arange(l, device=input_seq.device)   
        pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb
        causal_mask = torch.triu(torch.ones(l, l, device=input_seq.device), 1).bool()
        output = self.decoder_layers(tgt=embs, memory=encoder_output, tgt_mask=causal_mask, tgt_key_padding_mask=input_padding_mask,
                                     memory_key_padding_mask=encoder_padding_mask)
       
       
        
        return self.fc_out(output)

    
class VisionEncoder(nn.Module):
    def __init__(self, image_size, channels_in, patch_size=16, hidden_size=128, num_layers=3, num_heads=4):
        super(VisionEncoder, self).__init__()
        
        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, 
                                                   dim_feedforward=hidden_size * 4, dropout=0.0, 
                                                   batch_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers) 
                
    def forward(self, image):  
        bs = image.shape[0]
        patch_seq = extract_patches(image, patch_size=self.patch_size)       
        patch_emb = self.fc_in(patch_seq)
        embs = patch_emb + self.pos_embedding
        output = self.encoder_layers(embs)
      
        return output
    
    

class VisionEncoderDecoder(nn.Module):
    def __init__(self, image_size, channels_in, num_emb, patch_size=16, 
                 hidden_size=128, num_layers=(3, 3), num_heads=4):
        super(VisionEncoderDecoder, self).__init__()
        
        self.encoder = VisionEncoder(image_size=image_size, channels_in=channels_in, patch_size=patch_size,
                               hidden_size=hidden_size, num_layers=num_layers[0], num_heads=num_heads)
        
        self.decoder = Decoder(num_emb=num_emb, hidden_size=hidden_size, 
                               num_layers=num_layers[1], num_heads=num_heads)

    def forward(self, input_image, target_seq, padding_mask):
       
        bool_padding_mask = padding_mask == 0
        encoded_seq = self.encoder(image=input_image)
        
        decoded_seq = self.decoder(input_seq=target_seq, 
                                   encoder_output=encoded_seq, 
                                   input_padding_mask=bool_padding_mask)
        return decoded_seq
import torch.nn as nn
import torch
import math
from transformers import AutoBackbone, AutoConfig

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x) 
    

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden, enc_layers=1, dec_layers=1, nhead=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden = hidden
        
        self.backbone = self._init_textnet_backbone()
        
        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.decoder = nn.Embedding(vocab_size, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=hidden*4,
            dropout=dropout
        )
        self.fc_out = nn.Linear(hidden, vocab_size)

    def _init_textnet_backbone(self):
        config = AutoConfig.from_pretrained("jadechoghari/textnet-base")
        backbone = AutoBackbone.from_pretrained("jadechoghari/textnet-base", config=config)
        
        for param in backbone.parameters():
            param.requires_grad = True
        
        return backbone

    def _get_features(self, src):
    
        features = self.backbone(src)  
        x = features["feature_maps"][-1] 
        
        x = x.flatten(2)  
        x = x.permute(2, 0, 1) 
        
        return x

    def forward(self, src, trg):
        trg_mask = self.generate_square_subsequent_mask(trg.size(0)).to(trg.device)
        
        features = self._get_features(src)  
        
        src = self.pos_encoder(features)
        trg = self.pos_decoder(self.decoder(trg))
        
        output = self.transformer(
            src, trg,
            tgt_mask=trg_mask,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None
        )
        
        return self.fc_out(output)
    

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask.float().masked_fill(mask == 1, float('-inf'))
    
    def predict(self, src, vocab, max_len=100):
       
        self.eval()
        with torch.no_grad():
            features = self._get_features(src)  
            memory = self.transformer.encoder(self.pos_encoder(features))
            outputs = [vocab.stoi["<SOS>"]]
            for i in range(max_len):
                trg = torch.LongTensor(outputs).unsqueeze(1).to(src.device)
                output = self.fc_out(self.transformer.decoder(
                    self.pos_decoder(self.decoder(trg)),
                    memory
                ))
                
                next_token = output.argmax(2)[-1].item()
                outputs.append(next_token)
                if next_token == vocab.stoi["<EOS>"]:
                    break
            
            return outputs

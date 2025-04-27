import torch
from torch import optim
import torchvision.transforms as transforms
import config
from model import VisionEncoderDecoder
from dataloader import get_loader
import torchvision.transforms as transforms
from tqdm import tqdm
from dataloader import Vocabulary
import torch.nn as nn 


if __name__ == "__main__":

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
            transforms.ToTensor(),
        ]
    )

    vocab = Vocabulary(freq_threshold=config.FREQ_TRESHOLD)
    vocab.build_vocabulary('captions_train.csv')

    loader = get_loader(
            'val2017/val2017/', "captions_train.csv",vocab, transform=transform
        )
    

    vocab_size = len(vocab)

    caption_model = VisionEncoderDecoder(image_size=config.IMAGE_HEIGHT, channels_in=config.NUM_CHANNELS, 
                                        num_emb=vocab_size, patch_size=config.PATCH_SIZE, 
                                        num_layers=config.NUM_LAYER, hidden_size=config.HIDDEN_SIZE, 
                                        num_heads=config.NUM_HEADS).to(device)
    
    optimizer = optim.Adam(caption_model.parameters(), lr=config.LEARNING_RATE)

    scaler = torch.amp.GradScaler(device)


    loss_fn = nn.CrossEntropyLoss(reduction="none")

    training_loss_logger = []
    eval_loss_logger = []
    start_epoch = 0
    
    if config.LOAD_MODEL==True:
        print('Загрузка модели')
        cp = torch.load("captioning_model.pt")
        caption_model.load_state_dict(cp["model_state_dict"])
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        training_loss_logger = cp["train_data_logger"]
        eval_loss_logger = cp["eval_data_logger"]
        start_epoch = cp["epoch"] 
        print('Модель загружена')

    
    for epoch in tqdm(range(start_epoch, config.NUM_EPOCHS), leave=False, desc="Epoch"):

        caption_model.train()
        loop = tqdm(loader, desc="Training", leave=False)
        
        for imgs, captions in loop:
            
            captions = captions.permute(1, 0).to(device)
            imgs = imgs.to(device)
            
            bs = captions.shape[0]

            target_ids = torch.cat((captions[:, 1:], 
                                        torch.zeros(bs, 1, device=device).long()), 1)
            
            padding_mask = torch.where(captions != 0, torch.ones_like(captions), captions)
            
        
            with torch.amp.autocast('cuda'):
                pred = caption_model(imgs, captions, padding_mask=padding_mask)
            
            loss_mask = (~(target_ids == 0)).float()
            loss = (loss_fn(pred.transpose(1, 2), target_ids) * loss_mask).sum()/loss_mask.sum()
        
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            training_loss_logger.append(loss.item())
            loop.set_postfix(loss=loss.item())
            
        
        

        print('Сохранение модели')
        torch.save({'epoch': epoch + 1,
                    'train_data_logger': training_loss_logger,
                    'eval_data_logger': eval_loss_logger,
                    'model_state_dict': caption_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, "captioning_model_new.pt")
        
        
       
       
        
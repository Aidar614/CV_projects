import torch
from torch import optim
import config
from utils import AccuracyTracker
from model import TransformerModel
from dataloader import get_loader
from tqdm import tqdm
from dataloader import Vocabulary
import torch.nn as nn 
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocabulary()
    accuracy_tracker = AccuracyTracker(vocab)

    train_loader = get_loader(
    root_dir='ocr_dataset/train/',  
    annotation_file='ocr_dataset/train.tsv', 
    vocab=vocab
    )
    

    vocab_size = len(vocab)
    model = TransformerModel(
        vocab_size=len(vocab),
        hidden=config.HIDDEN_SIZE,
        enc_layers=config.NUM_LAYER[0],
        dec_layers=config.NUM_LAYER[1],
        nhead=config.NUM_HEAD,
        dropout=config.DROPOUT
    ).to(device)
 

    best_accuracy = 0.0
    num_epochs = config.NUM_EPOCHS

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    scaler = torch.amp.GradScaler(device)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)  

    if config.LOAD_MODEL:
        checkpoint = torch.load(config.MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  
        best_accuracy = checkpoint['accuracy']
        print(f"Loaded model from epoch {start_epoch-1} with accuracy {best_accuracy:.2%}")
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for images, captions in progress_bar:
            images = images.to(device).float() 
            captions = captions.to(device).long()
            
            trg_input = captions[:-1, :]
            targets = captions[1:, :]
           
            
            with torch.amp.autocast("cuda"):    
                output = model(images, trg_input)

                loss = criterion(output.view(-1, output.shape[2]), targets.reshape(-1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
           
            batch_accuracy = accuracy_tracker.calculate_accuracy(output, targets)
            moving_avg = accuracy_tracker.get_moving_avg()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'batch_acc': f'{batch_accuracy:.2%}',
                'avg_acc': f'{moving_avg:.2%}',
                'lr': optimizer.param_groups[0]['lr']
            })
        
        
        epoch_accuracy = accuracy_tracker.epoch_end()
        scheduler.step(epoch_accuracy)  
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': epoch_accuracy,
        }, f'model_epoch_{epoch+1}.pth')
        
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1} | Acc: {epoch_accuracy:.2%} | LR: {optimizer.param_groups[0]['lr']:.2e}")



    
       
       
        
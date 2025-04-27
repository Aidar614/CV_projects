import argparse
from PIL import Image
import torchvision.transforms as transforms
from model import TransformerModel
from dataloader import Vocabulary
import torch
import config
from torchvision import transforms

def initialize_model(device):
   
    vocab = Vocabulary()
    model = TransformerModel(
        vocab_size=len(vocab),
        hidden=config.HIDDEN_SIZE,
        enc_layers=config.NUM_LAYER[0],
        dec_layers=config.NUM_LAYER[1],
        nhead=config.NUM_HEAD,
        dropout=config.DROPOUT
    ).to(device)
    
    checkpoint = torch.load('model_epoch_6.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = torch.compile(model)
    model.eval() 
    
    return model, vocab

def predict_image(model, vocab, image_path, device):
    
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    src = transform(image).unsqueeze(0).to(device)
    predicted_indices = model.predict(src, vocab, max_len=100)
    predicted_tokens = [
        vocab.itos[idx] 
        for idx in predicted_indices 
        if idx not in (vocab.stoi["<SOS>"], vocab.stoi["<EOS>"])
    ]
    return "".join(predicted_tokens)

def main():
    parser = argparse.ArgumentParser(description='OCR Predictor')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab = initialize_model(device)
    
    print("The model is loaded. Enter the path to the image or 'exit' to exit:")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ('exit', 'quit'):
            break
        
        try:
            predicted_text = predict_image(model, vocab, user_input, device)
            print(f"Result: {predicted_text}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
import torch
import torchvision.transforms as transforms
import config
from model import VisionEncoderDecoder
import torchvision.transforms as transforms
from torch.distributions import Categorical
import pandas as pd
import argparse
from PIL import Image
import goslate


def tesing_model(image_path, target_language):

    gs = goslate.Goslate()  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
            transforms.ToTensor(),
        ]
    )

    df = pd.read_csv("vocab_itos.csv", header=0) 

    vocab_itos = {int(row["Index"]): row["Token"] for _, row in df.iterrows()}
    vocab_stoi = {row["Token"]: int(row["Index"]) for _, row in df.iterrows()}

    image = Image.open(image_path)
    transformed_image = transform(image).unsqueeze(0)

    vocab_size = len(vocab_itos)

    caption_model = VisionEncoderDecoder(image_size=config.IMAGE_HEIGHT, channels_in=config.NUM_CHANNELS, 
                                        num_emb=vocab_size, patch_size=config.PATCH_SIZE, 
                                        num_layers=config.NUM_LAYER, hidden_size=config.HIDDEN_SIZE, 
                                        num_heads=config.NUM_HEADS).to(device)


    cp = torch.load("captioning_model_new.pt")
    caption_model.load_state_dict(cp["model_state_dict"])

    sos_token = vocab_stoi["<SOS>"] * torch.ones(1, 1).long()

    temp = 0.3
    log_tokens = [sos_token]

    caption_model.eval()

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            image_embedding = caption_model.encoder(transformed_image.to(device))



    log_tokens = [sos_token]
    for i in range(30):

        input_tokens = torch.cat(log_tokens, 1)
        data_pred = caption_model.decoder(input_tokens.to(device), image_embedding)
        
        dist = Categorical(logits=data_pred[:, -1] / temp)
        next_tokens = dist.sample().reshape(1, 1)
        log_tokens.append(next_tokens.cpu())
        
        if next_tokens.item() == vocab_stoi["<EOS>"]:
            break

    pred_text = torch.cat(log_tokens, 1)

    words = [vocab_itos[int(token.item())] for token in pred_text[0]] 
    words = words[1:-1]
    words[0] = words[0].capitalize()
    sentence = ' '.join(words)
    
    translated_sentence = gs.translate(sentence, target_language)
    print(sentence)
    print(translated_sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Путь к изображению.")
    parser.add_argument("-l", type=str, default='ru', help="Код языка, на который нужно перевести (например, ru).")

    args = parser.parse_args()

    image_path = args.image_path
    target_language = args.l

    test = tesing_model(image_path, target_language)

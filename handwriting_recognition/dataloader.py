
import spacy
import os  
import pandas as pd 
import torch
from torch.nn.utils.rnn import pad_sequence  
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
import config
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
spacy_rus = spacy.load("ru_core_news_sm")  


class Vocabulary:
    
    
    def __init__(self):
        
        self.itos_file = "itos.pkl"
        self.stoi_file = "stoi.pkl"

        with open('itos.json', 'r', encoding='utf-8') as f:
            self.itos = json.load(f)
        self.itos = {int(k): v for k, v in self.itos.items()}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_ru(text):
        return list(str(text))

    

    def numericalize(self, text):
        tokenized_text = self.tokenizer_ru(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
    def denumericalize(self, numerical_sequence):
        return ''.join([self.itos[idx] for idx in numerical_sequence])



class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
    
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, sep='\t', header=None, names=['image', 'caption'])
        self.transform = transform
        self.vocab = vocab
        self.imgs = self.df["image"]  
        self.captions = self.df["caption"]  

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]  
        img_name = self.imgs[index] 
        img_path = os.path.join(self.root_dir, img_name)  
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)  

        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]  
        numericalized_caption += self.vocab.numericalize(caption)  
        numericalized_caption.append(self.vocab.stoi["<EOS>"])  

        return img, torch.tensor(numericalized_caption)  

class AlbumentationsTransform:
    def __init__(self):
        self.transforms = A.Compose([
        A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Rotate(limit=10, p=0.03),  
        A.Affine(scale=(0.85, 1.15), translate_percent=(0.09, 0.09), rotate=(-5, 5), shear=(-5, 5), p=0.03),  
        A.GaussianBlur(blur_limit=(3, 3), p=0.02),  
        A.OneOf([
            A.GaussNoise(per_channel=True, p=0.02),  
            A.CoarseDropout(p=0.02),  
        ], p=0.03), 
        A.Perspective(scale=(0.0, 0.2), keep_size=True, p=0.02),  
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.03), 
        A.ElasticTransform(alpha=0.5, sigma=5, p=0.05),
        A.MotionBlur(blur_limit=(3, 5), p=0.03),
        A.CLAHE(clip_limit=2.0, p=0.03),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.03),
        A.CoarseDropout(p=0.03),
        ToTensorV2()
    ])
        
    def __call__(self, img):
        img = np.array(img)
        augmented = self.transforms(image=img)
        return augmented['image']
    

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
    root_dir,  
    annotation_file,
    vocab,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    shuffle=True,
    pin_memory=False,
):
    transform = AlbumentationsTransform()
    dataset = FlickrDataset(root_dir, annotation_file, vocab, transform=transform)
    
    pad_idx = vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader

      
from diffusion_model import DiT
from dataloader_alternative import LatentDataset
import torch
import torch.nn.functional as F
import torch.optim as optim
import config
from tqdm import tqdm
from diffusers import AutoencoderKL
from PIL import Image
from utils import cosine_alphas_bar, count_parameters
from utils import cold_diffuse
import lpips
import torchvision

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dit = DiT(
            image_size=config.IMAGE_SIZE,
            channels_in=config.CHANNELS_IN,
            patch_size=config.PATCH_SIZE,
            highres_patch_size=config.HIGHRES_PATCH_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_features=config.NUM_FEATURES,
            num_layers=config.NUM_LAYERS,
            num_heads=config.NUM_HEADS
        ).to(device) 
    
    vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
    use_safetensors=True
    ).to(device)

    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    lpips_weight = 1

    total, trainable = count_parameters(dit)
    print(f"Общее количество параметров: {total:,}")
    print(f"Количество обучаемых параметров: {trainable:,}")

    timesteps = config.TIMESTEPS
    
    total_steps = 500
   
    train_epoch = config.TRAIN_EPOCH
    lr = config.LR 
    load_model = config.LOAD_MODEL
    checkpoint_path = config.CHECKPOINT_PATH

    optimizer = optim.AdamW(dit.parameters(), lr=config.LR)
    criterion = F.l1_loss
    scaler = torch.amp.GradScaler(device)

    start_epoch = 0
    loss_log = []

    if config.LOAD_MODEL:
        cp = torch.load(checkpoint_path)
        dit.load_state_dict(cp["model_state_dict"])
        loss_log = cp["train_data_logger"]
        start_epoch = cp["epoch"]
        print(f"Loaded model from epoch {start_epoch-1}")

    alphas = torch.flip(cosine_alphas_bar(timesteps), (0,)).to(device)

    batch_size = config.BATCH_SIZE
    data_set_root = config.DATASET_ROOT
    image_dir = config.IMAGE_DIR 
    trainset = LatentDataset(data_set_root, image_dir) 
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    dit.train()
    for epoch in range(start_epoch, train_epoch):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        mean_loss = 0
        mean_lpips_loss = 0

        for num_iter, (latents, images) in enumerate(progress_bar):
            
            latents = latents.to(device)
            latents = latents.squeeze(dim=1)
            images = images.to(device, dtype=torch.float16) 
            
            bs = latents.shape[0]

            rand_index = torch.randint(0, timesteps, (bs,), device=device)
            random_sample = torch.randn_like(latents)
            alpha_batch = alphas[rand_index].reshape(bs, 1, 1, 1)

            noise_input = alpha_batch.sqrt() * latents + (1 - alpha_batch).sqrt() * random_sample

            with torch.amp.autocast("cuda", dtype=torch.float16):
                latent_pred = dit(noise_input, rand_index)
                latent_pred = latent_pred.to(dtype=torch.float16)
                img_gen = vae.decode(latent_pred / 0.13025).sample 
                lpips_loss = loss_fn_lpips(img_gen, images).mean() 
                mse_loss = criterion(latent_pred, latents)
                loss = mse_loss + lpips_weight * lpips_loss 
                
            scaler.scale(loss).backward() 
            torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm=1.0)
            
            scaler.step(optimizer) 
            scaler.update()
            optimizer.zero_grad()    

            loss_log.append(loss.item())  
            mean_loss += loss.item()  
            mean_lpips_loss += lpips_loss.item()

            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': mean_loss / (num_iter + 1),
                'lr': optimizer.param_groups[0]['lr']
            })

        torch.save({
            'epoch': epoch + 1,
            'train_data_logger': loss_log,
            'model_state_dict': dit.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"final_vgg{epoch+1}.pt")
        latent_noise = 0.8 * torch.randn(6, 4, 12, 12, device=device)  
        latent_noise = latent_noise.to(dtype=torch.float32)
        with torch.no_grad():
            fake_latents = cold_diffuse(dit, latent_noise, total_steps=total_steps)
            fake_latents = fake_latents.to(dtype=torch.float16)
            fake_samples = vae.decode(fake_latents / 0.13025).sample

       
        grid = torchvision.utils.make_grid(
            (fake_samples * 0.5 + 0.5).clip(0, 1), 
            nrow=3  
        )
        decoded_output = (grid.permute(1, 2, 0).cpu().float() * 255).to(torch.uint8)
        decoded_image = Image.fromarray(decoded_output.numpy())
        decoded_image.save(f'final_vgg{epoch+1}.jpg')
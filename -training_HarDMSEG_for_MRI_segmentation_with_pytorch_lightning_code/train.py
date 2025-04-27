import torch
import pytorch_lightning as pl
from dataset import DataModule
import config
from HarDMSEG import HarDMSEG
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from utils import load_checkpoint


torch.set_float32_matmul_precision("medium") 

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="HarDMSEG")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=20),
    )
    model = HarDMSEG(
    )

    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    dm = DataModule(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, config.VAL_IMG_DIR,
        config.VAL_MASK_DIR, config.BATCH_SIZE, config.NUM_WORKERS
    )
    trainer = pl.Trainer(
        profiler=profiler,
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
    )
   
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    

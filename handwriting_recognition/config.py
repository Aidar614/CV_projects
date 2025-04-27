import os
#  Training hyperparameters

LEARNING_RATE = 0.0001
BATCH_SIZE = 12
NUM_EPOCHS = 100

IMAGE_SIZE = (80, 272)

HIGHRES_PATCH_SIZE = 8
HIDDEN_SIZE = 512
NUM_LAYER = (3, 3) #  (Encoder, Decoder)
NUM_HEAD = 8
DROPOUT = 0.1

PIN_MEMORY = True

LOAD_MODEL = True
MODEL_PATH = 'model_epoch_6.pth'

NUM_WORKERS = 0#os.cpu_count()


# Training hyperparameters
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 10
NUM_EPOCHS = 3
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "I:/VS_CodeProjects/U-net_implementation/images/images"
TRAIN_MASK_DIR = "I:/VS_CodeProjects/U-net_implementation/masks/masks"
VAL_IMG_DIR = "I:/VS_CodeProjects/U-net_implementation/test/images"
VAL_MASK_DIR = "I:/VS_CodeProjects/U-net_implementation/test/masks"

NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16

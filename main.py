from MEGABYTE import MEGABYTE
from config import CONFIG, configure_device, set_seed
from data import load_shakespeare, preprocess, train_validation_split, TextDataset
from train import train

def main():
    set_seed(CONFIG.seed)
    configure_device()
    model = MEGABYTE(CONFIG)

if __name__ == '__main__':
    main()

import math

config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'select_all': True,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'num_epochs': 10,  # Number of epochs.
    'batch_size': 128,
    'learning_rate': 0.1,
    'early_stop': 600,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './Vgg/best_model.pth',  # Your model will be saved here.
    'best_loss': math.inf,
    'img_size': 224,
    'workers': 4,
}
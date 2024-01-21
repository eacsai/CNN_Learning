import math

config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'select_all': True,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 10,  # Number of epochs.
    'batch_size': 256,
    'learning_rate': 0.05,
    'early_stop': 600,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt',  # Your model will be saved here.
    'best_loss': math.inf
}
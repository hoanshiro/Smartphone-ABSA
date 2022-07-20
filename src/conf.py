import torch

class AbsaConfig():
    def __init__(self):
        self.rdrsegmenter_path = 'VnCoreNLP/VnCoreNLP-1.1.1.jar'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Hyperparameters
        self.mode_path = 'vinai/phobert-base'
        self.MAX_LEN = 128
        self.TRAIN_BATCH_SIZE = 32
        self.VALID_BATCH_SIZE = 32
        self.TEST_BATCH_SIZE = 32
        self.EPOCHS = 10
        self.LEARNING_RATE = 1e-05
        self.THRESHOLD = 0.5  # threshold for the sigmoid
        self.data_dir = 'data/model'


from .trainer import Trainer

class ClassicalTrainer(Trainer):
    def __init__(self, model):
        super().__init__(model)

    def train(self):
        print('Classical training (placeholder)')


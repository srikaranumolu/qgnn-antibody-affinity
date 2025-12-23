from .trainer import Trainer

class QuantumTrainer(Trainer):
    def __init__(self, model):
        super().__init__(model)

    def train(self):
        print('Quantum training (placeholder)')


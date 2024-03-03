from lion_pytorch import Lion
from sklearn.model_selection import train_test_split
from custom_metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss, total_loss
import os
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
from model import custom_model
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torchsummary import summary
import torch.optim as optim
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class DataModule(pl.LightningDataModule):
    def __init__(self, train, val, test):
        super().__init__()
        self.train_data = train
        self.val_data = val
        self.test_data = test

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.val_data

    def test_dataloader(self):
        return self.test_data

class customModel(pl.LightningModule):
    def __init__(self, parameters):
        super().__init__()
        self.model = parameters['custom_model']  
        self.criterion = parameters['criterion']
        self.metrics = parameters['metrics']
        self.starter_learning_rate = parameters['starter_learning_rate']
        self.end_learning_rate = parameters['end_learning_rate']
        self.decay_steps = parameters['decay_steps']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y,y_hat)
        self.log('train_loss', loss)
        for name, metric in self.metrics.items():
            self.log(f'train_{name}', metric(y,y_hat))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y,y_hat)
        self.log('val_loss', loss)
        for name, metric in self.metrics.items():
            self.log(f'val_{name}', metric(y,y_hat))
        return loss

    def configure_optimizers(self):
        #weight_decay_lambda = lambda step: ((self.starter_learning_rate - self.end_learning_rate) *
        #                        (1 - step / self.decay_steps) ** 0.2
        #                        ) + self.end_learning_rate
        optimizer = AdamW(self.model.parameters(), lr=self.starter_learning_rate, weight_decay=0.01)
        reduce_lr_callback = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2, min_lr=self.end_learning_rate),
            'monitor': 'val_loss'
        }
        return [optimizer], [reduce_lr_callback]

img_size = 256
BATCH_SIZE = 1
SEED = 42
save_path = "best_model.pth"

valid_size = 0.1
test_size = 0.15
epochs = 350
save_weights_only = True
starter_learning_rate = 1e-4
end_learning_rate = 1e-6
decay_steps = 1000


route = 'TrainDataset'
X_path = 'TrainDataset/images/'
Y_path = 'TrainDataset/masks/'

# route = './Kvasir-SEG'
# X_path = './Kvasir-SEG/images/'
# Y_path = './Kvasir-SEG/masks/'
valid_extensions = ['.jpg', '.jpeg', '.png']
#X_full = sorted(os.listdir(f'{route}/images'))
#Y_full = sorted(os.listdir(f'{route}/masks'))
X_full = sorted([file for file in os.listdir(f'{route}/images') if file.lower().endswith(('.jpg', '.jpeg', '.png'))])
Y_full = sorted([file for file in os.listdir(f'{route}/masks') if file.lower().endswith(('.jpg', '.jpeg', '.png'))])
X_train, X_valid = train_test_split(X_full, test_size=valid_size, random_state=SEED)
Y_train, Y_valid = train_test_split(Y_full, test_size=valid_size, random_state=SEED)

X_train, X_test = train_test_split(X_train, test_size=test_size, random_state=SEED)
Y_train, Y_test = train_test_split(Y_train, test_size=test_size, random_state=SEED)

X_train = [X_path + x for x in X_train]
X_valid = [X_path + x for x in X_valid]
X_test = [X_path + x for x in X_test]

Y_train = [Y_path + x for x in Y_train]
Y_valid = [Y_path + x for x in Y_valid]
Y_test = [Y_path + x for x in Y_test]

print("N Train:", len(X_train))
print("N Valid:", len(X_valid))
print("N test:", len(X_test))

train_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='png', segment=True, ext2='jpg')
train_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=train_decoder, 
                            augmentAdv=False, augment=False, augmentAdvSeg=True)

valid_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='png', segment=True, ext2='jpg')
valid_dataset = build_dataset(X_valid, Y_valid, bsize=BATCH_SIZE, decode_fn=valid_decoder, 
                            augmentAdv=False, augment=False, repeat=False, shuffle=False,
                            augmentAdvSeg=False)

test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='png', segment=True, ext2='jpg')
test_dataset = build_dataset(X_test, Y_test, bsize=BATCH_SIZE, decode_fn=test_decoder, 
                            augmentAdv=False, augment=False, repeat=False, shuffle=False,
                            augmentAdvSeg=False)

data_module = DataModule(train_dataset, valid_dataset, test_dataset)

model = custom_model(256,1).to(device=device)
summary(model.to(device), (3,256,256))
criterion = dice_loss
#Metrics: dice_coeff,bce_dice_loss, IoU, zero_IoU
metrics = {
    "dice_coeff": dice_coeff,
    "bce_dice_loss":bce_dice_loss,
    "IoU":IoU,
    "zero_IoU":zero_IoU
}
parameters = {
    'custom_model': model,
    'criterion': criterion,
    'metrics': metrics,
    'starter_learning_rate': starter_learning_rate,
    'end_learning_rate': end_learning_rate,
    'decay_steps': decay_steps
}
metapoly_model = customModel(parameters)

checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='./checkpoints', filename='best_model')
lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
csv_logger = CSVLogger('logs', name='csv_log')
tb_logger = TensorBoardLogger("tb_logs", name="tensorboard_log")

trainer = pl.Trainer(
    max_epochs=epochs,
    accelerator="auto",
    callbacks=[checkpoint_callback, lr_monitor_callback],
    logger=[csv_logger, tb_logger],
)

trainer.fit(metapoly_model, datamodule=data_module)

trainer.test(datamodule=data_module)
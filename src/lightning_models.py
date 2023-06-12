from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torch.optim

from nn import UNet


class HeatmapRegressor(LightningModule):
    """
    UNet model which can handle variable numbers of input channels.
    """

    def __init__(self, in_channels=1, num_classes=1, key=None, loss=F.mse_loss, lr=0.01, checkpoint=None):
        super().__init__()
        self.model = UNet(in_channels=in_channels, num_classes=num_classes)
        self.key = key
        self.loss = loss
        self.lr = lr

    def forward(self, x):
        # input of size (512, 512)
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        """
        In summary, the main differences between BCELoss and BCEWithLogitsLoss are:
        - BCELoss expects the input to the loss function to be in the range [0, 1]
        - BCEWithLogitsLoss expects the input to be the logits (i.e. unactivated outputs of the model).
        - BCELoss applies a stable log function to the loss calculatio
        - whereas BCEWithLogitsLoss applies an unstable log function.
        - This can make BCEWithLogitsLoss more numerically unstable, but this is not usually an issue in practice.
        :param batch:
        :param batch_idx:
        :return:
        """
        # get data and target values
        x = batch["image"]
        y = batch[self.key]

        # predict target values
        y_hat = self.model(x)

        # reshape target values
        y = y.reshape(y_hat.shape)
        return self.loss(y_hat, y)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["image"]
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


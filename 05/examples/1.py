# import torch

# from torch.utils.data import random_split


# random_split()

# from pytorch_lightning import Trainer
# from pytorch_lightning.demos import BoringModel, BoringDataModule

# model = BoringModel()
# data_radio = .75
# trainer = Trainer(
#     max_epochs=50,
#     limit_train_batches=data_radio,
#     limit_val_batches=data_radio,
#     limit_test_batches=data_radio,
#     limit_predict_batches=data_radio,
# )

# datamodule = BoringDataModule()
# trainer.fit(model, datamodule=datamodule)

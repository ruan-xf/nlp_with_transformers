from _shared import MyCLI

from data import MyDefaultDataModule
from model import NER_Model 


cli = MyCLI(
    NER_Model,
    MyDefaultDataModule,
    trainer_defaults={
        # 'fast_dev_run': True,
        'max_epochs': 10,
    }
)
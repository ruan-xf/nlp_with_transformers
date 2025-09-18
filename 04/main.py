from _shared import MyCLI
from model import SentencePairClassifier


# model = SentencePairClassifier()
# tensorboard = pl_loggers.TensorBoardLogger('tb_logs', None)
# data_used_radio = .15
# trainer = pl.Trainer(
#     fast_dev_run=True,
#     logger=tensorboard,
#     callbacks=[TensorBoardLauncher()],
#     log_every_n_steps=1,
#     limit_train_batches=data_used_radio,
#     limit_val_batches=data_used_radio,
#     limit_test_batches=data_used_radio,
#     max_epochs=6,
#     precision="bf16-mixed",  # 或 "bf16-mixed"（如果GPU支持）
#     accelerator="auto",
#     devices="auto",
# )

# trainer.fit(model)
# trainer.test(model)

cli = MyCLI(
    SentencePairClassifier,
    trainer_defaults={
        # 'fast_dev_run': True,
        'max_epochs': 6,
        'precision': "bf16-mixed",
    }
)
import os
import subprocess
import webbrowser
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import Callback

class TensorBoardLauncher(Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.fast_dev_run: return
        subprocess.Popen(
            ["tensorboard", "--logdir=tb_logs", "--port=6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        webbrowser.open("http://localhost:6006")


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.default_config_files.append(
            os.path.join(os.path.dirname(__file__), 'default.yaml')
        )

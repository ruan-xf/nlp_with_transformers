import subprocess


subprocess.Popen(
    ["tensorboard", "--logdir=tb_logs", "--port=6006"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
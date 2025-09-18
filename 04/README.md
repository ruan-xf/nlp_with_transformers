
使用pytorch lightning，预计将做出的改进：
- 超参数的设置、访问和保存
  - 进一步的，trainer的参数也要一并保存
- 完成混合精度训练
- 设置get_linear_schedule_with_warmup时需要用到的总步数用内置属性而不是自己算

## 各参数的保存
在 PyTorch Lightning 中，超参数管理主要通过 self.save_hyperparameters() 和 self.hparams 实现

trainer的参数也常改，也应该保存

学习cli
- [Auto Structuring Deep Learning Projects with the Lightning CLI | by Aaron (Ari) Bornstein | PyTorch Lightning Developer Blog](https://devblog.pytorchlightning.ai/auto-structuring-deep-learning-projects-with-the-lightning-cli-9f40f1ef8b36)
- [Configure hyperparameters from the CLI — PyTorch Lightning 2.5.2 documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)

用了cli后，最方便的就是从config启动fit，然后顺便把这次指定的参数通通保存下来了


## 混合精度训练
原生 PyTorch 混合精度训练
scaler = GradScaler()  # 手动梯度缩放
with autocast():       # 手动 autocast 上下文
    logits = net(...)
    loss = criterion(...)
scaler.scale(loss).backward()  # 手动缩放梯度
scaler.step(opti)              # 手动更新参数
scaler.update()                # 手动调整缩放器


而使用pl只需设置trainer的初始化参数


关于梯度累积：一定条件下，batchsize越大训练效果越好，梯度累加则实现了batchsize的变相扩大
- [【PyTorch】PyTorch中的梯度累加 - lart - 博客园](https://www.cnblogs.com/lart/p/11628696.html)

## 总训练步数的计算
由于：
- 使用的数据量
- batch size
- 梯度累积
等的改变，手动计算总训练步数会显得很烦人
好在pytorch lightning提供了内置属性 estimated_stepping_batches
可在 configure_optimizers 中直接访问

## 其他
关于fast_dev_run
- [PyTorch Lightning教程五：Debug调试_pytorch lightning怎么debug以及训练-CSDN博客](https://blog.csdn.net/qq_33952811/article/details/132073542)


```py
trainer.fit(model)
trainer.test(model)
```
这样就可以跑一个batch进行训练、验证和测试流程的测试
并且自动禁用logger

另外，由于tensorboard的启动是用on_train_start的callback实现的，而在该模式下还会执行
需要设置这时不启动tensorboard


关于data_module
- [LightningDataModule — PyTorch Lightning 2.5.2 documentation](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)

用法
```
mnist_dm = MNISTDataModule(data_dir="./data", batch_size=64)
model = LitClassifier()

trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule=mnist_dm)
trainer.test(datamodule=mnist_dm)
```

我也希望在本次任务中用这个模块把数据集准备的操作分离出来
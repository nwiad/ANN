# README

需要安装wandb：`pip install wandb

新增plot.py，导入到main.py中用于绘图

main.py新增命令行参数`batch_norm`和`dropout`用于设置是否使用batch normalization和dropout，并新增绘图和记录最终测试准确率的代码

实验操作：执行run_cnn.sh脚本，出现wandb选项时输入2，提示输入api key时输入9c66b8ab9e3f544bb192c6d3dce4401b8a4724bb，若wandb出现错误只需将main.py中含有wandb的语句注释掉即可
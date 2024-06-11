# README

## 关于额外修改的声明

我额外修改了run_mlp.py、solve_net.py、loss.py的代码。

run_mlp.py：增加了对命令行参数的处理和绘图操作。

solve_net.py：增加了辅助绘图和以文本形式记录关键数据的操作。

loss.py：修改了`FocalLoss`的构造函数以适应我对FocalLoss的调用与实现。

## 新增源文件

我新增了plot.py和plot_focalloss.py，其中plot.py用于绘制所有独立实验的图表，将会自动导入至run_mlp.py并被调用。plot_focalloss.py用于绘制不同参数取值的FocalLoss和SoftmaxCrossEntropyLoss的对比图，需要手动填入数据并执行。

## 代码的运行

只需在/codes目录下运行run.sh和plot_focalloss.py即可重现报告所述的所有实验。运行过程中产生results.txt和res目录。results.txt以文本形式保存了所有独立实验的关键数据。所有实验图表存储在res/目录下。
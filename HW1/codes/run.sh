# 1 hidden layer
# Selu/Swish/Gelu + MSE
echo "1 hidden layer" >> results.txt
echo "Selu+MSELoss" >> results.txt
python run_mlp.py --activation=Selu
echo "" >> results.txt

echo "Swish+MSELoss" >> results.txt
python run_mlp.py --activation=Swish
echo "" >> results.txt

echo "Gelu+MSELoss" >> results.txt
python run_mlp.py --activation=Gelu
echo "" >> results.txt

# Swish + SoftmaxCrossEntropyLoss/HingeLoss/FocalLoss
echo "Swish+SoftmaxCrossEntropyLoss" >> results.txt
python run_mlp.py --activation=Swish --loss=SoftmaxCrossEntropyLoss
echo "" >> results.txt

echo "Swish+HingeLoss" >> results.txt
python run_mlp.py --activation=Swish --loss=HingeLoss
echo "" >> results.txt

# 2 hidden layers
# Selu/Swish/Gelu + MSE
echo "2 hidden layers" >> results.txt
echo "Selu+MSELoss" >> results.txt
python run_mlp.py --activation=Selu --hidden_layers=2
echo "" >> results.txt

echo "Swish+MSELoss" >> results.txt
python run_mlp.py --activation=Swish --hidden_layers=2
echo "" >> results.txt

echo "Gelu+MSELoss" >> results.txt
python run_mlp.py --activation=Gelu --hidden_layers=2
echo "" >> results.txt

# Swish + SoftmaxCrossEntropyLoss/HingeLoss/FocalLoss
echo "Swish+SoftmaxCrossEntropyLoss" >> results.txt
python run_mlp.py --activation=Swish --loss=SoftmaxCrossEntropyLoss --hidden_layers=2
echo "" >> results.txt

echo "Swish+HingeLoss" >> results.txt
python run_mlp.py --activation=Swish --loss=HingeLoss --hidden_layers=2
echo "" >> results.txt

# ------------------------------------------------------------------------------------
# Selu+MSELoss+1, 100 epochs
echo "Selu+MSELoss+1, 100 epochs" >> results.txt
python run_mlp.py --activation=Selu --loss=MSELoss --epochs=100
echo "" >> results.txt

# ------------------------------------------------------------------------------------
# FocalLoss & SoftmaxCrossEntropyLoss
echo "Relu+SoftmaxCrossEntropyLoss+1" >> results.txt
python run_mlp.py --activation=Relu --loss=SoftmaxCrossEntropyLoss
echo "" >> results.txt

echo "Relu+FocalLoss+1+0.1" >> results.txt
python run_mlp.py --activation=Relu --loss=FocalLoss --alpha=0.1
echo "" >> results.txt

echo "Relu+FocalLoss+1+0.25" >> results.txt
python run_mlp.py --activation=Relu --loss=FocalLoss --alpha=0.25
echo "" >> results.txt

echo "Relu+FocalLoss+1+0.5" >> results.txt
python run_mlp.py --activation=Relu --loss=FocalLoss --alpha=0.5
echo "" >> results.txt

echo "Relu+FocalLoss+1+0.75" >> results.txt
python run_mlp.py --activation=Relu --loss=FocalLoss --alpha=0.75
echo "" >> results.txt

echo "Relu+FocalLoss+1+0.9" >> results.txt
python run_mlp.py --activation=Relu --loss=FocalLoss --alpha=0.9
echo "" >> results.txt

echo "Relu+SofmaxCrossEntropyLoss+1" >> results.txt
python run_mlp.py --activation=Relu --loss=SoftmaxCrossEntropyLoss
echo "" >> results.txt
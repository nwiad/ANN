# 训练 2 个基础模型
python main.py --name="Tfmr-scratch"

python main.py --name="Tfmr-finetune" --model_config=pretrained/model_3_layers/config.json --pretrain_dir=pretrained/model_3_layers

# 测试 8 种解码策略
python main.py --test="Tfmr-scratch" --decode_strategy="random" --temperature=1

python main.py --test="Tfmr-scratch" --decode_strategy="random" --temperature=0.7

python main.py --test="Tfmr-scratch" --decode_strategy="top-p" --top_p=0.9 --temperature=1

python main.py --test="Tfmr-scratch" --decode_strategy="top-p" --top_p=0.9 --temperature=0.7

python main.py --test="Tfmr-finetune" --decode_strategy="random" --temperature=1

python main.py --test="Tfmr-finetune" --decode_strategy="random" --temperature=0.7

python main.py --test="Tfmr-finetune" --decode_strategy="top-p" --top_p=0.9 --temperature=1

python main.py --test="Tfmr-finetune" --decode_strategy="top-p" --top_p=0.9 --temperature=0.7

# 选取预训练模型的 1、6、12 层，训练后会直接进行测试
python main.py --name="Tfmr-finetune-1-6-12" --model_config=pretrained/model_12_layers/config.json --pretrain_dir=pretrained/model_12_layers

# 测试不同注意力头数
python main.py --name="Tfmr-scratch_6" --model_config="config_6.json"

python main.py --test="Tfmr-scratch_6"

python main.py --name="Tfmr-scratch_16" --model_config="config_16.json"

python main.py --test="Tfmr-scratch_16"

python main.py --name="Tfmr-scratch_24" --model_config="config_24.json"

python main.py --test="Tfmr-scratch_24"


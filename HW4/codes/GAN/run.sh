############ train
echo "train latent_dim=16, hidden_dim=16"
python main.py --do_train --latent_dim=16 --generator_hidden_dim=16 --discriminator_hidden_dim=16
# 76.450
echo "train latent_dim=16, hidden_dim=100"
python main.py --do_train --latent_dim=16 --generator_hidden_dim=100 --discriminator_hidden_dim=100
# 60.943
echo "train latent_dim=100, hidden_dim=16"
python main.py --do_train --latent_dim=100 --generator_hidden_dim=16 --discriminator_hidden_dim=16
# 67.694
echo "train latent_dim=100, hidden_dim=100"
python main.py --do_train --latent_dim=100 --generator_hidden_dim=100 --discriminator_hidden_dim=100
# 44.026

############ 10000 steps
echo "latent_dim=100, hidden_dim=100, num_training_steps=10000"
python main.py --do_train --latent_dim=100 --generator_hidden_dim=100 --discriminator_hidden_dim=100 --num_training_steps=10000
# 76.084

############ linear interpolation and random generation
export linear_interpolation=1 random_generation=1
echo "linear interpolation latent_dim=100, hidden_dim=100"
python main.py --latent_dim=100 --generator_hidden_dim=100 --discriminator_hidden_dim=100
export linear_interpolation=0 random_generation=0
# 41.219

############ MLP-based GAN
export MLP=1
echo "train MLP-based latent_dim=100, hidden_dim=100"
python main.py --do_train --latent_dim=100 --generator_hidden_dim=100 --discriminator_hidden_dim=100 --ckpt_dir="results_mlp" --log_dir="runs_mlp"
export MLP=0
# 419.622

export remove_bn=1
echo "train CNN-based remove BatchNorm latent_dim=100, hidden_dim=100"
python main.py --do_train --latent_dim=100 --generator_hidden_dim=100 --discriminator_hidden_dim=100 --ckpt_dir="results_remove" --log_dir="runs_remove"
export remove_bn=0
# 55.902
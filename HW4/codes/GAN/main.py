import GAN
from trainer import Trainer
from dataset import Dataset
from tensorboardX import SummaryWriter

from pytorch_fid import fid_score

import torch
import torch.optim as optim
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true') # 是否训练
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--latent_dim', default=16, type=int) # 隐变量维度
    parser.add_argument('--generator_hidden_dim', default=16, type=int) # 生成器隐藏层维度
    parser.add_argument('--discriminator_hidden_dim', default=16, type=int) # 判别器隐藏层维度
    parser.add_argument('--batch_size', default=64, type=int) # 批大小
    parser.add_argument('--num_training_steps', default=5000, type=int) # 训练步数
    parser.add_argument('--logging_steps', type=int, default=10) # 记录步数
    parser.add_argument('--saving_steps', type=int, default=1000) # 保存步数
    parser.add_argument('--learning_rate', default=0.0002, type=float) # 学习率
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--data_dir', default='../data', type=str, help='The path of the data directory') # 数据集路径
    parser.add_argument('--ckpt_dir', default='results', type=str, help='The path of the checkpoint directory') # 检查点路径
    parser.add_argument('--log_dir', default='./runs', type=str) # 日志路径
    args = parser.parse_args()

    if not os.path.exists("./pics/"):
        os.makedirs("./pics/")

    # config = 'z-{}_batch-{}_num-train-steps-{}'.format(args.latent_dim, args.batch_size, args.num_training_steps)
    config = 'latent_{}_hidden_{}_steps_{}'.format(args.latent_dim, args.generator_hidden_dim, args.num_training_steps)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = Dataset(args.batch_size, args.data_dir)
    netG = GAN.get_generator(1, args.latent_dim, args.generator_hidden_dim, device)
    netD = GAN.get_discriminator(1, args.discriminator_hidden_dim, device)
    tb_writer = SummaryWriter(args.log_dir)

    if args.do_train:
        optimG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        optimD = optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        trainer = Trainer(device, netG, netD, optimG, optimD, dataset, args.ckpt_dir, tb_writer)
        trainer.train(args.num_training_steps, args.logging_steps, args.saving_steps)

    restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir))))
    netG.restore(restore_ckpt_path)

    num_samples = 3000
    real_imgs = None
    real_dl = iter(dataset.training_loader)
    while real_imgs is None or real_imgs.size(0) < num_samples:
        imgs = next(real_dl)
        if real_imgs is None:
            real_imgs = imgs[0]
        else:
            real_imgs = torch.cat((real_imgs, imgs[0]), 0)
    real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5

    with torch.no_grad():
        samples = None
        while samples is None or samples.size(0) < num_samples:
            imgs = netG.forward(torch.randn(args.batch_size, netG.latent_dim, 1, 1, device=device))
            if samples is None:
                samples = imgs
            else:
                samples = torch.cat((samples, imgs), 0)
    samples = samples[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
    samples = samples.cpu()

    fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
    tb_writer.add_scalar('fid', fid)
    print("FID score: {:.3f}".format(fid), flush=True)

    from torchvision.utils import make_grid, save_image
    # linear interpolation
    if os.getenv('linear_interpolation') == '1':
        print('linear interpolation')
        all_img_list = []
        for _ in range(5):
            z1 = torch.randn(1, netG.latent_dim, 1, 1, device=device)
            z2 = torch.randn(1, netG.latent_dim, 1, 1, device=device)
            K = 9
            img_list = []
            for i in range(K + 1): # i = 0, 1, ..., K(=9)
                z = z1 + (z2 - z1) * i / K # z = z1, z1 + (z2 - z1) / K, ..., z2
                img_list.append(netG.forward(z))
            all_img_list.append(torch.cat(img_list, 0))
        output_imgs = make_grid(torch.cat(all_img_list, 0), nrow=K + 1) * 0.5 + 0.5
        save_image(output_imgs, "./pics/" + config + "_interpolation.png")

    # random generation
    if os.getenv('random_generation') == '1':
        print('random generation')
        all_img_list = []
        for _ in range(5):
            img_list = []
            for _ in range(10):
                z = torch.randn(1, netG.latent_dim, 1, 1, device=device)
                img_list.append(netG.forward(z))
            all_img_list.append(torch.cat(img_list, 0))
        output_imgs = make_grid(torch.cat(all_img_list, 0), nrow=10) * 0.5 + 0.5
        save_image(output_imgs, "./pics/" + config + "_random_generation.png")
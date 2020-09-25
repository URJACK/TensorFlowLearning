import env_tf_2_3.pytorch_learn.gan_picture.dataset as datasetModule
import env_tf_2_3.pytorch_learn.gan_picture.model as model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import os
import matplotlib.pyplot as plt

GPATH = os.path.join(model.WORKSPACEDIR, f'gan_g.pth')
DPATH = os.path.join(model.WORKSPACEDIR, f'gan_d.pth')


def main():
    batch_size = 64
    z_dim = 100
    lr = 1e-4
    n_epoch = 10
    save_dir = os.path.join(model.WORKSPACEDIR, 'logs')
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)

    G = model.Generator(z_dim).cuda()
    D = model.Discriminator(3).cuda()
    G.train()
    D.train()

    criterion = nn.BCELoss()

    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    datasetModule.same_seeds(0)
    dataset = datasetModule.get_dataset(model.DATADIR)
    # plt.imshow(dataset[10].numpy().transpose(1, 2, 0))
    # plt.show()
    dataloader = datasetModule.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    checkpoint_generator = torch.load(GPATH)
    checkpoint_discriminator = torch.load(DPATH)
    G.load_state_dict(checkpoint_generator)
    D.load_state_dict(checkpoint_discriminator)

    z_sample = Variable(torch.randn(100, z_dim)).cuda()
    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data.cuda()

            bs = imgs.size(0)

            """ Train D """
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)

            r_label = torch.ones(bs).cuda()
            f_label = torch.zeros(bs).cuda()

            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())

            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            z = Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs = G(z)

            f_logit = D(f_imgs)
            loss_G = criterion(f_logit, r_label)

            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            print(
                f'\rEpoch [{epoch + 1} / {n_epoch}] {i + 1} / {len(dataloader)} loss_D:{loss_D.item():.4f} loss_G:{loss_G.item():.4f}',
                end='')
        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(save_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
        # show generated image
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        G.train()
        if (e + 1) % 5 == 0:
            torch.save(G.state_dict(), GPATH)
            torch.save(D.state_dict(), DPATH)


if __name__ == '__main__':
    main()

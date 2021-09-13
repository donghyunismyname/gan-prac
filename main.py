import torch
import torchvision
from tqdm import tqdm
import cv2

BATCH_SIZE = 100
DATA_ROOT = 'data_MNIST/'
DATA_W = 28
DATA_H = 28
DATA_DIM = DATA_W * DATA_H
SEED_DIM = 10
PATH_DATA_SYNTHETIC = 'img_synthetic/'


def write_as_image(img, num):
    path = PATH_DATA_SYNTHETIC + f'{num:04d}.png'
    img = img.detach().view(DATA_W, DATA_H).numpy() * 256
    cv2.imwrite(path, img)


def main():
    data = torchvision.datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    

    



    discriminator = torch.nn.Sequential(
        torch.nn.Linear(DATA_DIM, 1),
        torch.nn.Sigmoid()
    )
    discriminator_loss_fn = torch.nn.CrossEntropyLoss()

    generator = torch.nn.Sequential(
        torch.nn.Linear(SEED_DIM, DATA_DIM),
        torch.nn.Sigmoid()
    )
    generator_loss_fn = torch.nn.CrossEntropyLoss()
    


    for epoch in tqdm(range(10)):
        data_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True)

    





if __name__ == '__main__':
    main()
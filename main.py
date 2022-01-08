import torch
import torchvision
from tqdm import tqdm
import cv2

BATCH_SIZE = 100
DATA_ROOT = 'data_MNIST/'
PATH_DATA_SYNTHETIC = 'img_synthetic/'

DATA_W = 28
DATA_H = 28
DATA_DIM = DATA_W * DATA_H
SEED_DIM = 2
HIDDEN_DIM = 80

EPOCH = 20
EPOCH_DIS = 1
EPOCH_GEN = 200



device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def write_as_image(img, num):
    path = PATH_DATA_SYNTHETIC + f'{num:04d}.png'
    img = img.detach().view(DATA_W, DATA_H).numpy() * 256
    cv2.imwrite(path, img)


def main():
    real_data = torchvision.datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    real_data = [(x.view(-1), torch.tensor(1, dtype=torch.float32)) for x,_ in real_data]

    bce_loss = torch.nn.BCELoss()
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(DATA_DIM, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, 1),
        torch.nn.Sigmoid()
    )
    generator = torch.nn.Sequential(
        torch.nn.Linear(SEED_DIM, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, DATA_DIM),
        torch.nn.Sigmoid()
    )

    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_generator = torch.optim.Adam(generator.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in tqdm(range(EPOCH)):
        # Discriminator training
        print(epoch, 'Training discriminator...', end=' ')

        seed = torch.rand(len(real_data), SEED_DIM)
        fake_data = [(x.detach(), torch.tensor(0, dtype=torch.float32)) for x in generator(seed)]
        data_loader = torch.utils.data.DataLoader(
            dataset = real_data + fake_data,
            batch_size = BATCH_SIZE,
            shuffle = True,
            drop_last = True)

        cnt = 0
        for x,y in data_loader:
            pred = discriminator(x).view(BATCH_SIZE)
            cnt += sum(abs(pred - y) < 0.5)
        accuracy = cnt / len(data_loader.dataset)
        print(accuracy, end=' ')

        for depoch in range(EPOCH_DIS):
            for x,y in data_loader:
                pred = discriminator(x).view(BATCH_SIZE)
                loss = bce_loss(pred, y)

                opt_discriminator.zero_grad()
                loss.backward()
                opt_discriminator.step()
        
        cnt = 0
        for x,y in data_loader:
            pred = discriminator(x).view(BATCH_SIZE)
            cnt += sum(abs(pred - y) < 0.5)
        accuracy = cnt / len(data_loader.dataset)
        print(accuracy)

        
        
        # Generator training
        print(epoch, 'Training generator...')
        for gepoch in range(EPOCH_GEN):
            seed = torch.rand(BATCH_SIZE, SEED_DIM)
            pred = discriminator(generator(seed)).view(BATCH_SIZE)
            loss = bce_loss(pred, torch.ones(BATCH_SIZE))

            opt_generator.zero_grad()
            loss.backward()
            opt_generator.step()

        seed = torch.rand(SEED_DIM)
        img = generator(seed)
        write_as_image(img, epoch)


    # for i in range(10):
    #     seed = torch.rand(SEED_DIM)
    #     img = generator(seed)
    #     write_as_image(img, i)
        
        
    




if __name__ == '__main__':
    main()
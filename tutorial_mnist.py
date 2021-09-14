import torch
import torchvision
from tqdm import tqdm
import cv2

BATCH_SIZE = 1000
DATA_ROOT = 'data_MNIST/'
DATA_DIM = 784
NUM_CLASS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def main():
    mnist_train = torchvision.datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    mnist_test = torchvision.datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    for i in range(100):
        x,y = mnist_train[i]
        x = x.squeeze().numpy()
        x = x * 256
        cv2.imwrite(f"img/mnist{i:04}.png", x)
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=[(x.view(-1),y) for x,y in mnist_train],
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=[(x.view(-1),y) for x,y in mnist_test],
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True)

    lin = torch.nn.Linear(DATA_DIM, NUM_CLASS)
    model = torch.nn.Sequential(
        lin,
        torch.nn.Softmax(dim = -1))
    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for epoch in tqdm(range(10)):
        for x, y in train_data_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cnt_correct = 0
        for x, y in test_data_loader:
            pred = model(x)
            cnt_correct += sum(pred.argmax(1) == y)
        print(f"accuracy {cnt_correct/len(test_data_loader.dataset)}")






if __name__ == '__main__':
    main()  
import numpy as np
import torchvision
import cv2

DATA_ROOT = 'data_MNIST/'
PATH_DATA_SYNTHETIC = 'img_synthetic/'
DATA_W = 28
DATA_H = 28
DATA_DIM = DATA_W * DATA_H


def write_as_image(img, num):
    path = PATH_DATA_SYNTHETIC + f'{num:04d}.png'
    img = img.reshape(DATA_H, DATA_W) * 256
    cv2.imwrite(path, img)


def main():
    real_data = torchvision.datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    X = np.array([p[0].numpy().reshape(-1) for p in real_data])
    mu = np.mean(X, axis=0)

    X -= mu
    cov = np.dot(X.T, X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)

    N = 30
    U = U[:, :N]
    #X_proj = np.dot(np.dot(X,U),U.T) / (np.sqrt(S) + 1e-2)
    X_proj = np.dot(np.dot(X, U) / (np.sqrt(S[:N]) + 1e-2), U.T)

    scaling = np.mean(np.linalg.norm(X, axis=1))
    for i in range(30):
        #img = mu + U[:,i]*scaling
        img0 = mu + X[i]
        img1 = mu + X_proj[i]
        write_as_image(img0, i*2)
        write_as_image(img1, i*2+1)


if __name__ == '__main__':
    main()

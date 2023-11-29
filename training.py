import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import time
from PIL import Image
import os
import cv2


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2535, 5)
        )

    def forward(self, x):
        return self.model(x)


class CustomDataset(Dataset):
    def __init__(self, root_dir1, transform=None):
        self.root_dir1 = root_dir1
        self.transform = transform
        self.data = []
        self.targets = []

        self.load_data()

    def load_data(self):
        # Assuming your data is organized in folders, each representing a class
        folder = self.root_dir1
        for filename in os.listdir(folder):
            if filename != '.DS_Store':
                img_path = os.path.join(folder, filename)
                image = cv2.imread(img_path)
                desired_height, desired_width = 125, 50
                image = cv2.resize(image, (desired_width, desired_height))
                label = filename.split('-')[0]
                self.data.append(image)
                self.targets.append(label)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class TrainClassifier:
    def __init__(self, model, batch_size=32, learning_rate=1e-3, num_epochs=5):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

        self.train_dataset, self.train_loader, self.test_dataset, self.test_loader = self.load_data()

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = CustomDataset(root_dir1='train_in_spots', transform=transform)
        test_dataset = CustomDataset(root_dir1='train_in_spots', transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_dataset, train_loader, test_dataset, test_loader

    def train_batch(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        batch_loss = self.loss_func(self.model(x), y)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.item()

    @torch.no_grad()
    def accuracy(self, x, y):
        self.model.eval()
        prediction = self.model(x)
        argmaxes = prediction.argmax(dim=1)
        s = torch.sum((argmaxes == y).float()) / len(y)
        return s.item()


    def train(self):
        train_losses, train_accuracies, test_losses, test_accuracies, time_per_epoch = [], [], [], [], []
        
        for epoch in range(self.num_epochs):
            print(f"Running epoch {epoch + 1} of {self.num_epochs}")
            start_time = time.time()

            epoch_losses, epoch_accuracies = [], []
            print("START")
            print(self.train_loader)
            for batch in self.train_loader:
                x, y = batch
                batch_loss = self.train_batch(x, y)
                epoch_losses.append(batch_loss)
                batch_acc = self.accuracy(x, y)
                epoch_accuracies.append(batch_acc)
            print("DONE")
            train_losses.append(np.mean(epoch_losses))
            train_accuracies.append(np.mean(epoch_accuracies))
            print("THERE")
            epoch_test_accuracies, epoch_test_losses = [], []
            for ix, batch in enumerate(iter(self.test_loader)):
                x, y = batch
                test_loss = self.train_batch(x, y)
                test_acc = self.accuracy(x, y)
                epoch_test_accuracies.append(test_acc)
                epoch_test_losses.append(test_loss)

            test_losses.append(np.mean(epoch_test_losses))
            test_accuracies.append(np.mean(epoch_test_accuracies))

            end_time = time.time()
            time_per_epoch.append(end_time - start_time)

    

def MLP_CNN_experiment():
    cnn_model = CNN()
    cnn_classifier = TrainClassifier(cnn_model.model)
    cnn_classifier.train()


if __name__ == "__main__":
    MLP_CNN_experiment()
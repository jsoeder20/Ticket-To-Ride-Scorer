import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split
import time
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import torchvision


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(21960, 6)
            # nn.Conv2d(3, 32, kernel_size=3),
            # nn.Conv2d(32, 32, kernel_size=3),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(32, 64, kernel_size=3),
            # nn.Conv2d(64, 64, kernel_size=3),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Flatten(),
            # nn.Linear(16128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 6),
        )

    def forward(self, x):
        return self.model(x)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.targets = []

        self.load_data()

    def load_data(self):
        # Assuming your data is organized in folders, each representing a class
        big_folder = self.root_dir
        for folder in os.listdir(big_folder):
            if folder != '.DS_Store':
                folder_path = os.path.join(big_folder, folder)
                for filename in os.listdir(folder_path):
                    if filename != '.DS_Store':
                        img_path = os.path.join(folder_path, filename)
                        image = cv2.imread(img_path)
                        desired_height, desired_width = 50, 125
                        image = cv2.resize(image, (desired_width, desired_height))
                        label = filename.split('-')[0]
                        self.data.append(image)
                        if (label == 'blue'):
                            self.targets.append(1)
                        elif (label == 'black'):
                            self.targets.append(2)
                        elif (label == 'green'):
                            self.targets.append(3)
                        elif (label == 'red'):
                            self.targets.append(4)
                        elif (label == 'yellow'):
                            self.targets.append(5)
                        else:
                            self.targets.append(0)

    def split_data(self, train_percentage=0.7):
        dataset_size = len(self.data)
        train_size = int(train_percentage * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        return train_dataset, test_dataset
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class TrainClassifier:
    def __init__(self, model, batch_size=32, learning_rate=1e-3, num_epochs=10):
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

        custom_dataset = CustomDataset(root_dir='test_train_data', transform=transform)
        train_dataset, test_dataset = custom_dataset.split_data()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_dataset, train_loader, test_dataset, test_loader

    def train_batch(self, x, y):
        # print("TRAIN")
        self.model.train()
        # print("1")
        self.optimizer.zero_grad()
        # print("2")
        # print(x.shape)
        # print(self.model(x))
        # print(y)
        batch_loss = self.loss_func(self.model(x), y)
        # print("3")
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
            #print("START")
            for batch in self.train_loader:
                x, y = batch
                batch_loss = self.train_batch(x, y)
                epoch_losses.append(batch_loss)
                batch_acc = self.accuracy(x, y)
                epoch_accuracies.append(batch_acc)
            #print("DONE")
            train_losses.append(np.mean(epoch_losses))
            train_accuracies.append(np.mean(epoch_accuracies))
            #print("THERE")
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
        print(train_accuracies)
        print(test_accuracies)
    
    @torch.no_grad()
    def visualize_predictions(self, loader, num_images=5):
        self.model.eval()

        images, labels = next(iter(loader))

        images, labels = images.to(self.device), labels.to(self.device)

        outputs = self.model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(num_images):
            image, label, prediction = images[i], labels[i], predicted[i]
            image_np = image.cpu().numpy().transpose((1, 2, 0))  # Assuming the tensor is in CHW format
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

            plt.imshow(image_np)
            plt.title(f'Actual label: {label.item()}, Predicted label: {prediction.item()}')
            plt.show()

    def save_model(self, save_path='trained_model2.pth'):
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at: {save_path}")
    

def MLP_CNN_experiment():
    cnn_model = CNN()
    cnn_classifier = TrainClassifier(cnn_model.model)
    cnn_classifier.train()
    cnn_classifier.visualize_predictions(cnn_classifier.test_loader)
        
    cnn_classifier.save_model()

    

if __name__ == "__main__":
    MLP_CNN_experiment()
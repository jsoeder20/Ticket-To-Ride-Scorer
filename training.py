import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split
import time
from scipy.ndimage import rotate

from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt


class TrainsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(36015, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )


    def forward(self, x):
        return self.model(x)


class TrainsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.targets = []
        self.label_mapping = {'blue': 1, 'black': 2, 'green': 3, 'red': 4, 'yellow': 5}

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
                        image = Image.open(img_path).convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                        
                        label = filename.split('-')[0]
                        target = self.label_mapping.get(label, 0)
                        
                        self.data.append(image)
                        self.targets.append(target)


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



class StationsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(36015, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, x):
        return self.model(x)


class StationsDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.data = []
        self.targets = []
        self.transform = transform
        self.color_mapping = {'blue': 1, 'black': 2, 'green': 3, 'red': 4, 'yellow': 5}

        self.load_data()

    def load_data(self):
        big_folder = self.root_dir
        for folder in os.listdir(big_folder):
            if folder != '.DS_Store':
                folder_path = os.path.join(big_folder, folder)
                for filename in os.listdir(folder_path):
                    if filename != '.DS_Store':
                        img_path = os.path.join(folder_path, filename)
                        original_image = cv2.imread(img_path)
                        
                        desired_height, desired_width = 100, 100
                        resized_image = cv2.resize(original_image, (desired_width, desired_height))
                        
                        for rotation_angle in [0, 90, 180, 270]:
                            rotated_image = rotate(resized_image, rotation_angle, reshape=False)
                            label = filename.split('-')[0]
                            self.data.append(rotated_image)
                            self.targets.append(self.color_mapping.get(label, 0))


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


class Classifier:
    def __init__(self, model, root_dir, batch_size=32, learning_rate=1e-3, num_epochs=10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.root_dir = root_dir
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

        custom_dataset = StationsDataset(self.root_dir, transform=transform)
        train_dataset, test_dataset = custom_dataset.split_data()
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

            for batch in self.train_loader:
                x, y = batch
                batch_loss = self.train_batch(x, y)
                epoch_losses.append(batch_loss)
                batch_acc = self.accuracy(x, y)
                epoch_accuracies.append(batch_acc)

            train_losses.append(np.mean(epoch_losses))
            train_accuracies.append(np.mean(epoch_accuracies))

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

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at: {save_path}")
    

def train_models():
    station_cnn_model = StationsCNN()
    station_dir = 'test_train_station_data'
    station_cnn_classifier = Classifier(station_cnn_model.model, station_dir)
    station_cnn_classifier.train()
    station_cnn_classifier.visualize_predictions(station_cnn_classifier.test_loader)
    
    station_path = 'models/station_spot_classifiers/trained_station_model_07.pth'
    station_cnn_classifier.save_model(station_path)

    train_cnn_model = TrainsCNN()
    train_dir = 'test_train_train_data'
    train_cnn_classifier = Classifier(train_cnn_model.model, train_dir)
    train_cnn_classifier.train()
    train_cnn_classifier.visualize_predictions(train_cnn_classifier.test_loader)
    
    train_path = 'models/train_spot_classifiers/trained_train_model_08.pth'
    train_cnn_classifier.save_model(train_path)

    
if __name__ == "__main__":
    train_models()
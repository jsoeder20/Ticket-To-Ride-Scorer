from training import CNN, TrainClassifier
from torchvision import transforms
import os
import cv2
import torch
import pandas as pd

def load_model(model, model_path='trained_model.pth'):
    model_state_dict = torch.load(model_path)
    
    # Remove the "model." prefix from the keys
    model_state_dict = {'model.' + k: v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    return model

def assign_label(model, folder_path, filename):
    label_map = ['blank', 'blue', 'black', 'green', 'red', 'yellow']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_path = os.path.join(folder_path, filename)
    cropped_image = cv2.imread(img_path)
    desired_height, desired_width = 50, 125
    image = cv2.resize(cropped_image, (desired_width, desired_height))
    
    # Apply the same transformations used during training
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform model evaluation
    with torch.no_grad():
        label = model(input_batch)
    
    predicted_label = label_map[torch.argmax(label, dim=1).item()]
    return predicted_label

def construct_gamestate(model):
    image_folder_path = 'messy_train_in_some_spots'
    model.eval()

    columns = ['name', 'fullName', 'length', 'points', 'colors', 'color']
    game_info = pd.DataFrame(columns=columns)

    for image_filename in os.listdir(image_folder_path):
        predicted_label = assign_label(model, image_folder_path, image_filename)
        name = image_filename[:-6]
        train_number = int(image_filename.split('-')[-1][:-4])
        
        if not game_info['name'].isin([name]).any():
            new_row = {'name': name, 'length': train_number, 'colors': [predicted_label]}
            game_info = game_info.append(new_row, ignore_index=True)
        else:
            idx = game_info.index[game_info['name'] == name][0]
            game_info.at[idx, 'colors'].append(predicted_label)
            
            if int(game_info.at[idx, 'length']) < int(train_number):
                game_info.at[idx, 'length'] = train_number
    
    return game_info

if __name__ == '__main__':
    cnn_model = CNN()
    loaded_classifier = load_model(cnn_model)
    game_state = construct_gamestate(loaded_classifier)
    print(game_state)

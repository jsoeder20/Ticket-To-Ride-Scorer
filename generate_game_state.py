from training_trains import CNN, TrainClassifier
from training_stations import CNN2
from torchvision import transforms
from collections import Counter
import os
import cv2
import torch
import pandas as pd

def load_train_model(model_path):
    print(model_path)
    cnn_model = CNN()
    model_state_dict = torch.load(model_path)
    
    model_state_dict = {'model.' + k: v for k, v in model_state_dict.items()}
    cnn_model.load_state_dict(model_state_dict)
    return cnn_model

def load_station_model(model_path):
    print(model_path)
    cnn_model = CNN2()
    model_state_dict2 = torch.load(model_path)
    
    model_state_dict = {'model.' + k: v for k, v in model_state_dict2.items()}
    cnn_model.load_state_dict(model_state_dict)

    return cnn_model

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

def assign_label_station(model, folder_path, filename):
    label_map = ['blank', 'blue', 'black', 'green', 'red', 'yellow']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_path = os.path.join(folder_path, filename)
    cropped_image = cv2.imread(img_path)
    desired_height, desired_width = 100, 100

    image = cv2.resize(cropped_image, (desired_width, desired_height))
    # Apply the same transformations used during training
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    
    with torch.no_grad():
        label = model(input_batch)

    predicted_label = label_map[torch.argmax(label, dim=1).item()]
    return predicted_label


def assign_points(df):
    point_map = {1 : 1, 2 : 2, 3 : 4, 4 : 7, 6 : 15, 8 : 21}
    for idx, row in df.iterrows():
        df.at[idx, 'points'] = point_map[df.at[idx, 'length']]
    return df

def assign_color(df):
    count = 0
    for idx, row in df.iterrows():
        colors_detected = df.at[idx, 'colors']
        if len(set(colors_detected)) == 1:
            df.at[idx, 'color'] = colors_detected[0]
        else:
            count += 1
            print(df.at[idx, 'name'], colors_detected)

            color_counter = Counter(colors_detected)
            max_value = max(color_counter.values())
            max_keys = [key for key, value in color_counter.items() if value == max_value]

            if len(max_keys) == 1:
                if max_keys[0] == 'blank' and 'yellow' in color_counter:
                    df.at[idx, 'color'] = 'yellow'
                else:
                    df.at[idx, 'color'] = max_keys[0]
            else:
                for color in max_keys:
                    if color != 'blank':
                        df.at[idx, 'color'] = color
                        break
    print(count)
    return df


def elaborate_names(df):
    cities_df = pd.read_csv('game_data/cities.csv')

    code_to_city = {}
    for city in cities_df['City']:
        if city == 'Khobenhaven':
            code_to_city['kob'] = 'Khobenhaven'
        elif city == 'Riga':
            code_to_city['ric'] = 'Riga'
        else:
            code_to_city[city[0:3].lower()] = city

    for idx, row in df.iterrows():
        curr_name =  df.at[idx, 'name']
        code1 = curr_name[0:3]
        code2 = curr_name[4:7]
        df.at[idx, 'location1'] = code_to_city[code1]
        df.at[idx, 'location2'] = code_to_city[code2]
    
    return df

def build_train_df(model, image_folder_path):
    model.eval()

    columns = ['name', 'location1', 'location2', 'length', 'points', 'colors', 'color']
    game_info = pd.DataFrame(columns=columns)

    for image_filename in os.listdir(image_folder_path):
        predicted_label = assign_label(model, image_folder_path, image_filename)
        name = image_filename[:-6]
        train_number = int(image_filename.split('-')[-1][:-4])
        
        if not game_info['name'].isin([name]).any():
            new_row = pd.DataFrame({'name': [name], 'length': [train_number], 'colors': [[predicted_label]]})
            game_info = pd.concat([game_info, new_row], ignore_index=True)

        else:
            idx = game_info.index[game_info['name'] == name][0]
            game_info.at[idx, 'colors'].append(predicted_label)
            
            if int(game_info.at[idx, 'length']) < int(train_number):
                game_info.at[idx, 'length'] = train_number

    game_info = assign_points(game_info)
    game_info = assign_color(game_info)
    game_info = elaborate_names(game_info)
    
    return game_info

def elaborate_names_stations(df):
    cities_df = pd.read_csv('game_data/cities.csv')

    code_to_city = {}
    for city in cities_df['City']:
        if city == 'Khobenhaven':
            code_to_city['kob'] = 'Khobenhaven'
        else:
            code_to_city[city[0:3].lower()] = city

    for idx, row in df.iterrows():
        curr_name =  df.at[idx, 'name']
        code1 = curr_name[0:3]
        df.at[idx, 'city'] = code_to_city[code1]
    
    return df

def build_station_df(model, image_folder_path):
    model.eval()

    columns = ['name', 'city', 'color']
    game_info = pd.DataFrame(columns=columns)

    for image_filename in os.listdir(image_folder_path):
        if image_filename != '.DS_Store':
            predicted_label = assign_label_station(model, image_folder_path, image_filename)

            name = image_filename[:-6]

            new_row = pd.DataFrame({'name': [name], 'color': [predicted_label]})
            game_info = pd.concat([game_info, new_row], ignore_index=True)

    game_info = elaborate_names_stations(game_info)
    
    return game_info

def create_game_state(train_input_file, station_input_file, train_model, station_model):
    loaded_classifier = load_train_model(train_model)
    train_game_state = build_train_df(loaded_classifier, train_input_file)
    # print(train_game_state.to_string())

    loaded_classifier2 = load_station_model(station_model)
    station_game_state = build_station_df(loaded_classifier2, station_input_file)
    # print(station_game_state[station_game_state['color'] != 'blank'].to_string())
    return train_game_state, station_game_state

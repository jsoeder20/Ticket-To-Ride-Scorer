from training import CNN, TrainClassifier
from torchvision import transforms
from collections import Counter
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
    # city_names = cities_df['City'].to_list()
    # alt_name_map = {'ric' : 'Riga', 'kob': 'Khobenhaven'}

    # for idx, row in df.iterrows():
    #     names = df.at[idx, 'name']
    #     name1, name2 = names.split('-')[0], names.split('-')[1]

    #     if name1 in alt_name_map.keys():
    #         df.at[idx, 'location1'] = alt_name_map[name1]
            
    #     if name2 in alt_name_map.keys():
    #         df.at[idx, 'location2'] = alt_name_map[name2]

    #     for full_name in city_names:
    #         if full_name[:3].lower() == name1:
    #             df.at[idx, 'location1'] = full_name
    #         if full_name[:3].lower() == name2:
    #             df.at[idx, 'location2'] = full_name
    
    # return df


def construct_gamestate(model, image_folder_path):
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

if __name__ == '__main__':
    cnn_model = CNN()
    loaded_classifier = load_model(cnn_model)

    image_folder_path = 'messy_train_in_some_spots'
    game_state = construct_gamestate(loaded_classifier, image_folder_path)
    print(game_state.to_string())

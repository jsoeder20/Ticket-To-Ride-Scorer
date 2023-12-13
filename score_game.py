from generate_game_state import create_game_state
from extract_train_images import extract_images
import pandas as pd
from itertools import product
from collections import Counter
import os


LONGEST_ROUTE_POINTS = 10
POINTS_PER_UNUSED_STATION = 4
NUM_STATIONS = 3

def print_train_point_scores(key, score):
    print("-----------------------------------")
    print(key.capitalize() + " scored " + str(score) + " points from trains")

def train_points(df, scores):
    for key in scores.keys():
        score = df[df['color']==key]['points'].sum()
        scores[key] += score
        print_train_point_scores(key, score)

def print_longest_route_scores(max_keys, longest_roads):
    print("-----------------------------------")
    if len(max_keys) > 1:
        print("The longest route winners are: " + ", ".join([f"{color.capitalize()}" for color in max_keys]) + " with " + str(longest_roads[max_keys][0]) + " roads!")
    else:
        print("Longest route winner is " + max_keys[0] + " with " + str(longest_roads[max_keys[0]]) + " consectutive trains!")

def single_longest_route(color_df, visited, city, curr_path_length, longest_path):    
    for index, row in color_df.iterrows():
        if row['location1'] == city and set([city, row['location2']]) not in visited:
            visited.append(set([city, row['location2']]))
            single_longest_route(color_df, visited, row['location2'], curr_path_length + row['length'], longest_path)
        elif row['location2'] == city and set([city, row['location1']]) not in visited:
            visited.append(set([city, row['location1']]))
            single_longest_route(color_df, visited, row['location1'], curr_path_length + row['length'], longest_path)

    # Update longest_path only after processing all rows for the current city
    longest_path[0] = max(curr_path_length, longest_path[0])

def longest_route(df, scores):
    longest_roads = {} 
    for key in scores.keys():
        color_df = df[df['color'] == key][['location1', 'location2', 'length']]
        longest_path = 0

        # Iterate through all cities because we don't know where the longest road starts
        for city in set(color_df['location1']).union(color_df['location2']):
            curr_path = [0]  # List to be iterable
            visited = []  # List of sets to keep track of visited cities
            single_longest_route(color_df, visited, city, 0, curr_path)
            longest_path = max(curr_path[0], longest_path)

        longest_roads[key] = longest_path

    longest_road_counter = Counter(longest_roads)
    max_value = max(longest_road_counter.values())
    max_keys = [key for key, value in longest_road_counter.items() if value == max_value]

    for color in max_keys:
        scores[color] += LONGEST_ROUTE_POINTS
    
    print_longest_route_scores(max_keys, longest_roads)

    return max_keys


def print_destination_ticket_scores(key, max_score, all_best_completed, all_best_failed, best_combination):
        print("-----------------------------------")
        print(key.capitalize() + " scored " + str(max_score) + " points from destination tickets")
        if len(all_best_completed) > 0:
            print("Completed: " + ", ".join([f"{start} to {finish}" for start, finish in all_best_completed]))
        else:
            print("Completed no routes :(")
        if len(all_best_failed) > 0:
            print("Failed: " + ", ".join([f"{start} to {finish}" for start, finish in all_best_failed]))
        else:
            print("Failed none")
        if len(best_combination) > 0:
            print("Used station connections: " + ", ".join([f"{start} to {finish}" for start, finish in best_combination]))
        else:
            print("Used no stations")

def destination_complete(color_df, curr, finish, visited):
    if curr == finish:
        return True
    if curr in visited:
        return False
    
    visited.add(curr)
    
    for index, row in color_df.iterrows():
        if row['location1'] == curr and row['location2'] not in visited:
            if destination_complete(color_df, row['location2'], finish, visited):
                return True 
        elif row['location2'] == curr and row['location1'] not in visited:
            if destination_complete(color_df, row['location1'], finish, visited):
                return True
        
    return False

def get_no_connections(train_df):
    return {(row['location1'], row['location2']) for _, row in train_df[train_df['color'] == 'blank'][['location1', 'location2']].iterrows()}

def get_surrounding_cities(station_cities, no_connections, all_connections_df):
    connections = {}
    for city in station_cities:
        adjacent_cities_df = all_connections_df[(all_connections_df['Source'] == city) | (all_connections_df['Target'] == city)]
        possible_segments = []
        for idx, row in adjacent_cities_df.iterrows():
            location1, location2 = row['Source'], row['Target']
            if (location1, location2) not in no_connections and (location2, location1) not in no_connections:
                possible_segments.append((location1, location2))
        connections[city] = possible_segments
    return connections
    

def destination_tickets(train_df, station_df, all_destination_tickets_df, all_connections_df, scores, tickets):

    no_connections = get_no_connections(train_df)
    num_tickets_completed_dict = {}

    for key in scores.keys():
        color_df_train = train_df[train_df['color']==key][['location1', 'location2']]
        station_cities = station_df[station_df['color']==key]['city'].values
        potential_station_routes = get_surrounding_cities(station_cities, no_connections, all_connections_df)

        max_score = float('-inf')
        best_combination, all_best_completed, all_best_failed = [], [], []
        max_num_tickets_completed = 0
        all_combinations = list(product(*potential_station_routes.values()))
        for combination in all_combinations:
            color_df_train_with_stations = color_df_train.copy()
            for connection in combination:
                new_row = pd.DataFrame({'location1': connection[0], 'location2': connection[1]},index=[0])
                color_df_train_with_stations = pd.concat([color_df_train_with_stations, new_row], ignore_index=True)
            score, num_tickets_completed = 0, 0
            current_all_best_completed, current_all_best_failed = [], []
            for start, end in tickets[key].items(): ###DELETE .items() WHEN DONE!!!
                start = start.lower().capitalize()
                end = end.lower().capitalize()
                df = all_destination_tickets_df
                points = df[((df['Source'] == start) & (df['Target'] == end)) | 
                            ((df['Source'] == end) & (df['Target'] == start))]['Points'].values[0]

                if destination_complete(color_df_train_with_stations, start, end, set()):
                    num_tickets_completed += 1
                    score += points
                    current_all_best_completed.append((start, end))
                else:
                    score -= points
                    current_all_best_failed.append((start, end))
            
            if score > max_score:
                all_best_completed = current_all_best_completed
                all_best_failed = current_all_best_failed
                max_score = score
                best_combination = combination
                max_num_tickets_completed = num_tickets_completed
        scores[key] += max_score
        num_tickets_completed_dict[key] = max_num_tickets_completed

        print_destination_ticket_scores(key, max_score, all_best_completed, all_best_failed, best_combination)


    return num_tickets_completed_dict

def print_remaining_station_scores(key, num_used_stations, score):
    print("-----------------------------------")
    print(key.capitalize() + " scored " + str(score) + " points from their " + str(NUM_STATIONS - num_used_stations) + " remaining stations")

def remaining_stations(station_df, scores):
    num_stations_left_dict = {}
    for key in scores.keys():
        num_used_stations = station_df[station_df['color'] == key].shape[0]
        score = POINTS_PER_UNUSED_STATION * (NUM_STATIONS - num_used_stations)
        scores[key] += score
        num_stations_left_dict[key] = scores[key]

        print_remaining_station_scores(key, num_used_stations, score)
        
    return num_stations_left_dict


def get_user_destination_tickets(color, tickets, num_tickets, all_cities, all_destination_tickets):
    print(all_destination_tickets)
    for i in range(num_tickets):
        start, end = "", ""
        while (start, end) not in all_destination_tickets and (end, start) not in all_destination_tickets:
            start, end = "", ""
            while start not in all_cities:
                start = input(f"Enter the starting city for destination ticket {i + 1} (no special characters): ").lower().capitalize()
                if start not in all_cities:
                    print(start + " is not a valid city. Try again!")
            while end not in all_cities:
                end = input(f"Enter the ending city for destination ticket {i + 1} (no special characters): ").lower().capitalize()
                if end not in all_cities:
                    print(end + " is not a valid city. Try again!")
            if (start, end) not in all_destination_tickets and (end, start) not in all_destination_tickets:
                print(start + " and " + end + " do not form a valid destination ticket (order does NOT matter).")
        tickets[color].append((start, end))
    return tickets

def get_all_user_destination_tickets(scores, all_cities_df, all_destination_tickets_df):
    tickets = {}
    all_destination_tickets = set(zip(all_destination_tickets_df['Source'], all_destination_tickets_df['Target']))
    all_cities = set(all_cities_df['City'])
    print(all_destination_tickets)
    for color in ['red', 'blue', 'green', 'black', 'yellow']:
        player_num_tickets = ""
        while not player_num_tickets.isdigit():
            player_num_tickets = input("How many desination tickets does " + color + " have?")
            if not player_num_tickets.isdigit():
                print("Input must be a number! Try again")
        player_num_tickets = int(player_num_tickets)
        if player_num_tickets > 0:
            tickets[color] = []
            scores[color] = 0
            get_user_destination_tickets(color, tickets, player_num_tickets, all_cities, all_destination_tickets)
    return tickets


def print_final_scores_and_winner(scores, num_tickets_completed_dict, num_stations_left_dict, longest_route_winner):
    scores_counter = Counter(scores)
    max_value = max(scores_counter.values())
    max_keys = [key for key, value in scores_counter.items() if value == max_value]

    print("-----------------------------------")
    print("-----------------------------------")
    for color, score in scores.items():
        print(color.capitalize() + " final score is " + str(score) + "!")
    print("-----------------------------------")
    if len(max_keys) == 1:
        print(max_keys[0].capitalize() + " is the winner!")
    else:
        num_tickets_counter = Counter(num_tickets_completed_dict)
        max_value = max(num_tickets_counter.values())
        max_keys = [key for key, value in num_tickets_counter.items() if value == max_value]
        if len(max_keys) > 1:
            num_stations_left_counter = Counter(num_stations_left_dict)
            max_value = max(num_stations_left_counter.values())
            max_keys = [key for key, value in num_stations_left_counter.items() if value == max_value]
            if len(max_keys) > 1:
                if len(longest_route_winner) > 1:
                    print("Holy guacamole! " + str(longest_route_winner) + " all won!")
                else:
                    print(longest_route_winner[0].capitalize() + " is the winner!")
        else:
            print(max_keys[0].capitalize() + " is the winner!")
    print("-----------------------------------")


def clear_dir(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
    else:
        # If the directory exists, empty it
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    image_path = input("Enter the path to the cropped board image: ")
    #image_path = 'cropped_board_images/cropped5s_12-12.jpg'
    
    train_dir = 'unlabeled_data/train_data/real_game_train_spots1'
    station_dir = 'unlabeled_data/station_data/real_game_station_spots1'

    clear_dir(train_dir)
    clear_dir(station_dir)

    extract_images(image_path, station_output=station_dir, train_output=train_dir)

    scores = {}

    all_cities_df = pd.read_csv('game_data/cities.csv')
    all_destination_tickets_df = pd.read_csv('game_data/destinations.csv')
    all_connections_df = pd.read_csv('game_data/routes.csv')

    ##USER SELECTED TICKETS
    # tickets = get_all_user_destination_tickets(scores, all_cities_df, all_destination_tickets_df)

    ##DELETE WHEN DONE
    scores = {'red':0, 'blue':0, 'yellow':0, 'green':0, 'black':0}
    tickets = {'blue':{'lisboa': 'danzic', 'paris':'wien', 'madrid':'zurich', 'berlin':'roma'},
            'yellow': {'erzurum':'rostov', 'sofia':'smyrna', 'riga':'bucuresti', 'Kobenhavn':'Erzurum'}, 
            'green':{'London':'Berlin', 'Sarajevo':'Sevastopol', 'Palermo':'Moskva'}, 
            'black':{'smolensk':'Rostov', 'athina':'wilno', 'edinburgh':'athina'}, 
            'red':{'Cadiz':'Stockholm', 'Berlin':'Bucuresti', 'Kyiv':'Sochi'}}

    
    train_model = 'models/train_spot_classifiers/trained_train_model_07.pth'
    station_model = 'models/station_spot_classifiers/trained_station_model_06.pth'
    train_game_state, station_game_state = create_game_state(train_dir, station_dir, train_model, station_model)

    train_points(train_game_state, scores)
    
    longest_route_winner = longest_route(train_game_state, scores)


    print("-----------------------------------")
    num_tickets_completed_dict= destination_tickets(train_game_state, station_game_state, all_destination_tickets_df, all_connections_df, scores, tickets)

    print("-----------------------------------")
    num_stations_left_dict = remaining_stations(station_game_state, scores)

    print_final_scores_and_winner(scores, num_tickets_completed_dict, num_stations_left_dict, longest_route_winner)

    clear_dir(train_dir)
    clear_dir(station_dir)

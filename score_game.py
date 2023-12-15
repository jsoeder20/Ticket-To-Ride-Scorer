from generate_game_state import create_game_state
from extract_train_images import extract_images
import pandas as pd
from itertools import product
from collections import Counter
import os


LONGEST_ROUTE_POINTS = 10
POINTS_PER_UNUSED_STATION = 4
NUM_STATIONS = 3
COLORS = ['red', 'blue', 'green', 'black', 'yellow']

def print_train_point_scores(key, score):
    """
    Print the train point scores for a specific color.

    Parameters:
    - key (str): The color key for which the scores are printed.
    - score (int): The total train points scored for the specified color.
    """
    print("-----------------------------------")
    print(key.capitalize() + " scored " + str(score) + " points from trains")

def train_points(df, scores):
    """
    Update and print the train points for each color based on the train dataframe.

    Parameters:
    - df (pd.DataFrame): DataFrame containing train information.
    - scores (dict): Dictionary to store and update scores for each color.
    """
    for color in scores.keyss():
        score = df[df['color']==color]['points'].sum()
        scores[color] += score
        print_train_point_scores(color, score)

def print_longest_route_scores(max_keys, longest_roads):
    """
    Print the longest route scores and winners.

    Parameters:
    - max_keys (list): List of colors that share the longest route.
    - longest_roads (dict): Dictionary containing longest route lengths for each color.
    """
    print("-----------------------------------")
    if len(max_keys) > 1:
        print("The longest route winners are: " + ", ".join([f"{color.capitalize()}" for color in max_keys]) + " with " + str(longest_roads[max_keys][0]) + " roads!")
    else:
        print("Longest route winner is " + max_keys[0] + " with " + str(longest_roads[max_keys[0]]) + " consectutive trains!")


def single_longest_route(color_df, visited, city, curr_path_length, longest_path):   
    """
    Helper function to longest_route() that recursively finds the length of the 
    longest route for a specific color starting from a given city.

    Parameters:
    - color_df (pd.DataFrame): DataFrame containing train information for a specific color.
    - visited (list): List of sets to keep track of visited cities.
    - city (str): Current city being explored.
    - curr_path_length (int): Current path length.
    - longest_path (list): List to store the longest path length.
    """ 
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
    """
    Calculate and update scores based on the longest route for each color.
    For each color, call the helper function, single_longest_route(), on each traveled
    city for that color in order to guarantee finding the longest route.

    Parameters:
    - df (pd.DataFrame): DataFrame containing train information.
    - scores (dict): Dictionary to store and update scores for each color.

    Returns:
    - max_keys (list): List of colors that share the longest route.
    """
    longest_roads = {} 
    for color in scores.keys():
        color_df = df[df['color'] == color][['location1', 'location2', 'length']] #get a single players data
        longest_path = 0

        # Iterate through all cities because we don't know where the longest road starts
        for city in set(color_df['location1']).union(color_df['location2']):
            curr_path = [0]  # List to be iterable
            visited = []  # List of sets to keep track of visited routes
            single_longest_route(color_df, visited, city, 0, curr_path) 
            longest_path = max(curr_path[0], longest_path)

        longest_roads[color] = longest_path

    # Extract the color(s) with the longest route
    longest_road_counter = Counter(longest_roads)
    max_value = max(longest_road_counter.values())
    max_keys = [key for key, value in longest_road_counter.items() if value == max_value]

    # Assign points to color(s) who got the longest route
    for color in max_keys:
        scores[color] += LONGEST_ROUTE_POINTS
    
    # Print scores
    print_longest_route_scores(max_keys, longest_roads)

    return max_keys


def print_destination_ticket_scores(key, max_score, all_best_completed, all_best_failed, best_combination):
    """
    Print destination ticket scores, completed routes, failed routes, and used station connections.

    Parameters:
    - key (str): The color key for which the scores are printed.
    - max_score (int): The total points scored from destination tickets for the specified color.
    - all_best_completed (list): List of completed routes.
    - all_best_failed (list): List of failed routes.
    - best_combination (list): List of used station connections.
    """
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
    """
    Check if a destination route (start to finish) cityis complete. 
    Helper function to destination_tickets().

    Parameters:
    - color_df (pd.DataFrame): DataFrame containing train information for a specific color.
    - curr (str): Current city being checked for completion.
    - finish (str): Destination city for the route.
    - visited (set): Set of visited cities.

    Returns:
    - bool: True if the destination route is complete, False otherwise.
    """
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
    """
    Get a set of connections that do not have assigned colors.
    Used to minimize number of station permutations because stations can only occur
    on routes with placed trains in destination_tickets().

    Parameters:
    - train_df (pd.DataFrame): DataFrame containing train information.

    Returns:
    - set: Set of connections without assigned colors.
    """
    return {(row['location1'], row['location2']) for _, row in train_df[train_df['color'] == 'blank'][['location1', 'location2']].iterrows()}

def get_surrounding_cities(station_cities, no_connections, all_connections_df):
    """
    Get possible train segments from the station city to all adjacent cities. Used to
    find all possible cities the station could extend to in order to create the permutations
    of stations in destination_tickets().

    Parameters:
    - station_cities (array): Array of cities with stations for a specific color.
    - no_connections (set): Set of connections without assigned colors.
    - all_connections_df (pd.DataFrame): DataFrame containing all train connections.

    Returns:
    - dict: Dictionary mapping station cities to possible train segments.
    """
    connections = {}
    for city in station_cities:
        adjacent_cities_df = all_connections_df[(all_connections_df['Source'] == city) | (all_connections_df['Target'] == city)]
        possible_segments = []
        for idx, row in adjacent_cities_df.iterrows():
            location1, location2 = row['Source'], row['Target']
            # Stations can't extend on unclaimed routes, so get rid of those to reduce running time
            if (location1, location2) not in no_connections and (location2, location1) not in no_connections:
                possible_segments.append((location1, location2))
        connections[city] = possible_segments
    return connections
    

def destination_tickets(train_df, station_df, all_destination_tickets_df, all_connections_df, scores, tickets):
    """
    Calculate and update scores based on destination tickets, considering station connections.
    Uses destination_complete() helper method to determine if a specific destination ticket has been completed.
    For each color, the algorithm gets all placed stations and their possible connections. Since a station
    can only connect to one city, we create all the permutations for the station combinations and then find the 
    combination of specific station created connections that maximizes the earned points from destination tickets.

    Parameters:
    - train_df (pd.DataFrame): DataFrame containing train information.
    - station_df (pd.DataFrame): DataFrame containing station information.
    - all_destination_tickets_df (pd.DataFrame): DataFrame containing destination ticket information.
    - all_connections_df (pd.DataFrame): DataFrame containing all train connections.
    - scores (dict): Dictionary to store and update scores for each color.
    - tickets (dict): Dictionary mapping colors to their respective destination tickets.

    Returns:
    - num_tickets_completed_dict (dict): Dictionary mapping colors to the number of completed destination tickets.
    """
    no_connections = get_no_connections(train_df)
    num_tickets_completed_dict = {}

    for color in scores.keys():
        # Get data associated with color
        color_df_train = train_df[train_df['color']==color][['location1', 'location2']]
        station_cities = station_df[station_df['color']==color]['city'].values
        potential_station_routes = get_surrounding_cities(station_cities, no_connections, all_connections_df)

        max_score = float('-inf')
        best_combination, all_best_completed, all_best_failed = [], [], []
        max_num_tickets_completed = 0

        # Create all combinations of valid station route connections
        all_combinations = list(product(*potential_station_routes.values()))

        # For each possible combination, get its score from desintation tickets
        for combination in all_combinations:
            color_df_train_with_stations = color_df_train.copy() # Create copy so appended combination doesn't stay
            for connection in combination:
                # Append this combination so destination_complete() helper treats it as a route
                new_row = pd.DataFrame({'location1': connection[0], 'location2': connection[1]},index=[0])
                color_df_train_with_stations = pd.concat([color_df_train_with_stations, new_row], ignore_index=True)

            score, num_tickets_completed = 0, 0
            current_all_best_completed, current_all_best_failed = [], []

            for start, end in tickets[color]:
                # Set up inputs so they can be compared to items in csv
                start = start.lower().capitalize()
                end = end.lower().capitalize()

                # Getting the number of points associated with a destination ticket from csv data frame
                df = all_destination_tickets_df # Renaming csv data so shorter
                points = df[((all_destination_tickets_df['Source'] == start) & (df['Target'] == end)) | 
                            ((df['Source'] == end) & (df['Target'] == start))]['Points'].values[0]

                # Keep track of stats
                if destination_complete(color_df_train_with_stations, start, end, set()):
                    num_tickets_completed += 1
                    score += points
                    current_all_best_completed.append((start, end))
                else:
                    score -= points
                    current_all_best_failed.append((start, end))
            
            # Keep stats associated with the station combinations that maximize points earned
            if score > max_score:
                all_best_completed = current_all_best_completed
                all_best_failed = current_all_best_failed
                max_score = score
                best_combination = combination
                max_num_tickets_completed = num_tickets_completed

        # Add maximized points to players scores
        scores[color] += max_score
        num_tickets_completed_dict[color] = max_num_tickets_completed

        # Print scores
        print_destination_ticket_scores(color, max_score, all_best_completed, all_best_failed, best_combination)


    return num_tickets_completed_dict

def print_remaining_station_scores(key, num_used_stations, score):
    """
    Print the remaining station scores for a specific color.

    Parameters:
    - key (str): The color key for which the scores are printed.
    - num_used_stations (int): The number of stations used by the specified color.
    - score (int): The total points scored from remaining stations for the specified color.
    """
    print("-----------------------------------")
    print(key.capitalize() + " scored " + str(score) + " points from their " + str(NUM_STATIONS - num_used_stations) + " remaining stations")

def remaining_stations(station_df, scores):
    """
    Calculate and update scores based on the remaining stations for each color.

    Parameters:
    - station_df (pd.DataFrame): DataFrame containing station information.
    - scores (dict): Dictionary to store and update scores for each color.

    Returns:
    - num_stations_left_dict (dict): Dictionary mapping colors to the remaining number of stations.
    """
    num_stations_left_dict = {}
    for color in scores.keys():
        num_used_stations = station_df[station_df['color'] == color].shape[0]
        score = POINTS_PER_UNUSED_STATION * (NUM_STATIONS - num_used_stations)
        scores[color] += score
        num_stations_left_dict[color] = scores[color]

        print_remaining_station_scores(color, num_used_stations, score)
        
    return num_stations_left_dict


def get_user_destination_tickets(color, tickets, num_tickets, all_cities, all_destination_tickets):
    """
    Get the destination tickets through for a specific color using input from the user.

    Parameters:
    - color (str): The color for which destination tickets are input.
    - tickets (dict): Dictionary to store destination tickets for each color.
    - num_tickets (int): The number of destination tickets to be input.
    - all_cities (set): Set of all valid cities.
    - all_destination_tickets (set): Set of all valid destination tickets.

    Returns:
    - tickets (dict): Updated dictionary containing user-input destination tickets for the specified color.
    """
    for i in range(num_tickets):
        start, end = "", ""

        # Keep asking for valid tickets provided destination card doesn't exist
        while (start, end) not in all_destination_tickets and (end, start) not in all_destination_tickets:
            start, end = "", ""
            while start not in all_cities: # Keep asking for valid city provided city doesn't exist
                start = input(f"Enter the starting city for destination ticket {i + 1} (no special characters): ").lower().capitalize()
                if start not in all_cities:
                    print(start + " is not a valid city. Try again!")
            while end not in all_cities:             # Keep asking for valid city provided city doesn't exist
                end = input(f"Enter the ending city for destination ticket {i + 1} (no special characters): ").lower().capitalize()
                if end not in all_cities:
                    print(end + " is not a valid city. Try again!")
            if (start, end) not in all_destination_tickets and (end, start) not in all_destination_tickets:
                print(start + " and " + end + " do not form a valid destination ticket (order does NOT matter).")
        tickets[color].append((start, end))
    return tickets

def get_all_user_destination_tickets(scores, all_cities_df, all_destination_tickets_df):
    """
    Get all destination tickets input by the user for each color.
    Calls the helper method get_user_destination_tickets() to get an individual colors input desintation tickets.

    Parameters:
    - scores (dict): Dictionary to store scores for each color.
    - all_cities_df (pd.DataFrame): DataFrame containing information about all cities.
    - all_destination_tickets_df (pd.DataFrame): DataFrame containing information about all destination tickets.

    Returns:
    - tickets (dict): Dictionary mapping colors to their respective destination tickets.
    """
    tickets = {}
    all_destination_tickets = set(zip(all_destination_tickets_df['Source'], all_destination_tickets_df['Target']))
    all_cities = set(all_cities_df['City'])
    for color in COLORS:
        player_num_tickets = ""
        # Error handling for integer input
        while not player_num_tickets.isdigit():
            player_num_tickets = input("How many desination tickets does " + color + " have?")
            if not player_num_tickets.isdigit():
                print("Input must be a number! Try again")
        player_num_tickets = int(player_num_tickets)
        if player_num_tickets > 0:
            # Initialize data
            tickets[color] = []
            scores[color] = 0

            # Ask for which destination tickets
            get_user_destination_tickets(color, tickets, player_num_tickets, all_cities, all_destination_tickets)
    return tickets


def print_final_scores_and_winner(scores, num_tickets_completed_dict, num_stations_left_dict, longest_route_winner):
    """
    Print the final scores and determine the winner based on route points, longest road, 
    remaining stations, and destination tickets.

    Parameters:
    - scores (dict): Dictionary containing the final scores for each color.
    - num_tickets_completed_dict (dict): Dictionary mapping colors to the number of completed destination tickets.
    - num_stations_left_dict (dict): Dictionary mapping colors to the remaining number of stations.
    - longest_route_winner (list): List of colors that share the longest route.
    """
    # Extract the color(s) with the highest score
    scores_counter = Counter(scores)
    max_value = max(scores_counter.values())
    max_keys = [key for key, value in scores_counter.items() if value == max_value]

    print("-----------------------------------")
    print("-----------------------------------")
    # Print final scores
    for color, score in scores.items():
        print(color.capitalize() + " final score is " + str(score) + "!")

    print("-----------------------------------")

    # If scores aren't tied, declare winner
    if len(max_keys) == 1:
        print(max_keys[0].capitalize() + " is the winner!")
    else:
        # Tie break by determining player with most completed desination tickets
        num_tickets_counter = Counter(num_tickets_completed_dict)
        max_value = max(num_tickets_counter.values())
        max_keys = [key for key, value in num_tickets_counter.items() if value == max_value]
        if len(max_keys) > 1:
            # Tie break (again) by determining player with most unused stations
            num_stations_left_counter = Counter(num_stations_left_dict)
            max_value = max(num_stations_left_counter.values())
            max_keys = [key for key, value in num_stations_left_counter.items() if value == max_value]
            if len(max_keys) > 1:
                # Tie break (again) by determining player with longest road
                if len(longest_route_winner) > 1:
                    print("Holy guacamole! " + str(longest_route_winner) + " all won!")
                else:
                    print(longest_route_winner[0].capitalize() + " is the winner!")
        else:
            print(max_keys[0].capitalize() + " is the winner!")
    print("-----------------------------------")


def create_clear_dir(directory_path):
    """
    Clear the contents of the stored game data (for the next game) or create it if it does not exist.

    Parameters:
    - directory_path (str): The path of the directory to be cleared or created.
    """
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
    """
    Main execution block to process and score the Ticket to Ride game.
    """
    # Get game file path from user
    try:
        image_path = input("Enter the path to the cropped board image: ")

        if os.path.exists(image_path) and os.path.isfile(image_path):
            print(f"Image path: {image_path}")
        else:
            raise FileNotFoundError(f"The file '{image_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Files to store game data
    train_dir = 'unlabeled_data/train_data/real_game_train_spots1'
    station_dir = 'unlabeled_data/station_data/real_game_station_spots1'

    # Store the game data
    create_clear_dir(train_dir)
    create_clear_dir(station_dir)

    # Extract images from game to find trains
    extract_images(image_path, station_output=station_dir, train_output=train_dir)

    scores = {}

    # Prepare to access all needed CSV files
    all_cities_df = pd.read_csv('game_data/cities.csv')
    all_destination_tickets_df = pd.read_csv('game_data/destinations.csv')
    all_connections_df = pd.read_csv('game_data/routes.csv')

    # Get destination tickets from user input
    tickets = get_all_user_destination_tickets(scores, all_cities_df, all_destination_tickets_df)

    # Create the game state using the best models
    train_model = 'models/train_spot_classifiers/trained_train_model_07.pth'
    station_model = 'models/station_spot_classifiers/trained_station_model_06.pth'
    train_game_state, station_game_state = create_game_state(train_dir, station_dir, train_model, station_model)

    # Do all scoring:
    train_points(train_game_state, scores)
    
    longest_route_winner = longest_route(train_game_state, scores)

    print("-----------------------------------")
    num_tickets_completed_dict= destination_tickets(train_game_state, station_game_state, all_destination_tickets_df, all_connections_df, scores, tickets)

    print("-----------------------------------")
    num_stations_left_dict = remaining_stations(station_game_state, scores)

    print_final_scores_and_winner(scores, num_tickets_completed_dict, num_stations_left_dict, longest_route_winner)

    # Clear the game data so another game can be processed.
    create_clear_dir(train_dir)
    create_clear_dir(station_dir)

from generate_game_state import create_game_state
import pandas as pd
from itertools import product


LONGEST_ROUTE_POINTS = 10
POINTS_PER_UNUSED_STATION = 4
NUM_STATIONS = 3

def train_points(df, scores):
    for key in scores.keys():
        scores[key] += df[df['color']==key]['points'].sum()

def single_longest_route(color_df, visited, city, curr_path_length, longest_path):
    if city in visited:
        return
    
    visited.add(city)
    
    for index, row in color_df.iterrows():
        if row['location1'] == city and row['location2'] not in visited:
            single_longest_route(color_df, visited, row['location2'], curr_path_length+row['length'], longest_path)
        elif row['location2'] == city and row['location1'] not in visited:
            single_longest_route(color_df, visited, row['location1'], curr_path_length+row['length'], longest_path)

    longest_path[0] = max(curr_path_length,longest_path[0])

def longest_route(df, scores):
    longest_roads = {} 
    for key in scores.keys():
        color_df = df[df['color']==key][['location1', 'location2', 'length']]
        longest_path = 0

        #iterate thru all cities bc we don't know where longest road starts
        for city in set(color_df['location1']).union(color_df['location2']): 
            curr_path = [0] #list to be iterable
            single_longest_route(color_df, set(), city, 0, curr_path)
            longest_path = max(curr_path[0],longest_path)
        longest_roads[key] = longest_path

    longest_road_color = max(longest_roads, key=longest_roads.get)
    scores[longest_road_color] += LONGEST_ROUTE_POINTS
    print("Longest road lengths: " + str(longest_roads))
    print("Longest road winner: " + longest_road_color)

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
    no_connections = set()
    blank_df_train = train_df[train_df['color']=='blank'][['location1', 'location2']]
    for index, row in blank_df_train.iterrows():
        no_connections.add((row['location1'], row['location2']))
    return no_connections

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
    

def destination_tickets(train_df, station_df, scores, tickets):
    destination_tickets_df = pd.read_csv('game_data/destinations.csv')
    all_connections_df = pd.read_csv('game_data/routes.csv')
    no_connections = get_no_connections(train_df)
    for key in scores.keys():
        color_df_train = train_df[train_df['color']==key][['location1', 'location2']]
        station_cities = station_df[station_df['color']==key]['city'].values
        potential_station_routes = get_surrounding_cities(station_cities, no_connections, all_connections_df)

        max_score = 0
        best_combination = []
        all_combinations = list(product(*potential_station_routes.values()))
        for combination in all_combinations:
            color_df_train_with_stations = color_df_train.copy()
            # print("YEET")
            # print(color_df_train_with_stations)
            for connection in combination:
                # new_row = pd.DataFrame(columns=color_df_train.columns)
                # new_row.at[0, 'location1'] = connection[0]
                # new_row.at[0, 'location2'] = connection[1]
                new_row = pd.DataFrame({'location1': connection[0], 'location2': connection[1]},index=[0])
                color_df_train_with_stations = pd.concat([color_df_train_with_stations, new_row], ignore_index=True)
            # print(color_df_train_with_stations)
            player_all_tickets = tickets[key]
            score = 0 
            for start, end in player_all_tickets.items():
                start = start.capitalize()
                end = end.capitalize()
                points = destination_tickets_df[((destination_tickets_df['Source'] == start) & (destination_tickets_df['Target'] == end)) | ((destination_tickets_df['Source'] == end) & (destination_tickets_df['Target'] == start))]['Points'].values[0]
                if points == 0:
                    raise Exception("Cities DNE")

                if destination_complete(color_df_train_with_stations, start, end, set()):
                    score += points
                else:
                    score -= points
            
            if score > max_score:
                max_score = score
                best_combination = combination
        # print(key)
        # print(max_score)
        # print(best_combination)
        scores[key] += max_score


def remaining_stations(station_df, scores):
    for key in scores.keys():
        num_used_stations = station_df[station_df['color'] == key].shape[0]
        scores[key] += POINTS_PER_UNUSED_STATION * (NUM_STATIONS - num_used_stations)
    
def get_user_destination_tickets(color, tickets, num_tickets):
    for i in range(num_tickets):
        start = input(f"Enter the first three letters of the starting city for destination ticket {i + 1}: ")
        end = input(f"Enter the first three letters of the ending city for destination ticket {i + 1}: ")
        tickets[color][start] = end
    print(color)
    print(tickets[color])
    return tickets

if __name__ == "__main__":
    scores = {'red':0, 'blue':0, 'yellow':0, 'green':0, 'black':0}

    train_file = 'unlabeled_data/real_game_train_spots'
    station_file = 'unlabeled_data/real_game_station_spots'
    train_model = 'models/train_spot_classifiers/trained_train_model_03.pth'
    station_model = 'models/train_spot_classifiers/trained_station_model_03.pth'
    train_game_state, station_game_state = create_game_state(train_file, station_file, train_model, station_model)

    # tickets = {'red':{}, 'blue':{}, 'yellow':{}, 'green':{}, 'black':{}}
    # #ISSUE WHEN SAME KEYS
    # for key in scores.keys():
    #     player_num_tickets = int(input("How many desination tickets does " + key + " have?"))
    #     get_user_destination_tickets(key, tickets, player_num_tickets)

    train_points(train_game_state, scores)
    print(scores)

    longest_route(train_game_state, scores)
    print(scores)


    tickets = {'blue':{'lisboa': 'danzic', 'paris':'wien', 'madrid':'zurich', 'berlin':'roma'},
                'yellow': {'erzurum':'rostov', 'sofia':'smyrna', 'riga':'bucuresti', 'Kobenhavn':'Erzurum'}, 
                'green':{'London':'Berlin', 'Sarajevo':'Sevastopol', 'Palermo':'Moskva'}, 
                'black':{'smolensk':'Rostov', 'athina':'wilno', 'edinburgh':'athina'}, 
                'red':{'Cadiz':'Stockholm', 'Berlin':'Bucuresti', 'Kyiv':'Sochi'}}
    destination_tickets(train_game_state, station_game_state, scores, tickets)
    print(scores)

    remaining_stations(station_game_state, scores)
    print(scores)

    '''
    SCORING STILL NEEDS:
        - account for stations (destination and +4 per not used)
        - destination tickets input method
    '''


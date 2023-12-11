from generate_game_state import create_game_state
import pandas as pd

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
        print("AAAAHAHAHAH")
        print(adjacent_cities_df.to_string())
        possible_segments = []
        for idx, row in adjacent_cities_df.iterrows():
            location1, location2 = row['Source'], row['Target']
            if (location1, location2) not in no_connections and (location2, location1) not in no_connections:
                possible_segments.append((location1, location2))
        connections[city] = possible_segments
    print(connections)


def destination_tickets(train_df, station_df, scores, tickets):
    destination_tickets_df = pd.read_csv('game_data/destinations.csv')
    all_connections_df = pd.read_csv('game_data/routes.csv')
    no_connections = get_no_connections(train_df)
    for key in scores.keys():
        color_df_train = train_df[train_df['color']==key][['location1', 'location2']]
        print(key)
        print(color_df_train)
        station_cities = station_df[station_df['color']==key]['city'].values
        dict = get_surrounding_cities(station_cities, no_connections, all_connections_df)
        print(station_cities)
        color_connections = tickets[key]
        for start, end in color_connections.items():
            start_to_finish_values = destination_tickets_df[(destination_tickets_df['Source'] == start) & (destination_tickets_df['Target'] == end)]['Points'].values
            finish_to_start_values = destination_tickets_df[(destination_tickets_df['Source'] == end) & (destination_tickets_df['Target'] == start)]['Points'].values
            points = 0
            if start_to_finish_values.size > 0:
                points = start_to_finish_values[0]
            elif finish_to_start_values.size > 0:
                points = finish_to_start_values[0]
            else:
                raise Exception("Cities DNE")
            
            if destination_complete(color_df_train, start, end, set()):
                scores[key] += points
            else:
                scores[key] -= points

def remaining_stations(station_df, scores):
    for key in scores.keys():
        num_used_stations = station_df[station_df['color'] == key].shape[0]
        scores[key] += POINTS_PER_UNUSED_STATION * (NUM_STATIONS - num_used_stations)
    
if __name__ == "__main__":
    train_game_state, station_game_state = create_game_state('unlabeled_data/clean_trains_in_some_spots', 'unlabeled_data/messy2_stations_in_some_spots')
    scores = {'red':0, 'blue':0, 'yellow':0, 'green':0, 'black':0}

    train_points(train_game_state, scores)
    print(scores)

    longest_route(train_game_state, scores)
    print(scores)

    ##ISSUE WHEN SAME KEYS
    # for key in scores.keys():
    #     x = input(key + "desination ticket 1 (input three letter code *space* then three letter code)")
    
    tickets = {'blue':{'Lisboa': 'Danzic', 'Danzic':'Bruxelles'}, 'yellow': {'Madrid':'Zurich','Madrid':'Dieppe'}, 
               'green':{'Edinburgh':'Paris', 'Athina':'Edinburgh', 'Rostov':'Smolensk'}, 'black':{'Cadiz':'Stockholm'}, 'red':{'London':'Berlin'}}
    destination_tickets(train_game_state, station_game_state, scores, tickets)
    print(scores)

    remaining_stations(station_game_state, scores)
    print(scores)

    '''
    SCORING STILL NEEDS:
        - account for stations (destination and +4 per not used)
        - destination tickets input method
    '''


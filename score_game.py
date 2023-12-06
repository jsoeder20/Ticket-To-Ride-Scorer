from generate_game_state import create_game_state
import pandas as pd

LONGEST_ROUTE_POINTS = 10

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


def destination_tickets(df, scores, tickets):
    destination_tickets_df = pd.read_csv('game_data/destinations.csv')

    for key in scores.keys():
        color_df = df[df['color']==key][['location1', 'location2']]
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
            
            if destination_complete(color_df, start, end, set()):
                scores[key] += points
            else:
                scores[key] -= points

    
if __name__ == "__main__":
    game_state = create_game_state('clean_trains_in_some_spots', 'trained_station_model')
    scores = {'red':0, 'blue':0, 'yellow':0, 'green':0, 'black':0}

    train_points(game_state, scores)
    print(scores)

    longest_route(game_state, scores)
    print(scores)

    ##ISSUE WHEN SAME KEYS
    tickets = {'blue':{'Lisboa': 'Danzic', 'Danzic':'Bruxelles'}, 'yellow': {'Madrid':'Zurich','Madrid':'Dieppe'}, 
               'green':{'Edinburgh':'Paris', 'Athina':'Edinburgh', 'Rostov':'Smolensk'}, 'black':{'Cadiz':'Stockholm'}, 'red':{'London':'Berlin'}}
    destination_tickets(game_state, scores, tickets)
    print(scores)

    '''
    SCORING STILL NEEDS:
        - account for stations (destination and +4 per not used)
        - destination tickets input method
    '''


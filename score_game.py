from generate_game_state import create_game_state

LONGEST_ROUTE_POINTS = 10

def train_points(df, scores):
    for key in scores.keys():
        scores[key] += df[df['color']==key]['points'].sum()

def single_longest_route(single_df, visited, city, curr_path_length, longest_path):
    if city in visited:
        return
    
    visited.add(city)
    
    for index, row in single_df.iterrows():
        if row['location1'] == city and row['location2'] not in visited:
            single_longest_route(single_df, visited, row['location2'], curr_path_length+row['length'], longest_path)
        elif row['location2'] == city and row['location1'] not in visited:
            single_longest_route(single_df, visited, row['location1'], curr_path_length+row['length'], longest_path)

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
    print(longest_road_color)
    print(longest_roads)

# def destination_tickets(df, scores):
#     print('hi')
    
if __name__ == "__main__":
    game_state = create_game_state('clean_trains_in_some_spots')
    scores = {'red':0, 'blue':0, 'yellow':0, 'green':0, 'black':0}

    train_points(game_state, scores)
    print(scores)

    longest_route(game_state, scores)
    print(scores)

    # tickets = {'red':{}}
    # destination_tickets(game_state, scores, tickets)


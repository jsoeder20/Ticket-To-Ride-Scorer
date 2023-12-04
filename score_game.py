from generate_game_state import create_game_state


if __name__ == "__main__":
    game_state = create_game_state('messy_train_in_some_spots')

    # Now you can use the game_state DataFrame in this file
    print(game_state.to_string())
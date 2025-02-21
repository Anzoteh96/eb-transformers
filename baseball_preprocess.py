import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    # Part 1: get the batting ones first, should get a list of all the batting hit count. 
    batting = pd.read_csv("datasets/baseball/batting.csv")
    batting['year'] = batting['game_id'].apply(lambda x: int(x[3:7]))
    batting['date'] = pd.to_datetime(batting['game_id'].apply(lambda x: x[3:11]))
    batting_hit = batting[batting['stat'] == 'H']
    batting_hit 
    
    # Part 2: do the same, but for pitching. 
    pitching = pd.read_csv("datasets/baseball/pitching.csv")
    pitching['year'] = pitching['game_id'].apply(lambda x: int(x[3:7]))
    pitching['date'] = pd.to_datetime(pitching['game_id'].apply(lambda x: x[3:11]))
    pitching_hit = pitching[pitching['stat'] == 'H']
    pitching_hit 

    # Part 3: dump it to CSV file. 
    for year in range(a, b): # fill in with your own year of choice. 
        bat_yr = batting_hit[(batting_hit['year'] == year)]
        filename = os.path.join("datasets/baseball", "batting_hitcount_{}.csv".format(year))
        bat_yr.to_csv(filename)
        pitch_yr = pitching_hit[(pitching_hit['year'] == year)]
        filename = os.path.join("datasets/baseball", "pitching_hitcount_{}.csv".format(year))
        pitch_yr.to_csv(filename)

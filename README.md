# bracket_buster_v2.0

## Pipeline

**Scrape Data**

1. sos_list_scraper.py
    - scraped strength of schedule data
2. gamelog_scraper.py
    - scrapes teams gamelog data
3. player_scraper.py
    - scrapes roster and playter per 100 posession data

**Transform Data**

4. gamelog_stats_transform.py
5. data_merger.py
    - concats seasons
    - merges roster and player_per100 data
6. position_cluster.py
    - creates position clusters
    - creates team experiecnce factor
7. matchup_creator.py
    - merges gamelogs, clustering and experience dataframes
    - slices up dataframe to create matchups
    - saves final_model_data

**Modelling**
- model_optimization.py - skipping this step for now
- model_dumper.py
- model_tests.py

**Predicting**
- win_or_lose.py
- bracket_generator.py

**Utils**
- scraping_utils.py
- filters.py

**TODO for next update**
- Add config so only need to  update season in one place
- add season to file names on files that need archived
    - player_per100_full_date.pkl
    - player_stats_full.pkl
    - rostaer_full_data.csv
    - season-full_game_log_stats_data.pkl
    - team_clusters.pkl
    - team_experience.pkl
    - exp_gamelog_clust
- add full update shell script
    - archive past year files
    - run all scripts in order

**TODO**
- Save final brackets from each tourney
- vectorize height data in data merger player_roster_merger func
- fill missing height data in rosters instead of dropping all wiht NaNs 
- Create a testing framework to see which models are best
    - for full bracket 
    - for each round
        - ie. is tcf better in early rounds?
        - Can I create some ensemble of these models for better performace?

**Annual Update Prcess**
1. Add new season to `seasons_list.txt`
2. `sos_list_scraper.py`
    - creates sos_list{season}.csv to 0_scraped_data dir
3. `gamelog_scraper.py`
    - update `add_game_type` func
        - add new year season/tourney start and end dates
        - update if else section with new conditions
    - saves `season_{season}_gamelog_data.pkl` to `0_scraped_data` dir
4. `player_scraper.py`
    - saves `player_per100_{season}_data.pkl` & `roster_{season}_data.csv` to `0_scraped_data` dir
5. `gamelog_stats_transform.py`
    - saves `season_{season}_gamelog_stats_data.pkl` and `season_{season}_gamelog_final_stats_data.pkl` to `1_transformed_data` dir
6. `data_merger.py`
    - Manual: archive data to year specific folder
    - saves files to `2_full_season_data` dir
        - `player_per100_full_data.pkl`
        - `roster_full_data.csv`
        - `season_full_gamelog_stats_data.pkl`
        - `player_stats_full.pkl`
7. `position_cluster.py`
    - saves files to `2_full_season_data` dir
        - `team_clusters.pkl`
        - `team_experience.pkl`
8. `matchup_creator.py`
    - archive data to year specific folder
    - creates: `gamelog_exp_clust.pkl` & `season{season}_final_stats.pkl` in `3_model_data` dir
9. `model_dumper.py`
    - saves models in `fit_models` dir
10. `model_test.py`
    - update season in `if __name__ == '__main__'` block
    - tests models in `fit_models` dir and prints results
11. `winner_predictor.py`
    - update season in `if __name__ == '__main__'` block
12. `bracket_generator.py`
    - update season in `if __name__ == '__main__'` block
    - archive past year's brackets
    - create new initial bracket for current season's tournament

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

**Annual Update Prcess**
1. sos_list_scraper.py
    - add new year to seasons list in `if __name__ == '__main__'` block
2. gamelog_scraper.py
    - add new year to seasons list in `if __name__ == '__main__'` block
    - `add_game_type` func
        - add new year season/tourney start and end dates
        - update if else section with new conditions
3. player_scraper.py
    - add new year to seasons list in `if __name__ == '__main__'` block
4. gamelog_stats_transform.py
5. data_merger.py
6. position_cluster.py
7. matchup_creator.py

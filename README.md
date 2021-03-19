# bracket_buster_v2.0

## Pipeline

**Scrape Data**
1. sos_list_scraper.py
    - scraped strength of schedule data
2. gamelog_scraper.py
    - scrpaes teams gamelog data
3. player_scraper.py
    - scrapes roster and playter per 100 posession data

**Transform Data**
4. gamelog_stats_transform.py
    - TODO: Add season final stats func (from game_df_creator.py)
5. data_merger.py
    - concats seasons
    - merges roster and player_per100 data
5. position_cluster.py
    - creates position clusters
    - creates team experiecnce factor
6. matchup_creator.py
    - merges gamelogs, clustering and experience dataframes
    - slices up dataframe to create matchups
    - saves final_model_data

**Modelling**
- model_optimization.py - skipping htis step for now
- model_dumper.py
- model_tests.py

**Predicting**
- win_or_lose.py
- bracket_generator.py

**Utils**
- scraping_utils.py
- filters.py

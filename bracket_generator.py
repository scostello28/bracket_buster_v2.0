import pickle
import pandas as pd
import numpy as np
from filters import pre_matchup_feature_selection
from scraping_utils import check_for_file

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class BracketGen:
    
    def __init__(self, bracket, pickled_model_path, final_stats_df, tcf=True):
        self.first_round = bracket
        self.second_round = None
        self.sweet16 = None
        self.elite8 = None
        self.final4 = None
        self.championship = None
        self.champion = None

        with open(pickled_model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.final_stats_df = final_stats_df
        self.tcf = tcf

    def merge(self, team1, team2):
        '''
        INPUT: DataFrame
        OUTPUT: DataFrame with matching IDs merged to same row
        '''
        if self.tcf:
            df = self.final_stats_df[
                [
                    'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
                    'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
                    'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0',
                    'G1', 'G2', 'G3'
                    ]
            ]

        else:
            df = self.final_stats_df[
                [
                    'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
                    'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos'
                    ]
                ]

        '''Create separate dataframes for 1st and 2nd instances of games'''
        df1 = df.loc[df['Tm'] == team1,:]
        df2 = df.loc[df['Tm'] == team2,:]

        df2cols = df2.columns.tolist()
        OPcols = ['OP{}'.format(col) for col in df2cols]
        df2.columns = OPcols

        '''Force unique ID to merge on'''
        df1['game'] = 'game'
        df2['game'] = 'game'

        '''Merge games instance DataFrames'''
        dfout = pd.merge(df1, df2, how='left', on='game')

        '''Drop uneeded columns'''
        dfout = dfout.drop(['game', 'Tm', 'OPTm'], axis=1)

        return dfout, team1, team2

    @staticmethod
    def game_predict(model, matchup, matchup_reversed, team1, team2):
        '''Predict on matchup'''
        prob = model.predict_proba(matchup)
        prob_reversed = model.predict_proba(matchup_reversed)
        team1_prob = (prob[0][1] + prob_reversed[0][0]) / 2 * 100
        team2_prob = (prob[0][0] + prob_reversed[0][1]) / 2 * 100

        if team1_prob > team2_prob:
            return team1
        else:
            return team2
    

    @staticmethod
    def _pick_winner_random(team1, team2):
        matchup = [team1, team2]
        winner = matchup[randint(0,1)]
        return winner

    
    def _pick_winner(self, team1, team2):
        matchup, team1, team2 = self.merge(team1, team2)
        matchup_reversed, team2_rev, team1_rev = self.merge(team2, team1)
        return BracketGen.game_predict(self.model, matchup, matchup_reversed, team1, team2)

    
    def _pick_round(self, round_list):
        next_round = []
        i = 0
        while i <= len(round_list)-2:
            team1, team2 = round_list[i], round_list[i+1]
            winner = self._pick_winner(team1, team2)
#             print(f"{team1} v {team2} - {winner} wins")
            next_round.append(winner)
            i += 2
        return next_round

    
    def gen_bracket(self, verbose=True, bracket_name=None):
        self.second_round = self._pick_round(self.first_round)
        self.sweet16 = self._pick_round(self.second_round)
        self.elite8 = self._pick_round(self.sweet16)
        self.final4 = self._pick_round(self.elite8)
        self.championship = self._pick_round(self.final4)
        self.champion = self._pick_round(self.championship)

        if bracket_name:

            f = open(f"brackets/{bracket_name}.txt", 'w')
            print("First Round", file=f)
            print("-----------", file=f)
            BracketGen.print_list(self.first_round, f)
            print("\n", file=f)
            print("Second Round", file=f)
            print("------------", file=f)
            BracketGen.print_list(self.second_round, f)
            print("\n", file=f)
            print("Sweet 16", file=f)
            print("-------", file=f)
            BracketGen.print_list(self.sweet16, f)
            print("\n", file=f)
            print("Elite 8", file=f)
            print("-------", file=f)
            BracketGen.print_list(self.elite8, f)
            print("\n", file=f)
            print("Final4", file=f)
            print("------", file=f)
            BracketGen.print_list(self.final4, f)
            print("\n", file=f)
            print("Championship", file=f)
            print("------------", file=f)
            BracketGen.print_list(self.championship, f)
            print("\n", file=f)
            print("Champion", file=f)
            print("--------", file=f)
            BracketGen.print_list(self.champion, f)
            f.close()
    
        if verbose:
            print(f"First_round: {self.first_round}")
            print("\n")
            print(f"Second_round: {self.second_round}")
            print("\n")
            print(f"Sweet16: {self.sweet16}")
            print("\n")
            print(f"Elite8: {self.elite8}")
            print("\n")
            print(f"Final4: {self.final4}")
            print("\n")
            print(f"Championship: {self.championship}")
            print("\n")
            print(f"Champion: {self.champion}")

    @staticmethod
    def print_list(l, f):
        i=0
        while i < len(l):
            print(" v ".join(l[i: i+2]), file=f)
            i+=2


if __name__ == '__main__':

    bracket_2021 = [
        # WEST
        'gonzaga',
        'norfolk-state',
        'oklahoma',
        'missouri',
        'creighton',
        'california-santa-barbara',
        'virginia',
        'ohio',
        'southern-california', 
        'drake',
        'kansas',
        'eastern-washington',
        'oregon',
        'virginia-commonwealth', 
        'iowa',
        'grand-canyon',
        # EAST
        'michigan',
        'texas-southern', 
        'louisiana-state',
        'st-bonaventure',
        'colorado',
        'georgetown',
        'florida-state',
        'north-carolina-greensboro',
        'brigham-young',
        'ucla', 
        'texas',
        'abilene-christian',
        'connecticut',
        'maryland',
        'alabama',
        'iona',
        # SOUTH
        'baylor',
        'hartford',
        'north-carolina',
        'wisconsin',
        'villanova',
        'winthrop',
        'purdue',
        'north-texas',
        'texas-tech',
        'utah-state',
        'arkansas',
        'colgate',
        'florida',
        'virginia-tech',
        'ohio-state', 
        'oral-roberts',
        # MIDWEST
        'illinois',
        'drexel',
        'loyola-il', 
        'georgia-tech',
        'tennessee',
        'oregon-state',
        'oklahoma-state',
        'liberty',
        'san-diego-state',
        'syracuse',
        'west-virginia',
        'morehead-state',
        'clemson',
        'rutgers',
        'houston',
        'cleveland-state'
    ]


    models = {
        "lr": "fit_models/lr_2021_fit_model.pkl",
        "rf": "fit_models/rf_2021_fit_model.pkl",
        "gb": "fit_models/gb_2021_fit_model.pkl",
        "lr_nc": "fit_models/lr_2021_fit_model_no_clust.pkl",
        "rf_nc": "fit_models/rf_2021_fit_model_no_clust.pkl",
        "gb_nc": "fit_models/gb_2021_fit_model_no_clust.pkl"
    }  

    final_stats_df = pd.read_pickle('3_model_data/season2021_final_stats.pkl')
    finalgames2021_data = final_stats_df[final_stats_df['GameType'] == 'season2021']
    finalgames2021_exp_tcf = pre_matchup_feature_selection(finalgames2021_data, 'exp_tcf')
    finalgames2021 = pre_matchup_feature_selection(finalgames2021_data, 'gamelogs')



    lr_tcf = BracketGen(
        bracket=bracket_2021, 
        pickled_model_path=models["lr"], 
        final_stats_df=finalgames2021_exp_tcf, 
        tcf=True)
    lr_tcf.gen_bracket(bracket_name="lr_tcf")

    rf_tcf = BracketGen(
        bracket=bracket_2021, 
        pickled_model_path=models["rf"], 
        final_stats_df=finalgames2021_exp_tcf, 
        tcf=True)
    rf_tcf.gen_bracket(bracket_name="rf_tcf")

    gb_tcf = BracketGen(
        bracket=bracket_2021, 
        pickled_model_path=models["gb"], 
        final_stats_df=finalgames2021_exp_tcf, 
        tcf=True)
    gb_tcf.gen_bracket(bracket_name="gb_tcf")

    lr = BracketGen(
        bracket=bracket_2021, 
        pickled_model_path=models["lr_nc"], 
        final_stats_df=finalgames2021, 
        tcf=False)
    lr.gen_bracket(bracket_name="lr")

    rf = BracketGen(
        bracket=bracket_2021, 
        pickled_model_path=models["rf_nc"], 
        final_stats_df=finalgames2021, 
        tcf=False)
    rf.gen_bracket(bracket_name="rf")

    gb = BracketGen(
        bracket=bracket_2021, 
        pickled_model_path=models["gb_nc"], 
        final_stats_df=finalgames2021, 
        tcf=False)
    gb.gen_bracket(bracket_name="gb")
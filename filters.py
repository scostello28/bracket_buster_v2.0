import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def games_up_to_2021_season_filter(df):
    '''Filter for games up to 2021 season'''
    notourney2021 = (df['GameType'] != 'tourney2021')
    noseason2021 = (df['GameType'] != 'season2021')
    games_up_to_2021_season = df[notourney2021 & noseason2021]
    return games_up_to_2021_season

def season2021_filter(df):
    '''Filter for 2021 season games'''
    season2021cond = (df['GameType'] == 'season2021')
    season2021 = df[season2021cond]
    return season2021

def games_up_to_2021_tourney_filter(df):
    '''Filter for games up to 2021 tourney'''
    notourney2021 = (df['GameType'] != 'tourney2021')
    games_up_to_2021_tourney = df[notourney2021]
    return games_up_to_2021_tourney

def tourney2021_filter(df):
    '''Filter for 2021 tourney games'''
    tourney2021cond = (df['GameType'] == 'tourney2021')
    tourney2021 = df[tourney2021cond]
    return tourney2021

def apply_filter(df, filter):
    return filter(df)


def pre_matchup_feature_selection(df, feature_set='gamelogs'):
    '''
    Inputs: Model DataFrame
    Outputs: DataFrame with features selected
    '''

    if feature_set == 'gamelogs':
        df = df[['W', 'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos']]

    elif feature_set == 'exp':
        df = df[['W', 'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor']]

    elif feature_set == 'tcf':
        df = df[['W', 'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1', 'G2', 'G3']]

    elif feature_set == 'exp_tcf':
        df = df[['W', 'Tm', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1',
            'G2', 'G3']]

    elif feature_set == 'odds':
        df = df[['W', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1',
            'G2', 'G3', 'final_p']]

    return df


def post_merge_feature_selection(df, feature_set='gamelogs'):
    '''
    Inputs: Model DataFrame
    Outputs: DataFrame with features selected
    '''

    if feature_set == 'gamelogs':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp', 'OPORBpg',
            'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg', 'OPPFpg',
            'OPsos']]

    elif feature_set == 'exp':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'exp_factor', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor']]

    elif feature_set == 'tcf':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp', '3Pp', 'FTp', 'ORBpg',
            'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'sos',
            'C0', 'C1', 'C2', 'F0', 'F1', 'F2', 'G0', 'G1', 'G2', 'G3',
            'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp', 'OPFTp', 'OPORBpg',
            'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg', 'OPTOpg', 'OPPFpg',
            'OPsos', 'OPC0', 'OPC1', 'OPC2', 'OPF0', 'OPF1', 'OPF2', 'OPG0',
            'OPG1', 'OPG2', 'OPG3']]

    elif feature_set == 'exp_tcf':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
            'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2',
            'OPF0', 'OPF1', 'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3']]

    elif feature_set == 'odds':
        df = df[['W', 'GameType', 'Wp', 'ppg', 'pApg', 'FGp',
            '3Pp', 'FTp', 'ORBpg', 'RBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg',
            'PFpg', 'sos', 'exp_factor', 'C0', 'C1', 'C2', 'F0', 'F1', 'F2',
            'G0', 'G1', 'G2', 'G3', 'OPWp', 'OPppg', 'OPpApg', 'OPFGp', 'OP3Pp',
            'OPFTp', 'OPORBpg', 'OPRBpg', 'OPASTpg', 'OPSTLpg', 'OPBLKpg',
            'OPTOpg', 'OPPFpg', 'OPsos', 'OPexp_factor', 'OPC0', 'OPC1', 'OPC2',
            'OPF0', 'OPF1', 'OPF2', 'OPG0', 'OPG1', 'OPG2', 'OPG3', 'final_p']]

    return df


def data_for_model(df, feature_set='gamelogs', train_filter=games_up_to_2021_season_filter, test_filter=season2021_filter):
    '''
    Inputs: Model DataFrame
    Outputs: train and test DataFrames
    '''

    df = post_merge_feature_selection(df, feature_set=feature_set)
    train_df = apply_filter(df, train_filter)
    test_df = apply_filter(df, test_filter)

    train_df = train_df.drop(['GameType'], axis=1)
    test_df = test_df.drop(['GameType'], axis=1)

    return train_df, test_df


def set_up_data(train_df, test_df, bracket=False):
    '''Set up features and targets'''
    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values

    '''Balance classes'''
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    '''Standardize data'''
    scale = StandardScaler()
    scale.fit(X_train)
    X_train = scale.transform(X_train)

    if not bracket:
        X_test = scale.transform(X_test)

        return X_train, y_train, X_test, y_test
    else:
        return X_train, y_train

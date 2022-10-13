import numpy as np
import pandas as pd



def get_str_labels(df_songs:pd.DataFrame, column_name:str="majorityGenre"):
    """
    Return np.array of labels y_str
    """
    return df_songs[column_name].to_numpy()
    

def get_labels(y_str:np.ndarray):
    """
    Return tuple ()
    """
    return np.unique(y_str, return_inverse=True, return_counts=True)
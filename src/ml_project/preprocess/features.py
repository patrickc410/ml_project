import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler


spotify_feature_list = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "mode",
    "speechiness",
    "tempo",
    "time_signature",
    "valence",
]

spotify_continuous_feature_list = [
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo",
    "valence",
]


def get_spotify_feature_matrix(df_songs, continous=False):
    """
    Parameters:
    - df_songs (pd.DataFrame)
    - continuous (boolean)

    Returns:
    - feature_matrix (np.array)
    """

    if continous == False:
        return df_songs[spotify_feature_list].to_numpy()
    else:
        return df_songs[spotify_continuous_feature_list].to_numpy()


def scale_feature_matrix(X, scaler_type="standard"):
    """
    Parameters:
    - X (np.array)
    - scaler_type: standard or minmax

    Returns:
    - X_scl
    """

    if scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type.lower() == "standard":
        scaler = StandardScaler()
    else:
        print(f"unsupported scaler_type: {scaler_type}")
        return X
    return scaler.fit_transform(X)


def poly_features(X, degree=2, interaction_only=False, include_bias=False):
    """
    Parameters:
    - X (np.array)
    - degree (int)
    - interaction_only (bool)
    - include_bias (bool)

    Returns:
    - X_poly (np.array)
    """

    poly = PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=include_bias
    )
    return poly.fit_transform(X)


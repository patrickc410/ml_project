import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
import os


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


def preprocess_df_features(df_songs):

    X = get_spotify_feature_matrix(df_songs)
    X_rdf = get_spotify_feature_matrix(df_songs, True)
    X_poly = poly_features(X)
    X_rdf_poly = poly_features(X_rdf)

    X_scl = scale_feature_matrix(X)
    X_rdf_scl = scale_feature_matrix(X_rdf)
    X_poly_scl = scale_feature_matrix(X_poly)
    X_rdf_poly_scl = scale_feature_matrix(X_rdf_poly)

    X_dict = {
        "X": X,
        "X_rdf": X_rdf,
        "X_poly": X_poly,
        "X_rdf_poly": X_rdf_poly,
        "X_scl": X_scl,
        "X_rdf_scl": X_rdf_scl,
        "X_poly_scl": X_poly_scl,
        "X_rdf_poly_scl": X_rdf_poly_scl,
    }
    return X_dict


def save_all(folder_path, X_dict, y_str, genres, y, genre_counts):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"directory created: {folder_path}")
    else:
        print(f"directory exists: {folder_path}")
        overwrite = input(
            "Are you sure you want to overwrite the contents of this directory? [y,n] "
        )
        if overwrite.lower() != "y":
            print("aborting save_all command")
            return

    for k, X in X_dict.items():
        save_npy(folder_path, k, X)

    save_npy(folder_path, "y", y)
    save_npy(folder_path, "y_str", y_str)
    save_npy(folder_path, "genres", genres)
    save_npy(folder_path, "genre_counts", genre_counts)


def save_npy(folder_path, file_name, array):
    with open(f"{folder_path}/{file_name}.npy", "wb") as f:
        np.save(f, array)
        print(f"saved: {folder_path}/{file_name}.npy")


def load_npy(folder_path, file_name):
    with open(f"{folder_path}/{file_name}.npy", "rb") as f:
        arr = np.load(f, allow_pickle=True)
        return arr


def load_npy_data(folder_path, X_path):
    X = load_npy(folder_path, X_path)
    genres = load_npy(folder_path, "genres")
    genre_counts = load_npy(folder_path, "genre_counts")
    y_str = load_npy(folder_path, "y_str")
    y = load_npy(folder_path, "y")

    return X, y, y_str, genres, genre_counts

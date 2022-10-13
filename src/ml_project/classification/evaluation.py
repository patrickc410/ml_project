import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


def df_confusion_matrix(
    classifier, X_test: np.ndarray, y_test: np.ndarray, genres: np.ndarray, nn=False
) -> pd.DataFrame:
    """
    Parameters:
    - classifier: an object with a .predict() function
    - X_test: test data to predict with
    - y_test: ground truth labels for X_test
    - genres: list of strings mapping y_test integer label values to genre string labels

    Returns:
    - heatmap confusion matrix for visualizing what the classifier predicts well and what it predicts poorly
    """
    columns = np.char.add(genres.astype(str), "_label")
    index = np.char.add(genres.astype(str), "_pred")
    if nn == True:
        pred_probs = classifier.predict(X_test)
        predictions = np.argmax(pred_probs, axis=1)
    else:
        predictions = classifier.predict(X_test)

    df_conf_matrix = pd.DataFrame(
        data=metrics.confusion_matrix(genres[y_test], genres[predictions]),
        columns=columns,
        index=index,
    )
    return df_conf_matrix.style.background_gradient(cmap="coolwarm")


def plot_feature_importances(model, feature_list):
    indices = np.argsort(model.feature_importances_)
    feature_list_sorted = np.array(feature_list)[indices]
    imp_sorted = np.array(model.feature_importances_)[indices]

    plt.barh(feature_list_sorted, imp_sorted)
    plt.xlabel("Importance in Model")
    plt.ylabel("Feature Name")
    plt.title("Feature Importances")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def plot_label_distribution(y_str: np.ndarray) -> None:
    counter = Counter(y_str)
    for k, v in counter.items():
        per = v / len(y_str) * 100

    # plot the distribution
    plt.barh(list(counter.keys()), counter.values())
    plt.xlabel("Record Count")
    plt.ylabel("Genre")
    plt.title("Distribution of Genres in Labeled Data")
    plt.show()

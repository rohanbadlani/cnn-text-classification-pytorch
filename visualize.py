import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


def get_labels_from_csv(csv_file: str, label: str) -> np.array:
    df = pd.read_csv(csv_file)
    return np.array(df[[label]].values)


def get_overlap(arr1: list, arr2: list) -> (int, int, int):
    arr1, arr2 = np.array(arr1), np.array(arr2)
    total_overlap = np.sum(arr1 == arr2)
    positive_overlap = np.sum(np.logical_and(arr1 == arr2, arr1 == 1))
    negative_overlap = np.sum(np.logical_and(arr1 == arr2, arr1 == 0))
    return total_overlap, positive_overlap, negative_overlap


if __name__ == "__main__":
    true_labels = [1,1,1,0,0]
    predictions = [1,0,1,1,0]
    print(get_overlap([1,1,1,0,0], [1,0,1,1,0]))
    print(classification_report(true_labels, predictions))

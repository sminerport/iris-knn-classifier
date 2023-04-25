import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# k-nearest neighbors on the Iris Flowers Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import re

# Load a CSV file


def load_csv(filename):
    """
    Load a CSV file and return the contents as a list of lists.

    Each row in the CSV file is represented as a list, and the entire dataset
    is returned as a list of these rows.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list of lists containing the rows of the CSV file.
    """

    dataset = list()
    with open(filename, "r") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float


def str_column_to_float(dataset, column):
    """
    Convert the values in a specified column of a dataset from strings to floats.

    This function modifies the input dataset in-place, converting the string
    values in the specified column to floating-point numbers.

    Args:
        dataset (list): A list of lists representing the dataset. Each row is
                        a list, and the values in the specified column should
                        be strings that can be converted to floats.
        column (int): The index of the column to convert.
    """

    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer


def str_column_to_int(dataset, column):
    """
    Convert the string values in a specified column of a dataset to integers.

    This function assigns a unique integer value to each unique string value
    in the specified column. It then modifies the input dataset in-place,
    replacing the string values with their corresponding integer values.

    Args:
        dataset (list): A list of lists representing the dataset. Each row is
                        a list, and the values in the specified column should
                        be strings.
        column (int): The index of the column to convert.

    Returns:
        dict: A dictionary mapping the original string values to their assigned
              integer values.
    """

    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    # create dictionary
    for i, value in enumerate(unique):
        lookup[value] = i
    # convert dataset column
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column


def dataset_minmax(dataset):
    """
    Calculate the minimum and maximum values for each column in the dataset.

    This function iterates through each column in the dataset, finding the
    minimum and maximum values. It returns a list of pairs (min, max) for
    each column.

    Args:
        dataset (list): A list of lists representing the dataset. Each row is
                        a list containing numerical values.

    Returns:
        list: A list of pairs (min, max) for each column in the dataset.
    """

    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1


def normalize_dataset(dataset, minmax):
    """
    Normalize the dataset using the provided minimum and maximum values.

    This function normalizes each value in the dataset using the min-max
    normalization method. It modifies the input dataset in-place, replacing
    each value with its normalized counterpart.

    Args:
        dataset (list): A list of lists representing the dataset. Each row is
                        a list containing numerical values.
        minmax (list): A list of pairs (min, max) for each column in the dataset.
                       This can be generated using the `dataset_minmax` function.
    """

    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds


def cross_validation_split(dataset, n_folds):
    """
    Split the dataset into k equally sized folds for cross-validation.

    This function creates a list of k folds, where each fold contains
    approximately the same number of rows from the dataset. The dataset
    is randomly divided among the folds.

    Args:
        dataset (list): A list of lists representing the dataset. Each row is
                        a list containing numerical values.
        n_folds (int): The number of folds to divide the dataset into.

    Returns:
        list: A list of folds, where each fold is a list of rows from the
              dataset. The dataset is randomly divided among the folds.
    """

    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage


def accuracy_metric(actual, predicted):
    """
    Calculate the classification accuracy of predictions compared to actual values.

    This function computes the percentage of correct predictions by comparing
    the actual values to the predicted values.

    Args:
        actual (list): A list of actual class labels.
        predicted (list): A list of predicted class labels.

    Returns:
        float: The percentage of correct predictions (accuracy) as a float value.
    """

    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """
    Evaluate a classification algorithm using k-fold cross-validation.

    This function splits the dataset into k folds, then iteratively trains
    and tests the classification algorithm on each fold. The classification
    accuracy is calculated for each test, and the average accuracy is returned.

    Args:
        dataset (list): A list of lists representing the dataset. Each row is
                        a list containing numerical values.
        algorithm (callable): The classification algorithm to evaluate. This
                              should be a function that takes a training set,
                              a test set, and any additional arguments, and
                              returns a list of predicted class labels.
        n_folds (int): The number of folds to divide the dataset into for
                       cross-validation.
        *args: Any additional arguments required by the classification algorithm.

    Returns:
        list: A list of accuracy scores, one for each fold of the cross-validation.
    """

    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        # create hold out set
        train_set.remove(fold)
        # combine train sets
        train_set = sum(train_set, [])
        # create test set on new hold
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            # remove prediction from hold out set
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Calculate the Euclidean distance between two vectors


def euclidean_distance(row1, row2):
    """
    Calculate the Euclidean distance between two rows.

    This function computes the Euclidean distance between two rows of equal
    length, excluding the last element (usually the class label).

    Args:
        row1 (list): The first row containing numerical values.
        row2 (list): The second row containing numerical values.

    Returns:
        float: The Euclidean distance between the two rows.
    """

    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors


def get_neighbors(train, test_row, num_neighbors):
    """
    Locate the k most similar neighbors in the training set for a given test row.

    This function computes the Euclidean distances between a test row and all
    rows in the training set. It then returns the k nearest neighbors.

    Args:
        train (list): A list of lists representing the training dataset. Each row
                      contains numerical values.
        test_row (list): A row from the test dataset containing numerical values.
        num_neighbors (int): The number of neighbors to return.

    Returns:
        list: A list of the k nearest neighbors from the training dataset.
    """

    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors


def predict_classification(train, test_row, num_neighbors):
    """
    Make a classification prediction for a test row using k-nearest neighbors.

    This function finds the k-nearest neighbors for a given test row in the
    training dataset, then returns the most common class label among those neighbors.

    Args:
        train (list): A list of lists representing the training dataset. Each row
                      contains numerical values.
        test_row (list): A row from the test dataset containing numerical values.
        num_neighbors (int): The number of neighbors to use for the prediction.

    Returns:
        int or str: The predicted class label for the test row.
    """

    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# kNN Algorithm


def k_nearest_neighbors(train, test, num_neighbors):
    """
    Make classification predictions for all rows in the test dataset using the k-nearest neighbors algorithm.

    This function iterates through each row in the test dataset and makes a classification prediction
    using the k-nearest neighbors algorithm on the training dataset.

    Args:
        train (list): A list of lists representing the training dataset. Each row
                      contains numerical values.
        test (list): A list of lists representing the test dataset. Each row
                     contains numerical values.
        num_neighbors (int): The number of neighbors to use for the predictions.

    Returns:
        list: A list of predicted class labels for all rows in the test dataset.
    """

    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions


# Test the kNN on the Iris Flowers dataset
seed(2)
filename = "data/iris.txt"
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset[1:], i)
# convert class column to integers
# versicolor : 0
# virginica: 1
# setosa: 2
lookup = str_column_to_int(dataset[1:], len(dataset[0]) - 1)

# evaluate algorithm
n_folds = 5
num_neighbors = 10
scores = evaluate_algorithm(
    dataset[1:], k_nearest_neighbors, n_folds, num_neighbors)
print(f"****************************************************************************************")
print(f"*")
print(f"*   K-Nearest Neighbor (KNN) algorithm with {num_neighbors} neighbors trained on a ")
print(f"*   dataset containing  {len(dataset)-1} rows and {len(dataset[1])-1} features, using {n_folds}-fold cross validation.")
print(f"*")
print(f"*   Users can adjust the n_folds and num_neighbors variables in the script.")
print(f"*")
print(f"****************************************************************************************")
print()
print(f"Accuracy per fold: {scores}")
print(f"Mean Accuracy: {sum(scores) / float(len(scores)):.3f}")

while True:
    try:
        sl, sw, pl, pw = [
            float(x)
            for x in re.split(
                r"\, | |\,",
                input(
                    "\nplease input four floating point numbers, representing, respectively,\n\
sepal length, sepal width, petal length, and petal width \n(e.g., 5.1, 3.5, 1.4, 0.2).  The model will guess the flower category\n\
(i.e., setosa, versicolor, or virginica) based on your input (CTRL-C to Exit): "
                ),
            )
        ]
        test_row = [sl, sw, pl, pw]
        test_row
        prediction = predict_classification(
            train=dataset[1:], test_row=test_row, num_neighbors=num_neighbors
        )
        for key, value in lookup.items():
            if prediction == value:
                prediction = key
        print()
        print(f"Prediction: {prediction}")
    except ValueError:
        print("\nNote: wrong input format.")

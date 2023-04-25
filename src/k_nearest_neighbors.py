from math import sqrt


class KNearestNeighbors:
    """
    A class for the k-nearest neighbors classification algorithm.

    Methods:
        euclidean_distance(row1: List[float], row2: List[float]): Calculate the Euclidean distance between two rows.
        get_neighbors(train: List[List[float]], test_row: List[float], num_neighbors: int): Locate the k most similar neighbors in the training set for a given test row.
        predict_classification(train: List[List[float]], test_row: List[float], num_neighbors: int): Make a classification prediction for a test row using k-nearest neighbors.
        predict_all(train: List[List[float]], test: List[List[float]], num_neighbors: int): Make classification predictions for all rows in the test dataset using the k-nearest neighbors algorithm.
    """

    def __init__(self, num_neighbors):
        """
        Initialize the KNearestNeighbors with the specified number of neighbors.

        Args:
            num_neighbors (int): The number of neighbors to use for the predictions.
        """
        self.num_neighbors = num_neighbors

    @staticmethod
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

    def get_neighbors(self, train, test_row):
        """
        Locate the k most similar neighbors in the training set for a given test row.

        This function computes the Euclidean distances between a test row and all
        rows in the training set. It then returns the k nearest neighbors.

        Args:
            train (list): A list of lists representing the training dataset. Each row
                          contains numerical values.
            test_row (list): A row from the test dataset containing numerical values.

        Returns:
            list: A list of the k nearest neighbors from the training dataset.
        """

        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def predict_classification(self, train, test_row):
        """
        Make a classification prediction for a test row using k-nearest neighbors.

        This function finds the k-nearest neighbors for a given test row in the
        training dataset, then returns the most common class label among those neighbors.

        Args:
            train (list): A list of lists representing the training dataset. Each row
                          contains numerical values.
            test_row (list): A row from the test dataset containing numerical values.

        Returns:
            int or str: The predicted class label for the test row.
        """

        neighbors = self.get_neighbors(train, test_row)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    def predict_all(self, train, test):
        """
        Make classification predictions for all rows in the test dataset using the k-nearest neighbors algorithm.

        This function iterates through each row in the test dataset and makes a classification prediction
        using the k-nearest neighbors algorithm on the training dataset.

        Args:
            train (list): A list of lists representing the training dataset. Each row
                          contains numerical values.
            test (list): A list of lists representing the test dataset. Each row
                         contains numerical values.

        Returns:
            list: A list of predicted class labels for all rows in the test dataset.
        """

        predictions = list()
        for row in test:
            output = self.predict_classification(train, row)
            predictions.append(output)
        return predictions

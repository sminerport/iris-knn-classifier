from random import randrange

class CrossValidator:
    """
    A class for evaluating classification algorithms using k-fold cross-validation.

    Methods:
        cross_validation_split(dataset: List[List[float]], n_folds: int): Split the dataset into k equally sized folds for cross-validation.
        accuracy_metric(actual: List[int], predicted: List[int]): Calculate the classification accuracy of predictions compared to actual values.
        evaluate_algorithm(dataset: List[List[float]], algorithm: Callable, n_folds: int, *args): Evaluate a classification algorithm using k-fold cross-validation.
    """

    def __init__(self, n_folds):
        """
        Initialize the CrossValidator with the specified number of folds.

        Args:
            n_folds (int): The number of folds to divide the dataset into for cross-validation.
        """
        self.n_folds = n_folds

    def cross_validation_split(self, dataset):
        """
        Split the dataset into k equally sized folds for cross-validation.

        This function creates a list of k folds, where each fold contains
        approximately the same number of rows from the dataset. The dataset
        is randomly divided among the folds.

        Args:
            dataset (list): A list of lists representing the dataset. Each row is
                            a list containing numerical values.

        Returns:
            list: A list of folds, where each fold is a list of rows from the
                  dataset. The dataset is randomly divided among the folds.
        """

        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / self.n_folds)
        for _ in range(self.n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    @staticmethod
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

    def evaluate_algorithm(self, dataset, algorithm, *args):
        """
        Evaluate a classification algorithm using k-fold cross-validation.

        Args:
            dataset (list): A list of lists representing the dataset. Each row is
                            a list containing numerical values.
            algorithm (callable): The classification algorithm to evaluate. This
                                  should be a function that takes a training set,
                                  a test set, and any additional arguments, and
                                  returns a list of predicted class labels.
            *args: Any additional arguments required by the classification algorithm.

        Returns:
            list: A list of accuracy scores, one for each fold of the cross-validation.
        """

        folds = self.cross_validation_split(dataset)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = CrossValidator.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

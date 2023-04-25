class Preprocessor:
    """
    A class for preprocessing datasets.

    Methods:
        normalize_dataset(dataset: List[List[float]], minmax: List[Tuple[float, float]]): Normalize the dataset using the provided minimum and maximum values.
        dataset_minmax(dataset: List[List[float]]): Calculate the minimum and maximum values for each column in the dataset.
    """

    @staticmethod
    def convert_to_float(dataset, column):
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

    @staticmethod
    def convert_to_int(dataset, column):
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
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup

    @staticmethod
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

    @staticmethod
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
            minmax.append((value_min, value_max))
        return minmax

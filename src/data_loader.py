from csv import reader


class DataLoader:
    """
    A class for loading and preprocessing a dataset from a CSV file.

    Attributes:
        filename (str): The path to the CSV file.

    Methods:
        load_data(): Load the CSV file and return its contents as a list of lists.
        convert_to_float(dataset: List[List[str]], column: int): Convert the values in a specified column to floats.
        convert_to_int(dataset: List[List[str]], column: int): Convert the string values in a specified column to integers.
    """

    def __init__(self, filename):
        """
        Initialize a DataLoader instance.

        Args:
            filename (str): The path to the CSV file.
        """

        self.filename = filename

    def load_data(self):
        """
        Load the CSV file and return its contents as a list of lists.

        Each row in the CSV file is represented as a list, and the entire dataset
        is returned as a list of these rows.

        Returns:
            list: A list of lists containing the rows of the CSV file.
        """

        dataset = list()
        with open(self.filename, "r") as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

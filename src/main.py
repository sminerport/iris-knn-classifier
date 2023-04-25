import re
from random import seed
from data_loader import DataLoader
from preprocessor import Preprocessor
from cross_validator import CrossValidator
from k_nearest_neighbors import KNearestNeighbors

# Test the kNN on the Iris Flowers dataset
seed(2)
filename = "data/iris.txt"
dl = DataLoader(filename)
dataset = dl.load_data()

# Assuming the dataset variable contains your dataset as a list of lists
header, data = dataset[0], dataset[1:]

# Iterate over all columns except the last one
for column_index in range(len(header) - 1):
    # Call the convert_to_float static method for each column
    Preprocessor.convert_to_float(data, column_index)

lookup = Preprocessor.convert_to_int(data, len(header) - 1)


# Cross-validation setup
n_folds = 5
num_neighbors = 10
cross_validator = CrossValidator(n_folds=n_folds)

# kNN setup
knn = KNearestNeighbors(num_neighbors)
scores = cross_validator.evaluate_algorithm(data, knn.predict_all)

# Print results
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
        prediction = knn.predict_classification(dataset[1:], test_row)
        for key, value in lookup.items():
            if prediction == value:
                prediction = key
        print()
        print(f"Prediction: {prediction}")
    except ValueError:
        print("\nNote: wrong input format.")

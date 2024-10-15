import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("/Desktop/forexforecast/adaptive-forex-forecast/Adaptive filters/")
sys.path.append("/Desktop/forexforecast/adaptive-forex-forecast/Feature extractors/")

from lms import lms
from feature_extractor_functions import simple_amv_extractor
from normalization_functions import simple_normalize

# Constants
WINDOW_SIZE = 10
MU = 0.000195

def load_data(filename):
    '''Load dataset from a CSV file.'''
    try:
        dataset = pd.read_csv(filename)
        return dataset['Price'].values
    except FileNotFoundError:
        print(f"File {filename} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("No data found in the file.")
        sys.exit(1)

def predict(inputs, weights):
    '''Predict the exchange rate based on inputs and weights.'''
    return np.dot(inputs, weights)

def plot_convergence_characteristics(errors):
    '''Plot the convergence characteristics.'''
    errors_squared = np.square(errors)
    plt.plot(errors_squared)
    plt.xlabel('Pattern number')
    plt.ylabel('(Error)^2')
    plt.title('Convergence Characteristics')
    plt.show()

def train_model(training_data, mu):
    '''Train the prediction model to find optimum weights.'''
    weights = np.zeros(len(training_data[0]) - 1)
    errors = []

    for i in training_data:
        x_k = i[:-1]
        d_k = i[-1]
        y_k = predict(x_k, weights)
        e_k = d_k - y_k
        errors.append(e_k)
        weights = lms(weights, mu, x_k, e_k)

    plot_convergence_characteristics(errors)
    return weights

def test_model(testing_data, weights):
    '''Test the performance of the model.'''
    errors = []
    desired = []
    predicted = []

    for i in testing_data:
        x_k = i[:-1]
        d_k = i[-1]
        y_k = predict(x_k, weights)
        predicted.append(y_k)
        desired.append(d_k)
        e_k = d_k - y_k
        errors.append(e_k)

    plt.plot(desired, 'g-', label="Desired Values")
    plt.plot(predicted, 'r-', label="Predicted Values")
    plt.xlabel('Pattern number')
    plt.ylabel('Normalized Exchange rate')
    plt.title('Predicted vs Desired Values')
    plt.legend(loc='best')
    plt.show()

def main():
    data = load_data('data.csv')
    n_data = simple_normalize(data)
    feature_table = simple_amv_extractor(WINDOW_SIZE, n_data[:-1])

    training_data = feature_table[:int(len(feature_table) * 0.8)]
    testing_data = feature_table[int(len(feature_table) * 0.8):]

    weights = train_model(training_data, MU)
    test_model(testing_data, weights)

if __name__ == "__main__":
    main()

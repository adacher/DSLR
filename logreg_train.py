import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def accuracy_score_(features, houses, weights):
    final_prediction = []
    for feature in np.insert(features, 0, 1, axis=1):
        dot = []
        for weight, house in weights:
            dot.append((feature.dot(weight), house))
        final_prediction.append(max(dot)[1])
    return sum(final_prediction == houses) / len(houses)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_(X, h, nb_rows, unique):
    a = np.dot(-unique.T, np.log(h))
    b = np.dot((1 - unique).T, np.log((1 - h),
                                      out=np.zeros_like(1 - h), where=(1 - h != 0)))
    cost = (a - b) / unique.size
    return cost


def fit_gradient_descent(X, houses):
    learning_rate = 0.01
    iterations = 10000
    X = np.insert(X, 0, 1, axis=1)
    nb_rows, nb_features = X.shape
    predictions = []
    for house in np.unique(houses):
        cost = []
        unique = np.where(houses == house, 1, 0)
        weights = np.zeros(nb_features)
        for _ in range(0, iterations):
            h = _sigmoid(np.dot(X, weights))
            cost.append(cost_(X, h, nb_rows, unique))
            dw = np.dot(X.T, (h - unique)) / unique.size
            weights -= learning_rate * dw
        predictions.append((weights, house))
        plt.plot(cost, label=house)
    plt.legend()
    plt.show()
    np.save('weights', predictions)
    return predictions


def scaling(x):
    for i in range(len(x)):
        x[i] = (x[i] - x.mean()) / x.std()
    return x


def delete_features(data):
    for feature in data:
        if feature != 'Hogwarts House':
            if data[feature].dtype != np.float64:
                del data[feature]
        if feature == 'Arithmancy':
            del data[feature]
        if feature == 'Care of Magical Creatures':
            del data[feature]
        if feature == 'Defense Against the Dark Arts':
            del data[feature]
    return data


def get_csv_data():
    nb_arg = len(sys.argv)
    if nb_arg != 2:
        print("Error: provide a csv file to test.")
        exit(1)
    try:
        data = pd.read_csv(sys.argv[1])
        return data
    except Exception as err:
        print("Error: " + str(err))
        exit(1)


def main():
    data = get_csv_data()
    data = delete_features(data)
    data = data.dropna()
    features = np.array(data.iloc[:, 1:])
    house_name = np.array(data.loc[:, "Hogwarts House"])
    np.apply_along_axis(scaling, 0, features)
    weights = fit_gradient_descent(features, house_name)
    acc = accuracy_score_(features, house_name, weights)
    print("Weights succesfully saved in weights.npy")
    print("accuracy score is : " + str(acc))
    return 0


if __name__ == "__main__":
    main()

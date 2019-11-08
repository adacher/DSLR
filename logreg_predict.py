import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def get_predictions(features, weights):
    final_prediction = []
    for feature in np.insert(features, 0, 1, axis=1):
        dot = []
        for weight, house in weights:
            dot.append((feature.dot(weight), house))
        final_prediction.append(max(dot)[1])
    return final_prediction


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
    if nb_arg != 3:
        print("Error: provide csv file and weights.")
        exit(1)
    try:
        data = pd.read_csv(sys.argv[1])
        return data
    except Exception as err:
        print("Error: " + str(err))
        exit(1)


def get_weights():
    try:
        weights = np.load(sys.argv[2], allow_pickle=True)
        return weights
    except:
        print("Error: couldn't load weights")
        exit(1)


def main():
    data = get_csv_data()
    data = delete_features(data)
    data = data.iloc[:, 1:]
    data.fillna(data.mean(), inplace=True)
    features = np.array(data)
    np.apply_along_axis(scaling, 0, features)
    weights = get_weights()
    predictions = get_predictions(features, weights)
    pd.DataFrame(predictions, columns=['Hogwarts House']).to_csv(
        "predictions.csv")
    return 0


if __name__ == "__main__":
    main()

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def accuracy_score_(features, houses, weights):
    final_prediction = []
    for feature in np.insert(features, 0, 1, axis=1):
        dot = []
        for weight, house in weights:
            dot.append((feature.dot(weight), house))
        final_prediction.append(max(dot)[1])
    return sum(final_prediction == houses) / len(houses)


def update_fig_axes_names(fig):
    fig.update_xaxes(title_text="Total marks", row=1, col=1)
    fig.update_yaxes(title_text="House", row=1, col=1)
    fig.update_xaxes(title_text="Total marks", row=1, col=2)
    fig.update_yaxes(title_text="Sigmoid", row=1, col=2)
    fig.update_xaxes(title_text="Iterations", row=2, col=1)
    fig.update_yaxes(title_text="Cost", row=2, col=1)
    fig.update_xaxes(title_text="Feature", row=2, col=2)
    fig.update_yaxes(title_text="Weight value", row=2, col=2)
    fig.update_layout(template="seaborn",
                      title_text="Logistic Regression", title_font_size=25)


def append_classification_coordinates(data):
    x, y, a, b = ([] for i in range(4))
    for i in range(0, len(data)):
        if data[i] == 0:
            a.append(i)
        else:
            x.append(i)
        if data[i] == 0:
            b.append(data[i])
        else:
            y.append(data[i])
    return [x, y, a, b]


def append_coordinates(data):
    x = []
    y = []
    for i in range(0, len(data)):
        x.append(i)
        y.append(data[i])
    return [x, y]


def add_subplots(fig, house, cost, h, weights, unique):
    if house == "Gryffindor":
        color = "red"
    if house == "Slytherin":
        color = "green"
    if house == "Ravenclaw":
        color = "blue"
    if house == "Hufflepuff":
        color = "yellow"
    costxy = append_coordinates(cost)
    sigmoidxy = append_coordinates(h)
    weightsxy = append_coordinates(weights)
    uniquexy = append_classification_coordinates(unique)
    fig.add_trace(go.Scatter(x=uniquexy[0], y=uniquexy[1], mode='markers',
                             name=house, marker_color=color), row=1, col=1)
    fig.add_trace(go.Scatter(x=uniquexy[2], y=uniquexy[3], mode='markers',
                             name=house, marker_color=color), row=1, col=1)
    fig.add_trace(go.Scatter(x=sigmoidxy[0], y=sigmoidxy[1], mode='markers',
                             name=house, marker_color=color), row=1, col=2)
    fig.add_trace(go.Scatter(x=costxy[0], y=costxy[1], mode='lines',
                             name=house, line=dict(color=color)), row=2, col=1)
    fig.add_trace(go.Scatter(x=weightsxy[0], y=weightsxy[1], mode='markers',
                             name=house, marker_color=color), row=2, col=2)


def cost_(X, h, nb_rows, unique):
    a = np.dot(-unique.T, np.log(h))
    b = np.dot((1 - unique).T, np.log((1 - h),
                                      out=np.zeros_like(1 - h), where=(1 - h != 0)))
    cost = (a - b) / unique.size
    return cost


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_gradient_descent(X, houses):
    predictions = []
    iterations = 10000
    learning_rate = 0.01
    X = np.insert(X, 0, 1, axis=1)
    nb_rows, nb_features = X.shape
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Classification one vs rest', 'Total repartition', 'Accuracy', 'Weight repartition'))
    for house in np.unique(houses):
        cost = []
        unique = np.where(houses == house, 1, 0)
        weights = np.zeros(nb_features)
        for _ in range(0, iterations):
            h = sigmoid(np.dot(X, weights))
            cost.append(cost_(X, h, nb_rows, unique))
            dw = np.dot(X.T, (h - unique)) / unique.size
            weights -= learning_rate * dw
        predictions.append((weights, house))
        add_subplots(fig, house, cost, h, weights, unique)
    update_fig_axes_names(fig)
    np.save('weights', predictions)
    fig.show()
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

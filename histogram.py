import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def add_histogram_sublot(house_marks, row, col, fig, display_border):
    fig.add_trace(go.Histogram(
        x=house_marks[0], name='Slytherin', marker_color='green'), row=row, col=col)
    fig.add_trace(go.Histogram(
        x=house_marks[1], name='Gryffindor', marker_color='red'), row=row, col=col)
    fig.add_trace(go.Histogram(
        x=house_marks[2], name='Hufflepuff', marker_color='yellow'), row=row, col=col)
    fig.add_trace(go.Histogram(
        x=house_marks[3], name='Ravenclaw', marker_color='blue'), row=row, col=col)
    if display_border == 1:
        fig.update_xaxes(title_text="marks", row=row, col=col, linecolor='red',
                         linewidth=2,
                         mirror=True)
        fig.update_yaxes(title_text="student count", row=row, col=col, linecolor='red',
                         linewidth=2,
                         mirror=True)
    else:
        fig.update_xaxes(title_text="marks", row=row, col=col)
        fig.update_yaxes(title_text="student count", row=row, col=col)
    fig.update_traces(opacity=0.6)
    fig.update_layout(template="seaborn",
                      title_text="Features homogeneousness", barmode='overlay', title_font_size=20)


def get_house_marks(data, feature):
    try:
        columns = data[['Hogwarts House', feature]]
    except KeyError as err:
        print("Error: " + str(err))
        exit(1)
    slyth, gryff, huffl, raven = ([] for i in range(4))
    for index, row in columns.iterrows():
        try:
            if not np.isnan(row[1]):
                if row[0] == 'Slytherin':
                    slyth.append(row[1])
                elif row[0] == 'Gryffindor':
                    gryff.append(row[1])
                elif row[0] == 'Hufflepuff':
                    huffl.append(row[1])
                elif row[0] == 'Ravenclaw':
                    raven.append(row[1])
                else:
                    continue
            else:
                continue
        except TypeError as err:
            print("Error: " + str(err))
            exit(1)

    return [slyth, gryff, huffl, raven]


def display_histograms(data):
    row = 1
    col = 1
    names = [feature for feature in data if feature != 'Hogwarts House']
    fig = make_subplots(rows=5, cols=3, subplot_titles=(names))
    for feature in data:
        if feature != 'Hogwarts House':
            if col == 4:
                row += 1
                col = 1
            house_marks = get_house_marks(data, str(feature))
            if feature == 'Care of Magical Creatures':
                # parameter 1 -> add border around sublot.
                add_histogram_sublot(house_marks, row, col, fig, 1)
            else:
                # parameter 0 -> don't add border around sublot.
                add_histogram_sublot(house_marks, row, col, fig, 0)
            col += 1
    fig.show()


def get_numeric_features(data):
    for feature in data:
        if feature != 'Hogwarts House':
            if data[feature].dtype != np.float64:
                del data[feature]  # supprime directement l'original ...
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
    data = get_numeric_features(data)
    display_histograms(data)
    return 0


if __name__ == "__main__":
    main()

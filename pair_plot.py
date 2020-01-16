import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def display(data, color, names):
    try:
        fig = go.Figure(data=go.Splom(
            dimensions=[dict(label=names[0],
                             values=data[names[0]]),
                        dict(label=names[1],
                             values=data[names[1]]),
                        dict(label=names[2],
                             values=data[names[2]]),
                        dict(label=names[3],
                             values=data[names[3]]),
                        dict(label=names[4],
                             values=data[names[4]]),
                        dict(label=names[5],
                             values=data[names[5]]),
                        dict(label=names[6],
                             values=data[names[6]]),
                        dict(label=names[7],
                             values=data[names[7]]),
                        dict(label=names[8],
                             values=data[names[8]]),
                        dict(label=names[9],
                             values=data[names[9]])],
            marker=dict(color=color, size=5),
        ))
        fig.update_layout(
            title='Features scatter plot matrix',
            template='seaborn',
            title_font_size=20,
            height=1500
        )
    except Exception as err:
        print("Error: " + str(err))
        sys.exit(1)
    fig.show()


def get_colors(data):
    colors = []
    try:
        for i in data['Hogwarts House']:
            if i == 'Slytherin':
                colors.append('green')
            elif i == 'Gryffindor':
                colors.append('red')
            elif i == 'Hufflepuff':
                colors.append('yellow')
            elif i == 'Ravenclaw':
                colors.append('blue')
            else:
                continue
    except KeyError as err:
        print('Error: Hogwarts House not in dataset.')
        sys.exit(1)
    return colors


def del_homogeneous_features(data):
    names = []
    for name in data:
        if name != 'Hogwarts House':
            if name != 'Arithmancy':
                if name != 'Care of Magical Creatures':
                    if name != 'Defense Against the Dark Arts':
                        names.append(name)
    return names


def get_numeric_features(data):
    for feature in data.columns:
        if feature != 'Hogwarts House':
            lenn = len(data[feature])
            nan = data[feature].isnull().sum()
            if data[feature].dtype != np.float64:
                del data[feature]
            if lenn == nan:
                del data[feature]
    return data


def get_csv_data():
    nb_arg = len(sys.argv)
    if nb_arg != 2:
        print('Error: provide a csv file to test.')
        sys.exit(1)
    try:
        data = pd.read_csv(sys.argv[1])
        return data
    except Exception as err:
        print('Error: ' + str(err))
        sys.exit(1)


def main():
    data = get_csv_data()
    try:
        data = get_numeric_features(data)
    except KeyError:
        print("Error: data missing in csv file.")
        sys.exit(1)
    names = del_homogeneous_features(data)
    color = get_colors(data)
    data = data[names]
    data = data.dropna()
    display(data, color, names)
    sys.exit(0)


if __name__ == "__main__":
    main()

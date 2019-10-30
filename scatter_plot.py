import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def display_scatter_plot(feature_a, feature_b):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feature_a[0], y=feature_b[0],
                             line=dict(color='green'),
                             mode='markers',
                             name='slytherin'))
    fig.add_trace(go.Scatter(x=feature_a[1], y=feature_b[1],
                             line=dict(color='red'),
                             mode='markers',
                             name='gryffindor'))
    fig.add_trace(go.Scatter(x=feature_a[2], y=feature_b[2],
                             line=dict(color='yellow'),
                             mode='markers',
                             name='hufflepuff'))
    fig.add_trace(go.Scatter(x=feature_a[3], y=feature_b[3],
                             line=dict(color='blue'),
                             mode='markers',
                             name='ravenclaw'))
    fig.update_xaxes(title_text="Astronomy")
    fig.update_yaxes(title_text="Defense Against the Dark Arts")
    fig.update_layout(template="seaborn",
                      title_text="Features similarity", title_font_size=20)
    fig.show()


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
        except TypeError as err:
            print("Error: " + str(err))
            exit(1)
        else:
            continue
    return [slyth, gryff, huffl, raven]


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
    data = data.dropna()
    display_scatter_plot(get_house_marks(data, 'Astronomy'),
                         get_house_marks(data, 'Defense Against the Dark Arts'))
    return 0


if __name__ == "__main__":
    main()

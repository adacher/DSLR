import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def display(features_name, features):
    information = ['', 'Count', 'Mean', 'Std',
                   'Min', '25%', '50%', '75%', 'Max']
    fig = go.Figure(data=[go.Table(
        header=dict(values=information,
                    fill_color='lightgrey',
                    line_color='black',
                    font=dict(color='black', size=12),
                    align=['left', 'center'], font_size=14, height=40),
        cells=dict(values=[features_name, features[0], features[1], features[2], features[3], features[4], features[5], features[6], features[7]],
                   fill_color=['lightgrey', 'white', 'cyan',
                               'white', 'cyan', 'white', 'cyan', 'white', 'cyan'],
                   line_color='black',
                   font=dict(color='black', size=12),
                   align=['left', 'center'], font_size=14, height=40))
    ])
    fig.update_layout(title="Numerical features",
                      template="seaborn", title_font_size=20)
    fig.show()


def get_count(array):
    count = 0
    for nb in np.nditer(array):
        if not np.isnan(nb):
            count += 1
    return count


def get_percentile(data, per):
    data.sort()
    scale = (len(data) - 1) * (per / 100)
    floor = np.floor(scale)
    ceil = np.ceil(scale)
    if floor == ceil:
        return data[int(scale)]
    result = (data[int(floor)] * (ceil - scale)) + \
        (data[int(ceil)] * (scale - floor))
    return result


def get_min(data):
    final_min = data[0]
    for nb in data:
        if nb < final_min:
            final_min = nb
    return final_min


def get_max(data):
    final_max = data[0]
    for nb in data:
        if nb > final_max:
            final_max = nb
    return final_max


def get_mean(data):
    mean = 0
    for nb in data:
        if np.isnan(nb):
            continue
        mean += nb
    return mean / len(data)


def get_std(data):
    std = 0
    mean = get_mean(data)
    for nb in data:
        if np.isnan(nb):
            continue
        std += (nb - mean) ** 2
    return (std / len(data)) ** 0.5


def get_numeric_features(data):
    for col in data.columns:
        lenn = len(data[col])
        nan = data[col].isnull().sum()
        if data[col].dtype != np.float64:
            del data[col]
        if lenn == nan:
            del data[col]
    return data


def get_information(data):
    names, min_, max_, count_, std_, mean_, per_1, per_2, per_3 = (
        [] for i in range(9))
    data = get_numeric_features(data)
    for content in data.columns:
        np_arr = np.array(data[content][~np.isnan(data[content])])
        per_1.append(get_percentile(np_arr, 25))
        per_2.append(get_percentile(np_arr, 50))
        per_3.append(get_percentile(np_arr, 75))
        min_.append(get_min(np_arr))
        max_.append(get_max(np_arr))
        count_.append(get_count(np_arr))
        mean_.append(get_mean(np_arr))
        std_.append(get_std(np_arr))
        names.append(content)
    return [names, count_, mean_, std_, min_, per_1, per_2, per_3, max_]


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
    try:
        features = get_information(data)
    except Exception:
        print("Error: data missing in csv file.")
        return 1
    display(features[0], features[1:])
    return 0


if __name__ == "__main__":
    main()

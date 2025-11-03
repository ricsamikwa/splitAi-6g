import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math


def split_and_store_dataset(df):
    episode_duration = 50   # in s
    num_input_files = 19
    running_idx = 0
    for i in range(1, num_input_files + 1):
        last_idx = i * episode_duration
        df_sub = df[running_idx:last_idx]
        df_sub.to_csv('input/episode_parameters/radio_parameters_moving_{}.csv'.format(i))
        running_idx = last_idx

def read_trace_file():
    path = 'input/episode_parameters/radio_parameters_moving.csv'
    # save headers in a list
    headers = ['Timestamp', 'Longitude', 'Latitude', 'Speed', 'Operatorname', 'CellID', 'NetworkMode', 'RSRP', 'RSRQ',
               'SNR', 'CQI', 'RSSI', 'DL_bitrate', 'UL_bitrate', 'State', 'PINGAVG', 'PINGMIN', 'PINGMAX', 'PINGSTDEV',
               'PINGLOSS', 'CELLHEX', 'NODEHEX', 'LACHEX', 'RAWCELLID', 'NRxRSRP', 'NRxRSRQ'
               ]
    df = pd.read_csv(path)
    #print(df)
    df = df.drop_duplicates(subset=['Timestamp'], keep='last', ignore_index=True)
    #print(df['Timestamp'])
    #print(df)
    df_subset = df[:950]
    #print(df_subset)
    df_subset.to_csv('input/episode_parameters/radio_parameters_moving_clean.csv')
    return df_subset

def plot_radio_metrics(df):
    fig = px.scatter_map(df, lat='Latitude', lon='Longitude')
    fig.update_layout(mapbox_style='open-street-map')
    fig.show()
    n_samples = df['Timestamp'].size
    # print(df_subset['SNR'])
    snr_numpy = df['SNR'].to_numpy(dtype=float)
    # print(snr_numpy)
    snr_linear = np.zeros(n_samples)
    for i in range(n_samples):
        snr_linear[i] = math.pow(snr_numpy[i] / 10, 10)
    figure, ax = plt.subplots()
    plt.plot([x for x in range(n_samples)], snr_numpy)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('SNR (dB)')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    #write_params_to_file()
    #read_params_from_file(episode=1, num_nodes=4)
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    df_subset = read_trace_file()
    split_and_store_dataset(df_subset)
    #plot_radio_metrics(df_subset)
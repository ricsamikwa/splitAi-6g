import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
import seaborn as sns


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
    #fig.show()
    n_samples = df['Timestamp'].size
    print(n_samples)
    df_time = pd.DataFrame({'Timestep': [x for x in range(1, n_samples+1)]})
    df = pd.concat([df_time, df], axis=1)
    speed_df = df.loc[df['Speed'] > 0]
    print(df_time)
    print(df)
    snr_numpy = df['SNR'].to_numpy(dtype=float)
    rsrq = df['RSRQ'].to_numpy(dtype=int)
    cqi_numpy = df['CQI'].to_numpy(dtype=int)
    # print(snr_numpy)
    snr_linear = np.zeros(n_samples)
    # this calculation is not required for now
    for i in range(n_samples):
        snr_linear[i] = math.pow(snr_numpy[i] / 10, 10)
    figure, ax1 = plt.subplots()
    # change here
    plt.plot([x for x in range(n_samples)], cqi_numpy)
    ax2 = ax1.twinx()
    ax2.scatter([x for x in range(n_samples)], df['Speed'], marker='*', color='black')
    #print(cqi_numpy)
    #plt.scatter(df['Latitude'], df['Longitude'])
    #plt.plot(cqi_numpy, df['DL_bitrate'])
    #plt.boxplot(df['DL_bitrate'], by=)
    #df.boxplot(column='UL_bitrate', by='CQI')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('CQI')
    ax2.set_ylabel('Speed (km/hr)')
    ax1.grid()
    # record timesteps when handovers occurred
    timesteps_handovers = []
    cell_id = df['CellID'][0]
    for i in range(n_samples):
        if cell_id != df['CellID'][i]:
            timesteps_handovers.append(df['Timestep'][i])
            cell_id = df['CellID'][i]
    print(timesteps_handovers)
    for i in range(len(timesteps_handovers)):
        plt.axvline(x=timesteps_handovers[i], color='red')
    plt.show()
    #cells = df['CellID']
    #cells = cells.drop_duplicates()
    sns.scatterplot(data=df, x='Latitude', y='Longitude', hue='CellID', palette='deep')
    plt.grid()
    #plt.show()

if __name__ == '__main__':
    #write_params_to_file()
    #read_params_from_file(episode=1, num_nodes=4)
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    df_subset = read_trace_file()
    #split_and_store_dataset(df_subset)
    plot_radio_metrics(df_subset)
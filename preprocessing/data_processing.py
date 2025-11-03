import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math

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
    #fig = px.scatter_map(df_subset, lat='Latitude', lon='Longitude')
    #fig.update_layout(mapbox_style='open-street-map')
    #fig.show()
    n_samples = df_subset['Timestamp'].size
    #print(df_subset['SNR'])
    snr_numpy = df_subset['SNR'].to_numpy(dtype=float)
    #print(snr_numpy)
    snr_linear = np.zeros(n_samples)
    for i in range(n_samples):
        snr_linear[i] = math.pow(snr_numpy[i]/10, 10)
    #df_SNR_linear = math.pow(snr_series/10, 10)
    #figure, ax = plt.subplots()
    #plt.plot([x for x in range(n_samples)], snr_numpy)
    #ax.set_xlabel('Timestamp')
    #ax.set_ylabel('SNR (dB)')
    #plt.grid()
    #plt.show()
    return df_subset

if __name__ == '__main__':
    #write_params_to_file()
    #read_params_from_file(episode=1, num_nodes=4)
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    read_trace_file()
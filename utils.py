import numpy as np
import pandas as pd

from os import listdir
from os.path import join

import torch
import torch.utils.data as data

from itertools import product
from sklearn.preprocessing import LabelEncoder


class CustomDataset(data.Dataset):
    def __init__(self, path):
        super(CustomDataset, self).__init__()
        self.x = torch.Tensor(torch.load(path).values)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        

def path_list(folder_path, dset_range, mode):
    
    in4path_list = []
    thinn_path_list = []

    for c in sorted(listdir(folder_path)):
        i_fordel_path = join(folder_path, c)
        for data_path in sorted(listdir(i_fordel_path)):
            if ('in4bc_amsua' in data_path): 
                i_data_path= join(i_fordel_path, data_path)
                in4path_list.append(i_data_path)
            elif ('amsua' in data_path)&('InnQC2' in data_path): 
                i_data_path= join(i_fordel_path, data_path)
                thinn_path_list.append(i_data_path)
                
    in4bc_idx_1 = [s for s in range(len(in4path_list)) if dset_range[mode][0] in in4path_list[s]]
    in4bc_idx_2 = [s for s in range(len(in4path_list)) if dset_range[mode][1] in in4path_list[s]]

    thinn_idx_1 = [s for s in range(len(thinn_path_list)) if dset_range[mode][0] in thinn_path_list[s]]
    thinn_idx_2 = [s for s in range(len(thinn_path_list)) if dset_range[mode][1] in thinn_path_list[s]]
    return in4path_list[in4bc_idx_1[0]:in4bc_idx_2[0]+1], thinn_path_list[thinn_idx_1[0]:thinn_idx_2[0]+1]


# 채널별 in4bc에 ob4da의 정상, 비정상 값 매칭
def coordinate_matching(in4df, InnQC2, select_list):
    bin_df = pd.DataFrame()

    for chan in select_list:
        in4bc = in4df[in4df['nchan']==chan].reset_index(drop=True)
        assert in4bc['lat'].equals(InnQC2['lat']) == True, "Only possible same 'lat' feature in between df1, df2"
        in4bc['REAL_QC'] = InnQC2.filter(regex=str(chan+1))
        
        bin_df = pd.concat([bin_df, in4bc], axis=0)
            
    bin_df.reset_index(drop=True, inplace=True)
    
    nan_list = ['obsTB', 'innov', 'bias_pred']
    for col in nan_list:
        nan_idx = bin_df[bin_df[col] <= -999].index
        bin_df[col].loc[nan_idx] = np.nan
    return bin_df


def calc_chan_mean(df, select_list):
    
    df[['chan_lat_mean', 'chan_lon_mean', 'chan_bias_pred_mean', 'chan_obsTB_mean', 'chan_innov_mean']] = 0
    for chan in select_list:
        normal_index = df[df['nchan'] == chan][df['REAL_QC'] == 0.0].index
        abnormal_index = df[df['nchan'] == chan][df['REAL_QC'] == 7.0].index
        
        normal_mean = df.iloc[normal_index].mean()
        abnormal_mean = df.iloc[abnormal_index].mean()

        for col in ['lat', 'lon', 'bias_pred', 'obsTB', 'innov']:
            df['chan_{}_mean'.format(col)].loc[normal_index] = normal_mean[col]
            df['chan_{}_mean'.format(col)].loc[abnormal_index] = abnormal_mean[col]
    return df
    
    
def calc_grid_mean(df, select_list):
        
    lat_labels = ['lat_1', 'lat_2', 'lat_3', 'lat_4']
    lon_labels = ['lon_1', 'lon_2', 'lon_3', 'lon_4', 'lon_5', 'lon_6']
    df[['grid', 'grid_lat_mean', 'grid_lon_mean', 'grid_bias_pred_mean', 'grid_obsTB_mean', 'grid_innov_mean']] = 0
    df['lat_table']=pd.cut(df.lat,[-90.0001,-45,0,45,90.0001], 4, labels=lat_labels)
    df['lon_table']=pd.cut(df.lon,[-0.0001,60,120,180,240,300,360.0001], 6, labels=lon_labels)

    count = 0
    for lat in lat_labels:
        for lon in lon_labels:
            count+=1
            df.loc[(df['lat_table'] == lat) & (df['lon_table'] == lon),'grid'] = 'grid_{}'.format(count)
    df.drop(['lat_table', 'lon_table'], axis=1, inplace=True)

    for chan, grid in list(product(select_list, range(1,24+1))):
            
        normal_index = df[df['nchan'] == chan][df['REAL_QC'] == 0.0][df['grid'] == 'grid_{}'.format(grid)].index
        abnormal_index = df[df['nchan'] == chan][df['REAL_QC'] == 7.0][df['grid'] == 'grid_{}'.format(grid)].index

        normal_mean = df.iloc[normal_index].mean()
        abnormal_mean = df.iloc[abnormal_index].mean()
        
        for col in ['lat', 'lon', 'bias_pred', 'obsTB', 'innov']:
            df['grid_{}_mean'.format(col)].loc[normal_index] = normal_mean[col]
            df['grid_{}_mean'.format(col)].loc[abnormal_index] = abnormal_mean[col]
                                                
            normal_nan_idx = df['obsTB'].loc[normal_index][df['obsTB'].loc[normal_index].isna()].index
            abnormal_nan_idx = df['obsTB'].loc[abnormal_index][df['obsTB'].loc[abnormal_index].isna()].index
            
            df['obsTB'].loc[normal_nan_idx] = normal_mean['obsTB']
            df['innov'].loc[normal_nan_idx] = normal_mean['innov']
            
            df['obsTB'].loc[abnormal_nan_idx] = abnormal_mean['obsTB']
            df['innov'].loc[abnormal_nan_idx] = abnormal_mean['innov']
    return df 
    
# cyclical embedding
def dummy_and_add_feature(x):
    hour = int(x.split("_")[-1].split(".")[0][-2:])
    day = int(x.split("_")[-1].split(".")[0][-4:-2])
    month = int(x.split("_")[-1].split(".")[0][-6:-4])

    sin_hour = np.sin((2*np.pi*hour*60*60)/(24*60*60))
    cos_hour = np.cos((2*np.pi*hour*60*60)/(24*60*60))
    sin_day = np.sin((2*np.pi*day*24*60*60)/(31*24*60*60))
    cos_day = np.cos((2*np.pi*day*24*60*60)/(31*24*60*60))
    sin_month = np.sin((2*np.pi*month*31*24*60*60)/(12*31*24*60*60))
    cos_month = np.cos((2*np.pi*month*31*24*60*60)/(12*31*24*60*60))
    
    return sin_hour, cos_hour, sin_day, cos_day, sin_month, cos_month


def feature_encoding(df):
    scanpos_encoder_list = [1,2,3,4,5,6,7,8,9,10,
                            11,12,13,14,15,16,17,18,19,20,
                            21,22,23,24,25,26,27,28,29,30,
                            31,32,33,34,35,36,37,38,39,40,
                            41,42,43,44,45,46,47,48,49,50,
                            51,52,53,54,55,56]
    
    nchan_encoder_list = [4,5,6,7,8,9,10,
                        11,12,13,14]
    
    sat_id_encoder_list = [3,4,206,209,223]
    
    grid_encoder_list = ['grid_8', 'grid_14', 'grid_20', 'grid_19', 'grid_21', 'grid_22',
                        'grid_24', 'grid_23', 'grid_17', 'grid_11', 'grid_5', 'grid_4',
                        'grid_3', 'grid_2', 'grid_13', 'grid_16', 'grid_10', 'grid_1',
                        'grid_7', 'grid_9', 'grid_12', 'grid_18', 'grid_6', 'grid_15']
        
        
    scanpos_station_encoder = LabelEncoder()
    nchan_station_encoder = LabelEncoder()
    sat_id_station_encoder = LabelEncoder()
    grid_station_encoder = LabelEncoder()
    
    scanpos_station_encoder.fit(scanpos_encoder_list)
    nchan_station_encoder.fit(nchan_encoder_list)
    sat_id_station_encoder.fit(sat_id_encoder_list)
    grid_station_encoder.fit(grid_encoder_list)

    df['scanpos'] = scanpos_station_encoder.transform(df['scanpos']) / (len(df['scanpos'].unique())-1)
    df['nchan'] = nchan_station_encoder.transform(df['nchan']) / (len(df['nchan'].unique())-1)
    df['sat_id'] = sat_id_station_encoder.transform(df['sat_id']) / (len(df['sat_id'].unique())-1)
    df['grid'] = grid_station_encoder.transform(df['grid']) / (len(df['grid'].unique())-1)

    df[['chqcflag_-999', 'chqcflag_0']] = 0
    dummy_chqcflagpd = pd.get_dummies(df['chqcflag'], prefix = 'chqcflag')

    df[['chqcflag_-999', 'chqcflag_0']] = dummy_chqcflagpd[['chqcflag_-999', 'chqcflag_0']]
    df.drop(['chqcflag'], axis=1, inplace=True)
    
    
    return df
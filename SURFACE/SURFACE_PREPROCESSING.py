import os
import math
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from SURFACE_UTILS import (calc_missing, press_to_alt, convert_week, dummy_and_add_feature)

import warnings
warnings.filterwarnings(action='ignore') 


def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Preprocessing', add_help=False)

    # Model parameters
    parser.add_argument('--buoy', default='/data/aML_final/DAIN/surface/infer/buoy_1h.csv', type=str)
    parser.add_argument('--metar', default='/data/aML_final/DAIN/surface/infer/metar_1h.csv', type=str)
    parser.add_argument('--ship', default='/data/aML_final/DAIN/surface/infer/ship_1h.csv', type=str)
    parser.add_argument('--synop', default='/data/aML_final/DAIN/surface/infer/synop_1h.csv', type=str)
    parser.add_argument('--mode', default='TRAIN', type=str)
    parser.add_argument('--scaler_path', default='/home/COMPUTER_VISION/SURFACE/SURFACE_min_max.pickle', type=str)

    return parser


def main(args):
    
    # ------------------------------------------------------------------------------------------
    buoy_1hr = pd.read_csv(args.buoy)
    metar_1hr = pd.read_csv(args.metar)
    ship_1hr = pd.read_csv(args.ship)
    synop_1hr = pd.read_csv(args.synop)
    
    stn_type = {
        'buoy': buoy_1hr,
        'metar': metar_1hr,
        'ship': ship_1hr,
        'synop': synop_1hr, 
        }
    mode = args.mode
    scaler_path = args.scaler_path
    seed = 10

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------------------------------
    
    # min-max scaling 시 관측종별 컬럼별 최소, 최대값을 저장할 사전 생성
    min_max_dict = {'buoy':{},
                    'metar':{},
                    'ship':{},
                    'synop':{}}


    for key in stn_type.keys():
        
        use_col = ['Date/Time', 'stnID', 'lat', 'lon' ]
        use_col_list = ['stnHgt','Pressure','Pmsl','T2m','Td2m','RH2m','U10m','V10m']

        if key == 'buoy': use_col_list=[i for i in use_col_list if i not in ['stnHgt', 'Td2m', 'RH2m']]
        if key == 'ship': use_col_list.remove('stnHgt')
        
        stn_type[key]['Date/Time'] = stn_type[key].Date.astype(str) + (stn_type[key].Time / 10000).astype(int).astype(str)
        stn_type[key] = stn_type[key][use_col+use_col_list]

        format_ = '%Y%m%d%H'
        stn_type[key]['Date/Time'] = stn_type[key]['Date/Time'].apply(lambda x: datetime.strftime(datetime.strptime(x, format_),'%Y-%m-%d %H:%M'))
        stn_type[key].replace(-999.99, np.nan, inplace=True)
        
        each_stn_col_nan_rate = {}
        del_stn_list = []
        
        for each_stn in tqdm(stn_type[key].stnID.unique()):
            index = stn_type[key][stn_type[key].stnID == each_stn].index
            
            # station별 특정 변수의 결측률을 계산  
            for col in use_col_list: 
                each_stn_col_nan_rate['{}_mssing_rate'.format(col)] = calc_missing(stn_type[key].iloc[index], col)
            
            # 모든 변수의 평균 결측률이 0.3 이상인 경우 해당 station 제거 / 한 변수라도 결측률이 0.5 이상인 경우 해당 station 제거 
            del_stn = True if (sum(each_stn_col_nan_rate.values()) / len(each_stn_col_nan_rate) >= 0.3) | (any(0.5<num for num in each_stn_col_nan_rate.values())) else False 
            if del_stn == True: del_stn_list.append(each_stn)
        
        # 제거할 station을 취합 후 사용할 station만으로 dataframe 생성
        stn_type[key] = stn_type[key][~stn_type[key]['stnID'].isin(del_stn_list)].sort_values(["stnID", "Date/Time"]).reset_index(drop=True)
        
        # station별로 보간을 수행 
        stn = stn_type[key].stnID
        stn_type[key] = stn_type[key].groupby('stnID').transform(lambda x: x.interpolate(limit_direction ='both'))
        
        # groupby를 통해 지워진 stnID 변수를 재생성
        stn_type[key]['stnID'] = stn
            
        # 기압 변수를 활용하여 고도 변수 생성 (Feet -> Meter)
        stn_type[key]['Altitude'] = stn_type[key]['Pressure'].apply(lambda x: press_to_alt(x))
        
        # 'Date/Time' 변수를 활용하여 'num_week' 변수 생성
        stn_type[key]['num_week'] = stn_type[key]['Date/Time'].apply(convert_week)
        
        # 'Date/Time' 변수를 활용하여 cyclical embedding을 통한 'sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month' 변수 생성
        stn_type[key][['sin_hour', 'cos_hour', 'sin_day', 'cos_day', 'sin_month', 'cos_month']] = stn_type[key]['Date/Time'].apply(dummy_and_add_feature).tolist()
        
        # 관측종별 정규화할 변수들을 리스트화
        globals()['{}_min_max_col'.format(key)] = [i for i in list(stn_type[key].columns) if i not in ['stnID', 'Date/Time']]
        
        if mode == 'TRAIN':
            # 변수별 최대, 최소값 구하기
            max_arr = stn_type[key][globals()['{}_min_max_col'.format(key)]].max().values
            min_arr = stn_type[key][globals()['{}_min_max_col'.format(key)]].min().values
            
            # 이미 만들어진 사전에 변수별 최대, 최소값 저장
            min_max_dict[key] = {globals()['{}_min_max_col'.format(key)][i]:[min_arr[i], max_arr[i]] for i in range(len(globals()['{}_min_max_col'.format(key)]))}
            
            # Train data에 1~12월 까지의 데이터가 존재하지 않기 때문에, 사전의 month 최대값을 1로 설정
            min_max_dict[key]['cos_month'][1] = 1.0
        
        elif mode == 'TEST':
            with open(scaler_path, 'rb') as fr:
                min_max_dict = pickle.load(fr)
                        
        # min-max scaling
        for col, (col_min, col_max) in min_max_dict[key].items():
            stn_type[key][col] = stn_type[key][col] - col_min
            stn_type[key][col] = stn_type[key][col] / (col_max-col_min)
        
        # 모든 전처리 후 채워지지 않은 Na값의 경우 최대값으로 채워넣음   
        stn_type[key] = stn_type[key].fillna(1)
        
        # Prepocessed data save 
        stn_type[key].to_csv("{}_Preprocessed_{}.csv".format(mode, key), index=False)
        
        print("{}_Preprocessed_{}.csv is saved!".format(mode, key))
        
        
    if mode == 'TRAIN':
        with open(scaler_path, 'wb') as fw:
            pickle.dump(min_max_dict, fw)
            
        print("scaler is saved at {}".format(scaler_path))
    
    # -------------------------------------------------------------------------------------------------------------------------
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Preprocessing script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
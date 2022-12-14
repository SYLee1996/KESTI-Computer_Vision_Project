# USAD for SURFACE data

+ SURFACE 데이터의 이상탐지를 위한 USAD 기반의 baseline 모델 작성

---- 

## Environment 
+ NIA 서버 기준(cudnn7, ubuntu18.04)
+ 사용한 Docker image는 Docker Hub에 업로드되어 환경을 제공합니다.
  + https://hub.docker.com/r/lsy2026/computer_vision/tags
  
## Libraries
  + python==3.7.10
  + pandas==1.3.5
  + numpy==1.20.2
  + xarray==0.20.1
  + tqdm==4.51.0
  + sklearn==1.0.1
  + torch==1.9.0
  
----

## Directory
        .
        ├── SURFACE_INFERENCE.py
        ├── SURFACE_MAIN.p
        ├── SURFACE_MODEL.py
        ├── SURFACE_PREPROCESSING.py
        ├── SURFACE_UTILS.py
        └── RESULTS

        1 directories, 5 files
----

## Summary
+ ### Data preprocessing

      1. coast line의 표기 없이 모든 -999.99데이터 Nan으로 변환
      2. Station별 변수들의 결측률을 계산
          2.1. 특정 station에서 모든 변수에 대한 결측률이 0.3이상인 경우 해당 station 제거
          2.2. 특정 station에서 한 변수라도 결측률이 0.5 이상인 경우 해당 station 제거

      3. Station별 보간(linear) 수행
      
      4. 기압 변수를 활용하여 고도(Altitude) 변수 생성 (Feet -> Meter)
      
      5. ‘Date/Time’ 변수를 활용하여 해당 데이터가 몇 번 째 주 인지를 나타내는 ‘num_week’ 변수 생성
      
      6. ‘Date/Time’ 변수를 활용하여 cyclical embedding을 통한 ‘sin_hour’, ‘cos_hour’, ‘sin_day’, ‘cos_day’, ‘sin_month’, ‘cos_month’ 변수 생성
      
      7. 데이터 정규화(Min-max scaling)
      
      8. 모든 전처리 후 채워지지 않은 Nan값의 경우 최대값(1) 채워 넣음
      
      9. 관측종별 변수 다르게 사용
          9.1 ‘buoy’: ‘stnHgt’, ‘Td2m’, ‘RH2m’ 제거
          9.2 ‘ship’: ’stnHgt’ 제거
----

+ ### Model    
    + 단일 Encoder Layer와 이중 Decoder Layer로 구성
    + 약 10%의 데이터를 검증데이터로 사용
    + Ensemble이나 cross-validation은 적용하지 않음
    
    + scheduler: CosineAnnealingLR
    + Loss : 학습 초반에는 reconstruction error에 가중치를 주고, 학습 후반에는 adversarial training에 가중치를 줌
    + optimizer : AdamW 
    + EarlyStopping

----

+ ### Train  
    + SONDE 데이터의 경우 윈도우 사이즈별로 데이터를 예측하기 때문에 특정 시간대에 재구성된 데이터가 겹치는 경우가 발생 -> 재구성 윈도우가 겹치는 경우, 겹친 부분의 평균값을 사용하도록 수정   
    
----

+ ### Result  
     + 관측종 'buoy'에 대한 USAD 학습 손실 

     ![image](https://user-images.githubusercontent.com/30611947/191474640-eee3d3c7-f373-4af5-b890-78dc336652fc.png)


     + ‘buoy’의 ‘6200092’ station에 대한 ‘Pressure’, ‘Pmsl’, ‘T2m’, ‘U10m’, ‘V10m’, ‘Altitude’, 및 Loss 시각화
     + 모델 학습 시 validation set에 대한 loss의 75% quantile 평균값을 threshold로 사용(0.014911915051857462)

     ![image](https://user-images.githubusercontent.com/30611947/191475317-4db030d8-8c8b-4b53-a7ac-8411112978d6.png)


     + 특정 지점 및 고도의 KPOP QC와 USAD 이상감지 결과 시계열 비교 (2mT) : 지점명 GMAD(METAR)

     <img width="1054" alt="image" src="https://user-images.githubusercontent.com/30611947/192328437-f3fdb383-80f9-4810-a931-451ad0db1839.png">



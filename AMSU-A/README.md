# KESTI-DeepSVDD(AMSU-A)

+ AMSU-A 위성 데이터의 이상탐지를 위한 DeepSVDD 기반의 baseline 모델 작성
+ 파일당 천만 건의 데이터가 존재하기 때문에 샘플 데이터를 이용하여 학습 및 검증을 진행(추론 X)

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
        ├── Preprocessing.ipynb
        ├── Train & Validation.ipynb
        ├── model.py
        ├── utils.py
        ├── weights
        └── Data

        2 directories, 4 files
----

## Summary
+ ### Data preprocessing

      1. InnQC2의 정상, 이상 위경도를 in4bc 데이터에 매치하여 정상 및 이상 분리
      
      2. 사용하지 않는 1, 2, 3, 4, 15 채널 제외
      
      3. 채널별 lat, lon 평균값 변수 생성
      
      4. 채널별 bias_pred, obsTB, innov 평균값 변수 생성
      
      5. 전지구를 총 24개의 grid로 분할하는 grid 변수 생성

    ![image](https://user-images.githubusercontent.com/30611947/191458165-63fd7194-5b71-4333-96f7-f1ddca693722.png)

      6. 각 grid에 속하는 데이터의 lat, lon 평균값 변수 생성
      
      7. 각 grid에 속하는 데이터의 bias_pred, obsTB, innov 평균값 변수 생성
      
      8. 데이터가 수집된 시각(00시, 06시, 12시, 18시), 월(6월, 7월)을 cyclical embedding을 통해 변수로 생성

    ![image](https://user-images.githubusercontent.com/30611947/187855556-a5fb2d77-cb60-48cf-8b06-e198ca141365.png)

      9. 56개의 scanpos, 10개 채널, 5개의 위성 및 정상, 비정상 변수들에 대해 label encoding을 통해 변수를 수치화
      
      10. Min-Max scaling을 통해 변수 값의 범위 정규화
      
      11. 위 전처리 후에도 NaN 값이 존재할 경우 1 값으로 대체
      
----

+ ### Model    
    + Baseline 모델이기 때문에, DeepSVDD를 이용
    + Ensemble이나 cross-validation은 적용하지 않음
    
    + AE parameter
        + scheduler: CosineAnnealingLR
        + Loss : MSELoss   
        + optimizer : AdamW 
        + lr : 1e-4
        + weight_decay : 1e-4     
        + epoch : 3
        
    + DeepSVDD parameter
        + scheduler: CosineAnnealingLR
        + Loss : MSELoss   
        + optimizer : AdamW 
        + lr : 1e-4
        + weight_decay : 1e-4    
        + epoch : 1
----

+ ### Train  
 
    + 대략 3억개의 데이터를 이용하여 학습 후, 약 1000만개의 데이터에 대하여 검증 진행
    + Result  
      {'precision': 0.9998805272707689,
       'recall': 0.9998738395336857,
       'f1-score': 0.9998771683052723,
       'support': 11748352}

     



# UCF-Crime-Anomaly-Detection

Repository for report submission.

3dResNet Features(depth=152, dim=2048) -> https://drive.google.com/file/d/17wdy_DS9UY37J9XTV5XCLqxOFgXiv3ZK/view?usp=sharing

# 0. Usage

- Feature extraction


 ```bash
 
    python feature_extractor.py -o features -i your_dataset_dir --m your_path_to_pretrined_model -t your_extractor_type

 ```
 
- Training, Test: See here for a detailed explanation.

https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch

# 1. Overview

코로나 19가 장기간 유행하고 언택트 트렌드가 확산되면서 다양한 업종의 무인매장이 들어서고 있다. 이에 따라 무인매장에서 도난을 비롯한 사건사고에 관한 소식이 끊이지 않고 있다. 무인매장에 설치되어 있는 CCTV를 활용하는 이상 탐지 모델을 개발 한다면 도난사고를 예방하고 이상 상황 발생 시 신속하게 대처 할 수 있는 보안 솔루션으로 사용될 수 있을 것이다.

# 2. Dataset

UCF Crime Dataset
https://www.crcv.ucf.edu/projects/real-world/
![image](https://user-images.githubusercontent.com/28619620/122185184-ddcdd500-cec7-11eb-904f-a7dc2b954def.png)
- Surveilance Videos(CCTV)
- 1900 videos, 13 types of anomaly
- 950 anomal videos/950 normal videos
- weakly labeled data 

# 3. Baseline Model

<b>Real-world Anomaly Detection in Surveillance Videos</b>
https://arxiv.org/abs/1801.04264
![image](https://user-images.githubusercontent.com/28619620/122191171-71ee6b00-cecd-11eb-83ec-2ffde455792e.png)

- 비디오를 32개의 segments로 쪼개고 각각의 segment에 대해 feature extracion을 한다. 
- 추출된 segment의 feature를 입력하여 anomaly score를 얻는다.
- fully connect layer는 3층으로 되어있있다. (inuput laye(ReLU) -> layer 1(Linear) -> layer 2(Sigmoid)-> output(amomaly score))
- 학습시 Deep MIL Ranking Model을 이용하여 loss를 계산한다.
 
# 4. Changes

## 4.1. Loss Function

 기존 loss function(1)은 다음과 같다.
 
![loss_function](https://user-images.githubusercontent.com/28619620/172195239-c214e666-c4c3-4548-b7e4-098c1a7a8723.png)

위의 loss function은 hinge-loss function이다. B는 영상의 조각을 담고 있는 가방(Bag), V는 영상의 특징 추출된 값, f는 모델을 의미한다. 이 loss function을 다음과 같이 바꾸었다.

![CodeCogsEqn (11)](https://user-images.githubusercontent.com/28619620/172195208-e2374d3a-795a-4d6a-a3bb-3abd377790db.png)


## 4.2. Featuere Extraction(C3D, I3D, R3D)

Baseline 모델은 Sports-m1 데이터셋에서 사전 학습된 C3D 모델로 특징추출을 했다. 더 깊은 모델과 방대한 데이터셋으로 학습된 사전 학습 모델을 이용하면 성능을 높일 수 있을 것이라 생각하 여 Kinetics-700,  Moments in Time 데이터셋을 이용하여 학습된 3D ResNet 모델로 특징을 추출했다. 

Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, and Yutaka Satoh,
"Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs",
arXiv preprint, arXiv:2004.04968, 2020.

the paper: https://arxiv.org/abs/2004.04968

git: https://github.com/kenshohara/3D-ResNets-PyTorch

# 5. Result

![image](https://user-images.githubusercontent.com/28619620/172195704-5d91922d-8b3e-4875-9a8e-16c69f4ff852.png)
R3D + top-k Ranking Loss Models

![compare_k](https://user-images.githubusercontent.com/28619620/172195422-ba7d4a5f-892f-4fc6-b4f0-d1061a91af70.png)



# 6. Conclusion

 사전학습모델을 개선하는 것으로 모델의 성능을 높일 수 있었다. 따라서 사전학습모델을 학습시키기 위한 데이터셋의 양질이 매우 중요하다. loss function을 수정하여 Test dataset에서 성능을 높였지만, 실제로 일반화 성능 측정하려면 추가로 테스트를 진행해야 한다. 기존 모델의 연산 속도와 큰 차이 없이 성능의 향상 이끌어 냈다는 점을 차별점이다. 
 
# 7. Reference

Training and test codes are borrowed from ekosman. https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch

# 2021-2-capstone-design


Repository for report submission.


I3D Features(dim=1024) ->
3dResNet Features(depth=152, dim=2048) -> https://drive.google.com/file/d/17wdy_DS9UY37J9XTV5XCLqxOFgXiv3ZK/view?usp=sharing


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

- Input: 16프레임으로 쪼개진 영상 
- Output: anomaly score
 
# 4. Changes

## 4.1. Loss Function

 기존 loss function(1)은 다음과 같다.
 
![image](https://user-images.githubusercontent.com/28619620/146714848-208670bc-13d2-4069-b006-c2a1f0597bc4.png)

위의 loss function은 hinge-loss function이다. B는 영상의 조각을 담고 있는 가방(Bag), V는 영상의 특징 추출된 값, f는 모델을 의미한다. 이 loss function을 다음과 같이 바꾸었다.

![image](https://user-images.githubusercontent.com/28619620/146714852-d392e74c-4115-42b1-9f8c-2e7af14f9116.png)

기존의 loss function은 가방 안에서 anomaly socre(모델의 출력값)의 최댓값을 사용해 loss를 계산했지만, 위 식에서는 상위 2개의 anomaly score를 사용해 loss를 계산한다. 이 변경을 통해 한 영상에서 더 많은 영상 조각을 학습에 사용할 수 있다. 이는 한 anomaly 영상에서 32개의 조각 중 최소 2개의 조각은 양성이라는 가정을 전제로 한다. 

![image](https://user-images.githubusercontent.com/28619620/146714861-13791dc3-437c-4b5a-b9ae-232152b9b713.png)

최종적으로 anomaly 영상에서 score기준 상위 4개의 영상 조각을 고르고, normal 영상에서 상위 2개의 영상, 하위 2개의 영상을 골라 loss를 계산하는 function을 만들었다.

## 4.2. Featuere Extraction

Baseline 모델은 Sports-m1 데이터셋에서 사전 학습된 C3D 모델로 특징추출을 했다. 더 깊은 모델과 방대한 데이터셋으로 학습된 사전 학습 모델을 이용하면 성능을 높일 수 있을 것이라 생각하Kinetics-700,  Moments in Time 데이터셋을 이용하여 학습된 3D ResNet 모델로 특징을 추출했다. 

Hirokatsu Kataoka, Tenga Wakamiya, Kensho Hara, and Yutaka Satoh,
"Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs",
arXiv preprint, arXiv:2004.04968, 2020.

the paper: https://arxiv.org/abs/2004.04968

git: https://github.com/kenshohara/3D-ResNets-PyTorch

# 5. Result

## 5.1. I3D
![i3d_soft_60_maxmid](https://user-images.githubusercontent.com/28619620/146715712-be4974c1-0934-47a2-a786-dc6aecd3d2b5.png)
I3D(the loss function in the paper)
![toptop2_180k](https://user-images.githubusercontent.com/28619620/146715721-7ac98d02-1f2b-4078-ae57-98d703038a80.png)
I3D(custom loss function)


I3D(original loss), I3D(custom loss)

## 5.2. R3D

![basic_roc_auc](https://user-images.githubusercontent.com/28619620/146715766-60a6be3c-7144-48db-b6dd-bba6abddfffa.png)
R3D(the loss function in the paper)

![roc_auc](https://user-images.githubusercontent.com/28619620/146715832-00cdf7ff-b7cf-4696-a8ed-078fe401807d.png)
R3D(custom loss function)

# 6. Conclusion


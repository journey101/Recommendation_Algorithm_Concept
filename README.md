# Recommendation_Algorithm_Concept_ver0.1
추천모델 알고리즘 개념 정리 페이지(ver_0.1)_ongoing...

### 단순한 평점 예측 모델 
※[python Surprise lib](https://danthetech.netlify.app/DataScience/how-does-recommendation-algorithms-work-using-surpriselib#normalpredictor)에서 제공하는 추천 알고리즘 

- NormalPredictor: 정규분포로 가정한 학습 데이터셋의 평점분포에서 랜덤하게 샘플링. 

- BaselineOnly: User와 Item의 Baseline(global_mean)을 이용한 평점 예측 알고리즘. (ALS, SGD 2가지 계산방식이 있음.)


### 컨텐츠 기반 추천

- 유저 U가 구매한 아이템 I와 유사한 속성을 가진 아이템을 추천하는 모델. 


### 기본 행렬분해(Matrix Factorization) 알고리즘

user와 item의 평점이 들어간 행렬데이터에서 비어있는 영화 i의 평점을 유추하기 위해 행렬분해식(Matrix Factorization)에서 발생되는 영화 i에 대한 유저 U의 user_latent벡터값과 유저U에 대한 영화i의 item_latent벡터값을 내적하여 영화i에 대한 유저 U의 평점을 유추함. 
 - 단점: 
  - 단순한 내적값으로 추정한 영화i에 대한 예측평점이 실제 유저U의 평점과 동일하다고 보기는 어렵다는 단점. 
  - 이러한 단점을 보완하기 위해, 유저U와 유사한 평점패턴을 보이는 유사유저들의 평점을 활용하려는 협업필터링 Collaborative Filtering 개념이 등장함.


### 행렬분해 개념+협업필터링(Collaborative Filtering) 알고리즘 (KNN, SVD, NMF, SGD, ALS)

- KNNBasic: 기본 협업필터링(collaborative filtering) 알고리즘. 
 - 타겟 유저 U와 similarity(cosine or pearson)가 유사한 k명의 유사 사용자들이 영화 i에 대해 평가한 평점을 가중합(유사유저k1,2,..의 타겟유저U에 대한 유사도*영화i에 대한 평점 = 가중평균합)하여 영화 i에 대한 유저 U의 평점을 예측. 
 - 장점: 타겟 유저U와 근접한 similarity를 가진 사람이 평가한 i에 대한 평점이 더 큰 가중치를 갖게 되는 구조. 
 - 단점: 유저U와 단순히 근접하다는 이유로 다른 사람의 평점이 유저U에게도 적합할 수 있겠느냐는 약점을 가지며, 이점을 보완하기 위해 나온 개념이 KNNWithMeans 개념. 

- KNNWithMeans: 기본 협업필터링에서 나온 (k명의 유사유저들의 평점 - 타겟유저U의 평점평균)을 가중합하여 예측평점을 계산함으로써, 유저U가 평점을 주는 패턴에 더 근접한 예측평점을 내기 위한 알고리즘.

- KNNWithZScore: 기본적 협업필터링 알고리즘, 추가적으로 Z-Score 분포를 적용한다. 
 - 영화i에 대한 k유사 유저들의 평점에서 각 k유사 유저의 평균을 빼주고 표준편차를 나눠준다. 그다음 예측 타겟 유저U의 표준편차를 곱하고, 평균을 더해준다.) ==> KNNwithmeans와 유사하면서 좀 더 나아간 버전. 단, RMSE의 평가기준으로 성적이 보장되지 않는다고 함. 

- KNNBaseline: KNNWithMeans가 타겟유저U의 평점평균을 빼준 개념이라면, 이 모델은 유사유저들의 평점에서 baseline을 빼서 가중합을 매긴 다음, 타겟 유저U의 baseline을 더해 영화i에 대한 예측평점을 구한다.  

- SVD: 특이값 분해(SVD) 알고리즘, Netflix Prize에서 Simon Funk에 의해서 유명해진 알고리즘. 행렬 분해방식으로 user-item matrix를 분해하되, MF가 2개의 행렬로 분해했다면, SVD는 3가지 행렬로 분해됨. 결측치가 많은 경우, NMF 알고리즘보다 좋은 결과를 내기 어렵다고 함

- SVD++: 특이값 분해(SVD++) 알고리즘 개념에 암시적 rating을 계산하는 과정이 추가된 개념. 

- NMF : Non-negative 행렬분해(음수를 포함하지 않는 행렬분해)

- SGD(Stochastic Gradient Descent) : 결측치에 관계없이 사용할 수 있음. 

 ※참고자료: https://eda-ai-lab.tistory.com/528

- ALS(Alternating Least Square) : 기존 SGD가 2개 행렬(user latent, item latent)를 동시에 최적화하는 방법이라면,  ALS는 두 행렬 중 하나를 고정시키고 다른 하나의 행렬을 순차적으로 반복하면서 최적화하는 방법입니다. 이렇게 하면, 기존의 최적화 문제가 convex 형태로 바뀌기에 수렴된 행렬을 찾을 수 있는 장점이 있음. (결측치가 있으면 에러가 나므로 0으로 변경하여 진행해야 함)
 - (1) 초기 아이템, 사용자 행렬을 초기화
 - (2) 아이템 행렬 고정하고 사용자 행렬을 최적화
 - (3) 사용자 행렬 고정하고 아이템 행렬을 최적화
 - (4) 위 2,3과정을 반복. 
  ==> convex형태의 문제가 되어, 수렴된 행렬의 '최적해'를 알게 됨. 
 
  ※참고자료: https://eda-ai-lab.tistory.com/529?category=736098

### 딥러닝 활용 추천알고리즘 
 - (1) [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)
 - (2) [Deep Autoencoders For Collaborative Filtering](https://towardsdatascience.com/deep-autoencoders-for-collaborative-filtering-6cf8d25bbf1d)
 - (2-1) [Training Deep Autoencoders For Collaborative Filtering_한국어_블로깅 자료](https://soobarkbar.tistory.com/124)
 - (2-2) [code](https://github.com/NVIDIA/DeepRecommender)

# LaundryClassifier

이 코드 파일은 세탁물의 이미지와 무게 데이터를 이용해 세탁물의 종류를 분류하는 AI 모델을 구축하고 학습하는 코드입니다.
이미지 데이터는 CNN(Convolutional Neural Network)을 통해 처리되며, 무게 데이터는 FNN(Feedforward Neural Network)을 사용해 처리합니다.
이미지 데이터는 ImageDataGenerator를 통해 증강 및 전처리되며, 무게 데이터는 CSV 파일에서 로드합니다.
두 입력 데이터를 결합하여 최종적으로 세탁물의 종류를 분류하는 모델을 학습합니다. 이 모델은 다중 분류 문제를 해결하기 위해 설계되었습니다.


[이미지 입력] -> CNN -> Dense(128) -> Flatten -> [이미지 임베딩 벡터]
                                                    |
                                                    V
[무게 입력]  -> FNN -> Dense(32) -> [무게 임베딩 벡터] 
                                                    |
                                    [Concatenate: 이미지 + 무게 벡터]
                                                    |
                                  Dense(128) -> Dense(64) -> Softmax

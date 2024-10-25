import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# CNN 부분 - 이미지 처리
cnn_input = Input(shape=(64, 64, 3))  # 이미지 크기는 임의로 64x64로 설정
cnn_layer = Conv2D(32, (3, 3), activation='relu')(cnn_input)
cnn_layer = MaxPooling2D((2, 2))(cnn_layer)
cnn_layer = Conv2D(64, (3, 3), activation='relu')(cnn_layer)
cnn_layer = MaxPooling2D((2, 2))(cnn_layer)
cnn_layer = GlobalAveragePooling2D()(cnn_layer)

# FNN 부분 - 무게 데이터 처리
weight_input = Input(shape=(1,))  # 무게 데이터는 단일 값으로 입력
weight_layer = Dense(16, activation='relu')(weight_input)
weight_layer = Dense(8, activation='relu')(weight_layer)

# 두 부분을 결합
combined = Concatenate()([cnn_layer, weight_layer])

# 완전 연결층을 추가하여 최종 출력을 얻음
fc = Dense(64, activation='relu')(combined)
fc = Dense(32, activation='relu')(fc)
output = Dense(5, activation='softmax')(fc)  # 예를 들어 5가지 세탁물 종류를 분류한다고 가정

# 최종 모델 정의
final_model = Model(inputs=[cnn_input, weight_input], outputs=output)

# 모델 컴파일
final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 요약 출력
final_model.summary()

# 이미지 데이터 제너레이터 생성
image_datagen = ImageDataGenerator(
    rescale=1./255,  # 이미지 정규화
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 데이터 경로 설정
image_data_directory = ''  
weight_data_path = '' 

# 이미지 데이터 제너레이터 적용 (디렉터리에서 데이터 로드)
image_data_augmented = image_datagen.flow_from_directory(
    image_data_directory,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# 무게 데이터 로드
weight_data_df = pd.read_csv(weight_data_path)
weight_data = weight_data_df['weight'].values.reshape(-1, 1)
labels = weight_data_df['label'].values
labels = tf.keras.utils.to_categorical(labels, num_classes=5)  # 원-핫 인코딩

# 모델 학습 (이미지와 무게 데이터를 함께 사용)
final_model.fit(
    [image_data_augmented.next(), weight_data], labels, epochs=10, batch_size=32
)

# 예측
test_image_data_directory = ''  
test_weight_data_path = ''  

# 테스트 이미지 데이터 로드
test_image_data_gen = ImageDataGenerator(rescale=1./255)
test_image_data = test_image_data_gen.flow_from_directory(
    test_image_data_directory,
    target_size=(64, 64),
    batch_size=10,
    class_mode=None,
    shuffle=False
)

# 테스트 무게 데이터 로드
test_weight_data_df = pd.read_csv(test_weight_data_path)
test_weight_data = test_weight_data_df['weight'].values.reshape(-1, 1)

# 예측 수행
predictions = final_model.predict([test_image_data.next(), test_weight_data])

# 예측 결과 출력
print("예측 결과:")
print(predictions)

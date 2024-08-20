import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model, load_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean
import os

# 파인 튜닝된 모델 로드 및 수정
def load_feature_extractor(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일 {model_path}을(를) 찾을 수 없습니다.")
    
    model = load_model(model_path)
    print(model.summary())
    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
    return feature_extractor

# 이미지 전처리
def preprocess_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일 {img_path}을(를) 찾을 수 없습니다.")

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    return img_array

# 특징 벡터 추출
def extract_features(img_path, model):
    img_array = preprocess_image(img_path)
    features = model.predict(img_array).flatten()  # 1차원 벡터로 변환
    return features

# 유사도 계산
def calculate_similarity(features1, features2):
    # 정규화 및 유클리드 거리 기반 유사도 계산
    features1 = normalize(features1.reshape(1, -1))
    features2 = normalize(features2.reshape(1, -1))
    distance = euclidean(features1.flatten(), features2.flatten())  # 1차원으로 변환하여 거리 계산
    similarity = 1 / (1 + distance)
    return similarity

# 모델 경로와 이미지 경로 설정
model_path = 'Fashion_MNIST_MobileNetV3Small_final.h5'
image_path1 = 'test_2.png'
image_path2 = 'Data_1.png'

try:
    feature_extractor = load_feature_extractor(model_path)
    features1 = extract_features(image_path1, feature_extractor)
    features2 = extract_features(image_path2, feature_extractor)

    # 이미지 간 유사도 계산
    similarity_score = calculate_similarity(features1, features2)
    print(f"유사도: {similarity_score*100:.2f}%")
except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)
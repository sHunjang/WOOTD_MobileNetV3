import tensorflow as tf
from keras.applications import MobileNetV3Small
from keras.models import Model
from keras.preprocessing import image
from keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from PIL import Image

# MobileNetV3 모델 로드 (최종 분류 레이어 제외)
input_shape = (224, 224, 3)
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(img_path, model):
    img_array = preprocess_image(img_path)
    features = model.predict(img_array)
    return features

def cosine_similarity_score(features1, features2):
    return cosine_similarity(features1, features2)[0][0]

def euclidean_distance_score(features1, features2):
    return euclidean(features1.flatten(), features2.flatten())

def pearson_correlation_score(features1, features2):
    return pearsonr(features1.flatten(), features2.flatten())[0]

def plot_images_with_similarity(style_img_path, wardrobe_img_path, cos_sim, euc_dist, pearson_corr):
    style_img = Image.open(style_img_path)
    wardrobe_img = Image.open(wardrobe_img_path)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(style_img)
    ax[0].axis('off')
    ax[0].set_title('Wannabe Style Image')
    
    ax[1].imshow(wardrobe_img)
    ax[1].axis('off')
    ax[1].set_title('User Clothing Combination Image')
    
    plt.suptitle(f'Cosine Similarity: {cos_sim:.2f} | Euclidean Distance: {euc_dist:.2f} | Pearson Correlation: {pearson_corr:.2f}')
    plt.show()

# 이미지 파일 경로 설정
style_image_path = 'Wannabe_Combinations/Image_22.png'
user_clothing_combination_path = 'Wannabe_Combinations/Image_22.png'

# 예시 이미지의 특징 벡터 추출
style_features = extract_features(style_image_path, model)
wardrobe_features = extract_features(user_clothing_combination_path, model)

# 유사도 측정
cos_sim = cosine_similarity_score(style_features, wardrobe_features)
euc_dist = euclidean_distance_score(style_features, wardrobe_features)
pearson_corr = pearson_correlation_score(style_features, wardrobe_features)

# 이미지와 유사도 점수 시각화
plot_images_with_similarity(style_image_path, user_clothing_combination_path, cos_sim, euc_dist, pearson_corr)

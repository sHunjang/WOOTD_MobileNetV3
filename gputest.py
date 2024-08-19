import tensorflow as tf

# GPU 사용 가능 여부 확인
print("MPS device available: ", tf.config.list_physical_devices('GPU'))

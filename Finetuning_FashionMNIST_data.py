import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.applications import MobileNetV3Small
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

# 하이퍼파라미터
batch_size = 32
epochs_initial = 20
epochs_finetune = 20
learning_rate_initial = 0.0001
learning_rate_finetune = 0.00001

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=-1))
x_test = tf.image.grayscale_to_rgb(tf.expand_dims(x_test, axis=-1))
x_train = tf.image.resize(x_train, [224, 224])
x_test = tf.image.resize(x_test, [224, 224])
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 모델 구성
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=learning_rate_initial), loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# 모델 학습
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_initial, validation_data=(x_test, y_test), callbacks=[checkpoint, early_stop])

# 미세 조정
for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=learning_rate_finetune), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_finetune, validation_data=(x_test, y_test), callbacks=[checkpoint, early_stop])

# 모델 저장
model.save('Fashion_MNIST_MobileNetV3Small_final.h5')
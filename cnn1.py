from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import UpSampling2D, Concatenate, ZeroPadding2D
from tensorflow.keras import layers
from tensorflow.keras.layers import Cropping2D
import json
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

import cv2
import numpy as np

train_category = np.load("train_category.npy")
train_diff_images=np.load("train_diff_images.npy")
test_diff_images=np.load("test_diff_images.npy")
test_category=np.load("test_category.npy")
val_diff_images=np.load("val_diff_images.npy")
val_category=np.load("val_category.npy")

from tensorflow.keras import layers, models, regularizers

def parallel_convolution(input_tensor):
    conv_1x1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(input_tensor)
    conv_3x3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_tensor)
    conv_5x5 = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(input_tensor)
    return layers.concatenate([conv_1x1, conv_3x3, conv_5x5], axis=-1)

input_tensor = layers.Input(shape=(400, 400, 1))
x = layers.Conv2D(90, (5, 5), activation='relu')(input_tensor)
x = layers.MaxPooling2D((3, 3))(x)
x = layers.Conv2D(90, (1, 1), activation='relu')(x)
x = layers.MaxPooling2D((3, 3))(x)
x = layers.SeparableConv2D(200, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((4, 4))(x)
x = parallel_convolution(x)  # 应用并行卷积
x = layers.Conv2D(384, (3, 3), activation='relu')(x)
x = layers.Conv2D(384, (3, 3), activation='relu')(x)
x = layers.Conv2D(256, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
output_tensor = layers.Dense(14, activation='softmax')(x)

final_model = models.Model(inputs=input_tensor, outputs=output_tensor)

# def parallel_convolution(input_tensor):
#     # 使用 1x1、3x3、5x5 尺寸的卷积核进行卷积
#     conv_1x1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(input_tensor)
#     conv_3x3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_tensor)
#     conv_5x5 = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(input_tensor)
#
#     # 添加深度可分离卷积层
#     depthwise_conv = layers.DepthwiseConv2D((3, 3), activation='relu', padding='same')(input_tensor)
#
#     # 将四个卷积的结果在最后一个轴上拼接起来
#     return layers.concatenate([conv_1x1, conv_3x3, conv_5x5, depthwise_conv], axis=-1)
#
#
# # 输入层
# input_tensor = layers.Input(shape=(400, 400, 1))
#
# # 网络结构
# x = layers.Conv2D(90, (5, 5), activation='relu')(input_tensor)
# x = layers.MaxPooling2D((3, 3))(x)
# x = layers.Conv2D(90, (1, 1), activation='relu')(x)
# x = layers.MaxPooling2D((3, 3))(x)
# x = layers.Conv2D(200, (3, 3), activation='relu')(x)
# x = layers.MaxPooling2D((4, 4))(x)
# x = parallel_convolution(x)  # 应用并行卷积，包含深度可分离卷积
# x = layers.Conv2D(384, (3, 3), activation='relu')(x)
# x = layers.Conv2D(384, (3, 3), activation='relu')(x)
# x = layers.Conv2D(256, (3, 3), activation='relu')(x)
# x = layers.Flatten()(x)
# x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
# output_tensor = layers.Dense(14, activation='softmax')(x)
#
# # 构建模型
# final_model = models.Model(inputs=input_tensor, outputs=output_tensor)

early_stopping = EarlyStopping(monitor='val_loss',patience=5, restore_best_weights=True)
early_stopping_acc = EarlyStopping(monitor='val_accuracy',
                                   patience=5,
                                   verbose=1,
                                   mode='max',
                                   restore_best_weights=True)
# 编译模型
final_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
final_model.fit(train_diff_images, train_category, epochs=30, batch_size=32,
          validation_data=(val_diff_images,val_category),callbacks=[early_stopping,early_stopping_acc])

# 评估模型性能
test_loss, test_accuracy = final_model.evaluate(test_diff_images, test_category)
print(f"Test accuracy: {test_accuracy}")

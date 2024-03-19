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

def align_images(image1, image2):
    # 将图像转换为灰度
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # gray1=image1
    # gray2=image2
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 寻找关键点和描述符
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 检查描述符是否为 None
    if des1 is None or des2 is None:
        print("无法在至少一幅图像中找到足够的特征点")
        return None

    # 确保每组描述符至少有 k 个特征点
    k = 2
    if len(des1) < k or len(des2) < k:
        print("至少一组描述符的特征点数量少于 k")
        return None

    # 创建FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 确保描述符是浮点型，对于SIFT这通常是不必要的
    if des1.dtype != np.float32:
        des1 = des1.astype(np.float32)
    if des2.dtype != np.float32:
        des2 = des2.astype(np.float32)

    matches = flann.knnMatch(des1, des2, k=k)

    # 使用Lowes比率测试过滤匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 至少需要4个好的匹配点来寻找单应性
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 确保h是有效的单应性矩阵
        if h is not None and h.shape == (3, 3):
            height, width = image2.shape[:2]
            aligned_image = cv2.warpPerspective(image1, h, (width, height))
            return aligned_image
        else:
            return None
    else:
        return None

def create_difference_image(image1, image2):
    # 确保两幅图像都是单通道的灰度图
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 确保两幅图像尺寸相同
    if image1.shape != image2.shape:
        print("警告: 图像尺寸不匹配，尝试调整尺寸。")
        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    # 计算绝对差异
    diff_image = cv2.absdiff(image1, image2)
    return diff_image

def rotate_image(image, angle):
    """ 使用OpenCV旋转图像 """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 进行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def flip_image(image, mode='horizontal'):
    """ 使用OpenCV翻转图像 """
    if mode == 'horizontal':
        return cv2.flip(image, 1)  # 水平翻转
    elif mode == 'vertical':
        return cv2.flip(image, 0)  # 垂直翻转

def enhance4_image(image):
    image1=rotate_image(image,45)
    image2=flip_image(image)
    image3=flip_image(image,mode='vertical')
    return image1,image2,image3

def enhance5678910_image(image):
    image1 = rotate_image(image, 45)
    image2 = flip_image(image)
    image3 = flip_image(image, mode='vertical')
    image4=rotate_image(image, 22.5)
    image5 = rotate_image(image, 67.5)
    return image1, image2, image3,image4,image5

def enhance1112_image(image):
    image1 = rotate_image(image, 45)
    image2 = flip_image(image)
    image3 = flip_image(image, mode='vertical')
    image4 = rotate_image(image, 22.5)
    image5 = rotate_image(image, 67.5)
    image6 = rotate_image(image, 135)
    return image1, image2, image3, image4, image5,image6

def enhance13_image(image):
    image1 = rotate_image(image, 45)
    image2 = flip_image(image)
    image3 = flip_image(image, mode='vertical')
    image4 = rotate_image(image, 22.5)
    image5 = rotate_image(image, 67.5)
    image6 = rotate_image(image, 135)
    image7 = rotate_image(image, 225)
    return image1, image2, image3, image4, image5,image6,image7
# 读取 JSON 文件
with open('flaw_data/train/train.json', 'r') as json_file:
    data = json.load(json_file)
images=data["images"]
train_diff_images=[]
#print(len(images))
#print(target_image_name)
train_category=[]
#创建文件名-类别 字典
name_dict={}
with open("flaw_data/train/train_cls.txt",'r') as file:
    for line in file:
        parts = line.split(',')
        name=parts[0]
        num = parts[1].strip()
        name_dict[name]=num

for i in range(len(images)):
    file_name = "flaw_data/train/sample/" + images[i]["file_name"]
    # image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(file_name)
    if image is None:
        continue
    template_name = "flaw_data/train/template/" + images[i]["template_name"]
    # template_image = cv2.imread(template_name, cv2.IMREAD_GRAYSCALE)
    template_image = cv2.imread(template_name)
    if template_image is None:
        continue
    # image = cv2.resize(image, (400, 400))
    # template_image = cv2.resize(template_image, (400, 400))
    # 对齐图像
    aligned_image = align_images(image, template_image)
    print(template_image.shape)
    if aligned_image is not None:
        diff_image = create_difference_image(aligned_image, template_image)
    else:
        continue
    diff_image = cv2.resize(diff_image, (400, 400))
    #根据不同类别写入
    if name_dict[images[i]["file_name"]]==4:
        train_diff_images.append(diff_image)
        image1, image2, image3=enhance4_image(diff_image)
        train_diff_images.extend([image1, image2, image3])
        for i in range(4):
            train_category.append(name_dict[images[i]["file_name"]])#循环加入标签
        continue
    if name_dict[images[i]["file_name"]] in [5,6,7,8,9,10]:
        train_diff_images.append(diff_image)
        image1, image2, image3,image4,image5=enhance5678910_image(diff_image)
        train_diff_images.extend([image1, image2, image3,image4,image5])
        for i in range(6):
            train_category.append(name_dict[images[i]["file_name"]])#循环加入标签
        continue
    if name_dict[images[i]["file_name"]] in [11,12]:
        train_diff_images.append(diff_image)
        image1, image2, image3,image4,image5,image6=enhance1112_image(diff_image)
        train_diff_images.extend([image1, image2, image3,image4,image5,image6])
        for i in range(7):
            train_category.append(name_dict[images[i]["file_name"]])#循环加入标签
        continue
    if name_dict[images[i]["file_name"]]==13:
        train_diff_images.append(diff_image)
        image1, image2, image3,image4,image5,image6,image7=enhance13_image(diff_image)
        train_diff_images.extend([image1, image2, image3,image4,image5,image6,image7])
        for i in range(8):
            train_category.append(name_dict[images[i]["file_name"]])#循环加入标签
        continue
    #写入其他类别
    train_diff_images.append(diff_image)
    train_category.append(name_dict[images[i]["file_name"]])  # 循环加入标签
train_category = np.array(train_category).astype(float)
train_diff_images=np.array(train_diff_images).astype(float)

print(train_diff_images.dtype)
print(train_diff_images.shape)
print(train_category.shape)
np.save("train_diff_images.npy",train_diff_images)
np.save("train_category.npy",train_category)

#test data
with open('flaw_data/test/test.json', 'r') as tjson_file:
    tdata = json.load(tjson_file)
timages=tdata["images"]
test_diff_images=[]
test_category=[]

name_dict={}
with open("flaw_data/test/test_cls.txt",'r') as file:
    for line in file:
        parts=line.split(',')
        name=parts[0]
        num=parts[1].strip()
        name_dict[name] = num

for i in range(len(timages)):
    file_name = "flaw_data/test/sample/" + timages[i]["file_name"]
    image = cv2.imread(file_name)
    template_name = "flaw_data/test/template/" + timages[i]["template_name"]
    template_image = cv2.imread(template_name)
    #image = cv2.resize(image, (400, 400))
    #template_image = cv2.resize(template_image, (400, 400))
    aligned_image = align_images(image, template_image)
    print(template_image.shape)
    if aligned_image is not None:
        diff_image = create_difference_image(aligned_image, template_image)
    else:
        continue
    diff_image = cv2.resize(diff_image, (400, 400))

    test_diff_images.append(diff_image)
    test_category.append(name_dict[timages[i]["file_name"]])
test_diff_images=np.array(test_diff_images).astype(float)
test_category=np.array(test_category).astype(float)
print(test_diff_images.dtype)
print(test_diff_images.shape)
print(test_category.shape)
np.save("test_diff_images.npy",test_diff_images)
np.save("test_category.npy",test_category)
#val data
with open('flaw_data/val/val.json', 'r') as tjson_file:
    tdata = json.load(tjson_file)
timages=tdata["images"]
val_diff_images=[]
val_category=[]

name_dict={}
with open("flaw_data/val/val_cls.txt",'r') as file:
    for line in file:
        parts=line.split(',')
        name=parts[0]
        num=parts[1].strip()
        name_dict[name] = num


for i in range(len(timages)):
    file_name = "flaw_data/val/sample/" + timages[i]["file_name"]
    image = cv2.imread(file_name)
    template_name = "flaw_data/val/template/" + timages[i]["template_name"]
    template_image = cv2.imread(template_name)
    aligned_image = align_images(image, template_image)
    print(template_image.shape)
    if aligned_image is not None:
        diff_image = create_difference_image(aligned_image, template_image)
    else:
        continue
    diff_image = cv2.resize(diff_image, (400, 400))

    val_diff_images.append(diff_image)
    val_category.append(name_dict[timages[i]["file_name"]])
val_diff_images=np.array(test_diff_images).astype(float)
val_category=np.array(test_category).astype(float)
print(val_diff_images.dtype)
print(val_diff_images.shape)
print(val_category.shape)
np.save("val_diff_images.npy",val_diff_images)
np.save("val_category.npy",val_category)

# #图像增强
# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
# 重新塑形以添加通道维度 (灰度通道)
# train_diff_images = train_diff_images.reshape(train_diff_images.shape[0], 400, 400, 1)
# test_diff_images = test_diff_images.reshape(test_diff_images.shape[0], 400, 400, 1)
#
# # 应用数据增强
# datagen.fit(train_diff_images)
#
# model = models.Sequential()
#
# # 初始层
# model.add(layers.Conv2D(90, (5, 5), activation='relu', input_shape=(400, 400, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
#
# # 第二层
# model.add(layers.Conv2D(90, (1, 1), activation='relu'))
# model.add(layers.MaxPooling2D((3, 3)))
#
# # 保存第二层的特征图用于后续融合
# feature_map_layer2 = model.output
#
# # 更多层
# model.add(layers.Conv2D(200, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((4,4)))
# model.add(layers.Conv2D(384, (3,3), activation='relu'))
# model.add(layers.Conv2D(384, (3,3), activation='relu'))
# model.add(layers.Conv2D(256, (3,3), activation='relu'))
#
# # 上采样以匹配特征图尺寸
# upsampled_feature_map = UpSampling2D(size=(6, 6))(model.output) # 调整尺寸因子以匹配第二层特征图尺寸
# # upsampled_feature_map = UpSampling2D(size=(2, 2))(upsampled_feature_map)  # 调整上采样比例
# # upsampled_feature_map = Cropping2D(cropping=((6, 6), (6, 6)))(upsampled_feature_map)  # 裁剪
# upsampled_feature_map = ZeroPadding2D(padding=((3, 3), (3, 3)))(upsampled_feature_map)  # 填充
# concatenated_feature_map = Concatenate()([feature_map_layer2, upsampled_feature_map])
#
# # 使用融合的特征图继续模型
# final_model_output = layers.Flatten()(concatenated_feature_map)
# final_model_output = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(final_model_output)
# final_model_output = layers.Dense(14, activation='softmax')(final_model_output)
#
# # 创建最终模型
# final_model = models.Model(inputs=model.input, outputs=final_model_output)

# final_model = models.Sequential(
#     layers.Conv2D(90, (5, 5), activation='relu', input_shape=(400, 400, 1)),
#     layers.MaxPooling2D((3, 3)),
#     layers.Conv2D(90, (1, 1), activation='relu'),
#     layers.MaxPooling2D((3, 3)),
#     layers.Conv2D(200, (3, 3), activation='relu'),
#     layers.MaxPooling2D((4,4)),
#     layers.Conv2D(384, (3,3), activation='relu'),
#     layers.Conv2D(384, (3,3), activation='relu'),
#     layers.Conv2D(256, (3,3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
#     layers.Dense(14, activation='softmax')
# )

final_model = models.Sequential([
    layers.Conv2D(90, (5, 5), activation='relu', input_shape=(400, 400, 1)),
    layers.MaxPooling2D((3, 3)),
    layers.Conv2D(90, (1, 1), activation='relu'),
    layers.MaxPooling2D((3, 3)),
    layers.Conv2D(200, (3, 3), activation='relu'),
    layers.MaxPooling2D((4,4)),
    layers.Conv2D(384, (3,3), activation='relu'),
    layers.Conv2D(384, (3,3), activation='relu'),
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(14, activation='softmax')
])

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


# 获取特定字段或对象
# file_name="flaw_data/train/sample/"+images[2]["file_name"]
# image= cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
# template_name="flaw_data/train/template/"+images[2]["template_name"]
# template_image = cv2.imread(template_name, cv2.IMREAD_GRAYSCALE)
# diff_image = cv2.absdiff(image, template_image)
# print(diff_image.shape)


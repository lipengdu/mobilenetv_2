'''
 数据增强（train，test）
'''
import os
from keras.preprocessing.image import ImageDataGenerator
import constants

train_datagen = ImageDataGenerator(   #数据增强
    rescale=1./255,
    rotation_range=30, #旋转范围
    width_shift_range=0.2, #宽度转移范围
    height_shift_range=0.2, #高度旋转范围
    shear_range=0.2, #剪切范围
    zoom_range=0.2, #缩放范围
    channel_shift_range=20, #通道旋转范围
    horizontal_flip=True, #水平线
    fill_mode='nearest', #
)

# Validation data should not be modified
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

train_dir = os.path.join(constants.BASE_DIR, 'train')
test_dir = os.path.join(constants.BASE_DIR, 'test')

def create_generators(height, width):
    train_generator = train_datagen.flow_from_directory(   #目录路径（数据增强）
        train_dir,
        target_size=(height, width),
        class_mode='categorical',
        batch_size=constants.GENERATOR_BATCH_SIZE  # 64
    )

    validation_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=(height, width),
        class_mode='categorical',
        batch_size=constants.GENERATOR_BATCH_SIZE
    )

    return[train_generator, validation_generator]

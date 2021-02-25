import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD,Adam   #优化器
from pathlib import Path
from keras.applications.mobilenet_v2 import MobileNetV2  #model
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D  #层
from keras import initializers, regularizers

# reusable stuff
import constants
import callbacks
import generators

# No kruft plz
clear_session()

from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

# Config
height = constants.SIZES['basic']
width = height
weights_file = "weights.best_mobilenet" + str(height) + ".hdf5"

conv_base = MobileNetV2(
    weights='imagenet', 
    include_top=False,   #包含头部
    input_shape=(height, width, constants.NUM_CHANNELS)
)

# First time run, no unlocking
conv_base.trainable = False  #可训练

# Let's see it
print('Summary')
print(conv_base.summary())

# Let's construct that top layer replacement
x = conv_base.output
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)  #按行将维度
x = Dense(256, activation='relu', kernel_initializer=initializers.he_normal(seed=None), kernel_regularizer=regularizers.l2(.0005))(x) #正则化
x = Dropout(0.5)(x)
# Essential to have another layer for better accuracy
x = Dense(128,activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
x = Dropout(0.25)(x)
predictions = Dense(constants.NUM_CLASSES,  kernel_initializer="glorot_uniform", activation='softmax')(x)

print('Stacking New Layers')
model = Model(inputs = conv_base.input, outputs=predictions)
model._get_distribution_strategy = lambda: None
# Load checkpoint if one is found
if os.path.exists(weights_file):
        print ("loading ", weights_file)
        model.load_weights(weights_file)

# Get all model callbacks  #模型回调
callbacks_list = callbacks.make_callbacks(weights_file)

print('Compile model')
# originally adam, but research says SGD with scheduler
opt = Adam(lr=0.001, amsgrad=True)
#opt = SGD(momentum=.9)
model.compile(  #编译
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Get training/validation data via generators
train_generator, validation_generator = generators.create_generators(height, width)

print('Start training!')

history = model.fit_generator(
    train_generator,
    callbacks=callbacks_list,   #回调函数
    epochs=constants.TOTAL_EPOCHS,
    steps_per_epoch=constants.STEPS_PER_EPOCH,
    shuffle=True,
    workers=4,
    use_multiprocessing=False,
    validation_data=validation_generator,
    validation_steps=constants.VALIDATION_STEPS
)

# Save it for later
print('Saving Model')
model.save("nsfw_mobilenet2_A1." + str(width) + "x" + str(height) + ".h5")

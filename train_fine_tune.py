import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from tensorflow.keras.models import Sequential, Model, load_model

# reusable stuff
import constants
import callbacks
import generators

# No kruft plz
clear_session()
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#----------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #0映射gpu设备
sess = tf.compat.v1.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

# Config
height = constants.SIZES['basic']
width = height
weights_file = "weights.best_mobilenet" + str(height) + ".hdf5"

print ('Starting from last full model run')
model = load_model("nsfw_mobilenet2." + str(width) + "x" + str(height) + ".h5")

# Unlock a few layers deep in Mobilenet v2  #解锁最深的几层
model.trainable = True
set_trainable = False
for layer in model.layers:  #层集
    if layer.name == 'block_11_expand':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Let's see it
print('Summary')
print(model.summary())  #参数总结

# Load checkpoint if one is found   #检查点
if os.path.exists(weights_file):
    print("loading ", weights_file)
    model.load_weights(weights_file)
model._get_distribution_strategy = lambda: None
# Get all model callbacks  #模型回调函数
callbacks_list = callbacks.make_callbacks(weights_file)

print('Compile model')  #模型编译

opt=Adam(learning_rate=0.01,beta_1=0.99,amsgrad=True)
model.compile(  #编译
    loss='categorical_crossentropy',  #交叉熵分类损失
    optimizer=opt,
    metrics=['accuracy']
)
# Get training/validation data via generators
train_generator, validation_generator = generators.create_generators(height, width)

print('Start training!')

history = model.fit_generator(
    train_generator,
    callbacks=callbacks_list, #回调函数
    epochs=constants.TOTAL_EPOCHS,   #100
    steps_per_epoch=constants.STEPS_PER_EPOCH,  #500
    shuffle=True,
    workers=4,
    use_multiprocessing=False,
    validation_data=validation_generator,  #TEST
    validation_steps=constants.VALIDATION_STEPS  #50
)

# Save it for later
print('Saving Model')
model.save("nsfw_mobilenet_v2." + str(width) + "x" + str(height) + ".h5")

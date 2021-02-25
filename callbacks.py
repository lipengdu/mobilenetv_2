from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from time import time

# Slow down training deeper into dataset  #放慢训练
def schedule(epoch):
    if epoch < 6:
        # Warmup model first
        return .0000032
    elif epoch < 10:
        return .01
    elif epoch < 18:
        return .002
    elif epoch < 32:
        return .0004
    elif epoch < 54:
        return .00008
    elif epoch < 74:
        return .000016
    elif epoch < 85:
        return .0000032        
    else:
        return .0000007

def make_callbacks(weights_file): #回调函数
    # checkpoint
    filepath = weights_file
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # Update info  #地址
    tensorboard = TensorBoard(log_dir=r"logs\{}\\".format(time()))

    # learning rate schedule   #学习率（损失）
    lr_scheduler = LearningRateScheduler(schedule)

    # all the goodies  #返回参数
    return [lr_scheduler, checkpoint, tensorboard]

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

def VGG16_Avgpool(shape):
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    new_model = tf.keras.Sequential()
    for layer in vgg.layers:
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            new_model.add(tf.keras.layers.AveragePooling2D())
        else:
            new_model.add(layer)
    return new_model

import tensorflow as tf
import tensorflow.keras as keras


class Conv_Upsample_Concat(keras.layers.Layer):
    def __init__(self, kernel_num_1, kernel_num_2,
                 kernel_size_1, kernel_size_2, activation,
                 name):
        super(Conv_Upsample_Concat, self).__init__()
        self.conv_1 = keras.layers.Conv2D(filters=kernel_num_1,
                                          kernel_size=kernel_size_1,
                                          name=name + '_conv1',
                                          strides=1,
                                          padding='same')
        self.conv_2 = keras.layers.Conv2D(filters=kernel_num_2,
                                          kernel_size=kernel_size_2,
                                          name=name + '_conv2',
                                          strides=1,
                                          activation=getattr(tf.nn, activation),
                                          padding='same')

    def call(self, inputs, **kwargs):
        input_tensor, concat_tensor = inputs
        output = self.conv_1(input_tensor)
        shape = tf.shape(concat_tensor)[1:3]
        output = tf.image.resize(output, size=shape, method=tf.image.ResizeMethod.GAUSSIAN)
        output = tf.concat(values=[output, concat_tensor], axis=3)
        output = self.conv_2(output)
        return output

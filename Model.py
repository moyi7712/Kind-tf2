import tensorflow as tf
import tensorflow.keras as keras
import Layer


class Decom(keras.Model):
    def __init__(self):
        super(Decom, self).__init__()
        self.conv_1 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                          padding='same', name='conv_1',
                                          activation=tf.nn.leaky_relu)
        self.pool_1 = keras.layers.MaxPool2D(name='pool_1')
        self.conv_2 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                          padding='same', name='conv_2',
                                          activation=tf.nn.leaky_relu)
        self.pool_2 = keras.layers.MaxPool2D(name='pool_2')
        self.conv_3 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                          padding='same', name='conv_3',
                                          activation=tf.nn.leaky_relu)
        # self.pool_3 = keras.layers.MaxPool2D(name='pool_3')
        self.up_1 = Layer.Conv_Upsample_Concat(kernel_num_1=64, kernel_num_2=64,
                                               kernel_size_1=3, kernel_size_2=3,
                                               activation='leaky_relu',
                                               name='up_1')
        self.up_2 = Layer.Conv_Upsample_Concat(kernel_num_1=32, kernel_num_2=32,
                                               kernel_size_1=3, kernel_size_2=3,
                                               activation='leaky_relu',
                                               name='up_2')
        self.relfer = keras.layers.Conv2D(filters=3, kernel_size=1,
                                          strides=1, padding='same',
                                          name='relfer')
        self.illum_1 = keras.layers.Conv2D(filters=32, kernel_size=3,
                                           strides=1, padding='same',
                                           activation=tf.nn.leaky_relu,
                                           name='illum_1')
        self.illum_2 = keras.layers.Conv2D(filters=1, kernel_size=1,
                                           strides=1, padding='same',
                                           name='illum_2')

    def call(self, inputs, training=None, mask=None):
        decom_1 = self.conv_1(inputs)
        decom_1_pool = self.pool_1(decom_1)
        decom_2 = self.conv_2(decom_1_pool)
        decom_2_pool = self.pool_2(decom_2)
        decom_3 = self.conv_3(decom_2_pool)
        # decom_3_pool = self.pool_3(decom_3)

        up_1 = self.up_1([decom_3, decom_2])
        up_2 = self.up_2([up_1, decom_1])
        relfer = tf.nn.sigmoid(self.relfer(up_2))
        illum_1 = tf.concat(values=[self.illum_1(decom_1), up_2], axis=3)
        illum = self.illum_2(illum_1)
        return relfer, illum


class Restor(keras.Model):
    def __init__(self):
        super(Restor, self).__init__()
        self.conv_1_1 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                            padding='same', name='conv_1_1',
                                            activation=tf.nn.leaky_relu)
        self.conv_1_2 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                            padding='same', name='conv_1_2',
                                            activation=tf.nn.leaky_relu)
        self.pool_1 = keras.layers.MaxPool2D(name='pool_1')

        self.conv_2_1 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                            padding='same', name='conv_2_1',
                                            activation=tf.nn.leaky_relu)
        self.conv_2_2 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                            padding='same', name='conv_2_2',
                                            activation=tf.nn.leaky_relu)
        self.pool_2 = keras.layers.MaxPool2D(name='pool_2')

        self.conv_3_1 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                            padding='same', name='conv_3_1',
                                            activation=tf.nn.leaky_relu)
        self.conv_3_2 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                            padding='same', name='conv_3_2',
                                            activation=tf.nn.leaky_relu)
        self.pool_3 = keras.layers.MaxPool2D(name='pool_3')

        self.conv_4_1 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1,
                                            padding='same', name='conv_4_1',
                                            activation=tf.nn.leaky_relu)
        self.conv_4_2 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1,
                                            padding='same', name='conv_4_2',
                                            activation=tf.nn.leaky_relu)
        self.pool_4 = keras.layers.MaxPool2D(name='pool_4')

        self.conv_5_1 = keras.layers.Conv2D(filters=512, kernel_size=3, strides=1,
                                            padding='same', name='conv_5_1',
                                            activation=tf.nn.leaky_relu)
        self.conv_5_2 = keras.layers.Conv2D(filters=512, kernel_size=3, strides=1,
                                            padding='same', name='conv_5_2',
                                            activation=tf.nn.leaky_relu)
        # self.pool_5 = keras.layers.MaxPool2D(name='pool_5')

        self.up_1 = Layer.Conv_Upsample_Concat(kernel_num_1=256, kernel_num_2=256,
                                               kernel_size_1=3, kernel_size_2=3,
                                               activation='leaky_relu',
                                               name='up_1')
        self.up_1_conv = keras.layers.Conv2D(filters=256, kernel_size=3, strides=1,
                                             padding='same', name='up_1_conv',
                                             activation=tf.nn.leaky_relu)
        self.up_2 = Layer.Conv_Upsample_Concat(kernel_num_1=128, kernel_num_2=128,
                                               kernel_size_1=3, kernel_size_2=3,
                                               activation='leaky_relu',
                                               name='up_2')
        self.up_2_conv = keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
                                             padding='same', name='up_2_conv',
                                             activation=tf.nn.leaky_relu)
        self.up_3 = Layer.Conv_Upsample_Concat(kernel_num_1=64, kernel_num_2=64,
                                               kernel_size_1=3, kernel_size_2=3,
                                               activation='leaky_relu',
                                               name='up_3')
        self.up_3_conv = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                             padding='same', name='up_3_conv',
                                             activation=tf.nn.leaky_relu)
        self.up_4 = Layer.Conv_Upsample_Concat(kernel_num_1=32, kernel_num_2=32,
                                               kernel_size_1=3, kernel_size_2=3,
                                               activation='leaky_relu',
                                               name='up_4')
        self.up_4_conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                             padding='same', name='up_4_conv',
                                             activation=tf.nn.leaky_relu)

        self.restor = keras.layers.Conv2D(filters=3, kernel_size=3, strides=1,
                                          padding='same', name='restor',
                                          activation=tf.nn.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        relfer, illum = inputs
        inputs = tf.concat(values=[relfer, illum], axis=3)
        conv_1 = self.conv_1_2(self.conv_1_1(inputs))
        conv_1_pool = self.pool_1(conv_1)
        conv_2 = self.conv_2_2(self.conv_2_1(conv_1_pool))
        conv_2_pool = self.pool_1(conv_2)
        conv_3 = self.conv_3_2(self.conv_3_1(conv_2_pool))
        conv_3_pool = self.pool_1(conv_3)
        conv_4 = self.conv_4_2(self.conv_4_1(conv_3_pool))
        conv_4_pool = self.pool_4(conv_4)
        conv_5 = self.conv_5_2(self.conv_5_1(conv_4_pool))
        up_1 = self.up_1_conv(self.up_1([conv_5, conv_4]))
        up_2 = self.up_2_conv(self.up_2([up_1, conv_3]))
        up_3 = self.up_3_conv(self.up_3([up_2, conv_2]))
        up_4 = self.up_4_conv(self.up_4([up_3, conv_1]))
        restor = tf.nn.sigmoid(self.restor(up_4))

        return restor


class Adjust(keras.Model):
    def __init__(self):
        super(Adjust, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                         padding='same', name='conv_1',
                                         activation=tf.nn.leaky_relu)
        self.conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                         padding='same', name='conv_2',
                                         activation=tf.nn.leaky_relu)
        self.conv3 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                         padding='same', name='conv_3',
                                         activation=tf.nn.leaky_relu)
        self.conv4 = keras.layers.Conv2D(filters=1, kernel_size=3, strides=1,
                                         padding='same', name='conv_4',
                                         activation=tf.nn.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        illum, ratio = inputs
        inputs = tf.concat(values=[illum, ratio], axis=3)
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)
        output = tf.nn.sigmoid(self.conv4(output))
        return output


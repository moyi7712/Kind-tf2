import tensorflow as tf
import json
import random, os


class PipeLine(object):
    def __init__(self, config, batch_size=None):
        self.config = config
        self.batch_size = batch_size if batch_size else config.batch_size

        with open(config.filelist, 'r') as f:
            self.filelist = json.load(f)
        self.train_inputs_list = [os.path.join(self.config.train_input_path, temp) for temp in self.filelist['input']]
        self.train_lables_list = [os.path.join(self.config.train_lable_path, temp) for temp in self.filelist['input']]
        self.test_inputs_list = [os.path.join(self.config.test_input_path, temp) for temp in self.filelist['input']]
        self.test_lables_list = [os.path.join(self.config.test_lable_path, temp) for temp in self.filelist['input']]

    def _imread(self, file_path, flage='png'):
        if flage == 'png':
            image = tf.image.convert_image_dtype(tf.image.decode_png(tf.io.read_file(file_path)
                                                                     , dtype=tf.uint16), tf.dtypes.float32)
        else:
            image = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(file_path)
                                                                      ), tf.dtypes.float32)

        img_max = tf.reduce_max(image)
        img_min = tf.reduce_min(image)
        image = (image - img_min) / tf.maximum((img_max - img_min), 0.001)
        return image

    def _prosecc_train_Decom(self, name_input, name_lable):
        input = self._imread(name_input)
        lable = self._imread(name_lable)
        shape = tf.shape(input)
        random_x = random.randint(0, shape[0] - self.config.patch_size_Decom)
        random_y = random.randint(0, shape[1] - self.config.patch_size_Decom)

        input = input[random_x:random_x + self.config.patch_size_Decom,
                random_y:random_y + self.config.patch_size_Decom, :]
        lable = lable[random_x:random_x + self.config.patch_size_Decom,
                random_y:random_y + self.config.patch_size_Decom, :]
        k = random.randint(0, 3)
        is_flip = random.randint(0, 1)
        input = tf.image.rot90(input, k)
        lable = tf.image.rot90(lable, k)
        if is_flip:
            input = tf.image.flip_left_right(input)
            lable = tf.image.flip_left_right(lable)
        return input, lable

    def _prosecc_train_Restor(self, name_input, name_lable):
        input = self._imread(name_input)
        lable = self._imread(name_lable)
        shape = tf.shape(input)
        random_x = random.randint(0, shape[0] - self.config.patch_size_Restor)
        random_y = random.randint(0, shape[1] - self.config.patch_size_Restor)

        input = input[random_x:random_x + self.config.patch_size_Restor,
                random_y:random_y + self.config.patch_size_Restor, :]
        lable = lable[random_x:random_x + self.config.patch_size_Restor,
                random_y:random_y + self.config.patch_size_Restor, :]
        k = random.randint(0, 3)
        is_flip = random.randint(0, 1)
        input = tf.image.rot90(input, k)
        lable = tf.image.rot90(lable, k)
        if is_flip:
            input = tf.image.flip_left_right(input)
            lable = tf.image.flip_left_right(lable)
        return input, lable

    def _prosecc_train_Adjust(self, name_input, name_lable):
        input = self._imread(name_input)
        lable = self._imread(name_lable)
        shape = tf.shape(input)
        random_x = random.randint(0, shape[0] - self.config.patch_size_Adjust)
        random_y = random.randint(0, shape[1] - self.config.patch_size_Adjust)

        input = input[random_x:random_x + self.config.patch_size_Adjust,
                random_y:random_y + self.config.patch_size_Adjust, :]
        lable = lable[random_x:random_x + self.config.patch_size_Adjust,
                random_y:random_y + self.config.patch_size_Adjust, :]
        k = random.randint(0, 3)
        is_flip = random.randint(0, 1)
        input = tf.image.rot90(input, k)
        lable = tf.image.rot90(lable, k)
        if is_flip:
            input = tf.image.flip_left_right(input)
            lable = tf.image.flip_left_right(lable)
        return input, lable

    def _process_test(self, name_input, name_lable):
        input = self._imread(name_input)
        lable = self._imread(name_lable)
        shape = tf.shape(input)
        shape = tf.cast(512 * shape[:-1] / tf.reduce_max(shape), dtype=tf.int32)
        input = tf.image.resize(input, shape, method=tf.image.ResizeMethod.GAUSSIAN)
        lable = tf.image.resize(lable, shape, method=tf.image.ResizeMethod.GAUSSIAN)
        return input, lable

    def Train(self, flage):
        assert flage in ['Decom', 'Restor', 'Adjust']
        dataset = tf.data.Dataset.from_tensor_slices((self.train_inputs_list, self.train_lables_list)).shuffle(
            self.config.shuffle, reshuffle_each_iteration=True)
        if flage == 'Decom':
            dataset = dataset.map(self._prosecc_train_Decom, num_parallel_calls=self.config.num_parallel_calls).batch(
                self.config.batch_size_Decom)
        elif flage == 'Restor':
            dataset = dataset.map(self._prosecc_train_Restor, num_parallel_calls=self.config.num_parallel_calls).batch(
                self.config.batch_size_Restor)
        elif flage == 'Adjust':
            dataset = dataset.map(self._prosecc_train_Adjust, num_parallel_calls=self.config.num_parallel_calls).batch(
                self.config.batch_size_Adjust)
        dataset = dataset.prefetch(buffer_size=self.config.buffer_size)

        return dataset

    def Test(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.test_inputs_list, self.test_lables_list))
        dataset = dataset.repeat(-1)
        dataset = dataset.map(self._process_test)
        dataset = dataset.batch(1).shuffle(self.config.shuffle)
        return dataset

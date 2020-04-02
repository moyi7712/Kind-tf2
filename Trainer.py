# from Model import Model
from Config import Config
from Datapipe import PipeLine
from Network import Train_Adjust, Train_Restor, Train_Decom
import tensorflow as tf


class Train(object):
    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)

        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.strategy = tf.distribute.MirroredStrategy()


class Decom_train(Train):
    def __init__(self, config_path='config.yaml'):
        super(Decom_train, self).__init__(config_path=config_path)
        with self.strategy.scope():
            self.net = Train_Decom(self.config.HyperPara(), self.strategy, epoch=0)
        datapipe = PipeLine(self.config.Dataset()).Train(flage='Decom', num=self.strategy.num_replicas_in_sync)
        self.epoch = self.net.epoch
        self.dataset = self.strategy.experimental_distribute_dataset(datapipe)

    def train(self):
        with self.strategy.scope():

            for epoch in range(self.epoch,self.epoch+2000):

                step = 0
                for inputs, labels in self.dataset:

                    loss_dict = self.net.distributed_step(inputs, labels, epoch)
                    if step % 20 == 0:
                        with self.net.summary_writer.as_default():
                            for key in loss_dict:
                                tf.summary.scalar(key, loss_dict[key], step=epoch*225 + step)
                                loss_dict[key] = loss_dict[key].numpy()
                    if step % 200 == 0:
                        print('INFO: epoch: {}, step{}, total_loss:{}'.format(epoch, step, loss_dict['total_loss']))
                        self.net.ckpt_manager.save()
                    step += 1

class Restor_train(Train):
    def __init__(self, config_path='config.yaml'):
        super(Restor_train, self).__init__(config_path=config_path)
        with self.strategy.scope():
            self.net = Train_Restor(self.config.HyperPara(), self.strategy, epoch=0)
        datapipe = PipeLine(self.config.Dataset()).Train(flage='Restor', num=self.strategy.num_replicas_in_sync)
        self.epoch = self.net.epoch
        self.dataset = self.strategy.experimental_distribute_dataset(datapipe)

    def train(self):
        with self.strategy.scope():

            for epoch in range(self.epoch,self.epoch+2000):

                step = 0
                for inputs, labels in self.dataset:

                    loss_dict = self.net.distributed_step(inputs, labels, epoch)
                    if step % 20 == 0:
                        with self.net.summary_writer.as_default():
                            for key in loss_dict:
                                tf.summary.scalar(key, loss_dict[key], step=epoch*225 + step)
                                loss_dict[key] = loss_dict[key].numpy()
                    if step % 200 == 0:
                        print('INFO: epoch: {}, step{}, total_loss:{}'.format(epoch, step, loss_dict['total_loss']))
                        self.net.ckpt_manager.save()
                    step += 1






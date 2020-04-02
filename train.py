from Trainer import Decom_train,Restor_train,Adjust_train
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
train = Restor_train('config.yaml')
train.train()
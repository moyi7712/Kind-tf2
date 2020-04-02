import tensorflow as tf
from Model import Decom, Restor, Adjust
import os, random
import Utils


class Train(object):
    def __init__(self, config, strategy):
        self.strategy = strategy
        self.config = config
        # self.batch_size = config.batch_size
        # if strategy:
        #     self.global_batch_size = strategy.num_replicas_in_sync * self.batch_size
        self.Decom = Decom()
        self.Restor = Restor()
        self.Adjust = Adjust()

    def strategy2Tensor(self, loss):
        return tf.reduce_mean(self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None))


    def regularizer_loss(self, model):
        return tf.nn.scale_regularization_loss(tf.add_n(getattr(model, 'losses')))


class Train_Decom(Train):
    def __init__(self, config, strategy, epoch=104):
        super(Train_Decom, self).__init__(strategy=strategy,
                                          config=config)
        self.base_lr = config.Decom_lr
        self.batch_size = config.batch_size_Decom
        if strategy:
            self.global_batch_size = strategy.num_replicas_in_sync * self.batch_size
        self.epoch = epoch
        self.optmizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr)
        checkpoint = tf.train.Checkpoint(optimizer=self.optmizer,
                                         Model=self.Decom)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(config.Decom_prefix, 'summary'))
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=config.Decom_prefix, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            checkpoint.restore(self.ckpt_manager.latest_checkpoint)

    def lr_schedul(self, epoch):
        if epoch > 50:
            return self.base_lr / 2
        else:
            return self.base_lr

    def average_loss(self, loss):
        return tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)

    def train(self, inputs, labels, epoch):
        with tf.GradientTape(persistent=True) as tape:
            relfer_i, illum_i = self.Decom(inputs, training=True)
            relfer_l, illum_l = self.Decom(labels, training=True)
            I_i_3 = tf.concat([illum_i, illum_i, illum_i], axis=3)

            I_l_3 = tf.concat([illum_l, illum_l, illum_l], axis=3)

            recon_loss_i = tf.reduce_mean(tf.abs(relfer_i * I_i_3 - inputs), axis=[1, 2, 3])
            recon_loss_l = tf.reduce_mean(tf.abs(relfer_l * I_l_3 - labels), axis=[1, 2, 3])
            equal_relfer_loss = tf.reduce_mean(tf.abs(relfer_i - relfer_l), axis=[1, 2, 3])
            mutual_loss_illum = Utils.mutual_i_loss(illum_i, illum_l)

            mutual_loss_label_illum = Utils.mutual_i_input_loss(illum_l, labels)
            mutual_loss_input_illum = Utils.mutual_i_input_loss(illum_i, inputs)
            loss_total = 1 * (recon_loss_i + recon_loss_l)
            loss_total += 0.01 * equal_relfer_loss
            loss_total += 0.2 * mutual_loss_illum
            loss_total += 0.15 * (mutual_loss_input_illum + mutual_loss_label_illum)
            loss_total = self.average_loss(loss_total)
        gradient = tape.gradient(loss_total, self.Decom.trainable_variables)
        self.optmizer.apply_gradients(zip(gradient, self.Decom.trainable_variables))
        loss_return = [recon_loss_i, recon_loss_l, equal_relfer_loss, mutual_loss_illum, mutual_loss_input_illum,
                       mutual_loss_label_illum]
        loss_return = [tf.reduce_mean(loss_temp)/self.global_batch_size for loss_temp in loss_return]
        loss_return.append(loss_total)
        return loss_return

    @tf.function
    def distributed_step(self, inputs, labels, epoch):
        self.epoch = epoch
        self.optmizer.learning_rate = self.lr_schedul(epoch)
        loss = self.strategy.experimental_run_v2(self.train, args=(inputs, labels, epoch,))
        loss = [self.strategy2Tensor(lossTemp) for lossTemp in loss]
        loss_dict = {'recon_loss_i': loss[0],
                     'recon_loss_l': loss[1],
                     'equal_relfer_loss': loss[2],
                     'mutual_loss_illum': loss[3],
                     'mutual_loss_input_illum': loss[4],
                     'mutual_loss_label_illum': loss[5],
                     'total_loss': loss[6]
                     }
        return loss_dict


class Train_Restor(Train):
    def __init__(self, config, strategy, epoch):
        super(Train_Restor, self).__init__(strategy=strategy,
                                           config=config)
        self.base_lr = config.Adjust_lr
        self.batch_size = config.batch_size_Restor
        self.epoch = epoch
        if strategy:
            self.global_batch_size = strategy.num_replicas_in_sync * self.batch_size
        self.optmizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr)
        checkpoint = tf.train.Checkpoint(optimizer=self.optmizer,
                                         Model=self.Restor)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(config.Restor_prefix, 'summary'))
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=config.Restor_prefix, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            checkpoint.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
        optmizer_Decom = tf.keras.optimizers.Adam(learning_rate=self.base_lr)
        checkpoint_Decom = tf.train.Checkpoint(Model=self.Decom,
                                               optmizer=optmizer_Decom)
        ckpt_Decom_manager = tf.train.CheckpointManager(checkpoint_Decom, directory=config.Decom_prefix, max_to_keep=5)
        if ckpt_Decom_manager.latest_checkpoint:
            checkpoint_Decom.restore(ckpt_Decom_manager.latest_checkpoint).expect_partial()

    def lr_schedul(self, epoch):

        if epoch <= 800:
            lr = self.base_lr
        elif epoch <= 1250:
            lr = self.base_lr / 2
        elif epoch <= 1500:
            lr = self.base_lr / 4
        else:
            lr = self.base_lr / 10
        return lr

    def average_loss(self, loss):
        return tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)

    def train(self, inputs, labels):

        with tf.GradientTape(persistent=True) as tape:
            relfer_i, illum_i = self.Decom(inputs, training=False)
            relfer_l, _ = self.Decom(labels, training=False)
            relfer_l = tf.pow(relfer_l, 1.2)
            relfer_restor = self.Restor([relfer_l, illum_i], training=True)
            square_loss = tf.reduce_mean(tf.square(relfer_restor - relfer_l), axis=[1, 2, 3])
            ssim_loss_r = tf.image.ssim(tf.expand_dims(relfer_restor[:,:,:,0],axis=3), tf.expand_dims(relfer_l[:,:,:,0],axis=3), max_val=1.0)
            ssim_loss_g = tf.image.ssim(tf.expand_dims(relfer_restor[:,:,:,1],axis=3), tf.expand_dims(relfer_l[:,:,:,1],axis=3), max_val=1.0)
            ssim_loss_b = tf.image.ssim(tf.expand_dims(relfer_restor[:,:,:,2],axis=3), tf.expand_dims(relfer_l[:,:,:,2],axis=3), max_val=1.0)
            ssim_loss = 1 - (ssim_loss_r+ssim_loss_g+ssim_loss_b)/3.0
            grad_loss = Utils.grad_loss(relfer_restor, relfer_l)
            loss_total = square_loss + ssim_loss + grad_loss
            loss_total = self.average_loss(loss_total)
        gradient = tape.gradient(loss_total, self.Restor.trainable_variables)
        self.optmizer.apply_gradients(zip(gradient, self.Restor.trainable_variables))
        loss_return = [square_loss, ssim_loss, grad_loss]
        loss_return = [tf.reduce_mean(loss_temp) / self.global_batch_size for loss_temp in loss_return]
        loss_return.append(loss_total)

        return loss_return

    @tf.function
    def distributed_step(self, inputs, labels, epoch):
        self.epoch = epoch
        self.optmizer.learning_rate = self.lr_schedul(epoch)
        loss = self.strategy.experimental_run_v2(self.train, args=(inputs, labels))
        loss = [self.strategy2Tensor(lossTemp) for lossTemp in loss]

        loss_dict = {'square_loss': loss[0],
                     'ssim_loss': loss[1],
                     'grad_loss': loss[2],
                     'total_loss': loss[3]
                     }
        return loss_dict


class Train_Adjust(Train):
    def __init__(self, config, strategy, epoch):
        super(Train_Adjust, self).__init__(strategy=strategy,
                                           config=config)
        self.base_lr = config.Adjust_lr
        self.batch_size = config.batch_size_Adjust
        self.epoch = epoch
        if strategy:
            self.global_batch_size = strategy.num_replicas_in_sync * self.batch_size
        self.optmizer = tf.keras.optimizers.Adam(learning_rate=self.base_lr)
        checkpoint = tf.train.Checkpoint(optimizer=self.optmizer,
                                         Model=self.Adjust)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(config.Adjust_prefix, 'summary'))
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=config.Adjust_prefix, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            checkpoint.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

        checkpoint_Decom = tf.train.Checkpoint(Model=self.Decom)
        ckpt_Decom_manager = tf.train.CheckpointManager(checkpoint_Decom, directory=config.Decom_prefix, max_to_keep=5)
        if ckpt_Decom_manager.latest_checkpoint:
            checkpoint_Decom.restore(ckpt_Decom_manager.latest_checkpoint).expect_partial()

    def lr_schedul(self, epoch):
        return self.base_lr

    def average_loss(self, loss):
        return tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)

    def train(self, inputs, labels):
        with tf.GradientTape(persistent=True) as tape:
            _, illum_i = self.Decom(inputs, training=False)
            _, illum_l = self.Decom(labels, training=False)
            k = random.randint(0, 1)
            ratio = tf.reduce_mean(illum_i / (illum_l + 0.0001), axis=[1, 2, 3]) + 0.0001
            ratio = tf.expand_dims(tf.expand_dims(tf.expand_dims(ratio, axis=1), axis=2), axis=3)

            if k:
                ratio = tf.divide(tf.ones_like(illum_i), ratio + 0.0001)
                illum_adjust = self.Adjust([illum_i, ratio], training=True)
                loss_illum = illum_l
            else:
                ratio = tf.ones_like(illum_i) * ratio
                illum_adjust = self.Adjust([illum_l, ratio], training=True)
                loss_illum = illum_i
            square_loss = tf.reduce_mean(tf.square(illum_adjust - loss_illum),axis=[1,2,3])
            grad_loss = Utils.grad_loss_Adjust(illum_adjust, loss_illum)
            loss_total = square_loss + grad_loss
            # print(loss_total)
            loss_total = self.average_loss(loss_total)
        gradient = tape.gradient(loss_total, self.Adjust.trainable_variables)
        self.optmizer.apply_gradients(zip(gradient, self.Adjust.trainable_variables))
        loss_return = [square_loss, grad_loss]
        loss_return = [tf.reduce_mean(loss_temp) / self.global_batch_size for loss_temp in loss_return]
        loss_return.append(loss_total)
        return loss_return

    @tf.function
    def distributed_step(self, inputs, labels, epoch):
        self.epoch=epoch
        self.optmizer.learning_rate = self.lr_schedul(epoch)

        loss = self.strategy.experimental_run_v2(self.train, args=(inputs, labels))
        loss = [self.strategy2Tensor(lossTemp) for lossTemp in loss]
        loss_dict = {'square_loss': loss[0],
                     'grad_loss': loss[1],
                     'total_loss': loss[2]
                     }
        return loss_dict

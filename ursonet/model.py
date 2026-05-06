import os
import re
import datetime
import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM

import utils
from ursonet.data.formatting import mold_image, compose_image_meta
from ursonet.nn.layers import log


from ursonet.nn.backbones_resnet import resnet_graph
from ursonet.nn.backbones_shallow import resnet_shallow_graph
from ursonet.nn.heads_loc import build_loc_graph
from ursonet.nn.heads_ori import build_ori_graph
from ursonet.data.generator import data_generator


class UrsoNet():

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build UrsoNet architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
            这里已经修改了，在开启概率分布的方向输出的时候，bayesian_loc打开和关闭都有对应的输出与损失输出（train下）
        """
        assert mode in ['training', 'inference']

        # Change Keras backend to use f16 precision
        if config.F16:
            K.set_floatx('float16')
            # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
            K.set_epsilon(1e-4)

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=[None, None, config.NR_IMAGE_CHANNELS], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

        tensor_dtype = tf.float32
        if config.F16:
            tensor_dtype = tf.float16

        if mode == "training":

            if config.REGRESS_LOC:
                input_gt_loc = KL.Input(shape=[3], name="input_gt_loc", dtype=tensor_dtype)
            else:
                input_gt_loc = KL.Input(shape=[config.LOC_BINS_PER_DIM ** 3], name="input_gt_loc", dtype=tensor_dtype)

            if config.REGRESS_KEYPOINTS:
                input_gt_k2 = KL.Input(shape=[3], name="input_gt_k2", dtype=tensor_dtype)
                input_gt_k3 = KL.Input(shape=[3], name="input_gt_k3", dtype=tensor_dtype)
            else:
                if config.REGRESS_ORI:
                    if config.ORIENTATION_PARAM == 'quaternion':
                        input_gt_ori = KL.Input(shape=[4], name="input_gt_ori", dtype=tensor_dtype)
                    else:
                        input_gt_ori = KL.Input(shape=[3], name="input_gt_ori", dtype=tensor_dtype)
                else:
                    input_gt_ori = KL.Input(shape=[config.ORI_BINS_PER_DIM ** 3], name="input_gt_ori",
                                            dtype=tensor_dtype)

        # Backbone architecture
        if config.BACKBONE in ['resnet50', 'resnet101']:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)
        else:
            C5 = resnet_shallow_graph(input_image, config.BACKBONE, train_bn=config.TRAIN_BN)

        # Original Resnet uses a 7x7 average pooling:
        # C6 = KL.GlobalAveragePooling2D()(C5)
        # but because we care about resolution, instead we perform here a convolution

        C6 = KL.Conv2D(config.BOTTLENECK_WIDTH, (3, 3), padding='SAME', strides=(2, 2), name='bottleneck_layer')(C5)
        nr_features = int(config.BOTTLENECK_WIDTH * config.IMAGE_SHAPE[0] * config.IMAGE_SHAPE[1] / (64 ** 2))

###修改head 输出后加解析
        loc_pred = build_loc_graph(C6, config, nr_features)
        ori_pred = build_ori_graph(C6, config, nr_features)

        loc_mu = None
        loc_logvar = None
        if config.REGRESS_LOC and (not config.REGRESS_KEYPOINTS) and getattr(config, "BAYESIAN_LOC", False):
            loc_mu, loc_logvar = loc_pred

        if mode == "training":

            # Experimental feature
            if config.LEARNABLE_LOSS_WEIGHTS:
                self.ori_weight = K.variable(-2.3, name='ori_weight')
                self.loc_weight = K.variable(0.0, name='loc_weight')
            else:
                # Default
                self.ori_weight = K.variable(0.0, name='ori_weight')
                self.loc_weight = K.variable(0.0, name='loc_weight')


###修改：训练模式下 loss 构造部分替换
            if config.REGRESS_KEYPOINTS:
                loc_loss = KL.Lambda(lambda x: self.mse_loss_graph(*x), name="loc_loss")([input_gt_loc, loc_pred[0]])
                k2_loss = KL.Lambda(lambda x: self.mse_loss_graph(*x), name="k2_loss")([input_gt_k2, loc_pred[1]])
                k3_loss = KL.Lambda(lambda x: self.mse_loss_graph(*x), name="k3_loss")([input_gt_k3, loc_pred[2]])
            else:
                if config.REGRESS_LOC:
                    if getattr(config, "BAYESIAN_LOC", False):
                        loc_loss = KL.Lambda(
                            lambda x: self.gaussian_nll_loc_loss_graph(*x),
                            name="loc_loss"
                        )([input_gt_loc, loc_mu, loc_logvar])
                    else:
                        loc_loss = KL.Lambda(
                            lambda x: self.rel_loss_graph(*x),
                            name="loc_loss"
                        )([input_gt_loc, loc_pred])
                else:
                    loc_loss = KL.Lambda(
                        lambda x: self.softmax_loss_graph(*x),
                        name="loc_loss"
                    )([input_gt_loc, loc_pred])

                if config.REGRESS_ORI:
                    ori_loss = KL.Lambda(
                        lambda x: self.one_minus_dot_prod_graph(*x),
                        name="ori_loss"
                    )([input_gt_ori, ori_pred])
                else:
                    ori_loss = KL.Lambda(
                        lambda x: self.softmax_loss_graph(*x),
                        name="ori_loss"
                    )([input_gt_ori, ori_pred])

            # Model
            if config.REGRESS_KEYPOINTS:
                inputs = [input_image, input_image_meta, input_gt_loc, input_gt_k2, input_gt_k3]
            else:
                inputs = [input_image, input_image_meta, input_gt_loc, input_gt_ori]

###修改：训练模式 outputs 替换
            if config.REGRESS_KEYPOINTS:
                outputs = [loc_pred[0], loc_pred[1], loc_pred[2], loc_loss, k2_loss, k3_loss]
            else:
                if config.REGRESS_LOC and getattr(config, "BAYESIAN_LOC", False):
                    outputs = [loc_mu, loc_logvar, ori_pred, loc_loss, ori_loss]
                else:
                    outputs = [loc_pred, ori_pred, loc_loss, ori_loss]

            model = KM.Model(inputs, outputs, name='urso_net')

            # Workaround to make weights trainable
            if config.LEARNABLE_LOSS_WEIGHTS:
                model.layers[-1].trainable_weights.extend([self.ori_weight, self.loc_weight])
        else:    # 即model = inference
            ###修改：inference 模式 outputs 替换
            if config.REGRESS_KEYPOINTS:
                model = KM.Model(input_image, [loc_pred[0], loc_pred[1], loc_pred[2]], name='urso_net')
            else:
                if config.REGRESS_LOC and getattr(config, "BAYESIAN_LOC", False):
                    model = KM.Model(input_image, [loc_mu, loc_logvar, ori_pred], name='urso_net')
                else:
                    model = KM.Model(input_image, [loc_pred, ori_pred], name='urso_net')

        # Add multi-GPU support.
        # if config.GPU_COUNT > 1:
        #     from mrcnn.parallel_model import ParallelModel
        #     model = ParallelModel(model, config.GPU_COUNT)

        return model

    ############################################################
    #  Loss Functions
    ############################################################

    def softmax_loss_graph(self, y_gt, y_pred):
        """Loss for classification prediction.
        """
        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss = tf.losses.softmax_cross_entropy(y_gt, y_pred)/tf.exp(self.ori_weight) + self.ori_weight

        # [Modified for TF2] Use tf.compat.v1.losses
        loss = tf.compat.v1.losses.softmax_cross_entropy(y_gt, y_pred)
        return loss

    def arcos_graph(self, y_true, y_pred):
        """Implements rotation error
        y_true and y_pred are typicallly: [N, 4], but could be any shape.
        """
        loss = tf.acos(K.abs(K.sum(y_true * y_pred, axis=-1, keepdims=True)))
        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss = loss/tf.exp(self.ori_weight) + self.ori_weight
        loss_mean = K.mean(loss)

        return loss_mean

    def one_minus_dot_prod_graph(self, y_true, y_pred):
        """Implements 1-dot-product.
        y_true and y_pred are typicallly: [N, 4], but could be any shape.
        """
        loss = 1 - K.abs(K.sum(y_true * y_pred, axis=-1, keepdims=True))
        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss = loss / tf.exp(self.ori_weight) + self.ori_weight
        loss_mean = K.mean(loss)

        return loss_mean

    def mse_loss_graph(self, y_gt, y_pred):
        """Loss for regression prediction.
        e.g.
        pose_gt: [batch, (x,y,z)]
        pose_pred: [batch, (x,y,z)]
        """
        loss = K.square(y_gt - y_pred)

        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss_mse = K.square(y_gt - y_pred)
        # loss = loss_mse/tf.exp(self.loc_weight) + self.loc_weight
        loss_mean = K.mean(loss)

        return loss_mean

###新增： Gaussian NLL 函数
    def gaussian_nll_loc_loss_graph(self, y_gt, y_mu, y_logvar):
        """
        Stable Gaussian negative log-likelihood loss for Bayesian location regression.

        y_gt:     [B, 3] ground-truth location
        y_mu:     [B, 3] predicted mean
        y_logvar: [B, 3] predicted log-variance (raw or semi-raw)
        """
        eps = getattr(self.config, "LOC_EPS", 1e-6)
        # 建议把范围收紧，不要再用 [-10, 10]
        logvar_min = getattr(self.config, "LOC_LOGVAR_MIN", -4.0)       #后面两个参数不一致时听LOC_LOGVAR_MIN的
        logvar_max = getattr(self.config, "LOC_LOGVAR_MAX", 4.0)
        # 1) 用 tanh 做平滑有界映射，比硬 clip 更稳
        # 先把网络输出映射到 [-1, 1]，再缩放到 [logvar_min, logvar_max]
        y_logvar = tf.tanh(y_logvar)
        y_logvar = logvar_min + 0.5 * (y_logvar + 1.0) * (logvar_max - logvar_min)
        # 2) 计算方差和逆方差
        var = tf.exp(y_logvar)
        var = tf.maximum(var, eps)
        inv_var = 1.0 / var
        # 3) 平方误差
        sq_error = tf.square(y_gt - y_mu)
        # 4) 每维 Gaussian NLL（省略常数项 0.5*log(2*pi)）
        per_dim_loss = 0.5 * (inv_var * sq_error + y_logvar)
        # 5) 可选：对单维损失做上限截断，防止极端异常样本把训练带炸
        #    如果你不想加这个保护，可以删除这一行
        max_per_dim = getattr(self.config, "LOC_MAX_PER_DIM_LOSS", 50.0)
        per_dim_loss = tf.minimum(per_dim_loss, max_per_dim)
        # 6) 每样本求和，再对 batch 求平均
        per_sample_loss = tf.reduce_sum(per_dim_loss, axis=-1)
        loss_mean = tf.reduce_mean(per_sample_loss)

        return loss_mean

    def rel_loss_graph(self, y_gt, y_pred):
        """Loss for regression prediction.
        e.g.
        pose_gt: [batch, (x,y,z)]
        pose_pred: [batch, (x,y,z)]
        """

        loss = tf.norm((y_gt - y_pred) / tf.norm(y_gt))

        # Experimental: Adaptive weighting based on Laplace likelihood (Kendall & Cipolla)
        # loss = loss/tf.exp(self.loc_weight) + self.loc_weight
        loss_mean = K.mean(loss)
        return loss_mean

    ############################################################
    #  Weights Loading Functions
    ############################################################

    def get_last_checkpoint(self, model_name):
        """Finds the last checkpoint file of a selected trained model in the
                model directory.
                Returns:
                    log_dir: The directory where events and weights are saved
                    checkpoint_path: the path to the last checkpoint file
                """
        dir_names = next(os.walk(self.model_dir))[1]

        assert model_name in dir_names

        model_path = os.path.join(self.model_dir, model_name)
        checkpoints = next(os.walk(model_path))[2]
        checkpoints = filter(lambda f: f.startswith("weights"), checkpoints)
        checkpoints = sorted(checkpoints)

        if not checkpoints:
            return model_path, None
        checkpoint = os.path.join(model_path, checkpoints[-1])

        return model_path, checkpoint

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("weights"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, weights_in_path, weights_out_path, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to exclude
        """
        import h5py
        # [Modified for TF2] topology is not directly exposed in tf.keras.engine
        # Use h5py directly or tf.keras.saving methods if possible.
        # However, for compatibility with old weights, we try to use the standard load_weights

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')

        # In TF2, we can usually just call load_weights on the model
        # But to support 'exclude', we might need manual loading.
        # For simplicity in TF2, we try the standard method first if no exclude.

        if not exclude:
            self.keras_model.load_weights(weights_in_path, by_name=by_name)
        else:
            # Fallback for exclude logic (simplified for TF2)
            # This is a complex part because TF2 structure differs.
            # We will use the built-in load_weights with by_name=True and skip mismatch
            # But 'exclude' explicitly wants to skip.
            print("Warning: 'exclude' argument in load_weights is partially supported in this TF2 port.")
            self.keras_model.load_weights(weights_in_path, by_name=True, skip_mismatch=True)

        # Update the log directory
        self.set_log_dir(weights_out_path)

    def get_imagenet_weights(self, architecture):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from tensorflow.keras.utils import get_file

        if architecture in ['resnet50', 'resnet101']:
            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                     'releases/download/v0.2/' \
                                     'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        elif architecture == 'resnet18':

            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/qubvel/classification_models/' \
                                     'releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5'
            weights_path = get_file('resnet18_imagenet_1000_no_top.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='318e3ac0cd98d51e917526c9f62f0b50')
        elif architecture == 'resnet34':

            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/qubvel/classification_models/' \
                                     'releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5'
            weights_path = get_file('resnet34_imagenet_1000_no_top.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='8caaa0ad39d927cb8ba5385bf945d582')
        return weights_path

    def get_urso_weights(self, dataset_name):
        """Downloads URSO trained weights from Keras.
        Returns path to weights file.
        """

        assert dataset_name in ['soyuz_hard', 'dragon_hard', 'speed']

        from tensorflow.keras.utils import get_file

        if dataset_name == "soyuz_hard":

            assert self.config.BACKBONE == 'resnet50'
            assert self.config.BOTTLENECK_WIDTH == 128
            assert self.config.ORI_BINS_PER_DIM == 24

            weights_name = 'resnet50_soyuz_hard_128_24.h5'

            TF_WEIGHTS_PATH = 'https://github.com/pedropro/UrsoNet/releases/download/v1.0/' + weights_name

        elif dataset_name == "dragon_hard":

            assert self.config.BACKBONE == 'resnet50'
            assert self.config.BOTTLENECK_WIDTH == 128
            assert self.config.ORI_BINS_PER_DIM == 24

            weights_name = 'resnet50_dragon_hard_128_24.h5'

            TF_WEIGHTS_PATH = 'https://github.com/pedropro/UrsoNet/releases/download/v1.0\/' + weights_name


        elif dataset_name == "speed":

            assert self.config.BACKBONE == 'resnet101'

            if self.config.ORI_BINS_PER_DIM == 32:

                assert self.config.BOTTLENECK_WIDTH == 528

                weights_name = 'resnet101_speed_528_32.h5'

                TF_WEIGHTS_PATH = 'https://github.com/pedropro/UrsoNet/releases/download/v1.0/' + weights_name


            elif self.config.ORI_BINS_PER_DIM == 64:

                assert self.config.BOTTLENECK_WIDTH == 800

                weights_name = 'resnet101_speed_800_64.h5'

                TF_WEIGHTS_PATH = 'https://github.com/pedropro/UrsoNet/releases/download/v1.0/' + weights_name

        weights_path = get_file(weights_name, TF_WEIGHTS_PATH, cache_subdir='models')

        return weights_path

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        if model_path:
            # Directory for training logs
            self.log_dir = os.path.dirname(model_path)
            self.epoch = int(model_path[-6:-3])
        else:
            self.epoch = 0
            now = datetime.datetime.now()
            self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
                self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "weights_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    ############################################################
    #  Weights Loading Functions
    ############################################################

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """

        # Optimizer object
        if self.config.OPTIMIZER == 'SGD':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum,
                                             clipnorm=self.config.GRADIENT_CLIP_NORM)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True,
                                              clipnorm=self.config.GRADIENT_CLIP_NORM)

        # Add Losses
        # [Modified for TF2] Instead of manually clearing _losses and re-adding them via add_loss,
        # which causes 'unhashable type: ListWrapper' error in TF2 graph mode,
        # we will construct the total loss tensor and pass it to model.compile(loss=...)
        # or add it via a Lambda layer during model construction if possible.
        # However, since the model is already built, we will use a workaround:
        # We calculate the losses and add them to the model's loss list safely.

        tensor_dtype = tf.float32
        if self.config.F16:
            tensor_dtype = tf.float16

        # 1. Identify loss layers
        if self.config.REGRESS_KEYPOINTS:
            loss_names = ["loc_loss", "k2_loss", "k3_loss"]
        else:
            loss_names = ["loc_loss", "ori_loss"]

        # 2. Collect all extra losses (custom losses + regularization)
        extra_losses = []

        # Add custom losses from specific layers
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            # Calculate weighted loss
            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
            extra_losses.append(loss)

            # Add metric for monitoring
            # In TF2, we can use add_metric, but in graph mode with disable_eager_execution,
            # we need to be careful.
            self.keras_model.add_metric(loss, name=name, aggregation='mean')

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tensor_dtype)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]

        if reg_losses:
            extra_losses.append(tf.add_n(reg_losses))

        # 3. Compile
        # We pass a dummy loss function because we handle losses manually via add_loss/add_metric
        # or by passing the total loss to compile.

        # In TF2 graph mode, adding a symbolic tensor as loss via add_loss is tricky.
        # The most robust way for this specific legacy architecture is to sum them up
        # and use a dummy target.

        # However, let's try the standard add_loss again but ensuring we don't trigger the ListWrapper bug.
        # The bug is triggered when modifying the graph after build.

        # Workaround: Sum all losses and add as a single loss tensor.
        total_extra_loss = tf.add_n(extra_losses)
        self.keras_model.add_loss(total_extra_loss)

        # Compile with empty loss for outputs, as we added the total loss globally
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs)
        )

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue

            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

#修改为保存最好的和最后的权重文件，且只保存权重，在此代码中还可以训练不同的神经网络部分
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        # All options except 'all' are only currently valid for resnet50 and renet101 models
        layer_regex = {
            # all layers but the backbone
            "heads": r"(ori\_.*)|(loc\_.*)|(fpn\_.*)|(bottleneck_layer)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(loc\_.*)|(ori\_.*)|(fpn\_.*)|(bottleneck_layer)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(loc\_.*)|(ori\_.*)|(fpn\_.*)|(bottleneck_layer)",
            "5+": r"(res5.*)|(bn5.*)|(loc\_.*)|(ori\_.*)|(fpn\_.*)|(bottleneck_layer)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators    在这里进行数据打乱，shuffle
        train_generator = data_generator(
            train_dataset,
            self.config,
            shuffle=True,
            batch_size=self.config.BATCH_SIZE
        )
        val_generator = data_generator(
            val_dataset,
            self.config,
            shuffle=True,
            batch_size=self.config.BATCH_SIZE
        )

        # Callbacks
        class BatchLogger(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.ori_loss_acc = []
                self.loc_loss_acc = []

            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                self.ori_loss_acc.append(logs.get('ori_loss'))
                self.loc_loss_acc.append(logs.get('loc_loss'))

        history_full = BatchLogger()

        best_ckpt_path = os.path.join(self.log_dir, "weights_best.h5")
        last_ckpt_path = os.path.join(self.log_dir, "weights_last.h5")

        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=False
            ),

            # ✅ 保存验证集最优模型
            keras.callbacks.ModelCheckpoint(
                filepath=best_ckpt_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),

            # ✅ 保存最后一轮模型
            # 每个 epoch 结束都会覆盖保存一次，训练结束后保留下来的就是最后一轮权重
            keras.callbacks.ModelCheckpoint(
                filepath=last_ckpt_path,
                save_best_only=False,
                save_weights_only=True,
                verbose=1
            ),

            history_full
        ]

        if self.config.CLR:
            import clr_callback

            clr_triangular = clr_callback.CyclicLR(
                self.config.BASE_LEARNING_RATE,
                self.config.MAX_LEARNING_RATE,
                self.config.CLR_STEP_SIZE,
                mode='triangular'
            )
            callbacks.append(clr_triangular)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Best Checkpoint Path: {}".format(best_ckpt_path))
        log("Last Checkpoint Path: {}".format(last_ckpt_path))

        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # TODO: print('Total FLOPs',get_flops(self))

        # print('Orientation var:', K.eval(K.exp(self.ori_weight)))
        # print('Location var:', K.eval(K.exp(self.loc_weight)))

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        hist = self.keras_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=(workers > 1),
        )

        self.epoch = max(self.epoch, epochs)

        return history_full

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale)
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def _apply_logvar_mapping(self, raw_logvar):
        """
        与训练时 NLL Loss 中的处理保持一致：
        raw -> tanh -> 线性缩放到 [logvar_min, logvar_max]
        """
        logvar_min = getattr(self.config, "LOC_LOGVAR_MIN", -4.0)
        logvar_max = getattr(self.config, "LOC_LOGVAR_MAX", 4.0)
        mapped = np.tanh(raw_logvar)
        mapped = logvar_min + 0.5 * (mapped + 1.0) * (logvar_max - logvar_min)
        return mapped

    # detect函数修改了
    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image.
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, \
            "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)

        results = []

        if self.config.REGRESS_KEYPOINTS:
            loc_pred, k1_pred, k2_pred = self.keras_model.predict(molded_images, verbose=0)

            for i, image in enumerate(images):
                results.append({
                    "loc": np.asarray(loc_pred[i]).reshape(-1),
                    "k1": np.asarray(k1_pred[i]).reshape(-1),
                    "k2": np.asarray(k2_pred[i]).reshape(-1),
                })

        else:
            if self.config.REGRESS_LOC and getattr(self.config, "BAYESIAN_LOC", False):
                loc_mu, loc_logvar, ori_pred = self.keras_model.predict(molded_images, verbose=0)

                for i, image in enumerate(images):
                    logvar_i = np.asarray(loc_logvar[i]).reshape(-1)
                    logvar_i = self._apply_logvar_mapping(logvar_i)         #logvar的clip与tanh映射都要与损失函数一致
                    var_i = np.exp(logvar_i)

                    results.append({
                        "loc": np.asarray(loc_mu[i]).reshape(-1),
                        "loc_logvar": logvar_i,
                        "loc_var": np.asarray(var_i).reshape(-1),
                        "loc_aleatoric_var": np.asarray(var_i).reshape(-1),
                        "ori": np.asarray(ori_pred[i]).reshape(-1),
                    })
            else:
                loc_pred, ori_pred = self.keras_model.predict(molded_images, verbose=0)

                for i, image in enumerate(images):
                    results.append({
                        "loc": np.asarray(loc_pred[i]).reshape(-1),
                        "ori": np.asarray(ori_pred[i]).reshape(-1),
                    })

        return results

    #加入了辅助函数和detect_mc()
    def _numpy_softmax(self, x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _predictive_entropy(self, probs, eps=1e-8):
        # 1. 强制转换为 float64，防止 float16 下 eps 被截断为 0.0
        probs = np.asarray(probs, dtype=np.float64)

        # 2. 安全裁剪
        probs = np.clip(probs, eps, 1.0)

        # 3. 计算熵
        return -np.sum(probs * np.log(probs), axis=-1)

    def detect_mc(self, images, mc_samples=None, verbose=0):
        """"
        在测试/推理阶段做多次开启 Dropout 的随机前向传播，从而估计模型预测的不确定性，尤其是：

        epistemic uncertainty（模型不确定性）
        如果位置分支本身输出 logvar，还能结合得到
        aleatoric uncertainty（数据噪声不确定性）
        total uncertainty（总不确定性）
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, \
            "len(images) must be equal to BATCH_SIZE"

        if mc_samples is None:
            mc_samples = getattr(self.config, "MC_SAMPLES", 30)

        if verbose:
            log("Processing {} images with MC Dropout (T={})".format(len(images), mc_samples))
            for image in images:
                log("image", image)

        molded_images, image_metas, windows = self.mold_inputs(images)

        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)

        results = []

        if self.config.REGRESS_KEYPOINTS:
            raise NotImplementedError("detect_mc() is not implemented for keypoint regression mode.")

        bayesian_loc = self.config.REGRESS_LOC and getattr(self.config, "BAYESIAN_LOC", False)

        import tensorflow.keras.backend as K

        # 构造一个 dropout 开启的推理函数
        mc_func = K.function(
            [self.keras_model.input, K.learning_phase()],
            self.keras_model.outputs
        )

        loc_samples = []
        loc_logvar_samples = []
        ori_samples = []

        for _ in range(mc_samples):
            outputs = mc_func([molded_images, 1])  # 1 表示 training=True / dropout开启

            if bayesian_loc:
                loc_mu_t, loc_logvar_t, ori_t = outputs
                loc_samples.append(np.asarray(loc_mu_t))
                loc_logvar_samples.append(np.asarray(loc_logvar_t))
                ori_samples.append(np.asarray(ori_t))
            else:
                loc_t, ori_t = outputs
                loc_samples.append(np.asarray(loc_t))
                ori_samples.append(np.asarray(ori_t))

        loc_samples = np.stack(loc_samples, axis=0)
        ori_samples = np.stack(ori_samples, axis=0)

        loc_mean = np.mean(loc_samples, axis=0)
        loc_epistemic_var = np.var(loc_samples, axis=0)

        if bayesian_loc:
            loc_logvar_samples = np.stack(loc_logvar_samples, axis=0)

            # 如果你暂时坚持 raw 输出，就不要 clip
            # 如果想数值安全，可只在 exp 前 clip
            safe_logvar_samples = self._apply_logvar_mapping(loc_logvar_samples)       #logvar的clip与tanh映射都要与损失函数一致
            loc_var_samples = np.exp(safe_logvar_samples)
            loc_aleatoric_var = np.mean(loc_var_samples, axis=0)
            loc_total_var = loc_aleatoric_var + loc_epistemic_var
        else:
            loc_aleatoric_var = None
            loc_total_var = loc_epistemic_var

        if self.config.REGRESS_ORI:
            ori_mean = np.mean(ori_samples, axis=0)
            ori_epistemic_var = np.var(ori_samples, axis=0)

            if getattr(self.config, "ORIENTATION_PARAM", None) == 'quaternion':
                norm = np.linalg.norm(ori_mean, axis=-1, keepdims=True) + 1e-8
                ori_mean = ori_mean / norm

            for i, image in enumerate(images):
                result = {
                    "loc": np.asarray(loc_mean[i]).reshape(-1),
                    "loc_mc_samples": np.asarray(loc_samples[:, i, :]),
                    "loc_epistemic_var": np.asarray(loc_epistemic_var[i]).reshape(-1),
                    "ori": np.asarray(ori_mean[i]).reshape(-1),
                    "ori_mc_samples": np.asarray(ori_samples[:, i, :]),
                    "ori_epistemic_var": np.asarray(ori_epistemic_var[i]).reshape(-1),
                }

                if bayesian_loc:
                    result.update({
                        "loc_logvar_mc_samples": np.asarray(loc_logvar_samples[:, i, :]),
                        "loc_aleatoric_var": np.asarray(loc_aleatoric_var[i]).reshape(-1),
                        "loc_total_var": np.asarray(loc_total_var[i]).reshape(-1),
                    })

                results.append(result)

        else:
            ori_prob_samples = self._numpy_softmax(ori_samples, axis=-1)
            # 1.1 求 30 次采样的平均概率分布
            ori_mean_prob = np.mean(ori_prob_samples, axis=0)
            # 1.2 对平均分布算熵(总不确定性)
            predictive_entropy = self._predictive_entropy(ori_mean_prob)
            # 2.1 对 30 个分布分别算熵 (得到 30 个熵值)
            sample_entropies = self._predictive_entropy(ori_prob_samples)
            # 2.2 把这 30 个熵值求平均(数据不确定性)
            expected_entropy = np.mean(sample_entropies, axis=0)
            # 模型不确定性 = 总不确定性 - 数据不确定性
            mutual_info = predictive_entropy - expected_entropy
            ori_prob_var = np.var(ori_prob_samples, axis=0)

            for i, image in enumerate(images):
                result = {
                    "loc": np.asarray(loc_mean[i]).reshape(-1),
                    "loc_mc_samples": np.asarray(loc_samples[:, i, :]),
                    "loc_epistemic_var": np.asarray(loc_epistemic_var[i]).reshape(-1),

                    "ori": np.asarray(ori_mean_prob[i]).reshape(-1),
                    "ori_mean_prob": np.asarray(ori_mean_prob[i]).reshape(-1),
                    "ori_mc_samples": np.asarray(ori_prob_samples[:, i, :]),
                    "ori_entropy": float(predictive_entropy[i]),
                    "ori_expected_entropy": float(expected_entropy[i]),
                    "ori_mutual_info": float(mutual_info[i]),
                    "ori_var": np.asarray(ori_prob_var[i]).reshape(-1),
                    "ori_var_mean": float(np.mean(ori_prob_var[i])),
                }

                if bayesian_loc:
                    result.update({
                        "loc_logvar_mc_samples": np.asarray(loc_logvar_samples[:, i, :]),
                        "loc_aleatoric_var": np.asarray(loc_aleatoric_var[i]).reshape(-1),
                        "loc_total_var": np.asarray(loc_total_var[i]).reshape(-1),
                    })

                results.append(result)

        return results

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

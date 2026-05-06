import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
from ursonet.nn.layers import BatchNorm


def build_ori_graph(feature_map, config, nr_features):
    """Builds the computation graph for orientation estimation on top of the Feature Network."""

    nr_fc_layers = config.NR_DENSE_LAYERS
    assert isinstance(nr_fc_layers, int) and nr_fc_layers >= 0

    x = KL.Reshape((nr_features,), name="ori_reshape")(feature_map)

    for i in range(nr_fc_layers):
        intermediate_fc_layer_name = 'ori_dense_' + str(i)
        x = KL.Dense(config.BRANCH_SIZE, name=intermediate_fc_layer_name)(x)

        if config.TRAIN_BN:
            bn_name = 'ori_bn_' + str(i)
            x = BatchNorm(name=bn_name)(x)

        x = KL.Activation('relu', name='ori_relu_' + str(i))(x)

        if getattr(config, "MC_DROPOUT", False):
            dropout_name = 'ori_dropout_' + str(i)
            dropout_rate = getattr(config, "DROPOUT_RATE", 0.2)
            x = KL.Dropout(rate=dropout_rate, name=dropout_name)(x)

    if config.REGRESS_ORI:
        if config.ORIENTATION_PARAM == 'quaternion':
            q = KL.Dense(4, activation='linear', name="ori_q")(x)
            q = KL.Lambda(lambda t: K.l2_normalize(t, axis=-1), name="ori_q_norm")(q)
        else:
            q = KL.Dense(3, activation='linear', name="ori_final")(x)
    else:
        q = KL.Dense(
            config.ORI_BINS_PER_DIM**3,
            activation=None,
            name="ori_final"
        )(x)

    return q

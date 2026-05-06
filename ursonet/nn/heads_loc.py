import tensorflow.keras.layers as KL
from ursonet.nn.layers import BatchNorm


def build_loc_graph(feature_map, config, nr_features):
    """Builds the computation graph for location estimation on top of the Feature Network."""

    nr_fc_layers = config.NR_DENSE_LAYERS
    assert isinstance(nr_fc_layers, int) and nr_fc_layers >= 0

    x = KL.Reshape((nr_features,), name="loc_reshape")(feature_map)

    for i in range(nr_fc_layers):
        intermediate_fc_layer_name = 'loc_dense_' + str(i)
        x = KL.Dense(config.BRANCH_SIZE, name=intermediate_fc_layer_name)(x)

        if config.TRAIN_BN:
            bn_name = 'loc_bn_' + str(i)
            x = BatchNorm(name=bn_name)(x)

        x = KL.Activation('relu', name='loc_relu_' + str(i))(x)

        if getattr(config, "MC_DROPOUT", False):
            dropout_name = 'loc_dropout_' + str(i)
            dropout_rate = getattr(config, "DROPOUT_RATE", 0.2)
            x = KL.Dropout(rate=dropout_rate, name=dropout_name)(x)

    if config.REGRESS_KEYPOINTS:
        k1 = KL.Dense(3, activation='linear', name="k1_final")(x)
        k2 = KL.Dense(3, activation='linear', name="k2_final")(x)
        k3 = KL.Dense(3, activation='linear', name="k3_final")(x)
        loc = [k1, k2, k3]

    else:
        if config.REGRESS_LOC:
            if getattr(config, "BAYESIAN_LOC", False):
                loc_mu = KL.Dense(3, activation='linear', name="loc_mu")(x)
                loc_logvar = KL.Dense(
                    3,
                    activation='linear',
                    name="loc_logvar",
                    bias_initializer='zeros'
                )(x)
                loc = [loc_mu, loc_logvar]
            else:
                loc = KL.Dense(3, activation='linear', name="loc_final")(x)
        else:
            loc = KL.Dense(
                config.LOC_BINS_PER_DIM**3,
                activation=None,
                name="loc_final"
            )(x)

    return loc

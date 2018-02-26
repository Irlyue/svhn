import tensorflow as tf
import tensorflow.contrib.slim as slim


def input_fn(xin, yin, use_gen=True, batch_size=32, n_epochs=1, shuffle=False):
    def gen():
        yield from zip(xin, yin)

    if use_gen:
        data = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32),
                                              (tf.TensorShape([32, 32, 3]), tf.TensorShape([1])))
    else:
        data = tf.data.Dataset.from_tensor_slices((xin, yin))
    if shuffle:
        data = data.shuffle(1000)
    data = data.repeat(n_epochs).batch(batch_size)
    batch_x, batch_y = data.make_one_shot_iterator().get_next()
    return batch_x, batch_y


def model_fn(features, labels, mode, params=None):
    def batch_norm(x):
        return slim.batch_norm(x,
                               center=True,
                               scale=True,
                               is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                               activation_fn=tf.nn.relu)

    def prep_fn(x, y):
        with tf.name_scope('pre_process'):
            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, tf.int32)
            c = tf.constant(255.0, name='c')
            x = tf.divide(x, c)
            return x, y

    def l2_regularizer():
        reg = 1.0 if params['reg'] is None else params['reg']
        return slim.l2_regularizer(reg)

    regularizer = l2_regularizer()

    # Pre-process
    xin, yin = prep_fn(features, labels)

    params = {} if params is None else params
    #####################################################################################################
    #                                            Model                                                  #
    #####################################################################################################
    with tf.variable_scope('conv1'):
        conv1_1 = slim.conv2d(xin, 32, (3, 3), weights_regularizer=regularizer)
        conv1_2 = slim.conv2d(conv1_1, 32, (3, 3), weights_regularizer=regularizer, activation_fn=None)
        conv1_bn = batch_norm(conv1_2)
    pool1 = slim.max_pool2d(conv1_bn, (2, 2), 2, 'VALID')

    with tf.variable_scope('conv2'):
        conv2_1 = slim.conv2d(pool1, 32, (3, 3), weights_regularizer=regularizer)
        conv2_2 = slim.conv2d(conv2_1, 32, (3, 3), weights_regularizer=regularizer, activation_fn=None)
        conv2_bn = batch_norm(conv2_2)
    pool2 = slim.max_pool2d(conv2_bn, (2, 2), 2, 'VALID')

    with tf.variable_scope('conv3'):
        conv3_1 = slim.conv2d(pool2, 64, (3, 3), weights_regularizer=regularizer)
        conv3_2 = slim.conv2d(conv3_1, 64, (3, 3), weights_regularizer=regularizer, activation_fn=None)
        conv3_bn = batch_norm(conv3_2)
    pool3 = slim.max_pool2d(conv3_bn, (2, 2), 2, 'VALID')

    with tf.variable_scope('conv4'):
        conv4_1 = slim.conv2d(pool3, 64, (3, 3), weights_regularizer=regularizer)
        conv4_2 = slim.conv2d(conv4_1, 64, (3, 3), weights_regularizer=regularizer, activation_fn=None)
        conv4_bn = batch_norm(conv4_2)

    with tf.variable_scope('conv5'):
        conv5 = slim.conv2d(conv4_bn, params['n_classes'], (1, 1), activation_fn=None)

    logits = tf.reduce_mean(conv5, axis=(1, 2), name='logits')
    output = tf.argmax(logits, axis=1, name='predictions')
    output = output[:, tf.newaxis]
    #####################################################################################################
    #                                           Model END                                               #
    #####################################################################################################

    accuracy = tf.metrics.accuracy(labels=yin, predictions=output, name='acc_op')

    #####################################################################################################
    #                                           Loss                                                    #
    #####################################################################################################
    data_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=yin)
    total_loss = data_loss
    if params['reg']:
        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.add(data_loss, reg_loss, name='total_loss')
    #####################################################################################################
    #                                           Loss End                                                #
    #####################################################################################################

    #####################################################################################################
    #                                           Solver                                                  #
    #####################################################################################################
    lr = tf.constant(params['lr'], name='learning_rate')
    tf.summary.scalar('lr', lr)
    solver = tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops + [accuracy[1]]):
        train_op = solver.minimize(total_loss, global_step=tf.train.get_or_create_global_step())
    #####################################################################################################
    #                                           Solver End                                              #
    #####################################################################################################

    if mode == tf.estimator.ModeKeys.PREDICT:
        preds = {
            'class': output
        }
        return tf.estimator.EstimatorSpec(mode, predictions=preds)
    elif mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

"""
Usages:
    python train.py --lr=1e-3 --delete=True --n_epochs=8
"""
import inputs
import argparse
import my_utils as mu
import tensorflow as tf

from model import model_fn, input_fn

logger = mu.get_default_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--delete', default=False, type=lambda x: x == 'True',
                    help='Delete checkpoints and train from scratch')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--n_epochs', default=8, type=int, help='Number of epochs')
parser.add_argument('--reg', default=None, type=float, help='Regularization strength')


def get_training_config():
    pass


def main():
    def train_input_fn():
        return input_fn(*data['train'],  batch_size=config['batch_size'], n_epochs=config['n_epochs'], shuffle=True)

    # load the data
    data = inputs.load_data(config['n_examples_for_train'], config['n_examples_for_cv'])
    logger.info('\n%s\n', data)

    if config['delete']:
        logger.info('Deleting existing checkpoint files...')
        mu.delete_if_exists(config['model_dir'])
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=config,
                                       model_dir=config['model_dir'])
    estimator.train(train_input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parser.parse_args()
    config = mu.load_config(path=None, **FLAGS.__dict__)
    logger.info('\n%s\n', mu.json_out(config))
    main()

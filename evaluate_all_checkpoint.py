import inputs
import argparse
import my_utils as mu
import tensorflow as tf

from model import model_fn, input_fn

logger = mu.get_default_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='cv', type=str, help='Dataset for evaluation')


def main():
    def eval_input_fn():
        return input_fn(*data[config['data']], batch_size=config['batch_size'], shuffle=False)
    data = inputs.load_data(config['n_examples_for_cv'])
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=config,
                                       model_dir=config['model_dir'])
    for ckpt in tf.train.get_checkpoint_state(config['model_dir']).all_model_checkpoint_paths:
        with mu.Timer() as timer:
            result = estimator.evaluate(eval_input_fn, checkpoint_path=ckpt)
        result['data'] = config['data']
        logger.info('Done in %.fs', timer.eclipsed)
        logger.info('\n%s\n%s%s%s\n', data, '*'*10, result, '*'*10)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parser.parse_args()
    config = mu.load_config(path=None, **FLAGS.__dict__)
    logger.info('\n%s\n', mu.json_out(config))
    main()

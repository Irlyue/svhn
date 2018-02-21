import inputs
import argparse
import my_utils as mu
import tensorflow as tf

from model import model_fn, input_fn

logger = mu.get_default_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='cv', type=str, help='Dataset for evaluation')


def ckpt_gen():
    def get_all_checkpoints():
        try:
            return tf.train.get_checkpoint_state(config['model_dir']).all_model_checkpoint_paths
        except:
            return []
    ckpts = set()
    while True:
        current_ckpts = set(get_all_checkpoints())
        if ckpts == current_ckpts:
            # wait for some time
            logger.info('Waiting for %d seconds...', config['checkpoint_every_secs'])
            mu.sleep_secs(config['checkpoint_every_secs'])

        new_ckpts = current_ckpts - ckpts
        ckpts = current_ckpts
        yield from new_ckpts


def main():
    def return_bigger(best_result, cur_result):
        return best_result if best_result['accuracy'] > cur_result['accuracy'] else cur_result

    def eval_input_fn():
        return input_fn(*data[config['data']], batch_size=config['batch_size'], shuffle=False)

    data = inputs.load_data(config['n_examples_for_train'], config['n_examples_for_cv'])
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=config,
                                       model_dir=config['model_dir'])
    best = {'accuracy': 0.0}
    try:
        for ckpt in ckpt_gen():
            with mu.Timer() as timer:
                result = estimator.evaluate(eval_input_fn, checkpoint_path=ckpt)
            best = return_bigger(best, result)
            result['data'] = config['data']
            logger.info('Done in %.fs', timer.eclipsed)
            logger.info('\n%s\n%s%s%s\n', data, '*'*10, result, '*'*10)
    except KeyboardInterrupt:
        logger.info('Best result: \n%s\n', best)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parser.parse_args()
    config = mu.load_config(path=None, **FLAGS.__dict__)
    logger.info('\n%s\n', mu.json_out(config))
    main()

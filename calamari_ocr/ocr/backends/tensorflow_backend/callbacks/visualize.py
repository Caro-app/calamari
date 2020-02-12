from calamari_ocr.utils import RunningStatistics
from ..util import sparse_to_lists
import numpy as np
import tensorflow as tf
import time
keras = tf.keras


class VisCallback(keras.callbacks.Callback):
    def __init__(self, 
                 training_callback, 
                 codec, 
                 data_gen, 
                 val_data_gen, 
                 predict_func, 
                 checkpoint_params, 
                 steps_per_epoch, 
                 text_post_proc,
                 train_summary_writer,
                 test_summary_writer):
        self.training_callback = training_callback
        self.codec = codec
        self.data_gen = data_gen
        self.predict_func = predict_func
        self.checkpoint_params = checkpoint_params
        self.steps_per_epoch = steps_per_epoch
        self.text_post_proc = text_post_proc
        self.val_data_gen = val_data_gen

        self.train_summary_writer = train_summary_writer
        self.test_summary_writer = test_summary_writer

        self.loss_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.loss_stats)
        self.ler_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.ler_stats)
        self.dt_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.dt_stats)

        display = checkpoint_params.display
        self.display_epochs = display <= 1
        if display <= 0:
            display = 0                                       # do not display anything
        elif self.display_epochs:
            display = max(1, int(display * steps_per_epoch))  # relative to epochs
        else:
            display = max(1, int(display))                    # iterations

        self.display = display
        self.iter_start_time = time.time()
        self.train_start_time = time.time()

    def on_train_begin(self, logs):
        self.iter_start_time = time.time()
        self.train_start_time = time.time()

    def on_train_end(self, logs):
        self.training_callback.training_finished(time.time() - self.train_start_time, self.checkpoint_params.iter)

    def on_batch_end(self, batch, logs):
        dt = time.time() - self.iter_start_time
        self.iter_start_time = time.time()
        self.dt_stats.push(dt)
        self.loss_stats.push(logs['loss'])
        self.checkpoint_params.iter += 1

        if self.display > 0 and self.checkpoint_params.iter % self.display == 0:
            # apply postprocessing to display the true output
            cer, target, decoded = self._generate(1)
            self.ler_stats.push(cer)
            pred_sentence = self.text_post_proc.apply("".join(self.codec.decode(decoded[0])))
            gt_sentence = self.text_post_proc.apply("".join(self.codec.decode(target[0])))

            self.training_callback.display(self.ler_stats.mean(), self.loss_stats.mean(), self.dt_stats.mean(),
                                           self.checkpoint_params.iter, self.steps_per_epoch, self.display_epochs,
                                           pred_sentence, gt_sentence
                                           )
            with self.train_summary_writer.as_default():
                tf.summary.scalar('iter', self.checkpoint_params.iter, step=self.checkpoint_params.iter)            
                tf.summary.scalar('dt', self.dt_stats.mean(), step=self.checkpoint_params.iter)
                tf.summary.scalar('train_batch_loss', data=self.loss_stats.mean(), step=self.checkpoint_params.iter)
                tf.summary.scalar('train_batch_cer', data=self.ler_stats.mean(), step=self.checkpoint_params.iter)

    def on_epoch_end(self, epoch, logs):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('train_epoch_loss', data=logs['loss'], step=epoch)
        
        if self.val_data_gen is not None:
            if self.display > 0:
                val_cer, _, _ = self._generate_val(20) # 20 batches for generating validation metrics
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('val_epoch_loss', data=logs['val_loss'], step=epoch)
                    tf.summary.scalar('val_cer', data=val_cer, step=epoch)

    def _generate(self, count):
        it = iter(self.data_gen)
        cer, target, decoded = zip(*[self.predict_func(next(it)) for _ in range(count)])
        return np.mean(cer), sum(map(sparse_to_lists, target), []), sum(map(sparse_to_lists, decoded), [])

    def _generate_val(self, count):
        if self.val_data_gen is None:
            pass
        else:
            it = iter(self.val_data_gen)
            cer, target, decoded = zip(*[self.predict_func(next(it)) for _ in range(count)])
            return np.mean(cer), sum(map(sparse_to_lists, target), []), sum(map(sparse_to_lists, decoded), [])

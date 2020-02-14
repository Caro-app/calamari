import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from calamari_ocr.utils import RunningStatistics
from ..util import sparse_to_lists
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops
keras = tf.keras



class CustomTensorBoard(TensorBoard):


    def __init__(self, 
                 training_callback, 
                 codec, 
                 train_data_gen, 
                 validation_data_gen, 
                 predict_func, 
                 checkpoint_params, 
                 steps_per_epoch, 
                 text_post_proc,
                 log_dir='logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 update_freq='epoch',
                 embeddings_freq=0,
                 embeddings_metadata=None,
                 **kwargs):
        
        super().__init__(
                       log_dir=log_dir,
                       histogram_freq=histogram_freq,
                       write_graph=write_graph,
                       write_images=write_images,
                       update_freq=update_freq,
                       embeddings_freq=embeddings_freq,
                       embeddings_metadata=embeddings_metadata,
                       **kwargs)
        self.training_callback = training_callback
        self.codec = codec
        self.train_data_gen = train_data_gen
        self.validation_data_gen = validation_data_gen
        self.predict_func = predict_func
        self.checkpoint_params = checkpoint_params
        self.steps_per_epoch = steps_per_epoch
        self.text_post_proc = text_post_proc

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
        super().on_train_begin(logs)
        
        self.iter_start_time = time.time()
        self.train_start_time = time.time()

    def on_train_end(self, logs):
        super().on_train_end(logs)

        self.training_callback.training_finished(time.time() - self.train_start_time, self.checkpoint_params.iter)

    def on_train_batch_end(self, batch, logs=None):        
        assert self._total_batches_seen == self.checkpoint_params.iter

        self.checkpoint_params.iter += 1

        if self.update_freq == 'epoch' and self._profile_batch is None:
            return

        dt = time.time() - self.iter_start_time
        self.iter_start_time = time.time()
        self.dt_stats.push(dt)
        self.loss_stats.push(logs['loss'])

        logs = logs or {}
        if self.update_freq != 'epoch' and self.display > 0 and self.checkpoint_params.iter % self.display == 0:            
            cer, target, decoded = self._generate(self.train_data_gen, 1)
            self.ler_stats.push(cer)
            pred_sentence = self.text_post_proc.apply("".join(self.codec.decode(decoded[0])))
            gt_sentence = self.text_post_proc.apply("".join(self.codec.decode(target[0])))
            self._log_metrics({"loss": self.loss_stats.mean()}, prefix='batch_', step=self.checkpoint_params.iter)
            self._log_metrics({"cer": self.ler_stats.mean()}, prefix='batch_', step=self.checkpoint_params.iter)

            self.training_callback.display(self.ler_stats.mean(), self.loss_stats.mean(), self.dt_stats.mean(),
                                        self.checkpoint_params.iter, self.steps_per_epoch, self.display_epochs,
                                        pred_sentence, gt_sentence
                                        )
        self._total_batches_seen += 1

        if context.executing_eagerly():
            if self._is_tracing:
                self._log_trace()
            elif (not self._is_tracing and
                    math_ops.equal(self.checkpoint_params.iter, self._profile_batch - 1)):
                self._enable_trace()


    def on_epoch_end(self, epoch, logs=None):
        self._log_metrics(logs, prefix='epoch_', step=epoch)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)

        if self.update_freq == 'epoch':
            train_cer, _, _ = self._generate(self.train_data_gen, 20) # 20 batches for generating training metrics
            self._log_metrics({"cer": train_cer}, prefix='train_', step=epoch)

        if self.validation_data_gen is not None:
            val_cer, _, _ = self._generate(self.validation_data_gen, 20) # 20 batches for generating validation metrics
            self._log_metrics({"cer": val_cer}, prefix='validation_', step=epoch)

    def _generate(self, data_gen, count):
        if data_gen is None:
            pass
        else:
            it = iter(data_gen)
            cer, target, decoded = zip(*[self.predict_func(next(it)) for _ in range(count)])
            return np.mean(cer), sum(map(sparse_to_lists, target), []), sum(map(sparse_to_lists, decoded), [])

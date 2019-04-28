# coding: utf-8


import os
import csv
import json
import numpy as np
import collections
import itertools
import functools
import inspect
import datetime
from threading import Thread

import tensorflow as tf

from .custom_callbacks import \
        BatchLearningRateScheduler, \
        NBatchLogger

from .custom_loss import \
        masked_softmax_cross_entropy_with_logit, \
        masked_sparse_categorical_accuracy

from .dnn import get_transformer_custom_objects
from .transformer import TransformerWrapper, Config


def get_custom_objects():
    obj = {
            'masked_softmax_cross_entropy_with_logit':masked_softmax_cross_entropy_with_logit,
            'masked_sparse_categorical_accuracy':masked_sparse_categorical_accuracy,
        }
    obj.update(get_transformer_custom_objects())
    return obj


def simple_json_serialize(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    #raise TypeError(repr(obj) + " is not JSON serializable")
    return str(obj)


class FitEnvironment(object):
    
    MODEL_NAME_TEMPLATE = 'model.{epoch:02d}.hdf5'

    def get_default_train_callbacks(self):
        callbacks = []
        
        def lr_fun(step_num, d_model, warm_step):
            return min(step_num/(warm_step**1.5), 1.0/(step_num+1e-8)**0.5)/(d_model**0.5)
        c = BatchLearningRateScheduler(
                functools.partial(lr_fun, d_model=self.model_config.embed_dim, warm_step=self.warm_step),
                verbose=0
            )
        callbacks.append(c)
        
        c = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, self.MODEL_NAME_TEMPLATE),
                monitor='loss',
                save_best_only=False,
                save_weights_only=False,
                mode='auto',
                period=5,
                verbose=1,
            )
        callbacks.append(c)
        
        c = tf.keras.callbacks.TensorBoard(
                log_dir=self.output_dir,
                update_freq=100,
            )
        callbacks.append(c)

        c = NBatchLogger(display=100, custom_metrics=['lr'])
        callbacks.append(c)        
        
        return callbacks
    
    @staticmethod
    def get_default_model_compiler():
        def _fun(model):
            model.compile(
                    # lr is a dummy and is set by Scheduler 
                    optimizer=tf.keras.optimizers.Adam(lr=0.0, beta_1=0.9, beta_2=0.98, epsilon=10e-9),
                    loss=masked_softmax_cross_entropy_with_logit,
                    metrics=[masked_sparse_categorical_accuracy],
                )
        return _fun 
    
    def __init__(
            self,
            model_config=None,
            use_tpu=None,
            input_len=None,
            work_dir='model/transformer',
            output_dir='model/transfromer',
            data_path=[],
            valid_path=None,
            show_model_summary=True,
            batch_size=8, # num of device 
            num_epoch=20,
            warm_step=4000,
            train_callbacks=None, # when no callbacks you want, set []
            model_compiler=None,
            resume_model_path=None,
            resume_initial_epoch=None,
            _train_examples_truncate_num=None,
        ):
        # set all arguments to self attributes
        self.arg_names = inspect.getfullargspec(self.__init__).args.copy()
        self.arg_names.remove('self')
        local_values = locals()
        for n in self.arg_names:
            setattr(self, n, local_values[n])
        
        if self.resume_model_path is not None and self.resume_initial_epoch is None:
            raise ValueError('specify initial_epoch when using resume_model')
        self.initial_epoch = self.resume_initial_epoch or 0
        
        if self.model_config is None:
            self.model_config = Config()
        
        if input_len is not None:
            self.model_config.input_len = self.input_len
        if use_tpu is not None:
            self.model_config.use_tpu = self.use_tpu
        self.model_config._check_condition_for_tpu()
        
        if self.model_compiler is None:
            self.model_compiler = self.get_default_model_compiler()
        
        if self.train_callbacks is None:
            self.train_callbacks = self.get_default_train_callbacks()
        
    def asdict(self):
        return {n:getattr(self, n) for n in self.arg_names}
    
    def __repr__(self):
        return json.dumps(self.asdict(), ensure_ascii=False, indent=2, default=simple_json_serialize)
    
    def _make_validation_data(self, wrapper, path):
        """
        Returns static data list made by generator        
        """
        
        generator_maker = FitGeneratorMaker(
                wrapper=wrapper,
                corpus_path=path,
            )
        generator_maker.set_params(
                input_len=self.input_len,
                batch_size=self.batch_size,
                num_epoch=1,
                file_prefix=os.path.join(self.work_dir, 'valid.tfrecord')
            )
        
        # After to_tpu, TPU configuration prevents graphs 
        # from running on non distributedTPU setting in the main thread.
        def subthread(iterator, data):
            for batch in iterator:
                data.append(batch)
        
        valid_data = []
        thread = Thread(
                target=subthread, 
                args=[generator_maker.make(repeat=False), valid_data]
            )
        thread.start()
        thread.join()
        len_valid_data = len(valid_data)
        
        def gen():
            i = 0
            while True:
                yield valid_data[i]
                i += 1
                if i >= len_valid_data:
                    i = 0
        
        return gen, len_valid_data
    
    def run(self):
        
        time_start = datetime.datetime.now()
        
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging.info('training starting with env=%s'%(self))
    
        tf.gfile.MakeDirs(self.work_dir)
        tf.gfile.MakeDirs(self.output_dir)
        
        # Model construction
        wrapper = TransformerWrapper(
                config=self.model_config, 
                model_path=self.resume_model_path,
                weights_only=False,
                custom_objects=get_custom_objects(),
            )
        
        if wrapper.model.optimizer is None:
            self.model_compiler(wrapper.model)
            
        if self.show_model_summary:
            wrapper.model.summary(line_length=150)

        # Validation data preparation
        if self.valid_path:
            tf.logging.info('Making validation data...')
            valid_gen, valid_len = self._make_validation_data(wrapper, self.valid_path)
        else:
            valid_gen = valid_len = None            
        
        # Training data preparation
        tf.logging.info('Making training data...')
        generator_maker = FitGeneratorMaker(
                wrapper=wrapper,
                corpus_path=self.data_path,
                truncate_num=self._train_examples_truncate_num,
            )
        generator_maker.set_params(
                input_len=self.input_len,
                batch_size=self.batch_size,
                num_epoch=self.num_epoch,
                file_prefix=os.path.join(self.work_dir, 'train.tfrecord')
            )
        generator_maker.summarize()
        
        time_prepared = datetime.datetime.now()
        tf.logging.info('Elapsed_sec_before_fit_generator=%.3f'%(
                (time_prepared-time_start).total_seconds()
            ))
        
        # Fitting starts     
        with tf.keras.utils.custom_object_scope(get_custom_objects()):
            history = wrapper.model.fit_generator(
                    generator_maker.make(),
                    validation_data=valid_gen(),
                    validation_steps=valid_len,
                    steps_per_epoch=generator_maker.estimated_steps_per_epoch,
                    epochs=self.initial_epoch + self.num_epoch,
                    callbacks=self.train_callbacks,
                    initial_epoch=self.initial_epoch,
                    verbose=0,
                )
            
            # Save the last model
            last_epoch = self.initial_epoch + self.num_epoch
            last_model_path = os.path.join(
                    self.output_dir, 
                    self.MODEL_NAME_TEMPLATE.format(epoch=last_epoch)
                )
            wrapper.model.save(last_model_path, overwrite=True)
            tf.logging.info('Epoch %05d: saving model to %s'%(last_epoch, last_model_path))
        
        history_file_path = os.path.join(self.output_dir, 'epoch_history.json')
        with open(history_file_path, 'w') as f:
            json.dump(history.history, f, default=simple_json_serialize)
        # Note: Default history file contains epoch-level logs
        # Copy batch-level logs from stdout if needed
        
        time_end = datetime.datetime.now()
        tf.logging.info('Elapsed_time=%.3f'%((time_end-time_start).total_seconds()))
        
        return wrapper


class Example(object):
    
    def __init__(self, guid, texts, wrapper):
        
        self.guid = guid
        self.text = {}
        self.token_ids = {}
        self.pos_ids = {}
        self.input_len = {}
        
        for key, text in texts.items():
            feat = wrapper.get_input_features(text, end_name=key)
            self.text[key] = text
            self.token_ids[key] = feat['Token-Ids']
            self.pos_ids[key] = feat['Pos-Ids']
            self.input_len[key] = len(feat['Token-Ids'])
        
        self.target = self.token_ids['tar'][1:]
        self.target.append(wrapper.pad_id['tar'])

            
class FitGeneratorMaker(object):
    
    @staticmethod
    def load_examples(corpus_path, wrapper, truncate_num):
        
        if isinstance(corpus_path, str):
            corpus_path = [corpus_path]
        
        examples = []
        for fid, fpath in enumerate(corpus_path):
            with open(fpath, 'r') as f:
                for i, row in enumerate(csv.reader(f)):
                    if truncate_num is not None and len(examples) >= truncate_num:
                        break
                    examples.append(Example(
                            guid='train-%d-%d'%(fid, i),
                            texts={'src':row[0], 'tar':row[1]},
                            wrapper=wrapper
                        ))
        
        return examples
    
    @staticmethod
    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    
    @staticmethod
    def create_float_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    
    def __init__(self, wrapper, corpus_path, truncate_num=None):
        # All examples are placed on the memory
        # In each epoch, they are randomly packed into batches and outputed as tfrecord
        tf.logging.info("Loading examples")
        self.examples = self.load_examples(corpus_path, wrapper, truncate_num)
        self.sum_src_tokens =sum(_.input_len['src'] for _ in self.examples)
        self.sum_tar_tokens =sum(_.input_len['tar'] for _ in self.examples)
    
    def set_params(self, input_len, batch_size, num_epoch, file_prefix, record_per_file=10000):
        self.input_len = input_len
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.file_prefix = file_prefix
        self.record_per_file = record_per_file
        
        src_steps_per_epoch = self.sum_src_tokens/(self.input_len[0]*self.batch_size)
        tar_steps_per_epoch = self.sum_tar_tokens/(self.input_len[1]*self.batch_size)
        self.packing_factor = 0.95
        
        self.estimated_steps_per_epoch = \
                int(max(src_steps_per_epoch, tar_steps_per_epoch)/self.packing_factor)
        
        self.feature_spec = feature_spec = collections.OrderedDict()
        feature_spec['Encoder-Token-Ids'] = tf.FixedLenFeature([self.input_len[0]], tf.int64)
        feature_spec['Encoder-Pos-Ids'] = tf.FixedLenFeature([self.input_len[0]], tf.int64)
        feature_spec['Encoder-Group-Ids'] = tf.FixedLenFeature([self.input_len[0]], tf.int64)
        feature_spec['Decoder-Token-Ids'] = tf.FixedLenFeature([self.input_len[1]], tf.int64)
        feature_spec['Decoder-Pos-Ids'] = tf.FixedLenFeature([self.input_len[1]], tf.int64)
        feature_spec['Decoder-Group-Ids'] = tf.FixedLenFeature([self.input_len[1]], tf.int64)
        feature_spec['Target'] = tf.FixedLenFeature([self.input_len[1]], tf.int64)
    
    def summarize(self):
        tf.logging.info("***** FitGenerator Summary *****")
        tf.logging.info("Num_all_examples=%d"%(len(self.examples)))
        tf.logging.info("sum_src_tokens=%d"%(self.sum_src_tokens))
        tf.logging.info("sum_tar_tokens=%d"%(self.sum_tar_tokens))
        tf.logging.info("Batch_size=%d"%self.batch_size)
        tf.logging.info("encoder_input_len=%d"%(self.input_len[0]))
        tf.logging.info("decoder_input_len=%d"%(self.input_len[1]))
        tf.logging.info('packing_factor=%.3f'%(self.packing_factor))
        tf.logging.info('approximate_steps_per_epoch=%d'%(self.estimated_steps_per_epoch))
        tf.logging.info('num_epoch=%d'%(self.num_epoch))
        tf.logging.info('total_step=%d'%(self.num_epoch*self.estimated_steps_per_epoch))
    
    def make_record(self, verbose=False):
        # shuffle examples and packs them in batchs for 1 epoch
        # return path where data is output as tf records
           
        def log_record(eample_stack, feats):
            tf.logging.info("*** Example ***")
            tf.logging.info("guids:%s"%([_.guid for _ in eample_stack]))
            tf.logging.info("src_texts:%s"%([_.text['src'] for _ in eample_stack]))
            tf.logging.info("tar_texts:%s"%([_.text['tar'] for _ in eample_stack]))
            for k, v in feats.items():
                tf.logging.info("%s:%s" %(k, " ".join([str(x) for x in v])))
        
        def fit_len(input_len, list_list):
            flat_list = list(itertools.chain.from_iterable(list_list))
            l = len(flat_list)
            if l == input_len:
                return flat_list
            if l > input_len:
                return flat_list[:input_len]
            return flat_list + [-1]*(input_len - l)
        
        status = {
                'writer': None,
                'records_in_file': 0,
                'len_all_record': 0,
                'record_files': [],
            }
        
        def _write_record(example_stack):
            if status['records_in_file'] % self.record_per_file == 0:
                if status['writer'] is not None:
                    status['writer'].close()
                status['records_in_file'] = 0
                output_file = self.file_prefix + '.' + str(len(status['record_files']))
                status['writer'] = tf.python_io.TFRecordWriter(output_file)
                status['record_files'].append(output_file) 
                if verbose:
                    tf.logging.info("New file created %s" %(output_file))
            
            feats = {}
            feats['Target'] =  fit_len(self.input_len[1], [_.target for _ in example_stack])
            for key, input_len, prefix in zip(('src', 'tar'), self.input_len, ('Encoder', 'Decoder')):
                feats[prefix+'-Token-Ids'] = fit_len(input_len, [_.token_ids[key] for _ in example_stack])
                feats[prefix+'-Pos-Ids'] = fit_len(input_len, [_.pos_ids[key] for _ in example_stack])
                feats[prefix+'-Group-Ids'] = fit_len(input_len, [[g]*_.input_len[key] for g, _ in enumerate(example_stack)])
            
            features = collections.OrderedDict()
            for k in self.feature_spec.keys():
                features[k] = self.create_int_feature(feats[k])
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            status['writer'].write(tf_example.SerializeToString())
            
            status['len_all_record'] += 1
            status['records_in_file'] += 1
            
            if verbose and status['len_all_record'] <= 3:
                log_record(example_stack, feats)
        
        example_stack = []
        cur_len = (0, 0)
        
        idx = list(range(len(self.examples)))
        np.random.shuffle(idx)
        
        for i in idx:
            example = self.examples[i]
            ex_len = (example.input_len['src'], example.input_len['tar'])
            
            if any([c>l for c,l in zip(cur_len, self.input_len)]):
                continue
            
            if any([c+e>l for c,e,l in zip(cur_len, ex_len, self.input_len)]):
                _write_record(example_stack)
                example_stack.clear()
                cur_len = (0, 0)
            else:
                example_stack.append(example)
                cur_len = tuple(c+e for c,e in zip(cur_len, ex_len))
        
        if len(example_stack) > 0:
            _write_record(example_stack)
        
        while status['len_all_record'] % self.batch_size != 0:
            _write_record([])
        
        if status['writer'] is not None:
            status['writer'].close()
        
        return status['record_files']
        
    def make(self, repeat=True):
           
        def decode_record(record):
            example = tf.parse_single_example(record, self.feature_spec)
           
            for key in example.keys():
                v = example[key]
                if v.dtype == tf.int64:
                    example[key] = tf.cast(v, tf.int32)
            
            target = example.pop('Target')[:, None]
            
            return example, target
        
        # iterate unless a fit function stops calling
        is_first=True
        try:
            with tf.Session() as session:
                while True:
                    if (not repeat) and (not is_first):
                        break
                    tf.logging.info('Preparing new records')
                    record_files = self.make_record(is_first)
                    is_first=False
                
                    d = tf.data.Dataset.list_files(record_files)
                    d = d.flat_map(lambda _: tf.data.TFRecordDataset(_))
                    d = d.map(decode_record)
                    d = d.batch(
                            batch_size=self.batch_size,
                            drop_remainder=True,
                        )
                    try:   
                        next_batch = d.make_one_shot_iterator().get_next()
                        while True:
                            yield session.run(next_batch)
                    except tf.errors.OutOfRangeError:
                        # when prepared data runs out
                        # pass and make next records
                        pass
        except IndexError:
            # To avoid unclear error "IndexError: pop from empty list"
            pass



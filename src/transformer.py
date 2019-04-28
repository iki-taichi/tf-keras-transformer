# coding: utf-8


import os
import json
import inspect
import numpy as np
import itertools

from .dnn import get_transformer_model
from .tokenization import sp_uncase_en_ja_40000


class Config(object):
    
    def __init__(
            self,
            use_tpu=False,
            tpu_grpc_url='COLAB',
            src_tokenizer='sp_uncase_en_ja_40000',
            tar_tokenizer='sp_uncase_en_ja_40000',
            use_same_embed=True,
            block_num=(6, 6),
            embed_dim=768,
            hidden_dim=3072,
            head_num=12,
            attention_activation='relu',
            feed_forward_activation='gelu',
            dropout_rate=0.10,
            input_len=(None, None), # None is accepbale when not using TPU
            token_num=(None, None), # automatically decided
        ):
        # set all named-arguments to self attributes
        self.arg_names = inspect.getfullargspec(self.__init__).args.copy()
        self.arg_names.remove('self')
        local_values = locals()
        for n in self.arg_names:
            setattr(self, n, local_values[n])
        
        self._check_condition_for_tpu()
        
        if self.tpu_grpc_url == 'COLAB':
            self.tpu_grpc_url = "grpc://"+os.environ.get("COLAB_TPU_ADDR", '')
    
    def _check_condition_for_tpu(self):
        if self.use_tpu:
            if self.input_len is None or any([_ is None for _ in self.input_len]):
                raise ValueError('tpu model requires fixed input len')
    
    def asdict(self):
        return {n:getattr(self, n) for n in self.arg_names}
    
    def __repr__(self):
        return json.dumps(self.asdict(), ensure_ascii=False, indent=2)


class TransformerWrapper(object):
    
    TOKEN_PAD = '<pad>'
    TOKEN_START = '<^>'
    TOKEN_END = '<$>'
    
    @classmethod
    def get_default_tokenizer(cls, name):
        if name == 'sp_uncase_en_ja_40000':
            return sp_uncase_en_ja_40000.FullTokenizer(
                    mapping={
                            'unused_0':cls.TOKEN_START,
                            'unused_1':cls.TOKEN_END,
                        }
                )
        
        raise ValueError('unknown tokenizer %s'%(name))
    
    @staticmethod
    def truncate_seq(seq, max_seq):
        if len(seq) > max_seq:
            print('the length of text excesses max len: %s'%(seq))
        while len(seq) > max_seq:
            seq.pop()
   
    def __init__(self, model_path=None, config=None, weights_only=True, custom_objects=None):
        
        self.config = config or Config(input_len=(None, None))
        
        if isinstance(self.config.src_tokenizer, str):
            self.src_tokenizer = self.get_default_tokenizer(self.config.src_tokenizer)
        else:
            self.src_tokenizer = self.config.src_tokenizer
            
        if isinstance(self.config.tar_tokenizer, str):
            self.tar_tokenizer = self.get_default_tokenizer(self.config.tar_tokenizer)
        else:
            self.tar_tokenizer = self.config.tar_tokenizer
        
        self.pad_id = {
                'src':self.src_tokenizer.vocab[self.TOKEN_PAD], 
                'tar':self.tar_tokenizer.vocab[self.TOKEN_PAD],
            }
        
        self.config.token_num = (
                len(self.src_tokenizer.vocab), 
                len(self.tar_tokenizer.vocab),
            )
        
        self.model = get_transformer_model(self.config, model_path, weights_only, custom_objects)
    
    def get_input_features(self, text, group_id=0, end_name='src'):
        
        if end_name=='src':
            tokenizer = self.src_tokenizer
            max_len = self.config.input_len[0]
        elif end_name=='tar':
            tokenizer = self.tar_tokenizer
            max_len = self.config.input_len[1]
        else:
            raise ValueError('end_name should be src or tar')
        
        tokens = tokenizer.tokenize(text)
        if max_len is not None:
            self.truncate_seq(tokens, max_len-2)
        token_ids = tokenizer.convert_tokens_to_ids(
                itertools.chain([self.TOKEN_START], tokens, [self.TOKEN_END])
            )
        
        seq_len = len(token_ids)
        pos_ids = list(range(0, seq_len))
        group_ids = [group_id]*seq_len
        
        return {
                'Token-Ids':token_ids, 
                'Pos-Ids':pos_ids,
                'Group-Ids':group_ids,
            }
    
    def __call__(self, src_text, max_output_ratio=1.5, debug_output=False):
        
        batch_len = 1        
        src_max_len, tar_max_len = self.config.input_len
        
        tar_start_id = self.tar_tokenizer.vocab[self.TOKEN_START]
        tar_end_id = self.tar_tokenizer.vocab[self.TOKEN_END]
           
        src_ids = self.get_input_features(src_text)
        
        if src_max_len is None:
            src_max_len = len(src_ids['Token-Ids'])
        
        if tar_max_len is None:
            tar_max_len = int(src_max_len*max_output_ratio)
        
        inputs = {}
        for k, v in src_ids.items():
            mat = np.full((batch_len, src_max_len), -1, dtype=np.int32)
            mat[0, :len(v)] = np.asarray(v, dtype=np.int32)
            inputs['Encoder-'+k] = mat
        
        for k in ('Decoder-Token-Ids', 'Decoder-Pos-Ids', 'Decoder-Group-Ids'):
            inputs[k] = np.full((batch_len, tar_max_len), -1, dtype=np.int32)
        
        output_token_ids = []
        output_token_ids.append(
                np.asarray([tar_start_id]*batch_len, dtype=np.int32)
            )
           
        for i in range(0, tar_max_len):
            inputs['Decoder-Token-Ids'][:, i] = output_token_ids[-1]
            inputs['Decoder-Pos-Ids'][:, i] = i
            inputs['Decoder-Group-Ids'][:, i] = 0
            
            p = self.model.predict(inputs)
            
            # debug
            if debug_output:
                _p = (-p[0,i,:]).argsort()
                print('-----')
                for order in range(0, 3):
                    j = int(_p[order])
                    print(self.tar_tokenizer.inv_vocab[j], p[0,i][j])
            
            current_ids = p[:,i,:].argmax(axis=-1)
            current_ids = current_ids.astype(dtype=np.int32)
            output_token_ids.append(current_ids)
            
            if np.all(current_ids == tar_end_id):
                break
        
        tar_token_ids = [int(_[0]) for _ in output_token_ids]
        tar_tokens = self.tar_tokenizer.convert_ids_to_tokens(tar_token_ids)
        tar_text = ''.join(tar_tokens[1:-1])
        
        return tar_text
           
    

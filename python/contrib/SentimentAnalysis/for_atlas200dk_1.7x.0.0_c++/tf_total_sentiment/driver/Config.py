# coding=utf-8

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
from configparser import ConfigParser


class Configurable(object):
    def __init__(self, config_file, extra_args):
        print(config_file)
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([
                (k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])
            ])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        if not os.path.isdir(self.save_model_path):
            os.makedirs(self.save_model_path)
        #config.write(open(self.config_file, 'w'))
        print('Load config file successfully.\n')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

# Data

    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    @property
    def vocab_size(self):
        return self._config.getint('Data', 'vocab_size')

    @property
    def max_length(self):
        return self._config.getint('Data', 'max_length')

    @property
    def shuffle(self):
        return self._config.getboolean('Data', 'shuffle')

    @property
    def embedding_file(self):
        return self._config.get('Data', 'embedding_file')

# Save

    @property
    def decode_path(self):
        return self._config.get('Save', 'decode_path')

    @property
    def decode(self):
        return self._config.getboolean('Save', 'decode')

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def save_dirs(self):
        return self._config.get('Save', 'save_dirs')

    @property
    def word_path(self):
        return self._config.get('Save', 'word_path')

    @property
    def label_path(self):
        return self._config.get('Save', 'label_path')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_feature_voc(self):
        return self._config.get('Save', 'save_feature_voc')

    @property
    def save_label_voc(self):
        return self._config.get('Save', 'save_label_voc')

    @property
    def train_pkl(self):
        return self._config.get('Save', 'train_pkl')

    @property
    def dev_pkl(self):
        return self._config.get('Save', 'dev_pkl')

    @property
    def test_pkl(self):
        return self._config.get('Save', 'test_pkl')

    @property
    def embedding_pkl(self):
        return self._config.get('Save', 'embedding_pkl')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def load_dir_1(self):
        return self._config.get('Save', 'load_dir_1')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def load_feature_voc(self):
        return self._config.get('Save', 'load_feature_voc')

    @property
    def load_label_voc(self):
        return self._config.get('Save', 'load_label_voc')

# Network

    @property
    def embed_dim(self):
        return self._config.getint('Network', 'embed_dim')

    @property
    def num_layers(self):
        return self._config.getint('Network', 'num_layers')

    @property
    def hidden_dim(self):
        return self._config.getint('Network', 'hidden_dim')

    @property
    def attention_size(self):
        return self._config.getint('Network', 'attention_size')

    @property
    def dropout_embed(self):
        return self._config.getfloat('Network', 'dropout_embed')

    @property
    def dropout(self):
        return self._config.getfloat('Network', 'dropout')

    @property
    def max_norm(self):
        return self._config.getfloat('Network', 'max_norm')

    @property
    def which_model(self):
        return self._config.get('Network', 'which_model')

# Optimizer

    @property
    def learning_algorithm(self):
        return self._config.get('Optimizer', 'learning_algorithm')

    @property
    def lr(self):
        return self._config.getfloat('Optimizer', 'lr')

    @property
    def lr_scheduler(self):
        return self._config.get('Optimizer', 'lr_scheduler')

    @property
    def weight_decay(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def clip_norm(self):
        return self._config.getfloat('Optimizer', 'clip_norm')


# Run

    @property
    def use_cuda(self):
        return self._config.getboolean('Run', 'use_cuda')

    @property
    def load_model(self):
        return self._config.getboolean('Run', 'load_model')

    @property
    def epochs(self):
        return self._config.getint('Run', 'epochs')

    @property
    def batch_size(self):
        return self._config.getint('Run', 'batch_size')

    @property
    def test_interval(self):
        return self._config.getint('Run', 'test_interval')

    @property
    def save_interval(self):
        return self._config.getint('Run', 'save_interval')

    @property
    def log_interval(self):
        return self._config.getint('Run', 'log_interval')

    @property
    def sentence_max_length(self):
        return self._config.getint('Run', 'sentence_max_length')

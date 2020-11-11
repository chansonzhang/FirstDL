# -*- coding: utf-8 -*-

# Copyright 2018 Zhang, Chen. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# @Time    : 1/13/2019 21:52
# @Author  : Zhang, Chen (chansonzhang)
# @Email   : ZhangChen.Shaanxi@gmail.com
# @FileName: test_network2.py.py

import mnist_loader
import network2
# notice! y in training_data is a one hot vector, but y in test_data is a single number
# this may lead to some differences in training code and evaluation code
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 10],cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(list(training_data), 30, 10, 0.5, evaluation_data=list(test_data), monitor_evaluation_accuracy=True)
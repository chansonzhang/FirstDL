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
# @Time    : 12/26/2018 16:49
# @Author  : Zhang, Chen (chansonzhang)
# @Email   : ZhangChen.Shaanxi@gmail.com
# @FileName: test_numpy.py

import  numpy as np
A= np.array([[1,2],
             [3,4]])
B= np.array([[5,6],
             [7,8]])
print(np.einsum('ij,jk',A,B))
print(np.einsum('ij,jk->k',A,B))
print(A.sum(0))

b=np.array([[1],
            [2]])
print(b.shape)
print(b)
nb=np.array([3,4])
print(nb.shape)
print(nb)
c=b*nb
print(c.shape)
print(c)

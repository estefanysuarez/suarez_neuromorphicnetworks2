# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 19:04:05 2021

@author: Estefany Suarez
"""

import numpy as np
import matplotlib.pyplot as plt

from reservoir.tasks import io


#%%
io_kwargs = {'task':'sgnl_recon',
              'task_ref':'T1',
              'time_len':2050  # total number of training+test samples
              } 

inputs, outputs = io.get_io_data(**io_kwargs)

plt.plot(inputs[0], label='input')
plt.plot(outputs[0], label='target')
plt.legend()
plt.show()
plt.close()


#%%
# io_kwargs = {'task':'sgnl_recon',
#              'task_ref':'T2',
#              'step_len':50,
#              'bias':5,
#              'n_repeats':3 
#              } 

# inputs, outputs = io.get_io_data(**io_kwargs)

# plt.plot(inputs[0], label='input')
# plt.plot(outputs[0], label='target')
# plt.legend()
# plt.show()
# plt.close()


#%%
# io_kwargs = {'task':'pttn_recog',
#               'task_ref':'T1',
#               'n_input_nodes':5, 
#               'gain':3,       # number of input nodes
#               'n_patterns':3, # number of patterns to classify
#               'n_repeats':50, # total number of training+test samples
#               'time_len':20   # length of each pattern
#               } 

# inputs, outputs = io.get_io_data(**io_kwargs)

# plt.plot(inputs[0][:20,0], label='pattern 1- node 1', c='yellow', marker='o')
# plt.plot(inputs[0][:20,1], label='pattern 1- node 2', c='blue', marker='o')
# plt.plot(inputs[0][:20,2], label='pattern 1- node 3', c='red', marker='o')

# plt.legend()
# plt.show()
# plt.close()


#%%
io_kwargs = {'task':'pttn_recog',
             'task_ref':'T2',
             'n_patterns':3,    # number of patterns to classify
             'n_repeats':50,    # total number of training+test samples
             'time_len':20      # length of each pattern
             } 

inputs, outputs = io.get_io_data(**io_kwargs)

plt.plot(inputs[0][:20],   label='pattern 1', c='yellow', marker='o')
plt.plot(inputs[0][20:40], label='pattern 2', c='blue', marker='o')
plt.plot(inputs[0][40:60], label='pattern 3', c='red', marker='o')
plt.legend()
plt.show()
plt.close()




#%%\
def child_fcn(word='HOLA', **kwargs):
    print(word)


def parent_fcn(a, b, c, **kwargs):
    
    child_fcn(**kwargs)
    
    return a+b+c

params = {}
d = parent_fcn(1,2,3, **params)

params = {'word':'HELLO'}
d = parent_fcn(1,2,3, **params)

d = parent_fcn(1,2,3, word='HELLO')

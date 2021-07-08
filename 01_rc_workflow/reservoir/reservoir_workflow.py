# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:27:07 2020

@author: Estefany Suarez
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import configparser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import numpy as np

import pandas as pd
import multiprocessing as mp

import scipy.io as sio
from scipy.linalg import (eig, eigh)
from scipy.spatial.distance import cdist

from reservoir.network import nulls
from reservoir.tasks import (io, coding, tasks)
from reservoir.simulator import sim_lnm

from netneurotools import networks

#%% --------------------------------------------------------------------------------------------------------------------
# DYNAMIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
params_config = configparser.ConfigParser()
params_config.read('pttn_recog.ini')
# params_config.read('mem_cap.ini')

INPUTS = params_config['network']['network_inputs']
CLASS  = params_config['network']['network_partition']

TASK = params_config['task_params']['task']
SPEC_TASK = params_config['task_params']['specific_task']
TASK_REF = params_config['task_params']['task_reference']
FACTOR = float(params_config['task_params']['input_gain'])

N_PROCESS = int(params_config['parall_params']['num_process'])
N_RUNS = int(params_config['parall_params']['num_iters'])


#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%% --------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_metada(connectome):

    if CLASS == 'functional':
        class_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping', f'rsn_{connectome}.npy'))

    elif CLASS == 'cytoarch':
        class_mapping = np.load(os.path.join(DATA_DIR, 'cyto_mapping', f'cyto_{connectome}.npy'))

    return class_mapping


def consensus_network(connectome, coords, hemiid, path_res_conn, iter_id):

    conn_file = f'consensus_{iter_id}.npy'
    resampling_file = f'resampling_{iter_id}.npy'
    if not os.path.exists(os.path.join(path_res_conn, conn_file)):

        def bootstrapp(n, exclude=None):

            # create a list of samples
            samples = np.arange(n)

            # discard samples indicated in exclude
            if exclude is not None: samples = np.delete(samples, exclude)

            # bootstrapp resampling
            samples = np.random.choice(samples, size=len(samples), replace=True)

            return samples

        # load connectivity data
        CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'individual')
        stru_conn = np.load(os.path.join(CONN_DIR, connectome + '.npy'))

        # remove bad subjects
        bad_subj = [7, 12, 43] #SC:7,12,43 #FC:32
        bad_subj.extend(np.unique(np.where(np.isnan(stru_conn))[-1]))

        # bootstrapp resampling
        resampling = bootstrapp(n=stru_conn.shape[2], exclude=bad_subj)
        stru_conn_avg = networks.struct_consensus(data=stru_conn.copy()[:,:,resampling],
                                                  distance=cdist(coords, coords, metric='euclidean'),
                                                  hemiid=hemiid[:, np.newaxis]
                                                  )

        stru_conn_avg = stru_conn_avg*np.mean(stru_conn, axis=2)

        np.save(os.path.join(path_res_conn, conn_file), stru_conn_avg)
        np.save(os.path.join(path_res_conn, resampling_file), resampling)


def rewire_network(path_res_conn, iter_id, model_name, **kwargs):

    conn_file = f'{model_name}_{iter_id}.npy'

    if not os.path.exists(os.path.join(path_res_conn, conn_file)):

        new_conn = nulls.construct_null_model(model_name, **kwargs)

        np.save(os.path.join(path_res_conn, conn_file), new_conn)


def workflow(conn_name, io_nodes, readout_modules, \
             path_res_conn, path_io, path_res_sim, path_res_tsk, \
             iter_id=None, bin=False, alphas=None, partition_name=None, \
             iter_conn=True, iter_io=False, iter_sim=False, \
             encode=True, decode=True, **kwargs):
    """
        Runs the full pipeline: loads and scales connectivity matrix, generates
        input/output data for the task, simulates reservoir states, and trains
        the readout module.

        Parameters
        ----------

        conn_name: str, {'consensus', 'rewired'}
            Specifies the name of the connectivity matrix file

        io_nodes: (N,) numpy.darray
            Binary array that indicates input and output nodes in the recurrent
            network. 1 for input, 0 for output.
            N: number of nodes in the network

        readout_modules: (N, ) numpy.darray
            Array that indicates the module at which each output node belongs
            to. Modules can be int or str
            N: number of output nodes

        path_res_conn : str
            Path to conenctivity matrix

        path_io : str
            Path to simulation results

        path_res_sim : str
            Path to simulation results

        path_res_tsk : str
            Path to task scores

        bin : bool
            If True, the binary matrix will be used

        partition_name : str, {'functional', 'cytoarch'}
            Name of the partition used

        iter_id : int
            Number/name of the iteration

        iter_{conn,io,sim} : bool
            If True, specific instances (i.e., connectivity, input/output data,
            network states) related to the iteration indicated by iter_id will
            be used.

        encode,decode : bool
            If True, encoding,decoding will run
    """

    # --------------------------------------------------------------------------------------------------------------------
    # DEFINE FILE NAMES
    # ----------------------------------------------------------------------------------------------------------------------

    # define file connectivity data
    if np.logical_and(iter_id is not None, iter_conn):
        conn_file = f'{conn_name}_{iter_id}.npy'
    else: conn_file = f'{conn_name}.npy'

    # define file I/O data
    if np.logical_and(iter_id is not None, iter_io):
        input_file  = f'inputs_{iter_id}.npy'
        output_file = f'outputs_{iter_id}.npy'
    else:
        input_file  = 'inputs.npy'
        output_file = 'outputs.npy'

    # define file simulation data (reservoir states)
    if np.logical_and(iter_id is not None, iter_sim):
        res_states_file = f'reservoir_states_{iter_id}.npy'
    else:
        res_states_file  = 'reservoir_states.npy'

    # define file encoding/decoding scores data
    if np.logical_and(iter_id is not None, partition_name is not None):
        encoding_file = f'{partition_name}_encoding_score_{iter_id}.csv'
        decoding_file = f'{partition_name}_decoding_score_{iter_id}.csv'

    elif np.logical_and(iter_id is not None, partition_name is None):
        encoding_file = f'encoding_score_{iter_id}.csv'
        decoding_file = f'decoding_score_{iter_id}.csv'

    elif np.logical_and(iter_id is None, partition_name is not None):
        encoding_file = f'{partition_name}_encoding_score.csv'
        decoding_file = f'{partition_name}_decoding_score.csv'

    else:
        encoding_file = 'encoding_score.csv'
        decoding_file = 'decoding_score.csv'

    #%% --------------------------------------------------------------------------------------------------------------------
    # IMPORT CONNECTIVITY DATA
    # ----------------------------------------------------------------------------------------------------------------------

    # load connectivity data
    conn = np.load(os.path.join(path_res_conn, conn_file))

    # scale weights [0,1]
    if bin: conn = conn.astype(bool).astype(int)
    else:   conn = (conn-conn.min())/(conn.max()-conn.min())

    # normalize by the spectral radius
    if nulls.check_symmetric(conn):
        ew, _ = eigh(conn)
    else:
        ew, _ = eig(conn)

    conn  = conn/np.max(ew)

    # --------------------------------------------------------------------------------------------------------------------
    # CREATE I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    # memory capacity task kwargs
    if TASK == 'sgnl_recon':
        io_kwargs = {'task':TASK,
                     'task_ref':TASK_REF,
                     'time_len':2000     # number of training/test samples
                     }

        pttn_recog_kwargs = {}

    # pattern recognition task kwargs
    elif TASK == 'pttn_recog':
        n_patterns = 10
        n_repeats = 500
        time_len = 20 #20

        io_kwargs = {'task':TASK,
                     'task_ref':TASK_REF,
                     'n_patterns':n_patterns,  # number of patterns to classify
                     'n_repeats':n_repeats,    # total number of training+test samples per pattern
                     'time_len':time_len,      # length of each pattern
                     'gain':3.0,
                     }

        # 0.5 is the training/text proportion
        pttn_recog_kwargs = {'time_lens':time_len*np.ones(int(0.5*n_patterns*n_repeats), dtype=int)}

    if not os.path.exists(os.path.join(path_io, input_file)):

        inputs, outputs = io.get_io_data(**io_kwargs)

        np.save(os.path.join(path_io, input_file), inputs)
        np.save(os.path.join(path_io, output_file), outputs)

    # --------------------------------------------------------------------------------------------------------------------
    # NETWORK SIMULATION - LINEAR MODEL
    # ----------------------------------------------------------------------------------------------------------------------
    if alphas is None: alphas = tasks.get_default_alpha_values(SPEC_TASK)
    if not os.path.exists(os.path.join(path_res_sim, res_states_file)):

        input_train, input_test = np.load(os.path.join(path_io, input_file))

        # create input connectivity matrix
        w_in = np.zeros((input_train.shape[1],len(conn)))
        w_in[:,io_nodes == 1] = FACTOR # fully connected input layer

        # w_in = FACTOR * np.ones((input_train.shape[1],len(conn)))

        reservoir_states_train = sim_lnm.run_sim(w_in=w_in,
                                                 w=conn,
                                                 inputs=input_train,
                                                 alphas=alphas,
                                                )

        reservoir_states_test  = sim_lnm.run_sim(w_in=w_in,
                                                 w=conn,
                                                 inputs=input_test,
                                                 alphas=alphas,
                                                )

        reservoir_states = [(rs_train, rs_test) for rs_train, rs_test in zip(reservoir_states_train, reservoir_states_test)]
        np.save(os.path.join(path_res_sim, res_states_file), reservoir_states, allow_pickle=False)


    # --------------------------------------------------------------------------------------------------------------------
    # IMPORT I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    reservoir_states = np.load(os.path.join(path_res_sim, res_states_file), allow_pickle=True)
    reservoir_states = reservoir_states[:, :, :, io_nodes==0]
    reservoir_states = reservoir_states.squeeze()
    reservoir_states = np.split(reservoir_states, len(reservoir_states), axis=0)
    reservoir_states = [rs.squeeze() for rs in reservoir_states]

    outputs = np.load(os.path.join(path_io, output_file))
    
    # --------------------------------------------------------------------------------------------------------------------
    # PERFORM TASK - ENCODERS
    # ----------------------------------------------------------------------------------------------------------------------
    try:
        if np.logical_and(encode, not os.path.exists(os.path.join(path_res_tsk, encoding_file))):
            print('\nEncoding: ')
            df_encoding = coding.encoder(task=SPEC_TASK,
                                         target=outputs,
                                         reservoir_states=reservoir_states,
                                         readout_modules=readout_modules,
                                         alphas=alphas,
                                         **pttn_recog_kwargs
                                         )

            df_encoding.to_csv(os.path.join(path_res_tsk, encoding_file))
    except:
        pass

    # --------------------------------------------------------------------------------------------------------------------
    # PERFORM TASK - DECODERS
    # ----------------------------------------------------------------------------------------------------------------------
    try:
        if np.logical_and(decode, not os.path.exists(os.path.join(path_res_tsk, decoding_file))):
            # binarize cortical adjacency matrix
            conn_bin = conn.copy()[np.ix_(np.where(io_nodes==0)[0], np.where(io_nodes==0)[0])].astype(bool).astype(int)

            print('\nDecoding: ')
            df_decoding = coding.decoder(task=SPEC_TASK,
                                         target=outputs,
                                         reservoir_states=reservoir_states,
                                         readout_modules=readout_modules,
                                         bin_conn=conn_bin,
                                         alphas=alphas,
                                         **pttn_recog_kwargs
                                         )

            df_decoding.to_csv(os.path.join(path_res_tsk, decoding_file))
    except:
        pass

    # delete reservoir states to release memory storage
    if iter_sim:
        os.remove(os.path.join(path_res_sim, res_states_file))
        pass


#%% --------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
def run_workflow(analysis, connectome, iter_id, **kwargs):

    # --------------------------------------------------------------------------------------------------------------------
    # CREATE DIRECTORIES
    # ----------------------------------------------------------------------------------------------------------------------
    IO_TASK_DIR  = os.path.join(RAW_RES_DIR, SPEC_TASK, 'io_tasks')
    RES_CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_results', analysis, f'scale{connectome[-3:]}')
    RES_SIM_DIR  = os.path.join(RAW_RES_DIR, SPEC_TASK, 'sim_results', analysis, f'{INPUTS}_scale{connectome[-3:]}')
    RES_TSK_DIR  = os.path.join(RAW_RES_DIR, SPEC_TASK, 'tsk_results', analysis, f'{INPUTS}_scale{connectome[-3:]}')

    if not os.path.exists(IO_TASK_DIR):  os.makedirs(IO_TASK_DIR)
    if not os.path.exists(RES_CONN_DIR): os.makedirs(RES_CONN_DIR)
    if not os.path.exists(RES_SIM_DIR):  os.makedirs(RES_SIM_DIR)
    if not os.path.exists(RES_TSK_DIR):  os.makedirs(RES_TSK_DIR)

    # --------------------------------------------------------------------------------------------------------------------
    # LOAD METADATA
    # ----------------------------------------------------------------------------------------------------------------------
    class_mapping = load_metada(connectome)

    ctx = np.load(os.path.join(DATA_DIR, 'cortical', f'cortical_{connectome}.npy'))
    coords = np.load(os.path.join(DATA_DIR, 'coords', f'coords_{connectome}.npy'))
    hemiid = np.load(os.path.join(DATA_DIR, 'hemispheres', f'hemiid_{connectome}.npy'))

    # --------------------------------------------------------------------------------------------------------------------
    # CREATE CONSENSUS MATRICES
    # ----------------------------------------------------------------------------------------------------------------------
    consensus_network(connectome,
                      coords,
                      hemiid,
                      RES_CONN_DIR,
                      iter_id,
                      )

    # --------------------------------------------------------------------------------------------------------------------
    # DEFINE PARAMS
    # ----------------------------------------------------------------------------------------------------------------------
    if analysis == 'reliability':
        conn_name = 'consensus'
        class_mapping = class_mapping[ctx == 1]

    elif analysis == 'significance':
        conn_name = 'rand_mio'
        class_mapping = class_mapping[ctx == 1]

        # create rewired network
        conn = np.load(os.path.join(RES_CONN_DIR, f'consensus_{iter_id}.npy'))
        try:
            rewire_network(RES_CONN_DIR,
                           iter_id,
                           model_name=conn_name,
                           conn=conn,
                           swaps=10
                           )
        except:
            pass

    elif analysis == 'spintest':
        conn_name = 'consensus'
        spins = np.genfromtxt(os.path.join(DATA_DIR, 'spin_test', f'spin_{connectome}.csv'), delimiter=',').astype(int)
        class_mapping = class_mapping[ctx == 1][spins[:, iter_id]]

    # --------------------------------------------------------------------------------------------------------------------
    # RUN WORKFLOW
    # ----------------------------------------------------------------------------------------------------------------------
    try:
        workflow(conn_name=conn_name,
                 io_nodes=np.logical_not(ctx).astype(int), # input and output nodes i.e., ctx + subctx
                 readout_modules=class_mapping,  # only output nodes i.e., only for ctx
                 iter_id=iter_id,
                 iter_conn=True,
                 iter_io=False,
                 iter_sim=True,
                 encode=True,
                 decode=False,
                 path_res_conn=RES_CONN_DIR,
                 path_io=IO_TASK_DIR,
                 path_res_sim=RES_SIM_DIR,
                 path_res_tsk=RES_TSK_DIR,
                 partition_name=CLASS,
                 )

    except:
        pass


def main():

    ANALYSIS = [
                'reliability',
                # 'significance',
                # 'spintest'
                ]

    for analysis in ANALYSIS:
        connectome = 'human_500'

        print (f'INITIATING PROCESSING TIME - {analysis.upper()}')
        t0_1 = time.perf_counter()

        # run iteration No. 0
        run_workflow(analysis,
                     connectome,
                     iter_id=0, #np.random.randint(1,1000,1)[0],
                     )

        # # run iterations No. 1-1000 in parallel
        # start = 1
        # params = []
        # for i in range(start, N_RUNS):
        #     params.append({
        #                    'analysis':analysis,
        #                    'connectome':connectome,
        #                    'iter_id': i,
        #                    })

        # pool = mp.Pool(processes=N_PROCESS)
        # res = [pool.apply_async(run_workflow, (), p) for p in params]
        # for r in res: r.get()
        # pool.close()

        print (f'PROCESSING TIME - {analysis.upper()}')
        print (time.perf_counter()-t0_1, "seconds process time")
        # print (time.time()-t0_2, "seconds wall time")

        print('END')


if __name__ == '__main__':
    main()

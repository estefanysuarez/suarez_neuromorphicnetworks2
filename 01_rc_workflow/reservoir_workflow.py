# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:44:16 2021

@author: Estefany Suarez
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import sys, getopt
import configparser
import multiprocessing as mp

# from tqdm import tqdm

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from netneurotools import networks

from rnns import (io_data, rnns, sim, nulls, coding, lyapunov)

#%% --------------------------------------------------------------------------------------------------------------------
# LOAD CONFIGURATION FILE
# ----------------------------------------------------------------------------------------------------------------------
try:
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'c:', ['config_file='])
except getopt.GetoptError:
        print ('reservoir_workflow.py -c <config_file>')
        sys.exit(2)

for opt, arg in opts:
    if opt in ("-c", "--config_file"):
        config_file = arg
        # print ('CONFIGURATION FILE: ', config_file)

config = configparser.ConfigParser()
config.read(config_file)


#%% --------------------------------------------------------------------------------------------------------------------
# DYNAMIC VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
ANALYSIS   = config['GENERAL']['analysis']
CONNECTOME = config['GENERAL']['connectome']
TASK       = config['GENERAL']['task']
MODULES    = config['GENERAL']['modules']

N_RUNS    = int(config['GENERAL']['num_iters'])
N_PROCESS = int(config['GENERAL']['num_process'])


# --------------------------------------------------------------------------------------------------------------------
# CREATE DIRECTORIES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')

IO_TASK_DIR  = os.path.join(RAW_RES_DIR, TASK, 'io_task')
RES_CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_res', ANALYSIS, CONNECTOME)
RES_SIM_DIR  = os.path.join(RAW_RES_DIR, TASK, 'sim_res', ANALYSIS, CONNECTOME)
RES_TSK_DIR  = os.path.join(RAW_RES_DIR, TASK, 'tsk_res', ANALYSIS, CONNECTOME)

if not os.path.exists(IO_TASK_DIR):  os.makedirs(IO_TASK_DIR)
if not os.path.exists(RES_CONN_DIR): os.makedirs(RES_CONN_DIR)
if not os.path.exists(RES_SIM_DIR):  os.makedirs(RES_SIM_DIR)
if not os.path.exists(RES_TSK_DIR):  os.makedirs(RES_TSK_DIR)


#%% --------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def consensus_network(coords, hemiid, iter_id):

    conn_file = f'consensus_{iter_id}.npy'
    resampling_file = f'resampling_{iter_id}.npy'
    if not os.path.exists(os.path.join(RES_CONN_DIR, conn_file)):

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
        stru_conn = np.load(os.path.join(CONN_DIR, f'{CONNECTOME}.npy'))

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

        np.save(os.path.join(RES_CONN_DIR, conn_file), stru_conn_avg)
        np.save(os.path.join(RES_CONN_DIR, resampling_file), resampling)


def rewired_network(model_name, iter_id, **kwargs):
    conn_file = f'{model_name}_{iter_id}.npy'

    if not os.path.exists(os.path.join(RES_CONN_DIR, conn_file)):
        new_conn = nulls.construct_null_model(model_name, **kwargs)
        np.save(os.path.join(RES_CONN_DIR, conn_file), new_conn)


def workflow(conn_file, input_nodes, output_nodes, gain, readout_modules, id_number=None, \
             alphas=[1.0], io_kwargs={}, sim_kwargs={}, task_kwargs={}, \
             bin=False, iter_conn=True, iter_io=False, iter_sim=False, \
             encode=True, decode=True, **kwargs):

    # --------------------------------------------------------------------------------------------------------------------
    # DEFINE FILE NAMES
    # ----------------------------------------------------------------------------------------------------------------------
    # define file connectivity data
    if np.logical_and(id_number is not None, iter_conn):
        conn_file = f'{conn_file}_{id_number}.npy'
    else: conn_file = f'{conn_file}.npy'

    # define file I/O data
    if np.logical_and(id_number is not None, iter_io):
        input_file  = f'inputs_{id_number}.npy'
        output_file = f'outputs_{id_number}.npy'
    else:
        input_file  = 'inputs.npy'
        output_file = 'outputs.npy'

    # define file simulation data (reservoir states)
    if np.logical_and(id_number is not None, iter_sim):
        res_states_file = f'reservoir_states_{id_number}.npy'
    else:
        res_states_file  = 'reservoir_states.npy'

    # define file encoding/decoding scores data
    if id_number is not None:
        encoding_file = f'encoding_score_{id_number}.csv'
        decoding_file = f'decoding_score_{id_number}.csv'
    else:
        encoding_file = 'encoding_score.csv'
        decoding_file = 'decoding_score.csv'

    if os.path.exists(os.path.join(RES_TSK_DIR, encoding_file)):
        return

    #%% --------------------------------------------------------------------------------------------------------------------
    # IMPORT CONNECTIVITY DATA
    # ----------------------------------------------------------------------------------------------------------------------
    print("\tLoading connectivity data ...")

    # load connectivity data
    conn = np.load(os.path.join(RES_CONN_DIR, conn_file))

    # scale weights [0,1]
    if bin: conn = conn.astype(bool).astype(int)
    else:   conn = (conn-conn.min())/(conn.max()-conn.min())

    # normalize by the spectral radius
    ew, _ = eigh(conn)
    conn  = conn/np.max(ew)

    # --------------------------------------------------------------------------------------------------------------------
    # CREATE I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(os.path.join(IO_TASK_DIR, input_file)):
        print("\tGenerating IO data ...")

        x, y = io_data.generate_IOData(TASK, **io_kwargs)

        np.save(os.path.join(IO_TASK_DIR, input_file), x)
        np.save(os.path.join(IO_TASK_DIR, output_file), y)

    # --------------------------------------------------------------------------------------------------------------------
    # NETWORK SIMULATION
    # ----------------------------------------------------------------------------------------------------------------------
    reservoir_states = None
    if not os.path.exists(os.path.join(RES_SIM_DIR, res_states_file)):
        print("\tSimulating network ...")

        X = np.load(os.path.join(IO_TASK_DIR, input_file))

        # fully connected input layer
        w_in = np.zeros((len(conn), X.shape[-1]))
        w_in[input_nodes] = gain

        # simulate network
        states_train, states_test = sim.run_multiple_sim(w_ih=w_in,
                                                         w_hh=conn,
                                                         inputs=X,
                                                         alphas=alphas,
                                                         **sim_kwargs
                                                         )

        reservoir_states = [(rs_train, rs_test) for rs_train, rs_test in zip(states_train, states_test)]
        np.save(os.path.join(RES_SIM_DIR, res_states_file), reservoir_states, allow_pickle=False)

    # --------------------------------------------------------------------------------------------------------------------
    # IMPORT I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    if reservoir_states is None:
        print("\tUploading networks states ...")
        reservoir_states = np.load(os.path.join(RES_SIM_DIR, res_states_file), allow_pickle=True)
    else:
        reservoir_states = np.array(reservoir_states)

    reservoir_states = reservoir_states[:, :, :, output_nodes]
    reservoir_states = reservoir_states.squeeze()
    reservoir_states = np.split(reservoir_states, len(reservoir_states), axis=0)
    reservoir_states = [rs.squeeze() for rs in reservoir_states]

    y = np.load(os.path.join(IO_TASK_DIR, output_file))

    # --------------------------------------------------------------------------------------------------------------------
    # PERFORM TASK - ENCODERS
    # ----------------------------------------------------------------------------------------------------------------------    #
    # try:
    if np.logical_and(encode, not os.path.exists(os.path.join(RES_TSK_DIR, encoding_file))):
        print(" Performing task ...")
        df_encoding = coding.encoder(task=TASK,
                                     reservoir_states=reservoir_states,
                                     target=y,
                                     readout_modules=readout_modules,
                                     alphas=alphas,
                                     **task_kwargs
                                     )

        df_encoding.to_csv(os.path.join(RES_TSK_DIR, encoding_file))
    # except:
    #     pass
    #
    # delete reservoir states to release memory storage
    if iter_sim:
        # os.remove(os.path.join(path_res_sim, res_states_file))
        pass


def run_workflow(iter_id, **kwargs):

    # --------------------------------------------------------------------------------------------------------------------
    # CREATE CONSENSUS MATRICES
    # ----------------------------------------------------------------------------------------------------------------------
    print("Creating consensus network ...")

    ctx    = np.load(os.path.join(DATA_DIR, 'cortical', f'cortical_{CONNECTOME}.npy'))
    coords = np.load(os.path.join(DATA_DIR, 'coords', f'coords_{CONNECTOME}.npy'))
    hemiid = np.load(os.path.join(DATA_DIR, 'hemispheres', f'hemiid_{CONNECTOME}.npy'))

    consensus_network(coords, hemiid, iter_id)

    # --------------------------------------------------------------------------------------------------------------------
    # DEFINE MODULES
    # ----------------------------------------------------------------------------------------------------------------------
    print("Defining modules ...")

    if MODULES == 'functional':
        modules = np.load(os.path.join(DATA_DIR, 'rsn_mapping', f'rsn_{CONNECTOME}.npy'))

    elif MODULES == 'cytoarch':
        modules = np.load(os.path.join(DATA_DIR, 'cyto_mapping', f'cyto_{CONNECTOME}.npy'))

    # --------------------------------------------------------------------------------------------------------------------
    # DEFINE OTHER VARIABLES
    # ----------------------------------------------------------------------------------------------------------------------
    print("Defining other variables ...")

    if ANALYSIS == 'reliability':
        conn_name = 'consensus'
        modules = modules[ctx == 1]

    elif ANALYSIS == 'significance':
        conn_name = 'rand_mio'
        modules = modules[ctx == 1]

        # create rewired network
        conn = np.load(os.path.join(RES_CONN_DIR, f'consensus_{iter_id}.npy'))
        try:
            rewired_network(iter_id,
                            model_name=conn_name,
                            conn=conn,
                            swaps=10
                            )

        except:
            pass

    elif ANALYSIS == 'spintest':
        conn_name = 'consensus'
        spins = np.genfromtxt(os.path.join(DATA_DIR, 'spin_test', f'spin_{CONNECTOME}.csv'), delimiter=',').astype(int)
        modules = modules[ctx == 1][spins[:, iter_id]]


    # --------------------------------------------------------------------------------------------------------------------
    # RUN WORKFLOW
    # ----------------------------------------------------------------------------------------------------------------------
    print("\nRunning workflow ...")

    if 'IO_PARAMS' in config: io_kwargs = {k:int(v) for k,v in config['IO_PARAMS'].items()}
    else: io_kwargs = {}

    if 'ALT_SIM_PARAMS' in config: sim_kwargs = {k:v for k,v in config['ALT_SIM_PARAMS'].items()}
    else: sim_kwargs = {}

    if TASK == 'pttn_recog':
        time_lens = io_kwargs['time_len']*np.ones(int(0.5*io_kwargs['n_patterns']*io_kwargs['n_repeats']), dtype=int)
        task_kwargs = {'pttn_lens':time_lens}
    else:
        task_kwargs = {}


    try:
        workflow(conn_file=conn_name,
                 input_nodes=np.where(ctx==0)[0], # internal input nodes
                 output_nodes=np.where(ctx==1)[0], # internal output nodes
                 gain=float(config['SIM_PARAMS']['gain']),
                 readout_modules=modules,  # only output nodes i.e., only for ctx
                 id_number=iter_id,
                 alphas=eval(config['SIM_PARAMS']['alphas']),
                 io_kwargs=io_kwargs,
                 sim_kwargs=sim_kwargs,
                 task_kwargs=task_kwargs,
                 iter_conn=config.getboolean('WORKFLOW','iter_conn'),
                 iter_io=config.getboolean('WORKFLOW','iter_io'),
                 iter_sim=config.getboolean('WORKFLOW','iter_sim'),
                 encode=config.getboolean('WORKFLOW','encode'),
                 decode=config.getboolean('WORKFLOW','decode')
                 )

    except:
        pass


def main():

    print (f'\nINITIATING PROCESSING TIME - {ANALYSIS.upper()} for {CONNECTOME.upper()}')
    t0 = time.perf_counter()

    # # run iteration No. 0
    # run_workflow(iter_id=0)

    # run iterations No. 1-1000 in parallel
    start = 89
    params = [{'iter_id': i} for i in range(start, N_RUNS)]

    pool = mp.Pool(processes=N_PROCESS)
    res = [pool.apply_async(run_workflow, (), p) for p in params]
    for r in res: r.get()
    pool.close()

    print (f'\nTOTAL PROCESSING TIME - {ANALYSIS.upper()}')
    print (time.perf_counter()-t0, "seconds process time")
    print ('END')


if __name__ == '__main__':
    main()

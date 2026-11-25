"""
008c_T1_ge_points_sweep - how many points does the T1 fit need?

A variant of 008_T1_ge. The same T1 experiment is repeated over the same fixed
time range (start / stop of T1_ge in system_config.json) with a different
number of points each time, from N_MIN to N_MAX. Fewer points over the same
range means a coarser time step. A dense T1 with BASELINE_N points is taken
before and after the sweep as the reference the others are compared with; the
two baselines also show how much T1 itself moved during the run.

reps and py_avg are the T1_ge values from system_config.json and are the same
for every run, so a run with fewer points is also a shorter measurement.

Every run is saved as its own T1 file, all in one new numbered folder in
DATA_PATH, with n_points and is_baseline saved alongside. Analyse the folder
with analyze_t1_points_sweep in data_analysis_live.ipynb.

There is no flux pulse here and the HDAWG is not used. This runs on the one
configured qubit (qubit_index); meas_list is ignored.

We also have the option to perform single shot measurements for g-e calibration
after each T1. This is done by setting the SS variable to True in the
system_config.json file. Your pi pulse should be calibrated before running SS.

Live plotting is setup.
"""

import os
os.chdir(os.getcwd() + '/qick_tprocv2_experiments')

from qick import *
from qick.pyro import make_proxy

from malab import SlabFile, get_next_filename
# for now, all the tProc v2 classes need to be individually imported (can't use qick.*)

# the main program class
from qick.asm_v2 import AveragerProgramV2
# for defining sweeps
from qick.asm_v2 import QickSpan, QickSweep1D

import json
import datetime
import time

# Used for live plotting, need to run "python -m visdom.server" in the terminal and open the IP address in browser
import visdom

from build_task import *
from build_state import *
from qick_setup_funcs import *

from qualang_tools.plot import Fit
import pprint as pp

# connect to the rfsoc and print soccfg
from rfsoc_connect import *

if SS == 'True':
    # import single shot information for g-e calibration
    from SingleShot import SingleShotProgram_g, SingleShotProgram_e
    from SingleShot import config as ss_config

###########################
#### sweep parameters ####
###########################
N_MIN = 4           # fewest points in a T1 (the fit has 3 parameters, so at least 4)
N_MAX = 40        # most points in a T1
NUM_N = 18          # how many different point counts between N_MIN and N_MAX
BASELINE_N = 300    # points in the dense reference T1, taken before and after the sweep

# the point counts, spaced evenly on a log scale (more of them at the low end,
# where the fit changes fastest). Or write your own list: N_LIST = [5, 10, 20, 50]
N_LIST = np.unique(np.round(np.geomspace(N_MIN, N_MAX, NUM_N)).astype(int))

if N_MIN < 4:
    raise ValueError(f"N_MIN = {N_MIN}: the T1 fit has 3 parameters and needs at least 4 points")
if N_MAX < N_MIN:
    raise ValueError(f"N_MAX = {N_MAX} is below N_MIN = {N_MIN}")
if np.min(N_LIST) < N_MIN or np.max(N_LIST) > N_MAX:
    raise ValueError(f"N_LIST runs from {np.min(N_LIST)} to {np.max(N_LIST)}, outside N_MIN = {N_MIN} to N_MAX = {N_MAX}")

# ----- Experiment configurations ----- #
base_expt_name = "T1_ge"

##################
# Define Program #
##################
class T1Program(AveragerProgramV2):
    def _initialize(self, cfg):
        # initialize the readout channels and pulses for DAC and ADC
        initialize_ro_chs(self, cfg, suffix='_ge')

        self.add_loop("waitloop", cfg["steps"])

        # initialize the qubit generator channel
        declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')

        # initialize qubit pulse
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='qubit_pulse', suffix='_ge')

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  #play pulse

        self.delay_auto(cfg['wait_time']+0.01, tag='wait') # wait_time after last pulse

        # readout using proper configurations for MUX and non-MUX
        readout(self, cfg)


def run_t1(n_points, is_baseline, data_path):
    '''
    One T1 with n_points over the configured time range, saved in data_path.

    n_points: number of delay times (the 'steps' of the sweep)
    is_baseline: True for the dense reference runs (saved with the data)
    data_path: folder the file is saved in
    '''
    start_time = time.time()
    # the config is rebuilt for each run: the save block deletes the swept
    # parameter (a QickParam, which cannot be JSON-serialised)
    config, expt_name = initialize_configs(base_expt_name)
    config['steps'] = int(n_points)   # same start and stop, so only the spacing changes
    if SS == 'True':
        ss_config, _ = initialize_configs("IQ_plot")

    t1=T1Program(soccfg, reps=config['reps'], final_delay=config['relax_delay'], cfg=config)
    py_avg = config['py_avg']

    # for live plotting
    IS_VISDOM = True
    if IS_VISDOM:
        expt_I = expt_Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"
        viz.close(win=None) # close previous plots
        plot_title = f"T1 Experiment, {n_points} points" + (" (baseline)" if is_baseline else "")
        win1 = viz.line( X=np.arange(0, 1), Y=np.arange(0, 1),
            opts=dict(height=400, width=700, title=plot_title, showlegend=True, xlabel='expt_pts'))

        for ii in range(py_avg):
            iq_list = t1.acquire(soc, soft_avgs=1, progress=False)
            delay_times = t1.get_time_param('wait', "t", as_array=True)
            delay_times -= delay_times[0]

            if MUX == 'False':
                iq_list = iq_list[0][0].T
                # what is the correct shape/index?
                this_I = (iq_list[0])
                this_Q = (iq_list[1])
            else:
                this_I = (iq_list[QUBIT_INDEX - 1].T[0])
                this_Q = (iq_list[QUBIT_INDEX - 1].T[1])

            if expt_I is None: # ii == 0
                expt_I, expt_Q = this_I, this_Q
            else:
                expt_I = (expt_I * ii + this_I) / (ii + 1.0)
                expt_Q = (expt_Q * ii + this_Q) / (ii + 1.0)

            expt_mags = np.abs(expt_I + 1j * expt_Q)  # magnitude
            expt_phases = np.angle(expt_I + 1j * expt_Q)  # phase

            viz.line(X = delay_times, Y = expt_mags, win=win1, name='I',
            opts=dict(height=400, width=700, title=plot_title, showlegend=True, xlabel='expt_pts'))

        amps = np.abs(expt_I + 1j*expt_Q)

    else:
        iq_list = t1.acquire(soc, soft_avgs=py_avg, progress=True)
        delay_times = t1.get_time_param('wait', "t", as_array=True)
        delay_times -= delay_times[0]
        amps = np.abs(iq_list[0][0].dot([1,1j]))

    if SS == 'True':
        print('performing single shot for g-e calibration')

        ssp_g = SingleShotProgram_g(soccfg, reps=1, final_delay=ss_config['relax_delay'], cfg=ss_config)
        iq_list_g = ssp_g.acquire(soc, soft_avgs=1, progress=True)

        ssp_e = SingleShotProgram_e(soccfg, reps=1, final_delay=ss_config['relax_delay'], cfg=ss_config)
        iq_list_e = ssp_e.acquire(soc, soft_avgs=1, progress=True)

        if MUX == 'False':
            I_g = iq_list_g[0][0].T[0]
            Q_g = iq_list_g[0][0].T[1]
            I_e = iq_list_e[0][0].T[0]
            Q_e = iq_list_e[0][0].T[1]
        else:
            I_g = iq_list_g[QUBIT_INDEX - 1].T[0]
            Q_g = iq_list_g[QUBIT_INDEX - 1].T[1]
            I_e = iq_list_e[QUBIT_INDEX - 1].T[0]
            Q_e = iq_list_e[QUBIT_INDEX - 1].T[1]

    #####################################
    # ----- Saves data to a file ----- #
    #####################################

    prefix = str(datetime.date.today())
    exp_name = expt_name + '_Q' + str(QUBIT_INDEX) + '_' + prefix
    print('Experiment name: ' + exp_name)

    fname = get_next_filename(data_path, exp_name, suffix='.h5')
    print('Current data file: ' + fname)
    tot_time = time.time() - start_time

    with SlabFile(os.path.join(data_path, fname), 'a') as f:
        # 'a': read/write/create

        # - Adds data to the file - #
        f.append('expt_time', [int(tot_time)])
        f.append('delay_times', delay_times)
        f.append('amps', amps)
        f.append('n_points', [int(n_points)])
        f.append('is_baseline', [int(is_baseline)])
        if IS_VISDOM:
            f.append('avgi', expt_I)
            f.append('avgq', expt_Q)
        else:
            f.append('avgi', iq_list[0][0].T[0])
            f.append('avgq', iq_list[0][0].T[1])

        del config['wait_time']  # cant save QickParam
        # formats config into file as a single line
        f.attrs['config'] = json.dumps(config)

        if SS == 'True':
            # - Adds ss data to the file - #
            f.append('I_g', I_g)
            f.append('Q_g', Q_g)
            f.append('I_e', I_e)
            f.append('Q_e', Q_e)

            # formats config into file as a single line
            f.attrs['ss_config'] = json.dumps(ss_config)

    return fname


###################
# Run the Program
###################

# one new numbered folder in DATA_PATH for the whole sweep
folder_suffix = 'T1_ge_points_sweep_Q' + str(QUBIT_INDEX) + '_' + str(datetime.date.today())
existing = [d for d in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, d)) and d[:5].isdigit()]
next_num = max([int(d[:5]) for d in existing], default=-1) + 1
data_path = os.path.join(DATA_PATH, f"{next_num:05d}_{folder_suffix}")
os.makedirs(data_path)
print('saving into ' + data_path)
print(f"point counts: {list(N_LIST)}, baseline {BASELINE_N} points before and after")

run_t1(BASELINE_N, True, data_path)
for n_points in N_LIST:
    run_t1(int(n_points), False, data_path)
    print(f'done: {n_points} points')
run_t1(BASELINE_N, True, data_path)

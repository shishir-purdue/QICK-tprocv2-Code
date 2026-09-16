"""
014b_readout_fidelity_ro_length_ge - Readout Fidelity vs Readout Duration for ge state

This runs the single-shot g/e measurement once per readout duration, stepping
through the hardcoded RO_LENGTHS list below. res_length_ge is kept
RES_LENGTH_OFFSET above ro_length on every pass. Neither length can be a
QickParam - the ADC acquisition window and the readout pulse are both fixed
when the program is compiled - so the sweep is a Python loop that rebuilds the
single-shot programs each time.

Configurations are taken from the IQ_plot block (the same one SingleShot.py
uses); this experiment has none of its own. Every run of the script writes into
its own numbered subfolder of DATA_PATH, one file per readout length in the
same format SingleShot.py writes, which is what analyze_readout_fidelity_ro_length
reads.

Note: SingleShot.py runs its own measurement when it is imported if SS_ONLY is
'True' in system_config.json. Set it to 'False' before running this, as for
014_Readout_Optimization_ge.py.

Author: Shishir
Date: 2026-09-16
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

from build_task import *
from build_state import *
from qick_setup_funcs import *

# import single shot programs for g-e measurement
from SingleShot import SingleShotProgram_g, SingleShotProgram_e

# connect to the rfsoc and print soccfg
from rfsoc_connect import *

# ----- Experiment configurations ----- #
expt_name = "IQ_plot"                        # where the configurations come from
save_name = "readout_fidelity_ro_length_ge"  # what the files and the folder are called
config, expt_name = initialize_configs(expt_name)
print(expt_name + '\n')
print(config)

# ----- Output folder for this run of the script ----- #
# One numbered folder per invocation, so repeated scans stay separate. The
# number is one past the highest already sitting in DATA_PATH.
folder_suffix = save_name + '_Q' + str(QUBIT_INDEX) + '_' + str(datetime.date.today())
existing = [d for d in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, d)) and d[:5].isdigit()]
next_num = max([int(d[:5]) for d in existing], default=-1) + 1
scan_folder_name = f"{next_num:05d}_{folder_suffix}"
scan_data_path = os.path.join(DATA_PATH, scan_folder_name)
os.makedirs(scan_data_path, exist_ok=True)
print('Scan data folder: ' + scan_data_path)

###################
# Run the Program
###################

# Hardcoded for now - the readout durations (us) to step through
RO_LENGTHS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# res_length_ge is held this far above ro_length on every pass, so the readout
# pulse always outlasts the acquisition window by a fixed margin.
RES_LENGTH_OFFSET = 0.1

# Keep the name we looked the config up with: initialize_configs appends
# '_mux' to what it is handed, so feeding expt_name back in would grow a
# second suffix on the next pass.
base_expt_name = expt_name

for ro_i, ro_length in enumerate(RO_LENGTHS):
    res_length = ro_length + RES_LENGTH_OFFSET
    print("")
    print(f"=== Readout fidelity scan {ro_i + 1}/{len(RO_LENGTHS)}: "
          f"ro_length = {ro_length} us, res_length_ge = {res_length} us ===")

    # Fresh config each pass, so nothing carries over between lengths.
    config, expt_name = initialize_configs(base_expt_name)

    # Both lengths have to be set BEFORE the programs are built: ro_length sizes
    # the ADC window in initialize_ro_chs and res_length_ge sets the readout
    # pulse in declare_pulse, both while the program is compiled.
    config.update([("ro_length", ro_length), ("res_length_ge", res_length)])

    ssp_g = SingleShotProgram_g(soccfg, reps=1, final_delay=config['relax_delay'], cfg=config)
    iq_list_g = ssp_g.acquire(soc, soft_avgs=1, progress=True)

    ssp_e = SingleShotProgram_e(soccfg, reps=1, final_delay=config['relax_delay'], cfg=config)
    iq_list_e = ssp_e.acquire(soc, soft_avgs=1, progress=True)

    if MUX == 'False':
        I_g = iq_list_g[0][0].T[0]
        Q_g = iq_list_g[0][0].T[1]
        I_e = iq_list_e[0][0].T[0]
        Q_e = iq_list_e[0][0].T[1]
    else:
        I_g = iq_list_g[QUBIT_INDEX].T[0]
        Q_g = iq_list_g[QUBIT_INDEX].T[1]
        I_e = iq_list_e[QUBIT_INDEX].T[0]
        Q_e = iq_list_e[QUBIT_INDEX].T[1]

    #####################################
    # ----- Saves data to a file ----- #
    #####################################

    prefix = str(datetime.date.today())
    exp_name = save_name + '_Q' + str(QUBIT_INDEX) + '_' + prefix
    print('Experiment name: ' + exp_name)

    data_path = scan_data_path

    fname = get_next_filename(data_path, exp_name, suffix='.h5')
    print('Current data file: ' + fname)

    with SlabFile(data_path + '\\' + fname, 'a') as f:
        # 'a': read/write/create

        # - Adds data to the file - #
        f.append('I_g', I_g)
        f.append('Q_g', Q_g)
        f.append('I_e', I_e)
        f.append('Q_e', Q_e)

        # formats config into file as a single line
        f.attrs['config'] = json.dumps(config)

    data = data_path + '\\' + fname

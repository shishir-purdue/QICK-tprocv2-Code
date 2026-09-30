"""
SingleShot - perform single shot measurements for g-e calibration

This experiment is designed to perform single shot measurements. 

Make sure to set the 'SS_ONLY' variable to 'True' in the configuration file to run this experiment alone.

MUX and non-MUX versions are setup and work the same for readout. The only difference
is in the data processing. The MUX version uses the QUBIT_INDEX to index the
data, while the non-MUX version does not. TODO: MUX version data processing.

Author: Santi
Date: 2025-05-09
"""

import os
folder = os.getcwd()
if 'qick_tprocv2_experiments' not in folder:
    os.chdir(folder + '/qick_tprocv2_experiments')

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

# Used for live plotting, need to run "python -m visdom.server" in the terminal and open the IP address in browser
import visdom

from build_task import *
from build_state import *
from qick_setup_funcs import *

from qualang_tools.plot import Fit
import pprint as pp

# connect to the rfsoc and print soccfg
from rfsoc_connect import *


# ----- Experiment configurations ----- #
expt_name = "IQ_plot"
config, expt_name = initialize_configs(expt_name)
print(expt_name + '\n')
print(config)

##################
# Define Program #
##################

# ----- single shot g experiment ----- #
class SingleShotProgram_g(AveragerProgramV2):
    def _initialize(self, cfg):
        # initialize the readout channels and pulses for DAC and ADC
        initialize_ro_chs(self, cfg, suffix='_ge')

        self.add_loop("shotloop", cfg["steps"]) # number of total shots

    def _body(self, cfg):
        self.delay_auto(0.01, tag='wait')
        # readout using proper configurations for MUX and non-MUX
        readout(self, cfg)


# ----- single shot e experiment ----- #
class SingleShotProgram_e(AveragerProgramV2):
    def _initialize(self, cfg):
        # initialize the readout channels and pulses for DAC and ADC
        initialize_ro_chs(self, cfg, suffix='_ge')

        self.add_loop("shotloop", cfg["steps"]) # number of total shots
        if cfg['half_pi'] == 'True':
            qubit_pulse = 'X90'
        else:
            qubit_pulse = 'qubit_pulse'
        # initialize the qubit generator channel
        declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')
        # initialize qubit pulse
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit', 
                      pulse_style='gauss', pulse_name=qubit_pulse, suffix='_ge')

    def _body(self, cfg):
        if cfg['half_pi'] == 'True':
            qubit_pulse = 'X90'
        else:
            qubit_pulse = 'qubit_pulse'
        self.pulse(ch=self.cfg["qubit_ch"], name=qubit_pulse, t=0)  #play pulse
        self.delay_auto(0.01)
        if cfg['half_pi'] == 'True':
            self.pulse(ch=self.cfg["qubit_ch"], name=qubit_pulse, t=0)  #play pulse
            self.delay_auto(0.01)
        # readout using proper configurations for MUX and non-MUX
        readout(self, cfg)

###################
# Run the Program
###################
if config['SS_ONLY'] == 'True':
    # ----- Which qubits to run, and where the data goes ----- #
    # meas_list = 'True' in system_config.json runs every qubit in qubit_list one
    # after another into a numbered folder; otherwise just the configured qubit,
    # into DATA_PATH, as before.
    qubit_list, meas_data_path = setup_meas_list(config, expt_name)
    base_expt_name = expt_name

    for QUBIT_INDEX in qubit_list:
        # QUBIT_INDEX is the qubit being measured, numbered from 1. With meas_list
        # off the list holds just the configured qubit and this runs once.
        if len(qubit_list) > 1:
            print(f"\n=== {base_expt_name}: qubit {QUBIT_INDEX} ===")
        config, expt_name = initialize_configs(base_expt_name, qubit_index=QUBIT_INDEX)

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

        data_path = meas_data_path

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
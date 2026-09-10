"""
006b_ramsey_ro_length_ge - Ramsey vs Readout Duration for ge state

This runs the Ramsey experiment once per readout duration, stepping through
the hardcoded RO_LENGTHS list below at a fixed Ramsey frequency. res_length_ge
is kept RES_LENGTH_OFFSET above ro_length on every pass. Neither length can be
a QickParam - the ADC acquisition window and the readout pulse are both fixed
when the program is compiled - so the sweep is a Python loop that rebuilds the
program each time. The timing sweep within each run is still a QickParam.

Configurations are taken from the Ramsey_ge block; this experiment has none of
its own. Every run of the script writes into its own numbered subfolder of
DATA_PATH, which is what analyze_ramsey_ro_length reads.

We also have the option to perform single shot measurements for g-e calibration
after the experiment. This is done by setting the SS variable to True in the 
system_config.json file. Your pi pulse should be calibrated before running SS.

MUX and non-MUX versions are setup and work the same for readout. The only difference
is in the data processing. The MUX version uses the QUBIT_INDEX to index the
data, while the non-MUX version does not.

Live plotting is setup.

Author: Shishir
Date: 2026-09-10
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

# ----- Experiment configurations ----- #
expt_name = "Ramsey_ge"            # where the configurations come from
save_name = "ramsey_ro_length_ge"  # what the files and the folder are called
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

##################
# Define Program #
##################
## Add the option to this with square pulses.
class RamseyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        # initialize the readout channels and pulses for DAC and ADC
        initialize_ro_chs(self, cfg, suffix='_ge')

        self.add_loop("waitloop", cfg["steps"])
        
       # initialize the qubit generator channel
        declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')
        # initialize qubit pulse
        
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit', 
                      pulse_style='gauss', pulse_name='ramsey1', suffix='_ge') 
    
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit', 
                      pulse_style='gauss', pulse_name='ramsey2', suffix='_ge') 
        
    
    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="ramsey1", t=0)  #play probe pulse
        
        self.delay_auto(cfg['wait_time']+0.01, tag='wait') # wait_time after last pulse
        
        self.pulse(ch=self.cfg["qubit_ch"], name="ramsey2", t=0)  #play pulse
        self.delay_auto(0.01) # wait_time after last pulse
        
        # readout using proper configurations for MUX and non-MUX
        readout(self, cfg)

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
    print(f"=== Readout scan {ro_i + 1}/{len(RO_LENGTHS)}: "
          f"ro_length = {ro_length} us, res_length_ge = {res_length} us ===")

    # Refresh the config each pass. The save block below deletes 'wait_time'
    # (a QickSweep1D, which cannot be JSON-serialised), so reusing the same
    # dict makes the second pass fail.
    config, expt_name = initialize_configs(base_expt_name)

    # Both lengths have to be set BEFORE the program is built: ro_length sizes
    # the ADC window in initialize_ro_chs and res_length_ge sets the readout
    # pulse in declare_pulse, both while the program is compiled.
    config.update([("ro_length", ro_length), ("res_length_ge", res_length)])

    ramsey=RamseyProgram(soccfg, reps=config['reps'], final_delay=config['relax_delay'], cfg=config)
    py_avg = config['py_avg']

    # for live plotting
    IS_VISDOM = False # has bugs
    if IS_VISDOM:
        expt_I = expt_Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"
        viz.close(win=None) # close previous plots
        win1 = viz.line( X=np.arange(0, 1), Y=np.arange(0, 1),
            opts=dict(height=400, width=700, title='Ramsey Frequency Calibration', showlegend=True, xlabel='expt_pts'))

        for ii in range(py_avg):
            iq_list = ramsey.acquire(soc, soft_avgs=1, progress=False)
            delay_times = ramsey.get_time_param('wait', "t", as_array=True)
            delay_times -= delay_times[0]
        
            if MUX == 'False':
                iq_list = iq_list[0][0].T
                # what is the correct shape/index?
                this_I = (iq_list[0])
                this_Q = (iq_list[1])
            else:
                this_I = (iq_list[QUBIT_INDEX].T[0])
                this_Q = (iq_list[QUBIT_INDEX].T[1])
            
            if expt_I is None: # ii == 0
                expt_I, expt_Q = this_I, this_Q
            else:
                expt_I = (expt_I * ii + this_I) / (ii + 1.0)
                expt_Q = (expt_Q * ii + this_Q) / (ii + 1.0)

            expt_mags = np.abs(expt_I + 1j * expt_Q)  # magnitude
            expt_phases = np.angle(expt_I + 1j * expt_Q)  # phase

            viz.line(X = delay_times, Y = expt_mags, win=win1, name='I',
            opts=dict(height=400, width=700, title='Ramsey Frequency Calibration', showlegend=True, xlabel='expt_pts'))


        amps = expt_mags
    
    else:
        iq_list = ramsey.acquire(soc, soft_avgs=py_avg, progress=True)
        delay_times = ramsey.get_time_param('wait', "t", as_array=True)
        amps = np.abs(iq_list[0][0].dot([1,1j]))

    # ### Fit ###
    # fit = Fit()

    # # Choose the suitable fitting function
    # fit_result = fit.ramsey(delay_times, amps)

    # fit_result = {
    #     "f": fit_result['f'],
    #     "phase": fit_result['phase'],
    #     "T2": fit_result['T2'],
    #     "amp": fit_result['amp'],
    #     "initial_offset": fit_result['initial_offset'],
    #     "final_offset": fit_result['final_offset']
    #     }

    # pp.pprint(fit_result)

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
        f.append('delay_times', delay_times)
        f.append('amps', amps)
        if IS_VISDOM:
            f.append('avgi', expt_I)
            f.append('avgq', expt_Q)
        else:
            f.append('avgi', iq_list[0][0].T[0])
            f.append('avgq', iq_list[0][0].T[1])

        del config['wait_time']  # cant save QickParam
        # formats config into file as a single line
        f.attrs['config'] = json.dumps(config)
        # f.attrs['fit_result'] = json.dumps(fit_result)

        if SS == 'True':
            # - Adds ss data to the file - #
            f.append('I_g', I_g)
            f.append('Q_g', Q_g)
            f.append('I_e', I_e)
            f.append('Q_e', Q_e)    

            # formats config into file as a single line
            f.attrs['ss_config'] = json.dumps(ss_config)

    data = data_path + '\\' + fname

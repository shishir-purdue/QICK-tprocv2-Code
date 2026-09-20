"""
005b_repeat_X180_pulse_ge - Repeated X180 Pulse Calibration for ge state

This experiment sweeps the pi-pulse gain while playing the pi pulse 2N+1 times
before each readout, so that a small amplitude error accumulates into a
2N+1 times larger rotation error and becomes easy to see. N is read from the
system_config.json block for this experiment; the gain sweep is done using a
QickParam object.

We also have the option to perform single shot measurements for g-e calibration
after the experiment. This is done by setting the SS variable to True in the 
system_config.json file. Your pi pulse should be calibrated before running SS.

MUX and non-MUX versions are setup and work the same for readout. The only difference
is in the data processing. The MUX version uses the QUBIT_INDEX to index the
data, while the non-MUX version does not.

Live plotting is setup.

Author: Shishir
Date: 2026-09-02
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
expt_name = "repeat_X180_pulse_ge"
config, expt_name = initialize_configs(expt_name)
print(expt_name + '\n')
print(config)

##################
# Define Program #
##################

class RepeatPulseProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        # initialize the readout channels and pulses for DAC and ADC
        initialize_ro_chs(self, cfg, suffix='_ge')

        self.add_loop("gainloop", cfg["steps"])
        
        # initialize the qubit generator channel
        declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')
        # initialize qubit pulse
        
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit', 
                      pulse_style='gauss', pulse_name='qubit_pulse', suffix='_ge')

    
    def _body(self, cfg):
        for n in range(2 * (cfg['N']) + 1):
            self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  #play probe pulse
            self.delay_auto(t=0.01)
        # readout using proper configurations for MUX and non-MUX
        readout(self, cfg)

###################
# Run the Program
###################

# Hardcoded upper limit for now - the loop runs N = 0 ... N_MAX - 1
N_MAX = 5

# Keep the name we looked the config up with: initialize_configs appends
# '_mux' to what it is handed, so feeding expt_name back in would grow a
# second suffix on the next pass.
base_expt_name = expt_name

for n in range(N_MAX):
    print("")
    print(f"=== Repeat X180 run {n + 1}/{N_MAX}: N = {n} ===")

    # Refresh the config each pass. The save block below deletes
    # 'qubit_gain_ge' (a QickSweep1D, which cannot be JSON-serialised), so
    # reusing the same dict makes declare_pulse fail on the second pass.
    config, expt_name = initialize_configs(base_expt_name)

    # N has to be set BEFORE the program is built: _body reads cfg['N'] while
    # the program is compiled on the next line, not when it is acquired.
    config.update([("N", n)])

    repeat_X180_program=RepeatPulseProgram(soccfg, reps=config['reps'], final_delay=config['relax_delay'], cfg=config)
    py_avg = config['py_avg']

    # for live plotting
    IS_VISDOM = False
    if IS_VISDOM:
        expt_I = expt_Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"
        viz.close(win=None) # close previous plots
        win1 = viz.line( X=np.arange(0, 1), Y=np.arange(0, 1),
            opts=dict(height=400, width=700, title='Repeat X180 Pulse Experiment', showlegend=True, xlabel='expt_pts'))

        for ii in range(py_avg):
            iq_list = repeat_X180_program.acquire(soc, soft_avgs=1, progress=False)
            gains = repeat_X180_program.get_pulse_param('qubit_pulse', "gain", as_array=True)

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

            viz.line(X = gains, Y = expt_mags, win=win1, name='I',
            opts=dict(height=400, width=700, title='Repeat X180 Pulse Experiment', showlegend=True, xlabel='expt_pts'))

        amps = np.abs(expt_I + 1j*expt_Q)
        
    else:
        iq_list = np.array(repeat_X180_program.acquire(soc, soft_avgs=py_avg, progress=True))
        gains = repeat_X180_program.get_pulse_param('qubit_pulse', "gain", as_array=True)
        amps = np.abs(iq_list[0][0].dot([1,1j]))

    # fit = Fit()
    # # Choose the suitable fitting function - currently bug for rabi - fixed in analysis
    # fit_result = fit.rabi(gains, amps)

    # fit_result = {
    #         "f": fit_result['f'],
    #         "phase": fit_result['phase'],
    #         "T": fit_result['T'],
    #         "amp": fit_result['amp'],
    #         "offset": fit_result['offset']
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

    data_path = DATA_PATH

    fname = get_next_filename(data_path, exp_name, suffix='.h5')
    print('Current data file: ' + fname)

    with SlabFile(data_path + '\\' + fname, 'a') as f:
        # 'a': read/write/create

        # - Adds data to the file - #
        f.append('gains', gains)
        f.append('amps', amps)
        if IS_VISDOM:
            f.append('avgi', expt_I)
            f.append('avgq', expt_Q)
        else:
            f.append('avgi', iq_list[0][0].T[0])
            f.append('avgq', iq_list[0][0].T[1])

        del config['qubit_gain_ge']  # cant save QickParam
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

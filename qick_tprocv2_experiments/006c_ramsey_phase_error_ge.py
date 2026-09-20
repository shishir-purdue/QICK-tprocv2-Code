"""
006c_ramsey_phase_error_ge - Ramsey Phase Error Amplification for ge state

A Ramsey experiment with N identity blocks inserted after the first pi/2
pulse. Each identity is an X90 followed by an X90 whose phase is shifted by
180 degrees (the 'X90_neg' pulse in qick_setup_funcs), which undoes the first
exactly only if the two really are 180 degrees apart. Any phase error between
them survives as a small azimuthal rotation of the state on the equator, and
N identities stack it N times, so the Ramsey fringe phase moves linearly with
N. N = 0 is the plain Ramsey experiment; N = 1, 2, 3 add 2, 4, 6 extra pulses.

The delay sweep covers just one fringe period, sized from ramsey_freq the way
006a does. Configurations are taken from the Ramsey_ge block; this experiment
has none of its own, and N is set by the loop below. Every run of the script
writes into its own numbered subfolder of DATA_PATH, one file per N, which is
what analyze_ramsey_phase_error reads.

Author: Shishir
Date: 2026-09-17
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
expt_name = "Ramsey_ge"                # where the configurations come from
save_name = "ramsey_phase_error_ge"    # what the files and the folder are called
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
class RamseyPhaseErrorProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        # initialize the readout channels and pulses for DAC and ADC
        initialize_ro_chs(self, cfg, suffix='_ge')

        self.add_loop("waitloop", cfg["steps"])

        # initialize the qubit generator channel
        declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')

        # the two Ramsey pi/2 pulses, the second with the virtual phase advance
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='ramsey1', suffix='_ge')
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='ramsey2', suffix='_ge')

        # the identity pair: X90, then X90 with its phase shifted by 180 deg
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='X90', suffix='_ge')
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='X90_neg', suffix='_ge')

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="ramsey1", t=0)  # first pi/2

        # N identities on the equatorial state. With N = 0 nothing is played
        # here and the sequence is exactly 006_Ramsey_ge.py.
        for n in range(cfg['N']):
            self.delay_auto(0.01)
            self.pulse(ch=self.cfg["qubit_ch"], name="X90", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=self.cfg["qubit_ch"], name="X90_neg", t=0)

        self.delay_auto(cfg['wait_time'], tag='wait')  # wait_time after last pulse

        self.pulse(ch=self.cfg["qubit_ch"], name="ramsey2", t=0)  # second pi/2
        self.delay_auto(0.025)

        # readout using proper configurations for MUX and non-MUX
        readout(self, cfg)

###################
# Run the Program
###################

# Hardcoded upper limit for now - the loop runs N = 0 ... N_MAX - 1 identities,
# i.e. 0, 2, 4, ... extra pulses.
N_MAX = 4

# The delay sweep covers N_PERIODS fringes at TARGET_STEP_NS between points,
# sized from ramsey_freq as in 006a. Just the first period here.
N_PERIODS = 1
TARGET_STEP_NS = 5.0
MAX_STEP_NS = 20.0

# Keep the name we looked the config up with: initialize_configs appends
# '_mux' to what it is handed, so feeding expt_name back in would grow a
# second suffix on the next pass.
base_expt_name = expt_name

for n in range(N_MAX):
    print("")
    print(f"=== Ramsey phase error run {n + 1}/{N_MAX}: N = {n} "
          f"({2 * n} extra pulses) ===")

    # Refresh the config each pass. The save block below deletes 'wait_time'
    # (a QickSweep1D, which cannot be JSON-serialised), so reusing the same
    # dict makes the second pass fail.
    config, expt_name = initialize_configs(base_expt_name)

    # N has to be set BEFORE the program is built: _body reads cfg['N'] while
    # the program is compiled, not when it is acquired.
    config.update([("N", n)])

    # ----- Delay sweep: one fringe period ----- #
    # initialize_configs has already built wait_time out of the config's own
    # start/stop, so the sweep object has to be rebuilt here or the new
    # limits are ignored.
    ramsey_freq = config['ramsey_freq']
    start_us = 0.0
    stop_us = N_PERIODS / ramsey_freq
    steps = int(np.ceil(stop_us * 1e3 / TARGET_STEP_NS)) + 1
    step_ns = stop_us * 1e3 / (steps - 1)
    if step_ns > MAX_STEP_NS:
        raise ValueError(
            f"delay resolution came out at {step_ns:.2f} ns, coarser than the "
            f"{MAX_STEP_NS} ns ceiling")

    config.update([("start", start_us), ("stop", stop_us), ("steps", steps),
                   ("wait_time", QickSweep1D('waitloop', start_us, stop_us))])
    print(f"    delay 0 -> {stop_us * 1e3:.0f} ns over {steps} points "
          f"({step_ns:.2f} ns resolution, {N_PERIODS} period)")

    ramsey = RamseyPhaseErrorProgram(soccfg, reps=config['reps'], final_delay=config['relax_delay'], cfg=config)
    py_avg = config['py_avg']

    # for live plotting
    IS_VISDOM = False # has bugs
    if IS_VISDOM:
        expt_I = expt_Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"
        viz.close(win=None) # close previous plots
        win1 = viz.line( X=np.arange(0, 1), Y=np.arange(0, 1),
            opts=dict(height=400, width=700, title='Ramsey Phase Error', showlegend=True, xlabel='expt_pts'))

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
            opts=dict(height=400, width=700, title='Ramsey Phase Error', showlegend=True, xlabel='expt_pts'))

        amps = expt_mags

    else:
        iq_list = ramsey.acquire(soc, soft_avgs=py_avg, progress=True)
        delay_times = ramsey.get_time_param('wait', "t", as_array=True)
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

        if SS == 'True':
            # - Adds ss data to the file - #
            f.append('I_g', I_g)
            f.append('Q_g', Q_g)
            f.append('I_e', I_e)
            f.append('Q_e', Q_e)

            # formats config into file as a single line
            f.attrs['ss_config'] = json.dumps(ss_config)

    data = data_path + '\\' + fname

"""
005c_drag_alpha_ge - DRAG alpha calibration of the X90 pulse for ge state

Plays N identity pairs (X90_drag, then X90_drag_neg: the same pulse 180 deg
round) from the ground state and measures, sweeping the DRAG coefficient
alpha. Ideally every pair is an identity and the qubit stays in g for any N.
With the wrong alpha each pulse picks up a small phase error that the pair does
not cancel, so N pairs rotate the qubit by N times that error and the excited
population goes as 1 - cos(N * error). Every N has its minimum at the right
alpha, and larger N make the dip narrower - the curves for all N in N_list
touching zero at one alpha is the calibration.

The envelope comes from declare_pulse('X90_drag'): QICK's add_DRAG with the
qubit's sigma_ge, half_pi_gain_ge, alpha_ge and anharmonicity_ge from
system_config.json. alpha is baked into the envelope, so it cannot be a QICK
sweep: the program is rebuilt for every alpha. The sweep (start, stop, steps)
and N_list are in the drag_alpha_ge block of system_config.json.

Each qubit is saved as one file: 'alphas', 'N_list', and 'avgi' / 'avgq' /
'amps' with one row per N. meas_list = 'True' runs every qubit in qubit_list
into a numbered folder, as the other experiments do.

We also have the option to perform single shot measurements for g-e calibration
after the experiment. This is done by setting the SS variable to True in the
system_config.json file. Your pi pulse should be calibrated before running SS.

Author: Shishir
Date: 2026-09-30
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

# connect to the rfsoc and print soccfg
from rfsoc_connect import *

if SS == 'True':
    # import single shot information for g-e calibration
    from SingleShot import SingleShotProgram_g, SingleShotProgram_e

# ----- Experiment configurations ----- #
expt_name = "drag_alpha_ge"
config, expt_name = initialize_configs(expt_name)
print(expt_name + '\n')
print(config)

# ----- Which qubits to run, and where the data goes ----- #
# meas_list = 'True' in system_config.json runs every qubit in qubit_list one
# after another into a numbered folder; otherwise just the configured qubit,
# into DATA_PATH, as before.
qubit_list, meas_data_path = setup_meas_list(config, expt_name)
base_expt_name = expt_name

##################
# Define Program #
##################
class DragAlphaProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        # initialize the readout channels and pulses for DAC and ADC
        initialize_ro_chs(self, cfg, suffix='_ge')

        # initialize the qubit generator channel
        declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')

        # the identity pair, both with the DRAG envelope at cfg['alpha_ge']
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='X90_drag', suffix='_ge')
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='X90_drag_neg', suffix='_ge')

    def _body(self, cfg):
        # N identities, spaced as in 006c_ramsey_phase_error_ge.py
        for n in range(cfg['N']):
            self.pulse(ch=self.cfg["qubit_ch"], name="X90_drag", t=0)
            self.delay_auto(0.01)
            self.pulse(ch=self.cfg["qubit_ch"], name="X90_drag_neg", t=0)
            self.delay_auto(0.01)

        # readout using proper configurations for MUX and non-MUX
        readout(self, cfg)

###################
# Run the Program
###################
for QUBIT_INDEX in qubit_list:
    # QUBIT_INDEX is the qubit being measured, numbered from 1. With meas_list
    # off the list holds just the configured qubit and this runs once.
    if len(qubit_list) > 1:
        print(f"\n=== {base_expt_name}: qubit {QUBIT_INDEX} ===")
    config, expt_name = initialize_configs(base_expt_name, qubit_index=QUBIT_INDEX)
    if SS == 'True':
        # single-shot calibration configs for this qubit
        ss_config, _ = initialize_configs("IQ_plot", qubit_index=QUBIT_INDEX)

    alpha_configured = config['alpha_ge']
    alphas = np.linspace(config['start'], config['stop'], config['steps'])
    N_list = list(config['N_list'])
    ro_index = 0 if MUX == 'False' else QUBIT_INDEX - 1

    avgi = np.zeros((len(N_list), len(alphas)))
    avgq = np.zeros((len(N_list), len(alphas)))
    for i, N in enumerate(N_list):
        print(f"  N = {N}: {len(alphas)} alphas from {alphas[0]:g} to {alphas[-1]:g}")
        for j, alpha in enumerate(alphas):
            # alpha is part of the envelope, so the program is rebuilt for each
            config.update([('alpha_ge', float(alpha)), ('N', N)])
            prog = DragAlphaProgram(soccfg, reps=config['reps'],
                                    final_delay=config['relax_delay'], cfg=config)
            iq_list = prog.acquire(soc, soft_avgs=config['py_avg'], progress=False)
            avgi[i, j], avgq[i, j] = np.array(iq_list[ro_index]).reshape(-1, 2)[0]
    amps = np.abs(avgi + 1j * avgq)

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

    data_path = meas_data_path

    fname = get_next_filename(data_path, exp_name, suffix='.h5')
    print('Current data file: ' + fname)

    with SlabFile(data_path + '\\' + fname, 'a') as f:
        # 'a': read/write/create

        # - Adds data to the file - #
        f.append('alphas', alphas)
        f.append('N_list', np.array(N_list))
        f.append('avgi', avgi)      # one row per N
        f.append('avgq', avgq)
        f.append('amps', amps)

        # the config as configured, not as left by the sweep
        config.update([('alpha_ge', alpha_configured)])
        del config['N']
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

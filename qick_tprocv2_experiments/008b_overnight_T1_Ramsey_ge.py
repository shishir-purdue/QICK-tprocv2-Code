"""
008b_overnight_T1_Ramsey_ge - Overnight Ramsey + T1 for every qubit

Cycles through the qubits in qubit_list one at a time: for each qubit one
Ramsey (Ramsey_ge block of system_config.json), then one T1 (T1_ge block),
then on to the next qubit. Rounds repeat until run_hours have passed; the
round in progress is always finished, so every qubit ends with the same
number of measurements. meas_list is not used - this script sets the qubit
index itself.

All files go into one subfolder of DATA_PATH\\overnight, named after the
start time, e.g. overnight\\2026-09-30_2215_T1_Ramsey. Each file is named as
usual (00003_T1_ge_Q5_2026-09-30.h5) and also stores its start time
('timestamp', ISO string, and 'unix_time') plus 'round' in its attrs.

system_config.json is re-read before every measurement, so edits made
during the night are picked up by the next measurement.

A measurement that fails (e.g. a dropped connection) is printed and skipped,
and the run carries on with the next one.

No live plotting, so no visdom server is needed.
"""

import os
os.chdir(os.getcwd() + '/qick_tprocv2_experiments')

import time
import datetime
import json
import traceback

from qick import *
from malab import SlabFile, get_next_filename
from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D

from build_task import *
from build_state import *
from qick_setup_funcs import *

# connect to the rfsoc and print soccfg
from rfsoc_connect import *

if SS == 'True':
    # import single shot information for g-e calibration
    from SingleShot import SingleShotProgram_g, SingleShotProgram_e

# ----- Overnight settings ----- #
run_hours = 8.0
qubit_list = list(range(1, NUM_QUBITS + 1))   # all qubits, 1-based

start = datetime.datetime.now()
overnight_data_path = os.path.join(DATA_PATH, 'overnight',
                                   start.strftime('%Y-%m-%d_%H%M') + '_T1_Ramsey')
os.makedirs(overnight_data_path, exist_ok=True)
print('Overnight data folder: ' + overnight_data_path)

##################
# Define Programs #
##################
class RamseyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        initialize_ro_chs(self, cfg, suffix='_ge')
        self.add_loop("waitloop", cfg["steps"])
        declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')
        # detuned vs. phase-advanced ramsey2 is chosen inside declare_pulse
        # from cfg['detuned_pulse']
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='ramsey1', suffix='_ge')
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='ramsey2', suffix='_ge')

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="ramsey1", t=0)
        self.delay_auto(cfg['wait_time'], tag='wait')
        self.pulse(ch=self.cfg["qubit_ch"], name="ramsey2", t=0)
        self.delay_auto(0.025)
        readout(self, cfg)


class T1Program(AveragerProgramV2):
    def _initialize(self, cfg):
        initialize_ro_chs(self, cfg, suffix='_ge')
        self.add_loop("waitloop", cfg["steps"])
        declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')
        declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                      pulse_style='gauss', pulse_name='qubit_pulse', suffix='_ge')

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(cfg['wait_time']+0.01, tag='wait')
        readout(self, cfg)


PROGRAMS = {'Ramsey_ge': RamseyProgram, 'T1_ge': T1Program}


def iq_for_qubit(iq_list, qubit_index):
    '''I and Q arrays of this qubit's readout, for MUX and non-MUX.'''
    if MUX == 'False':
        return iq_list[0][0].T[0], iq_list[0][0].T[1]
    return iq_list[qubit_index - 1].T[0], iq_list[qubit_index - 1].T[1]


def measure(base_expt_name, qubit_index, round_i):
    '''Run one experiment on one qubit and save it; returns the file path.'''
    stamp = datetime.datetime.now()
    config, expt_name = initialize_configs(base_expt_name, qubit_index=qubit_index)

    prog = PROGRAMS[base_expt_name](soccfg, reps=config['reps'],
                                    final_delay=config['relax_delay'], cfg=config)
    # average py_avg single passes, as the live-plotting experiments do
    expt_I = expt_Q = None
    for ii in range(config['py_avg']):
        this_I, this_Q = iq_for_qubit(prog.acquire(soc, soft_avgs=1, progress=False), qubit_index)
        if expt_I is None:
            expt_I, expt_Q = this_I, this_Q
        else:
            expt_I = (expt_I * ii + this_I) / (ii + 1.0)
            expt_Q = (expt_Q * ii + this_Q) / (ii + 1.0)
    delay_times = prog.get_time_param('wait', "t", as_array=True)
    delay_times -= delay_times[0]
    amps = np.abs(expt_I + 1j * expt_Q)

    if SS == 'True':
        ss_config, _ = initialize_configs("IQ_plot", qubit_index=qubit_index)
        iq_g = SingleShotProgram_g(soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                   cfg=ss_config).acquire(soc, soft_avgs=1, progress=False)
        iq_e = SingleShotProgram_e(soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                   cfg=ss_config).acquire(soc, soft_avgs=1, progress=False)
        I_g, Q_g = iq_for_qubit(iq_g, qubit_index)
        I_e, Q_e = iq_for_qubit(iq_e, qubit_index)

    # ----- save ----- #
    exp_name = expt_name + '_Q' + str(qubit_index) + '_' + str(stamp.date())
    fname = get_next_filename(overnight_data_path, exp_name, suffix='.h5')
    with SlabFile(overnight_data_path + '\\' + fname, 'a') as f:
        f.append('delay_times', delay_times)
        f.append('amps', amps)
        f.append('avgi', expt_I)
        f.append('avgq', expt_Q)

        del config['wait_time']  # cant save QickParam
        f.attrs['config'] = json.dumps(config)
        f.attrs['timestamp'] = stamp.isoformat(timespec='seconds')
        f.attrs['unix_time'] = stamp.timestamp()
        f.attrs['round'] = round_i

        if SS == 'True':
            f.append('I_g', I_g)
            f.append('Q_g', Q_g)
            f.append('I_e', I_e)
            f.append('Q_e', Q_e)
            f.attrs['ss_config'] = json.dumps(ss_config)

    return overnight_data_path + '\\' + fname


###################
# Run the Program
###################
end = start + datetime.timedelta(hours=run_hours)
print(f'Running Ramsey + T1 on qubits {qubit_list} until {end:%Y-%m-%d %H:%M}')

round_i = 0
while datetime.datetime.now() < end:
    print(f'\n=== Round {round_i} ({datetime.datetime.now():%H:%M:%S}) ===')
    for QUBIT_INDEX in qubit_list:
        for base_expt_name in ('Ramsey_ge', 'T1_ge'):
            t0 = time.time()
            try:
                data = measure(base_expt_name, QUBIT_INDEX, round_i)
                print(f'  Q{QUBIT_INDEX} {base_expt_name:9s} {time.time() - t0:5.0f} s  '
                      f'{os.path.basename(data)}')
            except Exception:
                print(f'  Q{QUBIT_INDEX} {base_expt_name} FAILED, skipping:')
                traceback.print_exc()
    round_i += 1

print(f'\nDone: {round_i} rounds, finished {datetime.datetime.now():%Y-%m-%d %H:%M}')

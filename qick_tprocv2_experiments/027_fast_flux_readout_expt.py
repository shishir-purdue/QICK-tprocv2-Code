"""
027_fast_flux_readout_expt - a readout pulse at a fixed delay from the
rising (left) edge of a fixed fast flux pulse.

One measurement, nothing swept: the RFSoC triggers the HDAWG, which plays a
square flux pulse of fixed length and gain, and the readout starts
readout_delay after the flux pulse's rising edge. readout_delay can put the
readout inside the flux pulse or after it.

    Trigger  t = 0                               (RFSoC pin hdawg_trigger_pin)
    Flux     t = trig_offset                     length ff_length, gain ff_gain
    Readout  t = trig_offset + readout_delay     (inside the flux pulse if readout_delay < ff_length)
   [Qubit    t = trig_offset + q_delay           commented out below, see QUBIT PULSE]

The flux pulse is followed, ff_compensation_wait later, by the same pulse
inverted (compensation), as in 022 and 025.

All the timing and the flux gain come from the "fast_flux_readout" block of
system_config.json (times in us):
    ff_length             flux pulse length
    ff_gain               flux pulse amplitude, fraction of the HDAWG output range
    readout_delay         rising edge of the flux pulse to the start of the readout
    q_delay               rising edge of the flux pulse to the qubit pulse (only with the qubit pulse in)
    trig_offset           RFSoC trigger to the rising edge of the flux pulse
    ff_compensation_wait  end of the flux pulse to the compensation pulse
    max_ff_gain           hard limit on |ff_gain|
    hdawg_channel         HDAWG output, counted from 0 (output 1 -> 0)
    hdawg_trigger_pin     RFSoC trigger pin wired to the HDAWG trigger input
    hdawg_sample_rate     HDAWG sample clock, set and read back
    hdawg_output_range    HDAWG output range, V
    reps, py_avg, relax_delay   averaging and wait between shots

The flux pulse is rounded to 16 HDAWG samples; the rounded length is saved.
The HDAWG setup is in hdawg_helpers.py.

The averaged I and Q are saved with every parameter above in one file in
DATA_PATH.
"""

import os
os.chdir(os.getcwd() + '/qick_tprocv2_experiments')

from qick import *
from malab import SlabFile, get_next_filename

# the main program class
from qick.asm_v2 import AveragerProgramV2

import json
import datetime
import time
import numpy as np

from build_task import *
from build_state import *
from qick_setup_funcs import *

# connect to the rfsoc and print soccfg
from rfsoc_connect import *

from hdawg_helpers import (connect_hdawg, reset_and_configure, set_sample_clock, flux_samples,
                           load_flux_pulse, set_flux_gain, start_flux, stop_flux, check_trigger_spacing)

# ----- Experiment configurations ----- #
expt_name = "fast_flux_readout"
config, expt_name = initialize_configs(expt_name)
print(expt_name + '\n')
print(config)

# the flux pulse the HDAWG actually plays: a whole number of 16-sample blocks
ff_samples = flux_samples(config['ff_length'], config['hdawg_sample_rate'])
ff_length = ff_samples / config['hdawg_sample_rate'] * 1e6     # us
readout_delay = config['readout_delay']
trig_offset = config['trig_offset']

if readout_delay < 0:
    raise ValueError(f"readout_delay = {readout_delay} us: the readout must start after the flux pulse's rising edge")
check_trigger_spacing(ff_length, config['ff_compensation_wait'], trig_offset,
                      shot_period_us=trig_offset + readout_delay + config['ro_length'] + config['relax_delay'])
where = "inside" if readout_delay < ff_length else "after"
print(f"flux pulse {ff_length:.6f} us (asked {config['ff_length']} us), gain {config['ff_gain']}; "
      f"readout {readout_delay} us after the rising edge ({where} the flux pulse)")


##################
# Define Program #
##################
class FastFluxReadoutProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        # initialize the readout channels and pulses for DAC and ADC
        initialize_ro_chs(self, cfg, suffix='_ge')

        # ----- QUBIT PULSE (option): uncomment to declare the qubit pulse ----- #
        # declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')
        # declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
        #               pulse_style='gauss', pulse_name='qubit_pulse', suffix='_ge')

    def _body(self, cfg):
        t_ro = cfg['trig_offset'] + cfg['readout_delay']   # readout start, from the trigger

        # trigger the HDAWG: the flux pulse starts trig_offset later
        self.trigger(pins=[cfg['hdawg_trigger_pin']], t=0)

        # ----- QUBIT PULSE (option): uncomment to play it q_delay after the rising edge ----- #
        # self.pulse(ch=cfg['qubit_ch'], name='qubit_pulse', t=cfg['trig_offset'] + cfg['q_delay'])

        # readout readout_delay after the rising edge of the flux pulse
        if MUX == 'True':
            self.pulse(ch=cfg['mux_ch'], name='res_pulse', t=t_ro)
            self.trigger(ros=cfg['mux_ro_chs'], pins=[0], t=cfg['trig_time'] + t_ro)
        else:
            self.send_readoutconfig(ch=cfg['ro_ch'], name='myro', t=0)
            self.pulse(ch=cfg['res_ch'], name='res_pulse', t=t_ro)
            self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'] + t_ro)


##########################
# Set up the HDAWG       #
##########################
device = connect_hdawg()
reset_and_configure(device, output_range=config['hdawg_output_range'])
set_sample_clock(device, config['hdawg_sample_rate'])
flux = load_flux_pulse(device, config['hdawg_channel'], ff_samples,
                       config['ff_compensation_wait'], config['hdawg_sample_rate'])

###################
# Run the Program #
###################
start_time = time.time()
try:
    ff_gain = set_flux_gain(flux, config['ff_gain'], max_gain=config['max_ff_gain'])
    start_flux(flux)       # the HDAWG now waits for the RFSoC triggers

    prog = FastFluxReadoutProgram(soccfg, reps=config['reps'], final_delay=config['relax_delay'], cfg=config)
    iq_list = prog.acquire(soc, soft_avgs=config['py_avg'], progress=True)
finally:
    # flux off, also after an error
    stop_flux(device, flux)

# no loops in the program: one averaged (I, Q) per readout channel
ro_index = QUBIT_INDEX - 1 if MUX == 'True' else 0
I, Q = (float(v) for v in np.ravel(iq_list[ro_index])[:2])
amp, phase = abs(I + 1j * Q), float(np.angle(I + 1j * Q))
print(f"I = {I:.4f}, Q = {Q:.4f}, |IQ| = {amp:.4f}, phase = {phase:.4f} rad")

#####################################
# ----- Saves data to a file ----- #
#####################################
prefix = str(datetime.date.today())
exp_name = expt_name + '_Q' + str(QUBIT_INDEX) + '_' + prefix
print('Experiment name: ' + exp_name)

data_path = DATA_PATH
fname = get_next_filename(data_path, exp_name, suffix='.h5')
print('Current data file: ' + fname)

with SlabFile(os.path.join(data_path, fname), 'a') as f:
    # 'a': read/write/create
    f.append('expt_time', [int(time.time() - start_time)])
    f.append('avgi', [I])
    f.append('avgq', [Q])
    f.append('amps', [amp])
    f.append('ff_length', [ff_length])               # the length played (rounded to 16 samples)
    f.append('ff_gain', [ff_gain])                   # as read back from the HDAWG
    f.append('ff_volts', [ff_gain * config['hdawg_output_range']])
    f.append('readout_delay', [readout_delay])
    f.append('trig_offset', [trig_offset])
    # with the qubit pulse in, also save its delay:
    # f.append('q_delay', [config['q_delay']])

    # formats config into file as a single line
    f.attrs['config'] = json.dumps(config)

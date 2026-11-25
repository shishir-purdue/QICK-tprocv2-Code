"""
022a_fast_flux_distortion_clock_aligned_expt - 022 with the HDAWG and QICK clocks lined up.

Same experiment as 022_fast_flux_distortion_expt (sweep over qubit pulse timing
compared to the fast flux pulse), with the timing changed so nothing drifts:

- The HDAWG sequencer clock (sample clock / 8) is set to SEQ_PER_TPROC = 2/3 of
  the tProc clock: 286.72 MHz, sample clock 2.29376 GS/s. 3 tProc cycles are
  then exactly 2 sequencer cycles (16 HDAWG samples).
- Every time entered in us (q_time, flux length, readout delay, trigger lead
  and width, trig_time) is rounded by on_grid() to a whole number of those
  common steps: TIME_GRID_US = 3 / 430.08 us = 6.9754 ns. A time on the grid
  is the same physical time on both devices: 3 tProc cycles per step on the
  QICK (tproc_cycles), 2 sequencer cycles per step on the ZI (seq_cycles).
- The shot period is a fixed delay at the end of the body, a whole number of
  cycles of the tProc clock, the sequencer clock and the 599.04 MHz DAC fabric
  clock (84 tProc cycles = 56 sequencer = 117 fabric = 195.3125 ns). 022 used
  final_delay, which QICK plays as a delay_auto: its length follows the channel
  end times, so each shot would start on a different phase of the other clocks.
- The flux pulse length in the HDAWG sequence is generated from ff_length, so
  the two cannot disagree.

Both instruments must be locked to the same 10 MHz reference. factory_reset()
puts the HDAWG back on its internal reference, so the reference source found
before the reset is restored after it and the script waits for lock.

What this does not fix: the tProc program starts on an arbitrary tProc cycle,
so each run (each q_time file) starts on its own phase of the sequencer clock
(3 possibilities, up to 2.33 ns apart) and of the DAC fabric clock (up to
1.67 ns). Within a run neither changes.

In this experiment I trigger the HDAWG using RFSoC external trigger.
The HDAWG is set to output a square pulse on CH1, and the trigger input is on CH1.
The qubit drive pulse is swept in frequency and timing while the flux and readout pulses are fixed.

TODO: This code is a work in progress and many values are still hardcoded below.

Live plotting is setup. No single shot yet.

This experiment connects to the Zurich Instruments HDAWG. The IP addresses are
defined below in the code.

Author: Santi
Date: 2025-05-09
"""

import os
os.chdir(os.getcwd() + '/qick_tprocv2_experiments')
print(os.getcwd())
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
import numpy as np

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

#######################
#### common time grid ####
#######################
from fractions import Fraction
from math import gcd

F_TPROC_MHZ = soccfg['tprocs'][0]['f_time']    # tProc timing clock, 430.08 MHz
# HDAWG sequencer clock / tProc clock. Fraction(2, 3) -> 286.72 MHz sequencer, 2.29376 GS/s.
# Fraction(1, 2) is the earlier setting -> 215.04 MHz sequencer, 1.72032 GS/s.
SEQ_PER_TPROC = Fraction(2, 3)
HDAWG_RATE_HZ = round(8 * F_TPROC_MHZ * 1e6 * SEQ_PER_TPROC.numerator / SEQ_PER_TPROC.denominator, 3)
assert HDAWG_RATE_HZ <= 2.4e9, "the HDAWG sample clock cannot go above 2.4 GS/s"

# One common grid step is the shortest time that is a whole number of cycles on
# BOTH devices. It is one physical time, counted differently on each device:
TPROC_CYCLES_PER_STEP = SEQ_PER_TPROC.denominator  # QICK side: 3 tProc cycles per step
SEQ_CYCLES_PER_STEP = SEQ_PER_TPROC.numerator      # ZI side:   2 sequencer cycles (16 samples) per step
TIME_GRID_US = TPROC_CYCLES_PER_STEP / F_TPROC_MHZ # 6.9754 ns


def on_grid(t_us, step=TIME_GRID_US):
    """Round a time in us to the nearest whole number of grid steps."""
    return float(np.round(t_us / step) * step)


def grid_steps(t_us):
    """Number of common grid steps in a time in us (after rounding)."""
    return int(np.round(t_us / TIME_GRID_US))


def tproc_cycles(t_us):
    """A time on the common grid, in QICK units (tProc cycles)."""
    return grid_steps(t_us) * TPROC_CYCLES_PER_STEP


def seq_cycles(t_us):
    """A time on the common grid, in ZI units (sequencer cycles; x8 for samples)."""
    return grid_steps(t_us) * SEQ_CYCLES_PER_STEP


def shot_grid_us(gen_chs):
    """
    Shortest time that is a whole number of cycles of the tProc clock, the HDAWG
    sequencer clock AND the DAC fabric clocks of the given generator channels.

    A pulse leaves the tProc on a tProc cycle and is picked up by the generator on
    its own fabric clock (599.04 MHz here, 39 cycles per 28 tProc cycles), so the
    pickup delay depends on where in that 28-cycle pattern the pulse falls. A shot
    period on this grid puts every shot at the same place in the pattern.
    """
    cycles = TPROC_CYCLES_PER_STEP
    for ch in np.atleast_1d(gen_chs).ravel():
        ratio = Fraction(soccfg['gens'][int(ch)]['f_fabric'] / F_TPROC_MHZ).limit_denominator(1000)
        cycles = cycles * ratio.denominator // gcd(cycles, ratio.denominator)
    return cycles / F_TPROC_MHZ


#importing some functions from toolkit
from zhinst.toolkit import Session, Waveforms, Sequence, CommandTable
from zhinst.toolkit.waveform import Wave, OutputType
import numpy as np
import time


def fast_flux_qubit_spec(q_time, ff_length, readout_delay=2):
    '''
    Perform qubit spectroscopy, save data, and return qubit frequency found by looking for maximum amplitude.

    q_time, ff_length and readout_delay are in us and must already be on the
    common time grid (see on_grid); every other time used here is put on it.
    '''
    for name, value in (('q_time', q_time), ('ff_length', ff_length), ('readout_delay', readout_delay)):
        if abs(value - on_grid(value)) > 1e-9:
            raise ValueError(f"{name} = {value} us is not on the {TIME_GRID_US * 1e3:.4f} ns grid; pass on_grid({name})")
    start_time = time.time()
    # ----- Experiment configurations ----- #
    expt_name = "qubit_spec_ge"
    config, expt_name = initialize_configs(expt_name)
    print(expt_name + '\n')
    print(config)
    print(QUBIT_INDEX)
    # the shot period goes on the grid shared by the tProc, HDAWG sequencer and DAC fabric clocks
    shot_grid = shot_grid_us([config['qubit_ch'], config['res_ch']])
    print(f"shot period grid: {shot_grid * 1e3:.4f} ns ({round(shot_grid * F_TPROC_MHZ)} tProc cycles)")
    ##################
    # Define Program #
    ##################

    class PulseProbeSpectroscopyProgram(AveragerProgramV2):
        def _initialize(self, cfg):
            # initialize the readout channels and pulses for DAC and ADC
            initialize_ro_chs(self, cfg, suffix='_ge')

            # add a loop for frequency sweep of qubit pulse frequency
            self.add_loop("freqloop", cfg["steps"])

            # initialize the qubit generator channel
            declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')
            # initialize qubit pulse
            declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit', 
                        pulse_style='gauss', pulse_name='qubit_pulse', suffix='_ge')

        def _body(self, cfg):
            # every time below is a whole number of grid steps
            trig_lead = on_grid(0.02)       # HDAWG trigger to qubit pulse at q_time = 0
            trig_width = on_grid(1)         # HDAWG trigger pulse width
            trig_time = on_grid(self.cfg['trig_time'])
            t_offset = ff_length + readout_delay + trig_lead # (ro delay, trig delay) time offset for the readout pulse (fixed)
            shift = abs(q_time) if q_time < 0 else 0.0       # negative q_time: everything but the qubit pulse moves later

            # if non-MUX then send readout configurations - useful when freq sweeping
            self.send_readoutconfig(ch=self.cfg['ro_ch'], name="myro", t=0)

            # the readout trigger moves with the readout pulse (in 022 it stayed put for q_time < 0)
            self.pulse(ch=self.cfg['res_ch'], name="res_pulse", t=t_offset + shift)
            self.trigger(ros=[self.cfg['ro_ch']], pins=[0], t=trig_time + t_offset + shift)

            if q_time < 0:
                self.trigger(pins=[1], t=shift, width=trig_width)
                self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=trig_lead)
            else:
                self.trigger(pins=[1], width=trig_width, t=0)
                self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=trig_lead + q_time)

            # Fixed shot period, in place of final_delay (a delay_auto, whose length
            # depends on the channel end times). On shot_grid, so in every shot the
            # HDAWG trigger lands on the same phase of the HDAWG sequencer clock and
            # the qubit pulse on the same phase of its DAC fabric clock.
            ro_end = trig_time + t_offset + shift + self.cfg['ro_length']
            res_end = t_offset + shift + self.cfg['res_length_ge']
            self.delay(t=on_grid(max(ro_end, res_end) + self.cfg['relax_delay'], shot_grid))

    ###################
    # Run the Program #
    ###################

    # final_delay=None: the shot period is the fixed delay at the end of _body
    qspec=PulseProbeSpectroscopyProgram(soccfg, reps=config['reps'], final_delay=None, cfg=config)
    py_avg = config['py_avg']

    # for live plotting
    IS_VISDOM = True
    if IS_VISDOM:
        expt_I = expt_Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"
        viz.close(win=None) # close previous plots
        win1 = viz.line( X=np.arange(0, 1), Y=np.arange(0, 1),
            opts=dict(height=400, width=700, title='Qubit Spectroscopy', showlegend=True, xlabel='expt_pts'))

        for ii in range(py_avg):
            iq_list = qspec.acquire(soc, soft_avgs=1, progress=False)
            freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True)

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

            viz.line(X = freqs, Y = expt_mags, win=win1, name='I',
            opts=dict(height=400, width=700, title='Qubit Spectroscopy', showlegend=True, xlabel='expt_pts'))

        amps = np.abs(expt_I + 1j*expt_Q)
        
    else:
        iq_list = qspec.acquire(soc, soft_avgs=py_avg, progress=True)
        freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True)
        amps = np.abs(iq_list[0][0].dot([1,1j]))

    # print(amps.shape)
    if MUX == 'True':
        amps = amps.T[0]
    # # ### Fit ###
    # fit = Fit()
    # # Choose the suitable fitting function
    # fit_result = fit.transmission_resonator_spectroscopy(freqs, amps)

    # # data = fit_result

    # fit_result = {
    #         "f": fit_result['f'],
    #         "kc": fit_result['kc'],
    #         "ki": fit_result['ki'],
    #         "k": fit_result['k'],
    #         "offset": fit_result['offset']
    #     }

    # pp.pprint(fit_result)

    #####################################
    # ----- Saves data to a file ----- #
    #####################################

    prefix = str(datetime.date.today())
    exp_name = expt_name + '_Q' + str(QUBIT_INDEX) + '_' + prefix
    print('Experiment name: ' + exp_name)

    data_path = SWEEP_DATA_PATH  # this sweep's own folder, made below before the sweep starts

    fname = get_next_filename(data_path, exp_name, suffix='.h5')
    print('Current data file: ' + fname)
    end_time = time.time()
    tot_time = end_time - start_time

    with SlabFile(data_path + '\\' + fname, 'a') as f:
        # 'a': read/write/create
        f.append('expt_time', [int(tot_time)])
        # - Adds data to the file - #
        f.append('fpts', freqs)
        f.append('amps', amps)
        f.append('q_time', [q_time])
        f.append('ff_length', [ff_length])
        f.append('readout_delay', [readout_delay])
        if IS_VISDOM:
            f.append('avgi', expt_I)
            f.append('avgq', expt_Q)
        else:
            f.append('avgi', iq_list[0][0].T[0])
            f.append('avgq', iq_list[0][0].T[1])

        del config['qubit_freq_ge']  # cant save QickParam
        # formats config into file as a single line
        f.attrs['config'] = json.dumps(config) # cant save configs yet with QickParams
        # f.attrs['fit_result'] = json.dumps(fit_result)

    data = data_path + '\\' + fname

    qubit_freq = freqs[np.argmax(amps)]
    return qubit_freq

#################################
### Zurich-Instruments code #####
#################################

'''
This code will play a square pulse with external trigger
The output is on CH1, the trigger input is on CH1

Here the sequence + waveforms are defined in awg_program.code
'''

###############################
### Connecting the device #####
###############################

#defining a connection
device_id='devxxxx'

session = Session("xxx.xxx.xxx.xxx", "xxxx")  # IP address of rfsoc pc and port of the device
print("devices ready to be connected",session.devices.visible())
device = session.connect_device(device_id)
print("connected to",session.devices.connected())

###################################
#### load HDAWG configurations ####
###################################

# Setting things to default
grouping = 0  # Channel grouping 4X2
# factory_reset() puts the reference clock back on internal, which would unlock
# the HDAWG from the RFSoC. Remember the source in use and restore it.
reference_source = device.system.clocks.referenceclock.source()
device.factory_reset()
device.system.clocks.referenceclock.source(reference_source)
device.system.clocks.referenceclock.status.wait_for_state_change(0, timeout=30)  # 0 = locked
if int(device.system.clocks.referenceclock.source()) == 0:
    print("WARNING: the HDAWG is on its INTERNAL reference clock - it is not locked to the RFSoC, "
          "so the clocks will drift however the sample rate is set.")

# sample clock = 8 x SEQ_PER_TPROC x tProc clock (see the common time grid above)
device.system.clocks.sampleclock.freq(HDAWG_RATE_HZ)
device.system.clocks.sampleclock.status.wait_for_state_change(0, timeout=30)     # 0 = locked
hdawg_rate = device.system.clocks.sampleclock.freq()
print(f"HDAWG sample clock: {hdawg_rate:.3f} Hz (wanted {HDAWG_RATE_HZ:.3f} Hz), "
      f"time grid {TIME_GRID_US * 1e3:.4f} ns")
if abs(hdawg_rate - HDAWG_RATE_HZ) > 0.5:
    raise RuntimeError(f"The HDAWG set {hdawg_rate} Hz instead of {HDAWG_RATE_HZ} Hz: "
                       "the clocks would not line up, so not running.")

def set_awg_default(device=device,
                    grouping=grouping,  # Channel grouping 2x4
                    awg_group=0,  # AWG group
                    output_range=1.0,  # Output range [V]
                    ):
    awg_cores_i = awg_group * 2 ** grouping + np.arange(2 ** grouping)  # AWG cores
    channels_i = awg_group * 2 ** (grouping + 1) + np.arange(2 ** (grouping + 1))  # Output channels

    awg_cores = [device.awgs[awg_core] for awg_core in awg_cores_i]
    channels = [device.sigouts[channel] for channel in channels_i]

    # Per-core settings
    with device.set_transaction():  # This line can be omitted. it
        # "bundles all node set commands into a single transaction.
        # This reduces the network overhead and often increases the speed."

        # Grouping mode
        device.system.awg.channelgrouping(grouping)

        # Per-channel settings
        for channel in channels:
            channel.range(output_range)  # Select the output range
            # channel.on(True)                # Turn on the outputs. Should be the last setting
    return awg_cores_i, channels_i

# create channel list from number of cores
channels = []
awgs = [0, 1, 2, 3] # four cores for hdawg8 -> ideal operation is the 4x2 grouping (here)
for i in awgs:
    awg_cores_i, channels_i = set_awg_default(awg_group=i, output_range=1.0)
    channels.append(channels_i)
print(channels)

# make sure channels are off
for i in awgs:
    device.awgs[i].enable(0)


###########################
#### program sequencer ####
###########################
# sequencer programming and compilation for sequencer 1

# code works for all cores, just need to change index and trig
def make_awg_program(ff_length, wait_length=10, amp=0.3):
    """
    Square flux pulse on an external trigger, then the same pulse inverted.

    ff_length and wait_length are in us, on the common time grid. seq_cycles()
    converts them to ZI units (whole sequencer cycles, 8 samples each), so
    nothing is rounded inside the sequencer.
    """
    wait_length = on_grid(wait_length)
    pulse_cycles = seq_cycles(ff_length)   # sequencer cycles
    wait_cycles = seq_cycles(wait_length)
    program = Sequence()
    # !! HARDCODED !! - need to manually change index
    program.code = f"""
    // sample clock {HDAWG_RATE_HZ:.0f} Hz, sequencer clock = sample clock / 8
    const pulse_samples = {8 * pulse_cycles}; // {ff_length:.6f} us
    const wait_cycles = {wait_cycles}; // {wait_length:.6f} us, in sequencer cycles

    const amp = {amp}; // amplitude of the pulse

    wave w_rect  = amp*rect(pulse_samples, 1);
    const trig = 1; // always 1 for the odd trigger?
    const index = 1; // if channel is odd => =1, else =2

    while (true) {{
        waitDigTrigger(trig);
        playWave(index, w_rect);
        wait(wait_cycles + {pulse_cycles}); // wait for the pulse to finish plus some time (wait_cycles)
        playWave(index, -1*w_rect);
      }}
    """
    return program


###########################
#### Play waveforms ####
###########################
num_pts = 81
# enter times in us; on_grid() moves each to the nearest common grid step (TIME_GRID_US)
ff_length = on_grid(2) # fast flux length in us; the HDAWG sequence is generated from this
readout_delay = on_grid(2) # readout delay in us, from the end of the flux pulse
q_times = np.unique([on_grid(t) for t in np.linspace(-1, ff_length + readout_delay, num_pts)]) # in us
awg_program = make_awg_program(ff_length)
print(f"ff_length = {ff_length:.6f} us, readout_delay = {readout_delay:.6f} us, "
      f"{len(q_times)} q_times from {q_times[0]:.6f} to {q_times[-1]:.6f} us")


# one new numbered folder in DATA_PATH for the whole sweep
folder_suffix = 'fast_flux_distortion_Q' + str(QUBIT_INDEX) + '_' + str(datetime.date.today())
existing = [d for d in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, d)) and d[:5].isdigit()]
next_num = max([int(d[:5]) for d in existing], default=-1) + 1
SWEEP_DATA_PATH = os.path.join(DATA_PATH, f"{next_num:05d}_{folder_suffix}")
os.makedirs(SWEEP_DATA_PATH)
print('saving into ' + SWEEP_DATA_PATH)

for q_time in q_times:

    q = 0 # qubit index 
    awg_index = int(q/2)
    channel_index = q%2
    device.awgs[awgs[awg_index]].load_sequencer_program(awg_program.code)
    device.sigouts[channels[awgs[awg_index]][channel_index]].on(True)  ### turn the output port ON
    device.awgs[awgs[awg_index]].single(True)
    time.sleep(.5)
    device.awgs[awgs[awg_index]].enable(1) ### run the sequence. ## is this a command in core?
    

    qubit_freqs = fast_flux_qubit_spec(q_time, ff_length, readout_delay)
    print('q_time = ' + str(q_time) + ' us')

# make sure channels are off
for i in awgs:
    device.awgs[i].enable(0)
    print('turned off awg ' + str(i))
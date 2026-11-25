"""
023a_fast_flux_length_sweep_expt - qubit spectroscopy against fast flux pulse length.

A variant of 023_fast_flux_length_sweep_expt in which Python sets every flux
pulse length, the way 025 sets every amplitude. In 023 the HDAWG steps through
the lengths on its own by counting triggers, so the two instruments only stay
in step if the trigger count per length and the two lists of lengths are
matched by hand. Here the HDAWG sequence is recompiled for each length and
plays that one length on every trigger, so nothing depends on counting.

The RFSoC triggers the HDAWG, which plays a square flux pulse (followed later by
the same pulse inverted, as compensation). The qubit pulse stays a fixed time
before the END of the flux pulse and the readout a fixed delay after it, as in
023. Only the flux pulse length changes: at each length one qubit spectroscopy
is taken.

    Trigger  t = 0
    Flux     t = trig offset,                             length ff_length (swept)
    Qubit    t = trig offset + ff_length - q_before_end
    Readout  t = trig offset + ff_length + readout_delay

The flux pulse is a whole number of 16-sample blocks (the HDAWG waveform
granularity), so each length is rounded to the nearest 16 samples (6.7 ns at
2.4 GS/s). The rounded length is the one used for the RFSoC timing and saved
as ff_length; the length asked for is saved as ff_length_requested.

The HDAWG sample clock is set to SAMPLE_RATE and read back; the run stops if
the instrument reports a different rate. The amplitude is the HDAWG output
gain (full-scale waveform), fixed for the whole sweep and read back.

The HDAWG needs trig offset + 2 x ff_length + WAIT_LENGTH for the pulse and
its compensation, and a trigger that arrives before that is over is missed.
The script checks every length against the shot period (readout time +
relax_delay) before anything is played, and refuses to start otherwise.

Every length is saved as its own qubit spectroscopy file, all in one new
numbered folder in DATA_PATH. Analyse the folder with
analyze_fast_flux_length_sweep in data_analysis_live.ipynb.

The frequency window is the fixed qubit_spec_ge window in system_config.json
(span_below / span_above around qubit_freq_ge, steps).

With SS = 'True' in system_config.json a single shot g-e calibration is taken
after every spectroscopy and saved in the same file (I_g, Q_g, I_e, Q_e), as in
the other experiments. The single shot programs do not trigger the HDAWG, so
they run with no flux pulse; the readout in the sweep comes after the flux
pulse has ended, so that is the right reference. Your pi pulse should be
calibrated before running SS.

Live plotting is setup.

This experiment connects to the Zurich Instruments HDAWG. The IP addresses are
defined below in the code.
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

#importing some functions from toolkit
from zhinst.toolkit import Session, Waveforms, Sequence, CommandTable
from zhinst.toolkit.waveform import Wave, OutputType
import numpy as np
import time

###########################
#### sweep parameters ####
###########################
MAX_FF_AMP = 0.5        # hard limit on |amplitude| (fraction of the output range); the script refuses anything above
OUTPUT_RANGE = 1.0      # HDAWG output range [V]; flux pulse voltage = amplitude x OUTPUT_RANGE
ff_amp = -0.3           # flux pulse amplitude (fraction of the output range), fixed for the sweep

ff_lengths = np.arange(5, 81, 1.0)  # fast flux lengths in us

q_before_end = 3        # us, the qubit pulse starts this long before the END of the flux pulse (as in 023)
readout_delay = 1     # readout delay in us, from the end of the flux pulse (as in 023)
trig_offset = 0.02      # us, HDAWG trigger to the start of the flux pulse (same as 022)

SAMPLE_RATE = 2.4e9     # HDAWG sample clock [Hz], set on the instrument and read back
WAIT_LENGTH = 15        # us between the end of the flux pulse and the compensation pulse
GRANULARITY = 16        # samples; HDAWG waveforms are a whole number of these

if abs(ff_amp) > MAX_FF_AMP + 1e-12:
    raise ValueError(f"ff_amp = {ff_amp} is above MAX_FF_AMP = {MAX_FF_AMP}")


def flux_samples(length_us):
    """Samples in a flux pulse of length_us, rounded to the waveform granularity."""
    return GRANULARITY * int(round(length_us * 1e-6 * SAMPLE_RATE / GRANULARITY))


def check_lengths(lengths, config):
    """
    Refuse lengths the experiment cannot play: the qubit pulse must fit inside
    the flux pulse, and the HDAWG (flux pulse, wait, compensation pulse) must
    be done before the next trigger.
    """
    qubit_pulse_length = 5 * config['sigma_ge']   # a gauss pulse is 5 sigma long
    if qubit_pulse_length > q_before_end:
        raise ValueError(f"the qubit pulse ({qubit_pulse_length:.4f} us) is longer than q_before_end = {q_before_end} us: "
                         "it would run past the end of the flux pulse")
    for length_us in lengths:
        length = flux_samples(length_us) / SAMPLE_RATE * 1e6
        if length < q_before_end:
            raise ValueError(f"ff_length = {length_us} us is shorter than q_before_end = {q_before_end} us: "
                             "the qubit pulse would start before the flux pulse")
        hdawg_busy = trig_offset + 2 * length + WAIT_LENGTH
        # the shot is at least this long (the readout pulse length comes on top)
        shot_period = trig_offset + length + readout_delay + config['relax_delay']
        if hdawg_busy + 1 > shot_period:
            raise ValueError(f"ff_length = {length_us} us: the HDAWG needs {hdawg_busy:.2f} us per trigger (pulse, "
                             f"{WAIT_LENGTH} us wait, compensation) but the next trigger comes after about "
                             f"{shot_period:.2f} us. Increase relax_delay or shorten the sweep.")


def make_awg_program(n_samples):
    """SeqC for one flux pulse of n_samples and its compensation, on every trigger."""
    return f"""
    // sample clock {SAMPLE_RATE:.0f} Hz, sequencer clock = sample clock / 8
    const pulse_samples = {n_samples};  // {n_samples / SAMPLE_RATE * 1e6:.6f} us
    const wait_samples = {int(round(WAIT_LENGTH * 1e-6 * SAMPLE_RATE / 8))};  // sequencer cycles, {WAIT_LENGTH} us

    const amp = 1.0; // full scale; the output gain sets the pulse amplitude

    wave w_rect  = amp*rect(pulse_samples, 1);
    const trig = 1; // always 1 for the odd trigger?
    const index = 1; // if channel is odd => =1, else =2

    while (true) {{
        waitDigTrigger(trig);
        playWave(index, w_rect);
        wait(wait_samples + pulse_samples / 8); // wait for the pulse to finish plus some time (wait_samples)
        playWave(index, -1*w_rect);
      }}
    """


def fast_flux_qubit_spec(q_time, ff_length, readout_delay, ff_amp, data_path, ff_length_requested):
    '''
    Perform qubit spectroscopy, save data, and return qubit frequency found by looking for maximum amplitude.

    q_time: qubit pulse start after the leading flux edge, us
    ff_length: flux pulse length the HDAWG plays, us (not set here)
    readout_delay: readout delay from the end of the flux pulse, us
    ff_amp: flux pulse amplitude (saved with the data, not set here)
    data_path: folder the file is saved in
    ff_length_requested: the length asked for before rounding to the HDAWG granularity, us (saved)
    '''
    start_time = time.time()
    # ----- Experiment configurations ----- #
    expt_name = "qubit_spec_ge"
    config, expt_name = initialize_configs(expt_name)
    print(expt_name + '\n')
    print(config)
    print(QUBIT_INDEX)

    # the qubit pulse must sit inside the flux pulse (a gauss pulse is 5 sigma long)
    qubit_pulse_length = 5 * config['sigma_ge']
    if q_time < 0 or q_time + qubit_pulse_length > ff_length:
        raise ValueError(f"the qubit pulse ({q_time:.4f} to {q_time + qubit_pulse_length:.4f} us) "
                         f"does not fit inside the {ff_length} us flux pulse")
    print(f"q_time = {q_time:.4f} us, qubit pulse length = {qubit_pulse_length:.4f} us")
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
            t_offset = ff_length + readout_delay + trig_offset # (ro delay, trig delay) time offset for the readout pulse (fixed)

            # if non-MUX then send readout configurations - useful when freq sweeping
            self.send_readoutconfig(ch=self.cfg['ro_ch'], name="myro", t=0)

            self.pulse(ch=self.cfg['res_ch'], name="res_pulse", t=t_offset)
            self.trigger(ros=[self.cfg['ro_ch']], pins=[0], t=self.cfg['trig_time']+t_offset)

            self.trigger(pins=[1], t=0)
            self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=trig_offset + q_time)

    ###################
    # Run the Program #
    ###################

    qspec=PulseProbeSpectroscopyProgram(soccfg, reps=config['reps'], final_delay=config['relax_delay'], cfg=config)
    py_avg = config['py_avg']

    # for live plotting
    IS_VISDOM = True
    if IS_VISDOM:
        expt_I = expt_Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        assert viz.check_connection(timeout_seconds=5), "Visdom server not connected!"
        viz.close(win=None) # close previous plots
        plot_title = f'Qubit Spectroscopy, flux length = {ff_length:.4f} us'
        win1 = viz.line( X=np.arange(0, 1), Y=np.arange(0, 1),
            opts=dict(height=400, width=700, title=plot_title, showlegend=True, xlabel='expt_pts'))

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
            opts=dict(height=400, width=700, title=plot_title, showlegend=True, xlabel='expt_pts'))

        amps = np.abs(expt_I + 1j*expt_Q)

    else:
        iq_list = qspec.acquire(soc, soft_avgs=py_avg, progress=True)
        freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True)
        amps = np.abs(iq_list[0][0].dot([1,1j]))

    # print(amps.shape)
    if MUX == 'True':
        amps = amps.T[0]

    if SS == 'True':
        # single shot g-e calibration for this point. These programs do not fire
        # the HDAWG trigger pin, so no flux pulse is played during them.
        print('performing single shot for g-e calibration')
        ss_config, _ = initialize_configs("IQ_plot")

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
    end_time = time.time()
    tot_time = end_time - start_time

    with SlabFile(os.path.join(data_path, fname), 'a') as f:
        # 'a': read/write/create
        f.append('expt_time', [int(tot_time)])
        # - Adds data to the file - #
        f.append('fpts', freqs)
        f.append('amps', amps)
        f.append('ff_amp', [ff_amp])                       # fraction of the output range
        f.append('ff_amp_volts', [ff_amp * OUTPUT_RANGE])  # at the HDAWG output
        f.append('q_time', [q_time])
        f.append('ff_length', [ff_length])                 # the length played
        f.append('ff_length_requested', [ff_length_requested])
        f.append('q_before_end', [ff_length - q_time])
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

        if SS == 'True':
            # - Adds ss data to the file - #
            f.append('I_g', I_g)
            f.append('Q_g', Q_g)
            f.append('I_e', I_e)
            f.append('Q_e', Q_e)

            # formats config into file as a single line
            f.attrs['ss_config'] = json.dumps(ss_config)

    qubit_freq = freqs[np.argmax(amps)]
    return qubit_freq

#################################
### Zurich-Instruments code #####
#################################

'''
This code will play a square pulse with external trigger
The output is on CH1, the trigger input is on CH1

Here the sequence + waveform are defined in make_awg_program, one per length
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
device.factory_reset()

def set_awg_default(device=device,
                    grouping=grouping,  # Channel grouping 2x4
                    awg_group=0,  # AWG group
                    output_range=OUTPUT_RANGE,  # Output range [V]
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
    awg_cores_i, channels_i = set_awg_default(awg_group=i, output_range=OUTPUT_RANGE)
    channels.append(channels_i)
print(channels)

# make sure channels are off
for i in awgs:
    device.awgs[i].enable(0)

# sample clock: set it and check what the instrument reports
device.system.clocks.sampleclock.freq(SAMPLE_RATE)
device.system.clocks.sampleclock.status.wait_for_state_change(0, timeout=30)     # 0 = locked
hdawg_rate = device.system.clocks.sampleclock.freq()
print(f"HDAWG sample clock: {hdawg_rate:.3f} Hz (wanted {SAMPLE_RATE:.3f} Hz)")
if abs(hdawg_rate - SAMPLE_RATE) > 0.5:
    raise RuntimeError(f"The HDAWG set {hdawg_rate} Hz instead of {SAMPLE_RATE} Hz: the flux pulse lengths would be wrong")


###########################
#### Play waveforms ####
###########################
q = 0 # qubit index
awg_index = int(q/2)
channel_index = q%2
awg_core = device.awgs[awgs[awg_index]]
# output gain of this channel: the flux pulse amplitude, -1 to 1 of the output range
gain_node = awg_core.outputs[channel_index].gains[channel_index]

# one new numbered folder in DATA_PATH for the whole sweep
folder_suffix = 'fast_flux_length_sweep_Q' + str(QUBIT_INDEX) + '_' + str(datetime.date.today())
existing = [d for d in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, d)) and d[:5].isdigit()]
next_num = max([int(d[:5]) for d in existing], default=-1) + 1
data_path = os.path.join(DATA_PATH, f"{next_num:05d}_{folder_suffix}")
os.makedirs(data_path)
print('saving into ' + data_path)

# every length is checked before anything is played
check_config, _ = initialize_configs("qubit_spec_ge")
check_lengths(ff_lengths, check_config)

gain_node(0.0)   # no flux pulse until the sequence is loaded
device.sigouts[channels[awgs[awg_index]][channel_index]].on(True)  ### turn the output port ON

try:
    for length_requested in ff_lengths:
        length_requested = float(length_requested)
        n_samples = flux_samples(length_requested)
        ff_length = n_samples / SAMPLE_RATE * 1e6   # us, the length actually played

        # load this length with the output silent, then set the amplitude
        awg_core.enable(0)
        gain_node(0.0)
        awg_core.load_sequencer_program(make_awg_program(n_samples))
        gain_node(ff_amp)
        set_amp = float(gain_node())
        if abs(set_amp - ff_amp) > 1e-3:
            raise RuntimeError(f"asked the HDAWG for gain {ff_amp}, it reports {set_amp}: not running this point")
        awg_core.single(True)
        awg_core.enable(1) ### run the sequence (it waits for the RFSoC trigger)
        time.sleep(.5)

        qubit_freq = fast_flux_qubit_spec(ff_length - q_before_end, ff_length, readout_delay, set_amp,
                                          data_path, length_requested)
        print(f'ff_length = {ff_length:.4f} us (asked {length_requested:g} us), peak at {qubit_freq:.3f} MHz')
finally:
    # make sure the flux pulse is off and channels are off, also after an error
    gain_node(0.0)
    for i in awgs:
        device.awgs[i].enable(0)
        print('turned off awg ' + str(i))

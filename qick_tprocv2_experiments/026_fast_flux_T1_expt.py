"""
026_fast_flux_T1_expt - T1 at the flux-shifted qubit frequency, against flux pulse amplitude.

The efficient version of reading T1 off a 022 distortion run. In 022 a whole
qubit spectroscopy is taken at every qubit pulse time; here, for each flux
pulse amplitude:

  1. One qubit spectroscopy with the qubit pulse inside the flux pulse finds
     the flux-shifted qubit frequency.
  2. The qubit pulse is then played at that one frequency at a few positions
     (N_T1_POINTS) inside the flux pulse. The readout only comes after the
     flux pulse has ended, so each position is a different wait between
     excitation and readout, and the signal against that wait is a T1 decay
     at the flux-shifted frequency.

    Trigger  t = 0
    Flux     t = trig offset,                             length ff_length, amplitude swept
    Qubit    t = trig offset + q_time                     q_time: a few positions inside the flux pulse
    Readout  t = trig offset + ff_length + readout_delay

    wait = ff_length + readout_delay - (q_time + qubit pulse length)

One more point is taken with no qubit pulse at all (the flux pulse still
plays). It is the signal the decay ends at, so the analysis can fit the decay
without a free offset, which matters with only a few points.

The readout_delay part of the wait is spent back at the idle frequency. It is
the same for every point, so it scales the decay without changing the fitted T1.

The qubit pulse is the same pulse in both steps (qubit_gain_ge, sigma_ge). It
is calibrated at the idle frequency, so at the flux-shifted frequency it may
not be an exact pi pulse; that only changes the size of the signal.

The frequency is taken from a Lorentzian fit of the spectroscopy, checked
against the data: if the fit fails or settles on noise away from the line (it
can, for a narrow line in a wide window), the point of the spectrum furthest
from its median is used instead, and the script says so. How the frequency was
found is saved with the T1 data (freq_from).

The amplitude is the HDAWG output gain (full-scale waveform, loaded once),
read back at every point. The HDAWG sample clock is set to SAMPLE_RATE and
read back. The flux pulse is rounded to 16 samples and the rounded length is
used for the RFSoC timing. Before anything is played the script checks that
the HDAWG (flux pulse, wait, compensation pulse) is done before the next
trigger.

Everything goes in one new numbered folder in DATA_PATH: per amplitude a
qubit_spec_ge file and a fast_flux_T1 file. Analyse the folder with
analyze_fast_flux_t1_sweep in data_analysis_live.ipynb.

reps, py_avg and relax_delay are those of qubit_spec_ge (step 1) and T1_ge
(step 2) in system_config.json; the frequency window is the qubit_spec_ge one.

With SS = 'True' in system_config.json a single shot g-e calibration is taken
once per amplitude (with no flux pulse: those programs do not trigger the
HDAWG) and saved in both files. Your pi pulse should be calibrated before
running SS.

Live plotting is setup for the spectroscopy.

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
ff_amps = np.linspace(0, -0.5, 11)  # flux pulse amplitudes

ff_length = 12          # fast flux length in us
readout_delay = 1       # readout delay in us, from the end of the flux pulse
trig_offset = 0.02      # us, HDAWG trigger to the start of the flux pulse (same as 022)

N_T1_POINTS = 6         # qubit pulse positions inside the flux pulse, per amplitude
EDGE_MARGIN = 0.5       # us, keep the qubit pulse this far from both edges of the flux pulse
SPEC_Q_TIME = None      # us, qubit pulse position for the spectroscopy. None: the last T1 position
                        # (shortest wait before the readout, so the largest signal)

SAMPLE_RATE = 2.4e9     # HDAWG sample clock [Hz], set on the instrument and read back
WAIT_LENGTH = 15        # us between the end of the flux pulse and the compensation pulse
GRANULARITY = 16        # samples; HDAWG waveforms are a whole number of these

if np.max(np.abs(ff_amps)) > MAX_FF_AMP + 1e-12:
    raise ValueError(f"ff_amps reaches {np.max(np.abs(ff_amps))}, above MAX_FF_AMP = {MAX_FF_AMP}")
if N_T1_POINTS < 3:
    raise ValueError(f"N_T1_POINTS = {N_T1_POINTS}: a decay needs at least 3 positions")

# the flux pulse the HDAWG actually plays: a whole number of 16-sample blocks
FF_SAMPLES = GRANULARITY * int(round(ff_length * 1e-6 * SAMPLE_RATE / GRANULARITY))
ff_length_requested = ff_length
ff_length = FF_SAMPLES / SAMPLE_RATE * 1e6


def t1_q_times(config):
    """The qubit pulse positions (us after the leading flux edge), evenly spaced inside the flux pulse."""
    qubit_pulse_length = 5 * config['sigma_ge']   # a gauss pulse is 5 sigma long
    last = ff_length - qubit_pulse_length - EDGE_MARGIN
    if last <= EDGE_MARGIN:
        raise ValueError(f"no room for the qubit pulse ({qubit_pulse_length:.4f} us) inside the {ff_length:.4f} us "
                         f"flux pulse with EDGE_MARGIN = {EDGE_MARGIN} us")
    return np.linspace(EDGE_MARGIN, last, N_T1_POINTS)


def check_timing(*configs):
    """Refuse to run if the HDAWG (flux pulse, wait, compensation pulse) is not done before the next trigger."""
    hdawg_busy = trig_offset + 2 * ff_length + WAIT_LENGTH
    for config in configs:
        # the shot is at least this long (the readout pulse length comes on top)
        shot_period = trig_offset + ff_length + readout_delay + config['relax_delay']
        if hdawg_busy + 1 > shot_period:
            raise ValueError(f"the HDAWG needs {hdawg_busy:.2f} us per trigger (pulse, {WAIT_LENGTH} us wait, "
                             f"compensation) but with relax_delay = {config['relax_delay']} us the next trigger "
                             f"comes after about {shot_period:.2f} us. Increase relax_delay or shorten ff_length.")


def find_line(freqs, amps):
    """
    The qubit frequency in one spectrum.

    The Lorentzian fit is used if it lands on the line: the data at the fitted
    frequency must stand at least half as far from the median as the furthest
    point of the spectrum does. A fit of a narrow line in a wide window can
    settle on noise instead; then, or if the fit fails, the furthest point
    itself is used, and the script says so. It also warns when nothing in the
    spectrum stands clearly out of the noise.

    returns: frequency (MHz), and 'fit' or 'extreme'
    """
    dev = np.abs(amps - np.median(amps))
    k = int(np.argmax(dev))
    noise = 1.4826 * np.median(dev)               # robust standard deviation
    if dev[k] < 5 * noise:
        print(f"!! no clear line in this spectrum (largest deviation {dev[k]:.3g}, noise {noise:.3g}): "
              "the frequency below is not to be trusted")
    try:
        f0 = float(Fit().transmission_resonator_spectroscopy(freqs, amps, plot=False)['f'][0])
        if np.min(freqs) <= f0 <= np.max(freqs) and dev[int(np.argmin(np.abs(freqs - f0)))] >= 0.5 * dev[k]:
            return f0, 'fit'
        print(f"the Lorentzian fit ({f0:.3f} MHz) is not on the line")
    except Exception as e:
        print(f"spectroscopy fit failed ({type(e).__name__})")
    f0 = float(freqs[k])
    print(f"!! using the point furthest from the median instead, {f0:.3f} MHz")
    return f0, 'extreme'


def single_shot():
    """Single shot g-e calibration (no flux pulse: these programs do not fire the HDAWG trigger pin)."""
    print('performing single shot for g-e calibration')
    ss_config, _ = initialize_configs("IQ_plot")

    ssp_g = SingleShotProgram_g(soccfg, reps=1, final_delay=ss_config['relax_delay'], cfg=ss_config)
    iq_list_g = ssp_g.acquire(soc, soft_avgs=1, progress=True)

    ssp_e = SingleShotProgram_e(soccfg, reps=1, final_delay=ss_config['relax_delay'], cfg=ss_config)
    iq_list_e = ssp_e.acquire(soc, soft_avgs=1, progress=True)

    if MUX == 'False':
        blobs = (iq_list_g[0][0].T[0], iq_list_g[0][0].T[1], iq_list_e[0][0].T[0], iq_list_e[0][0].T[1])
    else:
        blobs = (iq_list_g[QUBIT_INDEX - 1].T[0], iq_list_g[QUBIT_INDEX - 1].T[1],
                 iq_list_e[QUBIT_INDEX - 1].T[0], iq_list_e[QUBIT_INDEX - 1].T[1])
    return blobs, ss_config


def save_single_shot(f, ss):
    """Add the single shot blobs and their config to an open data file."""
    if ss is None:
        return
    blobs, ss_config = ss
    for key, values in zip(('I_g', 'Q_g', 'I_e', 'Q_e'), blobs):
        f.append(key, values)
    f.attrs['ss_config'] = json.dumps(ss_config)


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


def fast_flux_t1(q_times, flux_freq, ff_amp, data_path, ss=None, freq_from=''):
    '''
    The qubit pulse at flux_freq at each of q_times inside the flux pulse, plus
    one point with no qubit pulse; saved as one file.

    q_times: qubit pulse starts after the leading flux edge, us
    flux_freq: qubit pulse frequency, MHz (from the spectroscopy at this amplitude)
    ff_amp: flux pulse amplitude (saved with the data, not set here)
    data_path: folder the file is saved in
    ss: (blobs, ss_config) from single_shot(), saved with the data; or None
    freq_from: how flux_freq was found ('fit' or 'extreme', from find_line); saved with the data
    '''
    start_time = time.time()
    config, _ = initialize_configs("T1_ge")
    del config['wait_time']                       # the T1_ge delay sweep is not used here (and cannot be saved)
    config['qubit_freq_ge'] = float(flux_freq)
    qubit_pulse_length = 5 * config['sigma_ge']
    ro_index = 0 if MUX == 'False' else QUBIT_INDEX - 1

    def run_point(q_time):
        """One point: q_time in us, or None for no qubit pulse. Returns averaged I, Q."""
        class FluxT1Program(AveragerProgramV2):
            def _initialize(self, cfg):
                # initialize the readout channels and pulses for DAC and ADC
                initialize_ro_chs(self, cfg, suffix='_ge')

                # initialize the qubit generator channel
                declare_gen_ch(self, cfg, cfg['qubit_ch'], usage='qubit', suffix='_ge')
                # initialize qubit pulse
                declare_pulse(self, cfg, cfg['qubit_ch'], usage='qubit',
                            pulse_style='gauss', pulse_name='qubit_pulse', suffix='_ge')

            def _body(self, cfg):
                t_offset = ff_length + readout_delay + trig_offset # (ro delay, trig delay) time offset for the readout pulse (fixed)

                # if non-MUX then send readout configurations
                self.send_readoutconfig(ch=self.cfg['ro_ch'], name="myro", t=0)

                self.pulse(ch=self.cfg['res_ch'], name="res_pulse", t=t_offset)
                self.trigger(ros=[self.cfg['ro_ch']], pins=[0], t=self.cfg['trig_time']+t_offset)

                self.trigger(pins=[1], t=0)
                if q_time is not None:
                    self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=trig_offset + q_time)

        prog = FluxT1Program(soccfg, reps=config['reps'], final_delay=config['relax_delay'], cfg=config)
        iq_list = prog.acquire(soc, soft_avgs=config['py_avg'], progress=False)
        iq = np.ravel(iq_list[ro_index])          # no loops in the program: one averaged (I, Q)
        return float(iq[0]), float(iq[1])

    for q_time in q_times:
        if q_time < 0 or q_time + qubit_pulse_length > ff_length:
            raise ValueError(f"the qubit pulse at q_time = {q_time:.4f} us does not fit inside the {ff_length:.4f} us flux pulse")

    ref_I, ref_Q = run_point(None)
    I, Q = np.array([run_point(float(q_time)) for q_time in q_times]).T
    delay_times = ff_length + readout_delay - (np.asarray(q_times) + qubit_pulse_length)
    amps = np.abs(I + 1j * Q)

    prefix = str(datetime.date.today())
    exp_name = 'fast_flux_T1_Q' + str(QUBIT_INDEX) + '_' + prefix
    fname = get_next_filename(data_path, exp_name, suffix='.h5')
    print('Current data file: ' + fname)

    with SlabFile(os.path.join(data_path, fname), 'a') as f:
        f.append('expt_time', [int(time.time() - start_time)])
        f.append('q_times', np.asarray(q_times))
        f.append('delay_times', delay_times)               # end of the qubit pulse to the readout, us
        f.append('avgi', I)
        f.append('avgq', Q)
        f.append('amps', amps)
        f.append('ref_i', [ref_I])                         # no qubit pulse
        f.append('ref_q', [ref_Q])
        f.append('flux_freq', [float(flux_freq)])
        f.append('ff_amp', [ff_amp])                       # fraction of the output range
        f.append('ff_amp_volts', [ff_amp * OUTPUT_RANGE])  # at the HDAWG output
        f.append('ff_length', [ff_length])
        f.append('readout_delay', [readout_delay])
        f.attrs['config'] = json.dumps(config)
        f.attrs['freq_from'] = freq_from
        save_single_shot(f, ss)
    return delay_times, np.abs((I - ref_I) + 1j * (Q - ref_Q))


def fast_flux_qubit_spec(q_time, ff_length, readout_delay, ff_amp, data_path, ss=None):
    '''
    Perform qubit spectroscopy inside the flux pulse, save data, and return find_line's (frequency, how it was found).

    q_time: qubit pulse start after the leading flux edge, us
    ff_length: flux pulse length the HDAWG plays, us (not set here)
    readout_delay: readout delay from the end of the flux pulse, us
    ff_amp: flux pulse amplitude (saved with the data, not set here)
    data_path: folder the file is saved in
    ss: (blobs, ss_config) from single_shot(), saved with the data; or None
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
        plot_title = f'Qubit Spectroscopy, flux amp = {ff_amp:+.4f}'
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

        save_single_shot(f, ss)

    return find_line(freqs, amps)
    qubit_freq = freqs[np.argmax(amps)]
    return qubit_freq

#################################
### Zurich-Instruments code #####
#################################

'''
This code will play a square pulse with external trigger
The output is on CH1, the trigger input is on CH1

Here the sequence + waveform are defined in make_awg_program
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
folder_suffix = 'fast_flux_T1_Q' + str(QUBIT_INDEX) + '_' + str(datetime.date.today())
existing = [d for d in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, d)) and d[:5].isdigit()]
next_num = max([int(d[:5]) for d in existing], default=-1) + 1
data_path = os.path.join(DATA_PATH, f"{next_num:05d}_{folder_suffix}")
os.makedirs(data_path)
print('saving into ' + data_path)

# timing is checked before anything is played
spec_config, _ = initialize_configs("qubit_spec_ge")
t1_config, _ = initialize_configs("T1_ge")
check_timing(spec_config, t1_config)
q_times = t1_q_times(t1_config)
spec_q_time = float(q_times[-1]) if SPEC_Q_TIME is None else SPEC_Q_TIME
print(f"flux pulse {ff_length:.4f} us (asked {ff_length_requested:g} us); qubit pulse positions {np.round(q_times, 3)} us; "
      f"spectroscopy at q_time = {spec_q_time:.3f} us")

gain_node(0.0)   # no flux pulse until the first amplitude is set
awg_core.load_sequencer_program(make_awg_program(FF_SAMPLES))
device.sigouts[channels[awgs[awg_index]][channel_index]].on(True)  ### turn the output port ON
awg_core.single(True)
time.sleep(.5)

try:
    for ff_amp in ff_amps:
        ff_amp = float(ff_amp)
        if abs(ff_amp) > MAX_FF_AMP + 1e-12:
            raise ValueError(f"ff_amp = {ff_amp} is above MAX_FF_AMP = {MAX_FF_AMP}")

        gain_node(ff_amp)
        set_amp = float(gain_node())
        if abs(set_amp - ff_amp) > 1e-3:
            raise RuntimeError(f"asked the HDAWG for gain {ff_amp}, it reports {set_amp}: not running this point")
        awg_core.enable(1) ### run the sequence (it waits for the RFSoC trigger)
        time.sleep(.1)

        ss = single_shot() if SS == 'True' else None

        # 1. where is the qubit at this amplitude
        flux_freq, freq_from = fast_flux_qubit_spec(spec_q_time, ff_length, readout_delay, set_amp, data_path, ss)
        print(f'ff_amp = {set_amp:+.4f} ({set_amp * OUTPUT_RANGE:+.4f} V): qubit at {flux_freq:.3f} MHz')

        # 2. a few qubit pulse positions at that frequency
        delays, signal = fast_flux_t1(q_times, flux_freq, set_amp, data_path, ss, freq_from)
        print('   wait (us):   ' + '  '.join(f'{d:7.3f}' for d in delays))
        print('   |IQ - ref|:  ' + '  '.join(f'{s:7.3f}' for s in signal))
finally:
    # make sure the flux pulse is off and channels are off, also after an error
    gain_node(0.0)
    for i in awgs:
        device.awgs[i].enable(0)
        print('turned off awg ' + str(i))

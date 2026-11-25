"""
028_fast_flux_distortion_amp_sweep_expt - the 022 distortion scan, repeated at
each flux pulse amplitude.

At every amplitude in ff_amps this runs exactly what 022 runs: a qubit
spectroscopy at each q_time, with the qubit pulse q_time after the flux
pulse's rising edge and the readout readout_delay after the flux pulse ends.
The HDAWG sequence is 022's (the same 2.29376e9 sample clock constant, the
same 10 us wait before the compensation pulse, the clock left at its default),
with only the amplitude changed between amplitudes.

    Trigger  t = 0
    Flux     t = 0.02 us,                       length ff_length, amplitude ff_amp (swept)
    Qubit    t = 0.02 us + q_time               q_time swept inside the flux pulse
    Readout  t = 0.02 us + ff_length + readout_delay

Folders: one new numbered folder in DATA_PATH for the whole sweep, and inside
it one numbered folder per amplitude holding that amplitude's qubit
spectroscopy files (each a 022 run on its own):

    DATA_PATH/00022_fast_flux_distortion_amp_sweep_Q1_<date>/
        00000_ff_amp_+0.000/   00000_qubit_spec_ge_Q1_<date>.h5 ...
        00001_ff_amp_-0.025/   ...

Each file also saves ff_amp. Analyse the sweep folder with
analyze_fast_flux_t1_amp_sweep in data_analysis_live.ipynb, or one amplitude
folder with analyze_fast_flux_t1_from_spec / analyze_fast_flux_qubit_spec.

With SS = 'True' in system_config.json a single shot g-e calibration is
taken at the start of every amplitude, before its scan, and saved in every
file of that amplitude (I_g, Q_g, I_e, Q_e, and ss_config). The single shot
programs do not trigger the HDAWG, so they run with no flux pulse; the readout
in the scan comes after the flux pulse has ended, so that is the right
reference. Analyse with SS=True. With SS = 'False' nothing is taken: use a
separate single shot file as the analysis' calibration_file. Your pi pulse
should be calibrated before running SS.

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
ff_length = 20          # fast flux length in us (as in 022)
readout_delay = 1       # readout delay in us, from the end of the flux pulse (as in 022)
num_pts = 20            # q_times per amplitude (as in 022)
q_times = np.linspace(1, ff_length - 0.3, num_pts)  # in us (as in 022)

MAX_FF_AMP = 0.5        # hard limit on |amplitude|; the script refuses anything above
ff_amps = np.linspace(0, -0.5, 21)  # flux pulse amplitudes, 0.025 apart

if np.max(np.abs(ff_amps)) > MAX_FF_AMP + 1e-12:
    raise ValueError(f"ff_amps reaches {np.max(np.abs(ff_amps))}, above MAX_FF_AMP = {MAX_FF_AMP}")

def fast_flux_qubit_spec(q_time, ff_length, readout_delay, ff_amp, data_path, ss=None):
    '''
    Perform qubit spectroscopy, save data, and return qubit frequency found by looking for maximum amplitude.

    ff_amp: the flux pulse amplitude this scan runs at (saved with the data, set in the HDAWG sequence)
    data_path: the amplitude's folder the file is saved in
    ss: (blobs, ss_config) from single_shot() for this amplitude, saved with the data; or None
    '''
    start_time = time.time()
    # ----- Experiment configurations ----- #
    expt_name = "qubit_spec_ge"
    config, expt_name = initialize_configs(expt_name)
    print(expt_name + '\n')
    print(config)
    print(QUBIT_INDEX)
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
            t_offset = ff_length + readout_delay +0.02 # (ro delay, trig delay) time offset for the readout pulse (fixed)

            if q_time < 0:
                # if non-MUX then send readout configurations - useful when freq sweeping
                self.send_readoutconfig(ch=self.cfg['ro_ch'], name="myro", t=0)
                
                self.pulse(ch=self.cfg['res_ch'], name="res_pulse", t=t_offset - q_time)
                self.trigger(ros=[self.cfg['ro_ch']], pins=[0], t=self.cfg['trig_time']+t_offset)

                self.trigger(pins=[1], t=np.abs(q_time), width=1)
                self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0.02)
            else:
                # if non-MUX then send readout configurations - useful when freq sweeping
                self.send_readoutconfig(ch=self.cfg['ro_ch'], name="myro", t=0)
                
                self.pulse(ch=self.cfg['res_ch'], name="res_pulse", t=t_offset)
                self.trigger(ros=[self.cfg['ro_ch']], pins=[0], t=self.cfg['trig_time']+t_offset)

                self.trigger(pins=[1], width=1,t=0)
                self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0.02 + q_time)

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
        plot_title = f'Qubit Spectroscopy, flux amp {ff_amp:+.3f}, q_time {q_time:.3f} us'
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

    # data_path: this amplitude's folder inside the sweep folder

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
        f.append('ff_amp', [ff_amp])
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

        if ss is not None:
            # - Adds this amplitude's single shot calibration to the file - #
            blobs, ss_config = ss
            for key, values in zip(('I_g', 'Q_g', 'I_e', 'Q_e'), blobs):
                f.append(key, values)
            f.attrs['ss_config'] = json.dumps(ss_config)

    data = data_path + '\\' + fname

    qubit_freq = freqs[np.argmax(amps)]
    return qubit_freq

def single_shot():
    '''
    Single shot g-e calibration (no flux pulse: these programs do not fire the HDAWG
    trigger pin). Returns (I_g, Q_g, I_e, Q_e) and the single shot config.
    '''
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
device.factory_reset()

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
# !! HARDCODED !! - need to manually change index
def make_awg_program(amp):
    """022's sequence, with the pulse amplitude amp (fraction of the output range)."""
    awg_program = Sequence()
    awg_program.code = f"""
    const sample_clock = 2.29376e9;
    const sequencer_clock = sample_clock/8;

    const pulse_length = {ff_length}e-6; // seconds
    const wait_length = 10e-6; // seconds

    const wait_samples = round(wait_length*sequencer_clock);
    const pulse_samples = round(pulse_length*sample_clock);

    const amp = {amp:.6f}; // amplitude of the pulse

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
    return awg_program


###########################
#### Play waveforms ####
###########################

# one new numbered folder in DATA_PATH for the whole sweep
folder_suffix = 'fast_flux_distortion_amp_sweep_Q' + str(QUBIT_INDEX) + '_' + str(datetime.date.today())
existing = [d for d in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, d)) and d[:5].isdigit()]
next_num = max([int(d[:5]) for d in existing], default=-1) + 1
SWEEP_DATA_PATH = os.path.join(DATA_PATH, f"{next_num:05d}_{folder_suffix}")
os.makedirs(SWEEP_DATA_PATH)
print('saving into ' + SWEEP_DATA_PATH)
print(f"{len(ff_amps)} amplitudes x {len(q_times)} q_times; flux pulse {ff_length} us, readout {readout_delay} us after it")

q = 0 # qubit index
awg_index = int(q/2)
channel_index = q%2

try:
    for i_amp, ff_amp in enumerate(ff_amps):
        ff_amp = float(ff_amp)
        if abs(ff_amp) > MAX_FF_AMP + 1e-12:
            raise ValueError(f"ff_amp = {ff_amp} is above MAX_FF_AMP = {MAX_FF_AMP}")

        # this amplitude's own folder inside the sweep folder
        amp_data_path = os.path.join(SWEEP_DATA_PATH, f"{i_amp:05d}_ff_amp_{ff_amp:+.3f}")
        os.makedirs(amp_data_path)
        print(f'\n=== ff_amp = {ff_amp:+.3f} ({i_amp + 1} of {len(ff_amps)}): saving into {amp_data_path}')
        awg_program = make_awg_program(ff_amp)

        # this amplitude's single shot calibration (no flux pulse), saved in all its files
        ss = single_shot() if SS == 'True' else None

        # the 022 scan at this amplitude
        for q_time in q_times:
            device.awgs[awgs[awg_index]].load_sequencer_program(awg_program.code)
            device.sigouts[channels[awgs[awg_index]][channel_index]].on(True)  ### turn the output port ON
            device.awgs[awgs[awg_index]].single(True)
            time.sleep(.5)
            device.awgs[awgs[awg_index]].enable(1) ### run the sequence. ## is this a command in core?

            qubit_freqs = fast_flux_qubit_spec(q_time, ff_length, readout_delay, ff_amp, amp_data_path, ss)
            print(f'ff_amp = {ff_amp:+.3f}, q_time = {q_time} us')
finally:
    # make sure channels are off, also after an error
    for i in awgs:
        device.awgs[i].enable(0)
        print('turned off awg ' + str(i))

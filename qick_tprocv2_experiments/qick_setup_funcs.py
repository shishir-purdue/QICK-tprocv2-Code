"""
qick_setup_funcs - contains functions to initialize the experiment configurations and declare channels and pulses.

This file provides a layer of abstraction to the experiment code so that we can easily change the expeirments without
having to deal with the details of the hardware configurations. This is useful for MUX and non-MUX experiments.

Please see the description of each function for more details.


Author: Santi
Date: 2025-05-09
"""
import numpy as np
import json
import os
import datetime

from build_task import *
from build_state import *

# check if this import method is an issue with the proxy since we use this file many times
# open configuration file
with open('system_config.json', 'r') as f:
    configs = json.load(f)

DATA_PATH = configs['DATA_PATH']

system_config = configs['config']
expt_cfg = configs['expt_cfg']

fast_gen_chs = configs['zcu216_info']['fast_gen_chs']
mux_gen_chs = configs['zcu216_info']['mux_gen_chs']
inter_gen_chs = configs['zcu216_info']['inter_gen_chs']
dyn_ro_chs = configs['zcu216_info']['dyn_ro_chs']
mux_ro_chs = configs['zcu216_info']['mux_ro_chs']

NUM_QUBITS = system_config['num_qubits']
QUBIT_INDEX = system_config['qubit_index']
MUX = system_config['MUX']
SS = system_config['SS']

hw_cfg = system_config['hw_cfg']
readout_cfg = system_config['readout_cfg']
qubit_cfg = system_config['qubit_cfg']


def _check_qubit_lists(system_config):
    '''
    _check_qubit_lists makes sure every per-qubit list in the hardware, readout
    and qubit sections has one entry per qubit, so a hand-edited list that lost
    or gained a line fails here with its name instead of as an IndexError deep
    inside build_state.

    system_config: the 'config' section of system_config.json
    '''
    n = system_config['num_qubits']
    for section in ('hw_cfg', 'readout_cfg', 'qubit_cfg'):
        for key, value in system_config[section].items():
            if isinstance(value, list) and len(value) != n:
                raise ValueError(
                    f"system_config.json: {section}.{key} has {len(value)} entries "
                    f"but num_qubits is {n} - every per-qubit list needs exactly "
                    f"{n} (qubits are numbered 1..{n})")


def _center_frequency(system_config, expt_name, qubit_index):
    '''
    _center_frequency returns the frequency (MHz) a spectroscopy sweep is centred
    on for this qubit, or None for experiments whose ranges are absolute.

    system_config: the 'config' section of system_config.json
    expt_name: the experiment name
    qubit_index: 1-based qubit number
    '''
    centers = {
        'res_spec_ge':   ('readout_cfg', 'res_freq_ge'),
        'res_spec_ef':   ('readout_cfg', 'res_freq_ef'),
        'qubit_spec_ge': ('qubit_cfg',   'qubit_freq_ge'),
        'qubit_spec_ef': ('qubit_cfg',   'qubit_freq_ef'),
    }
    if expt_name not in centers:
        return None
    section, key = centers[expt_name]
    return system_config[section][key][qubit_index - 1]


def initialize_configs(expt_name, qubit_index=None):
    '''
    initialize_configs initializes the experiment configurations based on if we are doing mux or not.

    expt_name: a string name for the experiment
    qubit_index: which qubit, numbered from 1 as everywhere else in the configs.
                 Defaults to system_config['qubit_index']; the meas_all loop
                 in each experiment passes the qubit it is on.

    returns:
        config: the full configuration dictionary for this experiment
        expt_name: the name of the experiment (now with '_mux' if applicable)
    '''

    with open('system_config.json', 'r') as f:
        configs = json.load(f)

    system_config = configs['config']
    expt_cfg = configs['expt_cfg']
    _check_qubit_lists(system_config)

    QUBIT_INDEX = system_config['qubit_index'] if qubit_index is None else qubit_index
    MUX = system_config['MUX']
    readout_cfg = system_config['readout_cfg']

    # add the parameters for the specific experiment we want to do. The
    # spectroscopy experiments give their sweep as a span around the qubit's
    # own frequency, so hand that centre over.
    center = _center_frequency(system_config, expt_name, QUBIT_INDEX)
    exp_cfg = add_qubit_experiment(expt_cfg, expt_name, QUBIT_INDEX, center=center)

    if MUX == 'True': # MUX experiments
        if expt_name == 'res_spec_ge' or expt_name == 'res_spec_ef': # special cases since we sweep ro freq
            exp_cfg = expt_cfg['res_spec_mux']
        
        hw_cfg_cpy = add_qubit_channel(system_config, QUBIT_INDEX)
        qubit_cfg_cpy = add_qubit_cfg(system_config, QUBIT_INDEX)
        
        config = {**hw_cfg_cpy, **qubit_cfg_cpy, **readout_cfg, **exp_cfg}

        expt_name = expt_name + '_mux'
    else: # non-MUX experiments
        q_config = all_qubit_state(system_config)
        config = {**q_config['Q' + str(QUBIT_INDEX)], **exp_cfg}

    # Record which qubit this is (1-based, as typed) and how many there are, so
    # the saved h5 config says so and the analysis can label its plots and
    # assemble per-qubit lists without parsing filenames.
    config.update([('qubit_index', QUBIT_INDEX),
                   ('num_qubits', system_config['num_qubits'])])

    return config, expt_name


def setup_meas_all(config, expt_name):
    '''
    setup_meas_all decides which qubits an experiment runs on and where it saves.

    With config['meas_all'] == 'True' the experiment runs on every qubit in
    config['qubit_list'] (1-based; all qubits if the key is absent), one after
    another, and every file goes into a fresh numbered subfolder of DATA_PATH
    named after the experiment. Otherwise it runs on the one configured qubit
    and saves straight into DATA_PATH, exactly as before.

    config: the configuration dictionary from initialize_configs
    expt_name: the experiment name, used for the folder

    returns:
        qubit_list: the qubits to run, in order
        data_path: the folder the data files go in
    '''
    if config.get('meas_all', 'False') != 'True':
        return [QUBIT_INDEX - 1], DATA_PATH

    qubit_list = list(config.get('qubit_list', range(1, NUM_QUBITS + 1)))

    # One numbered folder per invocation, so repeated runs stay separate. The
    # number is one past the highest already sitting in DATA_PATH.
    folder_suffix = expt_name + '_all_' + str(datetime.date.today())
    existing = [d for d in os.listdir(DATA_PATH)
                if os.path.isdir(os.path.join(DATA_PATH, d)) and d[:5].isdigit()]
    next_num = max([int(d[:5]) for d in existing], default=-1) + 1
    data_path = os.path.join(DATA_PATH, f"{next_num:05d}_{folder_suffix}")
    os.makedirs(data_path, exist_ok=True)
    print(f"meas_all: running on qubits {qubit_list}, saving into {data_path}")
    return qubit_list, data_path


def declare_gen_ch(expt_func, cfg, ch, usage = 'res', suffix = ''):
    '''
    declare_gen_ch declares a generator channel properly depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    ch: The channel index that you want to declare
    usage: 'res' or 'qubit'
    pulse_style: type of pulse you want to send 'const', 'arb', 'flat_top'
    suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

    returns:
        nothing...
    '''

    if ch in fast_gen_chs: # fast gen channels
        expt_func.declare_gen(ch=ch, nqz=cfg['nqz_' + usage])
    elif ch in mux_gen_chs: # MUX channel
        expt_func.declare_gen(ch=ch, nqz=cfg['nqz_' + usage],
                         ro_ch=cfg['mux_ro_chs'][0], 
                         mux_freqs=cfg[usage + '_freq' + suffix], 
                         mux_gains=cfg[usage + '_gain' + suffix],
                         mux_phases=cfg[usage + '_phase'],
                         mixer_freq=cfg['mixer_freq'])
    elif ch in inter_gen_chs: # interpolated channels
        expt_func.declare_gen(ch=ch, nqz=cfg['nqz_' + usage],
                         ro_ch=cfg['ro_ch'],
                         mixer_freq=cfg['mixer_freq'])


def declare_pulse(expt_func, cfg, ch, usage = 'res', pulse_style = 'const', pulse_name = 'pulse', suffix = ''):
    '''
    declare_pulse declares a pulse to be sent properly depending on the configurations

    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    ch: The channel index that you want to declare
    usage: 'res' or 'qubit'
    pulse_style: type of pulse you want to send 'const', 'arb', 'flat_top' ('gauss' makes a gaussian pulse for us)
    pulse_name: name of the pulse we will send ('ramsey1' and 'ramsey2' are special)
    suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

    returns:
        nothing...
    '''
    if pulse_name == 'ramsey1':
        expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix], length=cfg['sigma' + suffix]*5, even_length=True)
        expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=None, 
                    style="arb", 
                    envelope="ramp" + suffix, 
                    freq=cfg[usage + '_freq' + suffix], 
                    phase=cfg[usage + '_phase'],
                    gain=cfg['half_pi_gain' + suffix], 
                    )
    elif pulse_name == 'ramsey2':
        expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix], length=cfg['sigma' + suffix]*5, even_length=True)
        expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=None, 
                    style="arb", 
                    envelope="ramp" + suffix, 
                    freq=cfg[usage + '_freq' + suffix], 
                    phase=cfg[usage + '_phase'] + cfg['wait_time']*360.0*cfg['ramsey_freq'],
                    gain=cfg['half_pi_gain' + suffix], 
                    )
    elif pulse_name == 'X90':
        expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix], length=cfg['sigma' + suffix]*5, even_length=True)
        expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=None, 
                    style="arb", 
                    envelope="ramp" + suffix, 
                    freq=cfg[usage + '_freq' + suffix], 
                    phase=cfg[usage + '_phase'],
                    gain=cfg['half_pi_gain' + suffix], 
                    )
    elif pulse_name == 'X90_neg':
        # X90 with its phase shifted by 180 deg: rotates about -X, so it undoes an
        # X90 exactly only if the two pulses really are 180 deg apart. Used by the
        # Ramsey phase-error amplification experiment.
        expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix], length=cfg['sigma' + suffix]*5, even_length=True)
        expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=None, 
                    style="arb", 
                    envelope="ramp" + suffix, 
                    freq=cfg[usage + '_freq' + suffix], 
                    phase=cfg[usage + '_phase'] - 180,
                    gain=cfg['half_pi_gain' + suffix], 
                    )
    else:
        if pulse_style == 'gauss': # for a gaussian pulse
            if (ch in fast_gen_chs) or (ch in inter_gen_chs): # fast gen channels or interpolated channels
                expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix], length=cfg['sigma' + suffix]*5, even_length=True)
                expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=None, 
                        style="arb", 
                        envelope="ramp" + suffix, 
                        freq=cfg[usage + '_freq' + suffix], 
                        phase=cfg[usage + '_phase'],
                        gain=cfg[usage + '_gain' + suffix], 
                        )
            elif ch in mux_gen_chs: # MUX channel
                expt_func.add_gauss(ch=ch, name="ramp" + suffix, sigma=cfg['sigma' + suffix], length=cfg['sigma' + suffix]*5, even_length=True)
                expt_func.add_pulse(ch=ch, name=pulse_name, 
                        style="arb", 
                        envelope="ramp" + suffix, 
                        mask=range(len(cfg[usage + '_freq' + suffix])), 
                        )
        else: # for other styles of pulses
            if (ch in fast_gen_chs) or (ch in inter_gen_chs): # fast gen channels or interpolated channels
                expt_func.add_pulse(ch=ch, name=pulse_name, ro_ch=cfg['ro_ch'], 
                            style=pulse_style, 
                            length=cfg[usage + '_length' + suffix], 
                            freq=cfg[usage + '_freq' + suffix], 
                            phase=cfg[usage + '_phase'],
                            gain=cfg[usage + '_gain' + suffix],
                            )
            elif ch in mux_gen_chs: # MUX channel
                expt_func.add_pulse(ch=ch, name= pulse_name,
                            style=pulse_style, 
                            length=cfg[usage + '_length' + suffix],
                            mask=range(len(cfg[usage + '_freq' + suffix])),
                            )

   
def initialize_ro_chs(expt_func, cfg, suffix = ''):
    '''
    initialize_ro_chs initializes the readout signal generator and readout ADC channels properly depending on the configurations
    
    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains
    suffix: do the parameters have a suffix? ('', '_ge', or '_ef')

    returns:
        nothing...
    '''

    if MUX == 'True': # for MUX readout
        ro_chs = cfg['mux_ro_chs']
        res_ch = cfg['mux_ch']
        
        # declare a MUX gen channel
        declare_gen_ch(expt_func, cfg, res_ch, usage = 'res', suffix = suffix)
        
        for ch, f, ph in zip(ro_chs, cfg['res_freq' + suffix], cfg['res_phase']):
            expt_func.declare_readout(ch=ch, length=cfg['ro_length'], freq=f, phase=ph, gen_ch=res_ch)

        # declare a pulse for the MUX channel
        declare_pulse(expt_func, cfg, res_ch, pulse_name = 'res_pulse', suffix = suffix)

    else: # for non-MUX readout
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        
        # declare the proper readout channel using
        declare_gen_ch(expt_func, cfg, res_ch, usage = 'res', suffix = suffix)

        expt_func.declare_readout(ch=ro_ch, length=cfg['ro_length'])
        expt_func.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['res_freq' + suffix], gen_ch=res_ch)
        
        # declare a pulse for the non-MUX channel
        declare_pulse(expt_func, cfg, res_ch, pulse_name='res_pulse', suffix=suffix)


def readout(expt_func, cfg):
    '''
    readout function sends a readout pulse and triggers the adc depending on the configurations
    
    expt_func: is a variable for "self" (i.e. the experiment function) in the code
    cfg: The configuration file so that you can pull values for freqs and gains

    returns:
        nothing...

    Note: we may want to rewrite this to not depend on the 'MUX' setting but rather the readout channel
          that is input. For now this should work fine, but if firmware changes or other types of
          channels are used, then we want to adjust for this.
    '''

    if MUX == 'True': # for MUX readout
        expt_func.pulse(ch=cfg['mux_ch'], name="res_pulse", t=0)
        expt_func.trigger(ros=cfg['mux_ro_chs'], pins=[0], t=cfg['trig_time'])

    else: # for non-MUX readout
        # if non-MUX then send readout configurations - useful when freq sweeping
        expt_func.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        
        expt_func.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        expt_func.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


# NOT CURRENTLY USEFUL -> this makes a QickParam for every experiment.
# In the future we could change this to make a large coniguration file 
# that contains all experiments with full configurations in different dictionaries
def SingleQubitAllExperimentParameter():
    '''
    SingleQubitAllExperimentParameter initializes all experiment configurations for a single qubit.

    returns:
        updated_cfgs: a dictionary of all experiment configurations for a single qubit
    '''
    # all experiment names possible
    var = ["tof", "res_spec_ge", "qubit_spec_ge", "time_rabi_ge", "power_rabi_ge", "qubit_temp",
           "Ramsey_ge", "SpinEcho_ge", "T1_ge", "Ramsey_ef", "res_spec_ef", "qubit_spec_ef", "power_rabi_ef",
           "Readout_Optimization_ge", "Readout_Optimization_gef"]
    updated_cfgs = {}
    for expt_name in var:
        updated_cfgs[expt_name] = add_single_qubit_experiment(expt_cfg, expt_name, QUBIT_INDEX)
    
    return updated_cfgs
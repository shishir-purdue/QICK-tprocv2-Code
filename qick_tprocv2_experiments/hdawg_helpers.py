"""
hdawg_helpers - set up the Zurich Instruments HDAWG for fast flux pulses.

Each step of the HDAWG setup is its own function, so an experiment reads as:

    device = connect_hdawg()                         # 1. connect
    reset_and_configure(device, output_range=1.0)    # 2. known state, 4x2 grouping, output range
    set_sample_clock(device, 2.4e9)                  # 3. sample clock, read back
    n = flux_samples(10.0, 2.4e9)                    # 4. pulse length in samples
    flux = load_flux_pulse(device, 0, n, ...)        # 5. compile + upload the sequence
    set_flux_gain(flux, -0.3)                        # 6. amplitude, read back
    start_flux(flux)                                 # 7. wait for RFSoC triggers
    ...run the RFSoC program...
    stop_flux(device, flux)                          # 8. gain 0, cores off, output off

Channel numbering: hdawg_channel is the front panel output counted from 0
(output 1 -> 0, output 2 -> 1, ...). In the 4x2 grouping used here AWG core
n drives outputs 2n and 2n + 1.

Only core 0 / output 1 has been used on the instrument so far. Other
channels use the factory default trigger routing of their core, which has
not been checked.
"""

import time

import numpy as np
from zhinst.toolkit import Session

# HDAWG waveforms are a whole number of 16-sample blocks
GRANULARITY = 16


# ---------------------------------------------------------------- 1. connect
def connect_hdawg(device_id='devxxxx', host='xxx.xxx.xxx.xxx', port="xxxx"):
    """
    Connect to the HDAWG through the LabOne data server.

    device_id: the instrument serial, as LabOne shows it
    host, port: the data server (the PC running LabOne)

    returns: the zhinst.toolkit device
    """
    session = Session(host, port)
    print("HDAWG: devices visible:", session.devices.visible())
    device = session.connect_device(device_id)
    print("HDAWG: connected to", session.devices.connected())
    return device


# ------------------------------------------------------- 2. reset + outputs
def reset_and_configure(device, output_range=1.0, grouping=0):
    """
    Put the HDAWG in a known state: factory reset, channel grouping, output
    range on every output, and every AWG core off.

    The factory reset also resets the sample clock (to 2.4 GS/s) and the
    trigger input settings, so call set_sample_clock afterwards.

    output_range: output range of every channel, V; the flux pulse voltage
                  is gain x output_range
    grouping: 0 = 4x2 (four cores, two outputs each)
    """
    device.factory_reset()
    n_outputs = len(device.sigouts)
    with device.set_transaction():                 # one network round trip for all settings
        device.system.awg.channelgrouping(grouping)
        for ch in range(n_outputs):
            device.sigouts[ch].range(output_range)
    for core in range(len(device.awgs)):
        device.awgs[core].enable(0)
    print(f"HDAWG: reset, grouping {grouping}, output range {output_range} V on {n_outputs} outputs, all cores off")


# ------------------------------------------------------------ 3. sample clock
def set_sample_clock(device, rate):
    """
    Set the sample clock and check what the instrument reports.

    rate: samples per second (at most 2.4e9). The sequencer clock is rate / 8.

    raises RuntimeError if the instrument does not run at the rate asked
    for, since every pulse length would then be wrong.
    """
    if rate > 2.4e9:
        raise ValueError(f"sample clock {rate} Hz is above the HDAWG maximum of 2.4 GS/s")
    device.system.clocks.sampleclock.freq(rate)
    device.system.clocks.sampleclock.status.wait_for_state_change(0, timeout=30)   # 0 = locked
    actual = device.system.clocks.sampleclock.freq()
    print(f"HDAWG: sample clock {actual:.3f} Hz (asked {rate:.3f} Hz)")
    if abs(actual - rate) > 0.5:
        raise RuntimeError(f"the HDAWG runs at {actual} Hz instead of {rate} Hz")
    return actual


# ------------------------------------------------------ 4. lengths in samples
def flux_samples(length_us, rate):
    """
    Number of samples in a flux pulse of length_us, rounded to the nearest
    16 samples (6.67 ns at 2.4 GS/s). Use samples / rate for the length the
    HDAWG actually plays.
    """
    return GRANULARITY * int(round(length_us * 1e-6 * rate / GRANULARITY))


# ------------------------------------------------------- 5. sequence program
def flux_pulse_seqc(n_samples, compensation_wait_us, rate, local_channel=0):
    """
    SeqC for one square flux pulse per trigger, followed by the same pulse
    inverted (compensation) after compensation_wait_us.

    The waveform is full scale (amplitude 1): the pulse amplitude is the
    output gain, set with set_flux_gain, so changing it needs no recompile.

    n_samples: flux pulse length in samples (from flux_samples)
    local_channel: 0 or 1, the output within the AWG core
    """
    wait_cycles = int(round(compensation_wait_us * 1e-6 * rate / 8))    # sequencer cycles
    index = local_channel + 1                                          # playWave counts outputs from 1
    return f"""
    // sample clock {rate:.0f} Hz, sequencer clock = sample clock / 8
    const pulse_samples = {n_samples};    // {n_samples / rate * 1e6:.6f} us
    const wait_cycles = {wait_cycles};    // {compensation_wait_us} us, in sequencer cycles

    wave w_rect = 1.0*rect(pulse_samples, 1);   // full scale; the output gain sets the amplitude

    while (true) {{
        waitDigTrigger(1);                          // RFSoC trigger
        playWave({index}, w_rect);                  // flux pulse
        wait(wait_cycles + pulse_samples / 8);      // pulse length plus the compensation wait
        playWave({index}, -1*w_rect);               // compensation pulse
    }}
    """


def load_flux_pulse(device, hdawg_channel, n_samples, compensation_wait_us, rate):
    """
    Compile and upload the flux pulse sequence for one output, with the
    output gain at 0 and the core stopped. The output is switched on.

    hdawg_channel: front panel output counted from 0 (output 1 -> 0)

    returns: a dict with the AWG core, the gain node and the channel, for
             set_flux_gain / start_flux / stop_flux
    """
    core_index, local = divmod(hdawg_channel, 2)        # 4x2 grouping: two outputs per core
    awg_core = device.awgs[core_index]
    gain_node = awg_core.outputs[local].gains[local]    # /awgs/<core>/outputs/<local>/gains/<local>

    awg_core.enable(0)
    gain_node(0.0)                                      # silent while loading
    awg_core.load_sequencer_program(flux_pulse_seqc(n_samples, compensation_wait_us, rate, local))
    device.sigouts[hdawg_channel].on(True)
    print(f"HDAWG: flux pulse loaded on output {hdawg_channel + 1} (core {core_index}), "
          f"{n_samples} samples = {n_samples / rate * 1e6:.6f} us")
    return dict(awg_core=awg_core, gain_node=gain_node, channel=hdawg_channel)


# ------------------------------------------------------------- 6. amplitude
def set_flux_gain(flux, gain, max_gain=0.5):
    """
    Set the flux pulse amplitude (output gain, -1 to 1 of the output range)
    and read it back.

    max_gain: hard limit on |gain|; anything above is refused
    raises RuntimeError if the instrument reports a different gain
    """
    if abs(gain) > max_gain + 1e-12:
        raise ValueError(f"flux gain {gain} is above the limit of {max_gain}")
    flux['gain_node'](gain)
    actual = float(flux['gain_node']())
    if abs(actual - gain) > 1e-3:
        raise RuntimeError(f"asked the HDAWG for gain {gain}, it reports {actual}")
    print(f"HDAWG: flux gain {actual:+.4f}")
    return actual


# --------------------------------------------------------- 7. start / 8. stop
def start_flux(flux, settle_s=0.5):
    """Start the core: it plays the flux pulse on every RFSoC trigger from now on."""
    flux['awg_core'].single(True)
    flux['awg_core'].enable(1)
    time.sleep(settle_s)


def stop_flux(device, flux=None):
    """Gain to 0, every AWG core off, and the flux output off. Safe to call after an error."""
    if flux is not None:
        flux['gain_node'](0.0)
    for core in range(len(device.awgs)):
        device.awgs[core].enable(0)
    if flux is not None:
        device.sigouts[flux['channel']].on(False)
    print("HDAWG: flux off, all cores stopped")


# ------------------------------------------------------------ timing check
def check_trigger_spacing(ff_length_us, compensation_wait_us, trig_offset_us, shot_period_us):
    """
    Refuse to run if the HDAWG is still busy (flux pulse, wait, compensation
    pulse) when the next RFSoC trigger arrives: that trigger would be missed.

    shot_period_us: the shortest time between two triggers (roughly the
                    readout end plus relax_delay)
    """
    busy = trig_offset_us + 2 * ff_length_us + compensation_wait_us
    if busy + 1 > shot_period_us:
        raise ValueError(f"the HDAWG needs {busy:.2f} us per trigger (pulse, {compensation_wait_us} us wait, "
                         f"compensation) but triggers come every {shot_period_us:.2f} us. "
                         "Increase relax_delay or shorten the flux pulse.")

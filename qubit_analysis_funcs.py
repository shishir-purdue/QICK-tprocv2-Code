"""
qubit_analysis_funcs.py
=======================
Cleaned-up analysis and plotting functions for single-qubit characterization
experiments. Each function loads an HDF5 data file, plots the results, prints
key values, and (where noted) returns useful arrays.

Dependencies
------------
- numpy, matplotlib, json
- fitting (lab fitting library: fitdecaysin, etc.)
- qualang_tools.plot.Fit
- plotting_helpers (hist, SlabFile, array, etc.)
- dsfit_copy (local copy of malab.dsfit; fithanger_new_withQc, hangerfunc_new_withQc)
- circle_fit_dcm (notch-type circle fit, Diameter Correction Method)
- scipy (for curve_fit in double-Gaussian fitting)

Usage
-----
    from qubit_analysis_funcs import *

    # Example: run a resonator spectroscopy analysis
    analyze_resonator_spectroscopy(
        global_data_path=r"C:/Users/G41Lab/Dropbox/_Data/...",
        filename=r"00000_res_spec_ge_Q1_2026-06-06.h5",
    )

    # Example: convert a DAC gain to dBm/mW from a single calibration point
    dac_to_power(dac=0.5, dac_ref=0.2, dbm_ref=-10.0)               # -> dBm
    dac_to_power(dac=0.5, dac_ref=0.2, dbm_ref=-10.0, units='mW')   # -> mW
    power_to_dac(power=-6.0, dac_ref=0.2, dbm_ref=-10.0)            # -> DAC gain

    # Example: analyze an overnight folder of T1 runs vs. time
    analyze_T1_temporal_scan(
        global_data_path=r"C:/Users/G41Lab/Dropbox/_Data/...",
        folder_name="2026-08-17overnight _T1_ge",
    )
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import pprint as pp
import os
import datetime

from fitting import *
from qualang_tools.plot import Fit
from plotting_helpers import *
from dsfit_copy import fithanger_new_withQc, hangerfunc_new_withQc
from circle_fit_dcm import notch_circlefit, model_S21

# Default color cycle for multi-trace plots
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']


# =============================================================================
#  Unit conversion utilities
# =============================================================================

def dac_to_power(dac, dac_ref, dbm_ref, units='dBm'):
    """
    Convert a DAC gain value to output RF power, calibrated from a single
    reference point.

    Assumes DAC gain (in [0, 1]) scales linearly with the pulse's output
    voltage amplitude (true for QICK gen-channel gain), so power follows
    dBm = dbm_ref + 20*log10(dac/dac_ref).

    Parameters
    ----------
    dac : float or array
        DAC gain value(s) to convert, in [0, 1].
    dac_ref : float
        Reference DAC gain used for calibration, in [0, 1].
    dbm_ref : float
        Measured/known power in dBm at dac_ref.
    units : str
        'dBm' (default) or 'mW' for the returned power units.

    Returns
    -------
    power : float or array
        Power in the requested units.
    """
    dbm = dbm_ref + 20 * np.log10(dac / dac_ref)
    if units == 'dBm':
        return dbm
    elif units == 'mW':
        return 10 ** (dbm / 10)
    else:
        raise ValueError(f"Unknown units: {units!r} (use 'dBm' or 'mW')")


def power_to_dac(power, dac_ref, dbm_ref, units='dBm'):
    """
    Convert a target RF power to the DAC gain value that should produce it,
    calibrated from a single reference point (inverse of ``dac_to_power``).

    Parameters
    ----------
    power : float or array
        Target power, in the units given by ``units``.
    dac_ref : float
        Reference DAC gain used for calibration, in [0, 1].
    dbm_ref : float
        Measured/known power in dBm at dac_ref.
    units : str
        Units of ``power``: 'dBm' (default) or 'mW'.

    Returns
    -------
    dac : float or array
        DAC gain value(s) implied by ``power``. Not clipped to [0, 1] --
        check the result is in range before using it as a gain setting.
    """
    if units == 'dBm':
        dbm = power
    elif units == 'mW':
        dbm = 10 * np.log10(power)
    else:
        raise ValueError(f"Unknown units: {units!r} (use 'dBm' or 'mW')")
    return dac_ref * 10 ** ((dbm - dbm_ref) / 20)


# =============================================================================
#  Helper functions
# =============================================================================

def _build_path(global_data_path, filename):
    """Join the global data directory and the experiment filename."""
    return global_data_path + "\\" + filename


def _load_config(slab_file_handle):
    """Extract and parse the JSON config from an open SlabFile."""
    return json.loads(slab_file_handle.attrs['config'])


def _stamp_data_path(data_path, fig=None):
    """Print the source data path in small text at the bottom of the figure."""
    if fig is None:
        fig = plt.gcf()
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02, data_path, ha='center', va='bottom', fontsize=7, color='black')


def _normalize_population(I, Q, Ig, Qg, Ie, Qe,
                           ef=False, If=None, Qf=None):
    """
    Compute normalised qubit population from averaged IQ data and
    single-shot reference blobs.

    Parameters
    ----------
    I, Q : array
        Averaged I/Q quadrature data from the experiment sweep.
    Ig, Qg, Ie, Qe : array
        Single-shot reference means for |g> and |e>.
    ef : bool
        If True, normalise for e-f transition instead of g-e.
    If, Qf : array or None
        Single-shot reference for |f>, required when ef=True.

    Returns
    -------
    pop_norm : array
        Normalised population.
    """
    e = np.mean(Ie + 1j * Qe)
    g = np.mean(Ig + 1j * Qg)

    if ef:
        f = np.mean(If + 1j * Qf)
        pop_norm = np.abs(((I + 1j * Q) - e) * (f - e) / abs(f - e) ** 2)
    else:
        pop_norm = np.abs(((I + 1j * Q) - g) * (e - g) / abs(e - g) ** 2)

    return pop_norm


def _compute_fidelity_by_projection(I_g, Q_g, I_e, Q_e,
                                     bin_num=100,
                                     n_threshold_pts=1000,
                                     plot=True,
                                     title_prefix=""):
    """
    Fidelity estimate — Method 1 (manual IQ projection + threshold sweep).

    Projects both blobs onto the line connecting their centroids, then sweeps
    a discrimination threshold to find the maximum fidelity.

    Parameters
    ----------
    I_g, Q_g : array   Single-shot I/Q for |g>.
    I_e, Q_e : array   Single-shot I/Q for |e>.
    bin_num : int       Number of histogram bins.
    n_threshold_pts : int  Number of threshold points to sweep.
    plot : bool         Show the projected histogram.
    title_prefix : str  Optional prefix for the plot title.

    Returns
    -------
    fidelity : float    Best fidelity (contrast), range [0, 1].
    sigma_g : float     Std dev of projected |g> distribution.
    sigma_e : float     Std dev of projected |e> distribution.
    """
    vec_I = np.mean(I_e) - np.mean(I_g)
    vec_Q = np.mean(Q_e) - np.mean(Q_g)
    norm = abs(vec_I + 1j * vec_Q) ** 2

    gstate = ((I_g - np.mean(I_g)) * vec_I + (Q_g - np.mean(Q_g)) * vec_Q) / norm
    estate = ((I_e - np.mean(I_g)) * vec_I + (Q_e - np.mean(Q_g)) * vec_Q) / norm

    if plot:
        plt.figure()
        plt.hist(gstate, bins=bin_num, label='g', density=True)
        plt.hist(estate, bins=bin_num, label='e', alpha=0.7, density=True)
        plt.legend()

    th_list = np.linspace(min(gstate), max(estate), n_threshold_pts)
    fidelity_curve = [
        (np.sum(gstate < th) + np.sum(estate > th)) / len(gstate) / 2
        for th in th_list
    ]
    best_fid = max(fidelity_curve) * 2 - 1
    sigma_g = np.std(gstate)
    sigma_e = np.std(estate)

    if plot:
        plt.title(f"{title_prefix}Readout Fidelity: {best_fid:.3f}")
        plt.show()

    print(f"  Method 1 fidelity = {best_fid:.4f}")
    print(f"  sigma_g = {sigma_g:.4f}")
    print(f"  sigma_e = {sigma_e:.4f}")

    return best_fid, sigma_g, sigma_e


def _compute_fidelity_by_hist(I_g, Q_g, I_e, Q_e,
                               plot=True, ran=50):
    """
    Fidelity estimate — Method 2 (using the ``hist`` helper from
    plotting_helpers, which performs an optimal-angle rotation).

    Parameters
    ----------
    I_g, Q_g : array   Single-shot I/Q for |g>.
    I_e, Q_e : array   Single-shot I/Q for |e>.
    plot : bool         Show the rotated histogram.
    ran : float         Histogram range parameter passed to ``hist()``.

    Returns
    -------
    fid : float         Fidelity after rotation.
    threshold : float   Optimal discrimination threshold.
    angle : float       Rotation angle used.
    """
    fid, threshold, angle = hist(
        data=[I_g, Q_g, I_e, Q_e], plot=plot, ran=ran
    )
    print(f"  Method 2 fidelity = {fid:.3f}")
    print(f"  threshold = {threshold:.4f},  angle = {angle:.4f}")
    return fid, threshold, angle


def _compute_fidelity_gf_projection(I_g, Q_g, I_f, Q_f,
                                     bin_num=100,
                                     n_threshold_pts=1000,
                                     plot=True):
    """
    Fidelity by projection for g-f discrimination.

    Same projection method as Method 1 but between |g> and |f> blobs.
    """
    vec_I = np.mean(I_f) - np.mean(I_g)
    vec_Q = np.mean(Q_f) - np.mean(Q_g)
    norm = abs(vec_I + 1j * vec_Q) ** 2

    gstate = ((I_g - np.mean(I_g)) * vec_I + (Q_g - np.mean(Q_g)) * vec_Q) / norm
    fstate = ((I_f - np.mean(I_g)) * vec_I + (Q_f - np.mean(Q_g)) * vec_Q) / norm

    if plot:
        plt.figure()
        plt.hist(gstate, bins=bin_num, label='g')
        plt.hist(fstate, bins=bin_num, label='f', alpha=0.7, color='g')
        plt.legend()
        plt.title("Readout g-f")
        plt.show()

    th_list = np.linspace(min(gstate), max(fstate), n_threshold_pts)
    fidelity_curve = [
        (np.sum(gstate < th) + np.sum(fstate > th)) / len(gstate) / 2
        for th in th_list
    ]
    best_fid = max(fidelity_curve) * 2 - 1

    print(f"  g-f fidelity = {best_fid:.4f}")
    print(f"  sigma_g = {np.std(gstate):.4f}")
    print(f"  sigma_f = {np.std(fstate):.4f}")
    return best_fid


def _compute_fidelity_ef_projection(I_e, Q_e, I_f, Q_f,
                                     bin_num=100,
                                     n_threshold_pts=1000,
                                     plot=True):
    """
    Fidelity by projection for e-f discrimination.

    Same projection method as Method 1 but between |e> and |f> blobs.
    """
    vec_I = np.mean(I_f) - np.mean(I_e)
    vec_Q = np.mean(Q_f) - np.mean(Q_e)
    norm = abs(vec_I + 1j * vec_Q) ** 2

    estate = ((I_e - np.mean(I_e)) * vec_I + (Q_e - np.mean(Q_e)) * vec_Q) / norm
    fstate = ((I_f - np.mean(I_e)) * vec_I + (Q_f - np.mean(Q_e)) * vec_Q) / norm

    if plot:
        plt.figure()
        plt.hist(estate, bins=bin_num, label='e')
        plt.hist(fstate, bins=bin_num, label='f', alpha=0.7)
        plt.legend()
        plt.title("Readout e-f")
        plt.show()

    th_list = np.linspace(min(estate), max(fstate), n_threshold_pts)
    fidelity_curve = [
        (np.sum(estate < th) + np.sum(fstate > th)) / len(estate) / 2
        for th in th_list
    ]
    best_fid = max(fidelity_curve) * 2 - 1

    print(f"  e-f fidelity = {best_fid:.4f}")
    print(f"  sigma_e = {np.std(estate):.4f}")
    print(f"  sigma_f = {np.std(fstate):.4f}")
    return best_fid


def _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix=""):
    """
    Plot the g/e single-shot calibration blobs and compute their fidelity.

    Called by experiment functions when both SS=True and plot_cal=True.

    Parameters
    ----------
    Ig, Qg : array   Single-shot I/Q for |g> calibration.
    Ie, Qe : array   Single-shot I/Q for |e> calibration.
    title_prefix : str  Optional label prepended to plot titles.
    """
    print(f"  --- {title_prefix}g-e calibration ---")

    # IQ scatter
    plt.figure()
    plt.title(f"{title_prefix}Calibration IQ Blob")
    plt.plot(Ig, Qg, '.', label='g')
    plt.plot(Ie, Qe, '.', label='e')
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.legend()
    plt.axis('square')
    plt.show()

    # Fidelity via projection
    _compute_fidelity_by_projection(Ig, Qg, Ie, Qe,
                                     plot=True,
                                     title_prefix=f"{title_prefix}Cal ")


# =============================================================================
#  Experiment functions
# =============================================================================

def analyze_time_of_flight(global_data_path, filename, show_plot=True):
    """
    Time-of-flight measurement.

    Loads the TOF data and plots I, Q, and magnitude vs time.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00000_tof_Q1_2026-06-06.h5".
    show_plot : bool
        If True, add a legend and display the figure.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        t = array(f['t'])[0]
        I = array(f['avgi'])[0]
        Q = array(f['avgq'])[0]
        config = _load_config(f)

    fig = plt.figure()
    plt.plot(t, I, label="I value")
    plt.plot(t, Q, label="Q value")
    plt.plot(t, np.abs(I + 1j * Q), label="magnitude")
    if show_plot:
        plt.legend()
    plt.ylabel("ADC Units")
    plt.xlabel("us")
    plt.title("Time of Flight Experiment")
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_resonator_spectroscopy(global_data_path, filename,
                                    fit_resonator=True, fit_method='circle',
                                    logscale=False, show_plot=True):
    """
    Single-resonator spectroscopy (amplitude + phase).

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00000_res_spec_ge_Q1_2026-06-06.h5".
    fit_resonator : bool
        If True, fit the resonator lineshape using ``fit_method``.
    fit_method : str
        "circle" (default) - notch-type circle fit using the Diameter
            Correction Method (``circle_fit_dcm.notch_circlefit``), fit
            directly to the complex avgi/avgq data. This is the standard
            method for extracting Qi/Qc/Ql and correctly accounts for
            impedance-mismatch asymmetry and cable delay.
        "hanger" - hanger-with-asymmetry lineshape fit
            (``hangerfunc_new_withQc`` from dsfit_copy), fit in dB scale
            to the amplitude data only.
    logscale : bool
        If True, plot the amplitude data and fit in dB (log) scale instead
        of linear a.u.
    show_plot : bool
        If True, add a legend and display both figures.

    Returns
    -------
    fig_amp, fig_phase : matplotlib Figure
        Handles to the amplitude and phase figures, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        freqs = array(f['fpts'])[0]
        amps = array(f['amps'])[0]
        phases = array(f['phases'])[0]
        avgi = array(f['avgi'])[0]
        avgq = array(f['avgq'])[0]
        config = _load_config(f)

    amps_db = 20 * np.log10(amps)

    # --- Amplitude plot ---
    fig_amp = plt.figure(figsize=(12, 6))
    plt.plot(freqs, amps_db if logscale else amps, marker='o', label='data')
    plt.ylabel("dB" if logscale else "a.u.")
    plt.xlabel("Resonator Frequency (MHz)")
    plt.title(f"Resonator Spectroscopy, Gain = {config['res_gain_ge']}")

    if fit_resonator:
        freqs_fit = np.linspace(freqs[0], freqs[-1], 1000)

        if fit_method == 'circle':
            # Diameter Correction Method circle fit on the complex S21 data.
            z_data = avgi + 1j * avgq
            fit = notch_circlefit(freqs, z_data, correct_delay=True)
            fit_curve_lin = np.abs(model_S21(freqs_fit, fit['fr'], fit['Ql'],
                                              fit['Qc'], fit['phi0'],
                                              fit['P_off'], fit['delay']))
            fit_curve = 20 * np.log10(fit_curve_lin) if logscale else fit_curve_lin
            f0 = fit['fr']

            plt.plot(freqs_fit, fit_curve, c='red', label='fit')
            plt.axvline(f0, c='red', linestyle='--', label=f'fit f0: {f0:.3f}')
            plt.axvline(freqs[np.argmin(amps)], c='blue', linestyle='--',
                        label=f"min amp: {freqs[np.argmin(amps)]:.3f}")
            print(f"  Min amp at {freqs[np.argmin(amps)]} MHz")
            print("  Circle fit (DCM) results:")
            print(f"    f0 = {fit['fr']} MHz")
            print(f"    Ql = {fit['Ql']}")
            print(f"    Qc = {fit['Qc']}")
            print(f"    Qi = {fit['Qi']}")
            print(f"    phi0 = {fit['phi0']} rad")
            print(f"    delay = {fit['delay']} us")

        elif fit_method == 'hanger':
            # hangerfunc_new_withQc fits in dB, so fit on the log-scale data
            # regardless of the display scale, then undo the log for a linear
            # display curve via the inverse of the dB equation (amp = 10**(dB/20)).
            f0, Qi, Qc, df, scale = fithanger_new_withQc(freqs, amps_db, showfit=False)
            fit_curve_db = hangerfunc_new_withQc([f0, Qi, Qc, df, scale], freqs_fit)
            fit_curve = fit_curve_db if logscale else 10 ** (fit_curve_db / 20)

            plt.plot(freqs_fit, fit_curve, c='red', label='fit')
            plt.axvline(f0, c='red', linestyle='--', label=f'fit f0: {f0:.3f}')
            plt.axvline(freqs[np.argmin(amps)], c='blue', linestyle='--',
                        label=f"min amp: {freqs[np.argmin(amps)]:.3f}")
            print(f"  Min amp at {freqs[np.argmin(amps)]} MHz")
            print("  Hanger fit results:")
            print(f"    f0 = {f0} MHz")
            print(f"    Qi = {Qi}")
            print(f"    Qc = {Qc}")
            print(f"    df = {df} MHz")
            print(f"    scale = {scale:.4f}")

        else:
            raise ValueError(f"Unknown fit_method: {fit_method!r} (use 'circle' or 'hanger')")

    if show_plot:
        plt.legend()
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    # --- Phase plot ---
    fig_phase = plt.figure(figsize=(12, 6))
    plt.plot(freqs, np.unwrap(phases), marker='o', label='phase data')
    plt.ylabel("radians")
    plt.xlabel("Resonator Frequency (MHz)")
    plt.title("Resonator Spectroscopy Phase")
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig_amp, fig_phase


def analyze_qubit_spectroscopy(global_data_path, filename,
                                SS=False, ef=False, dac=0.0,
                                plot_cal=False, show_plot=True):
    """
    Single-qubit spectroscopy.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00000_qubit_spec_ge_Q1_2026-06-06.h5".
    SS : bool
        True if single-shot normalisation data was collected.
    ef : bool
        True if this is an e-f spectroscopy experiment.
    dac : float
        DAC current in mA (for plot labelling).
    plot_cal : bool
        If True (and SS=True), plot the g/e calibration IQ blobs and fidelity.
    show_plot : bool
        If True, add a legend and display the figure.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        freqs = array(f['fpts'])[0]
        I = array(f['avgi'])[0]
        Q = array(f['avgq'])[0]
        amps = array(f['amps'])[0]
        config = _load_config(f)

        if SS:
            Ig = array(f['I_g'])[0]
            Qg = array(f['Q_g'])[0]
            Ie = array(f['I_e'])[0]
            Qe = array(f['Q_e'])[0]
            If_data = array(f['I_f'])[0] if ef else None
            Qf_data = array(f['Q_f'])[0] if ef else None

    if SS:
        if plot_cal:
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix="Qubit Spec ")
        ydata = _normalize_population(I, Q, Ig, Qg, Ie, Qe,
                                       ef=ef, If=If_data, Qf=Qf_data)
    else:
        ydata = amps

    fig = plt.figure(figsize=(12, 6))
    plt.plot(freqs, ydata, marker='o', label='data')
    if SS:
        plt.ylabel("Qubit Population")
        plt.axhline(1, color='black', linestyle='--')
    else:
        plt.ylabel("a.u.")
    plt.xlabel("Qubit Frequency (MHz)")
    plt.title(f"Qubit Spectroscopy:  Gain = {config['qubit_gain_ge']}, "
              f"Sigma (us) = {config['sigma_ge']}, DAC (mA) = {dac}")

    fit = Fit()
    fit_results = fit.transmission_resonator_spectroscopy(freqs, ydata, plot=show_plot)
    plt.axvline(fit_results['f'][0], c='red', linestyle='--')
    print("  Fit results:")
    pp.pprint(fit_results)

    if show_plot:
        plt.legend()
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_length_rabi(global_data_path, filename,
                         SS=False, ef=False,
                         plot_cal=False, show_plot=True):
    """
    Length (time) Rabi oscillation experiment.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00000_time_rabi_ge_Q1_2026-05-31.h5".
    SS : bool
        True if single-shot normalisation data was collected.
    ef : bool
        True if this is an e-f Rabi experiment.
    plot_cal : bool
        If True (and SS=True), plot the g/e calibration IQ blobs and fidelity.
    show_plot : bool
        If True, add a legend and display the figure.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        lengths = array(f['lengths'])[0]
        I = array(f['avgi'])[0]
        Q = array(f['avgq'])[0]
        amps = array(f['amps'])[0]
        config = _load_config(f)

        if SS:
            Ig = array(f['I_g'])[0]
            Qg = array(f['Q_g'])[0]
            Ie = array(f['I_e'])[0]
            Qe = array(f['Q_e'])[0]
            If_data = array(f['I_f'])[0] if ef else None
            Qf_data = array(f['Q_f'])[0] if ef else None

    if SS:
        if plot_cal:
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix="Length Rabi ")
        ydata = _normalize_population(I, Q, Ig, Qg, Ie, Qe,
                                       ef=ef, If=If_data, Qf=Qf_data)
    else:
        ydata = amps

    fig = plt.figure(figsize=(12, 6))
    plt.plot(lengths, ydata, marker='o', label='data')
    if SS:
        plt.ylabel("Qubit Population")
        plt.axhline(1, linestyle='--', c='gray')
        plt.axhline(0, linestyle='--', c='gray')
    else:
        plt.ylabel("a.u.")
    plt.xlabel(r"Pulse Length ($\mu s$)")
    plt.title(f"Length Rabi Oscillations (Qubit gain = {config['qubit_gain_ge']:.2f})")

    # Fit with decaying sinusoid
    fit_results = fitdecaysin(lengths, ydata, showfit=show_plot)
    print("  fitdecaysin results:")
    pp.pprint(fit_results)

    peak = lengths[np.argmax(ydata[lengths < 0.07])]
    print(f"  Pi-pulse length (first peak): {peak:.4f} us")
    print(f"  Rabi frequency: {fit_results[1]:.4f} MHz")

    # Plot fit curve
    x = np.linspace(0, lengths[-1], 1000)
    y = (fit_results[0]
         * np.sin(2.0 * np.pi * fit_results[1] * x + fit_results[2] * np.pi / 180.0)
         * np.exp(-1.0 * (x - lengths[0]) / fit_results[3])
         + fit_results[4])
    plt.plot(x, y, label='fit')
    plt.axvline(peak, linestyle='--', c='red',
                label=f'pi pulse: {peak:.3f} us')
    if show_plot:
        plt.legend()
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_amplitude_rabi(global_data_path, filename,
                            SS=False, ef=False,
                            gain_cutoff_low=0.0, gain_cutoff_high=1.0,
                            plot_cal=False, show_plot=True,
                            full_pi_offset=0.0, half_pi_offset=0.0, title_append=""):
    """
    Amplitude (power) Rabi oscillation experiment.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00001_power_rabi_ge_Q1_2026-06-06.h5".
    SS : bool
        True if single-shot normalisation data was collected.
    ef : bool
        True if this is an e-f Rabi experiment.
    gain_cutoff_low : float
        Lower bound of gain range to include in the fit.
    gain_cutoff_high : float
        Upper bound of gain range to include in the fit.
    plot_cal : bool
        If True (and SS=True), plot the g/e calibration IQ blobs and fidelity.
    show_plot : bool
        If True, add a legend and display the figure.
    full_pi_offset : float
        Added to the fitted pi-pulse gain (peak).
    half_pi_offset : float
        Added to the fitted pi/2-pulse gain (half_pi).

    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        gains = array(f['gains'])[0]
        I = array(f['avgi'])[0]
        Q = array(f['avgq'])[0]
        amps = array(f['amps'])[0]
        config = _load_config(f)

        if SS:
            Ig = array(f['I_g'])[0]
            Qg = array(f['Q_g'])[0]
            Ie = array(f['I_e'])[0]
            Qe = array(f['Q_e'])[0]
            If_data = array(f['I_f'])[0] if ef else None
            Qf_data = array(f['Q_f'])[0] if ef else None

    if SS:
        if plot_cal:
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix="Amp Rabi ")
        ydata = _normalize_population(I, Q, Ig, Qg, Ie, Qe,
                                       ef=ef, If=If_data, Qf=Qf_data)
    else:
        ydata = amps

    fig = plt.figure(figsize=(12, 6))
    if SS:
        plt.ylabel("Qubit Population")
        plt.axhline(1, color='black', linestyle='--')
        plt.axhline(0, color='black', linestyle='--')
        plt.axhline(0.5, color='gray', linestyle='--')
    else:
        plt.ylabel("a.u.")
    plt.xlabel("Pulse Gain (DAC units)")

    # Fit within the specified gain window
    cutoff = (gains >= gain_cutoff_low) & (gains <= gain_cutoff_high)
    fit_results = fitdecaysin(gains[cutoff], ydata[cutoff], showfit=True)
    print("  fitdecaysin results:")
    pp.pprint(fit_results)

    peak = 0.5 / fit_results[1]
    half_pi = peak / 2.0
    peak += full_pi_offset
    half_pi += half_pi_offset
    print(f"  Pi-pulse gain:    {peak:.4f} DAC units")
    print(f"  Pi/2-pulse gain:  {half_pi:.4f} DAC units")
    print(f"  Max population:   {ydata[np.argmax(ydata)]:.4f}")
    print(f"  Phase offset (deg from fit): "
          f"{(fit_results[2] * 180 / np.pi) / (2 * np.pi * fit_results[1]):.4f}")

    plt.axvline(peak, c='red', label=f'pi pulse: {peak:.3f} DAC')
    plt.axvline(half_pi, c='red', linestyle='--',
                label=f'pi/2 pulse: {half_pi:.3f} DAC')
    plt.title(f"Amplitude Rabi Oscillations (Sigma = {config['sigma_ge']:.2f})\n" + title_append)
    if show_plot:
        plt.legend()
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_t2_ramsey(global_data_path, filename, my_offset=0.0,
                       SS=False, ef=False,
                       plot_cal=False, show_plot=True):
    """
    T2 Ramsey fringe experiment.

    Prints the fitted T2* and the corrected qubit frequency.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00001_Ramsey_ge_Q1_2026-06-07.h5".
    SS : bool
        True if single-shot normalisation data was collected.
    ef : bool
        True if this is an e-f Ramsey experiment.
    plot_cal : bool
        If True (and SS=True), plot the g/e calibration IQ blobs and fidelity.
    show_plot : bool
        If True, add a legend and display the figure.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        delay_times = array(f['delay_times'])[0]
        delay_times -= delay_times[0]
        I = array(f['avgi'])[0]
        Q = array(f['avgq'])[0]
        amps = array(f['amps'])[0]
        config = _load_config(f)

        if SS:
            Ig = array(f['I_g'])[0]
            Qg = array(f['Q_g'])[0]
            Ie = array(f['I_e'])[0]
            Qe = array(f['Q_e'])[0]
            If_data = array(f['I_f'])[0] if ef else None
            Qf_data = array(f['Q_f'])[0] if ef else None

    if SS:
        if plot_cal:
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix="Ramsey ")
        ydata = _normalize_population(I, Q, Ig, Qg, Ie, Qe,
                                       ef=ef, If=If_data, Qf=Qf_data)
    else:
        ydata = amps

    x_ns = delay_times * 1000  # convert us -> ns

    fig = plt.figure(figsize=(12, 6))
    plt.plot(x_ns, ydata, marker='o', label='data')
    if SS:
        plt.ylabel("Qubit Population")
        plt.axhline(1, color='black', linestyle='--')
        plt.axhline(0, color='black', linestyle='--')
    else:
        plt.ylabel("a.u.")
    plt.xlabel("Delay Time (ns)")
    plt.title(f"Ramsey Fringe Experiment, (Ramsey frequency = {config['ramsey_freq']:.2f} MHz)"
              f"\n$\\sigma$ = {config['sigma_ge']:.2f} $\\mu$s"
              f", $\\pi / 2$ Gain = {config['half_pi_gain']}")

    fit = Fit()
    fit_results = fit.ramsey(x_ns, ydata, plot=show_plot)
    print("  Ramsey fit results:")
    pp.pprint(fit_results)

    # Decay envelope
    T2 = fit_results['T2'][0]
    offset = fit_results['final_offset'][0] + my_offset
    amp = fit_results['amp'][0]
    exp_lower = (amp / 2) * (-np.exp(-x_ns / T2)) + offset
    exp_upper = (amp / 2) * np.exp(-x_ns / T2) + offset
    plt.plot(x_ns, exp_lower, label='fit lower bound')
    plt.plot(x_ns, exp_upper, label='fit upper bound')
    plt.axhline(offset, color='black', linestyle='--',
                label=f"offset: {offset:.3f}")
    if show_plot:
        plt.legend()
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    # Corrected qubit frequency
    new_freq_ge = (config['qubit_freq_ge']
                   - (fit_results['f'][0] * 1e3 - config['ramsey_freq']))
    print(f"  New Qubit frequency: {new_freq_ge:.6f} MHz")

    return fig


def analyze_t2_spin_echo(global_data_path, filename,
                          SS=False, plot_cal=False, show_plot=True):
    """
    T2 spin-echo experiment.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00001_SpinEcho_ge_Q3_2024-11-03.h5".
    SS : bool
        True if single-shot normalisation data was collected.
    plot_cal : bool
        If True (and SS=True), plot the g/e calibration IQ blobs and fidelity.
    show_plot : bool
        If True, add a legend and display the figure.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        delay_times1 = array(f['delay_times1'])[0]
        delay_times2 = array(f['delay_times2'])[0]
        I = array(f['avgi'])[0]
        Q = array(f['avgq'])[0]
        amps = array(f['amps'])[0]
        config = _load_config(f)

        if SS:
            Ig = array(f['I_g'])[0]
            Qg = array(f['Q_g'])[0]
            Ie = array(f['I_e'])[0]
            Qe = array(f['Q_e'])[0]

    if SS:
        if plot_cal:
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix="Spin Echo ")
        ydata = _normalize_population(I, Q, Ig, Qg, Ie, Qe)
    else:
        ydata = amps

    delay_times = delay_times1 + delay_times2

    fig = plt.figure(figsize=(12, 6))
    plt.plot(delay_times, ydata, marker='o', label='data')
    if SS:
        plt.ylabel("Qubit Population")
    else:
        plt.ylabel("a.u.")
    plt.xlabel(r"Delay Time ($\mu$s)")
    plt.title("Spin Echo Experiment")
    if show_plot:
        plt.legend()

    fit = Fit()
    fit_results = fit.ramsey(delay_times, ydata, plot=show_plot)
    print("  Spin echo fit results:")
    pp.pprint(fit_results)
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_t1(global_data_path, filename,
               SS=False, plot_cal=False, show_plot=True):
    """
    T1 relaxation experiment.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00001_T1_ge_Q1_2026-06-06.h5".
    SS : bool
        True if single-shot normalisation data was collected.
    plot_cal : bool
        If True (and SS=True), plot the g/e calibration IQ blobs and fidelity.
    show_plot : bool
        If True, add a legend and display the figure.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        delay_times = array(f['delay_times'])[0]
        I = array(f['avgi'])[0]
        Q = array(f['avgq'])[0]
        amps = array(f['amps'])[0]
        config = _load_config(f)

        if SS:
            Ig = array(f['I_g'])[0]
            Qg = array(f['Q_g'])[0]
            Ie = array(f['I_e'])[0]
            Qe = array(f['Q_e'])[0]

    if SS:
        if plot_cal:
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix="T1 ")
        ydata = _normalize_population(I, Q, Ig, Qg, Ie, Qe)
    else:
        ydata = amps

    x_ns = delay_times * 1000  # us -> ns

    fig = plt.figure(figsize=(12, 6))
    plt.plot(x_ns, ydata, marker='o', label='data')
    if SS:
        plt.ylabel("Qubit Population")
        plt.axhline(1, color='black', linestyle='--')
        plt.axhline(0, color='black', linestyle='--')
    else:
        plt.ylabel("a.u.")
    plt.xlabel("Delay Time (ns)")
    plt.title(f"T1 Experiment: Gain = {config['qubit_gain_ge']}, "
              f"Sigma (us) = {config['sigma_ge']}")
    if show_plot:
        plt.legend()

    fit = Fit()
    fit_results = fit.T1(x_ns, ydata, plot=show_plot)
    print("  T1 fit results:")
    pp.pprint(fit_results)
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_T1_temporal_scan(global_data_path, folder_name,
                              time_spacing_min=5.0, show_plot=True):
    """
    T1 temporal scan (e.g. a folder of overnight T1 runs).

    Loads every T1 data file in a folder, fits T1 for each one, and plots
    the fitted T1 values against elapsed time. Measurements are assumed to
    be evenly spaced ``time_spacing_min`` minutes apart (matching e.g. the
    night_wait interval used by 008a_overnight_T1_ge.py), in the order the
    files sort alphabetically (i.e. by their sequential file number).

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    folder_name : str
        Name of the subfolder (under global_data_path) containing the T1
        data files, e.g. "2026-08-17overnight _T1_ge".
    time_spacing_min : float
        Assumed time in minutes between consecutive measurements.
    show_plot : bool
        If True, add a legend and display the figure.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    T1_values : 1D np.array
        Fitted T1 (us) for each file, in the order plotted.
    """
    folder_path = os.path.join(global_data_path, folder_name)
    print(f"Data folder: {folder_path}")

    files = sorted(f for f in os.listdir(folder_path) if f.endswith('.h5'))
    if not files:
        raise ValueError(f"No .h5 files found in {folder_path}")

    T1_values = []
    for fname in files:
        with SlabFile(os.path.join(folder_path, fname), 'r') as f:
            delay_times = array(f['delay_times'])[0]
            amps = array(f['amps'])[0]

        x_ns = delay_times * 1000  # us -> ns

        fit = Fit()
        fit_results = fit.T1(x_ns, amps, plot=False)
        T1_values.append(fit_results['T1'][0])

    T1_values = np.array(T1_values)
    time_axis = np.arange(len(files)) * time_spacing_min

    # earliest file's timestamp, for the start-time stamp
    mtimes = [os.path.getmtime(os.path.join(folder_path, fname)) for fname in files]
    start_time = datetime.datetime.fromtimestamp(min(mtimes))

    fig = plt.figure(figsize=(12, 6))
    plt.plot(time_axis, T1_values, marker='o', linestyle='-', label='T1')
    plt.xlabel("Elapsed Time (min)")
    plt.ylabel(r"T1 ($\mu s$)")
    plt.title(f"T1 Temporal Scan ({len(files)} measurements, "
              f"{time_spacing_min:.1f} min spacing)")
    if show_plot:
        plt.legend()
    _stamp_data_path(
        f"{folder_path}\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        fig)
    if show_plot:
        plt.show()

    return fig, T1_values


def analyze_single_shot(global_data_path, filename,
                         ef=False,
                         fidelity_method="method1",
                         hist_range=50,
                         bin_num=100, show_plot=True):
    """
    Single-shot IQ readout experiment.

    Plots the IQ blobs and computes readout fidelity using the chosen method.
    If ef=True, also computes g-f and e-f fidelity.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00007_IQ_plot_Q1_2026-06-06.h5".
    ef : bool
        True if |f>-state data was also collected.
    fidelity_method : str
        "method1" — manual IQ projection + threshold sweep.
        "method2" — rotated histogram via plotting_helpers.hist().
        "both"    — run both and print both results.
    hist_range : float
        Range parameter for method2 (passed to ``hist()``).
    bin_num : int
        Number of histogram bins for method1.
    show_plot : bool
        If True, add a legend and display the IQ scatter figure.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the IQ scatter figure, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        I_g = array(f['I_g'])[0]
        Q_g = array(f['Q_g'])[0]
        I_e = array(f['I_e'])[0]
        Q_e = array(f['Q_e'])[0]
        if ef:
            I_f = array(f['I_f'])[0]
            Q_f = array(f['Q_f'])[0]
        config = _load_config(f)

    # IQ scatter plot
    fig = plt.figure()
    plt.title("IQ Plot")
    plt.plot(I_g, Q_g, '.', label='g')
    plt.plot(I_e, Q_e, '.', label='e')
    if ef:
        plt.plot(I_f, Q_f, '.', label='f')
    plt.xlabel("I")
    plt.ylabel("Q")
    if show_plot:
        plt.legend()
    plt.axis('square')
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    # g-e fidelity
    print("--- g-e fidelity ---")
    if fidelity_method in ("method1", "both"):
        _compute_fidelity_by_projection(I_g, Q_g, I_e, Q_e,
                                         bin_num=bin_num, plot=True)
    if fidelity_method in ("method2", "both"):
        _compute_fidelity_by_hist(I_g, Q_g, I_e, Q_e,
                                   plot=True, ran=hist_range)

    # g-f and e-f fidelity if ef data present
    if ef:
        print("--- g-f fidelity ---")
        _compute_fidelity_gf_projection(I_g, Q_g, I_f, Q_f,
                                         bin_num=bin_num, plot=True)
        print("--- e-f fidelity ---")
        _compute_fidelity_ef_projection(I_e, Q_e, I_f, Q_f,
                                         bin_num=bin_num, plot=True)

    return fig


def analyze_readout_optimization(global_data_path, filename,
                                  ge=True, gf=False, ef=False,
                                  fidelity_method="method1",
                                  hist_range=50, show_plot=True):
    """
    Readout optimisation sweep (frequency x gain).

    Computes a 2D fidelity map, plots it, prints the optimal indices,
    and returns the fidelity array so it can be inspected further with
    ``inspect_readout_optimization_point()``.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00001_Readout_Optimization_ge_Q1_2026-05-31.h5".
    ge : bool
        Discriminate g-e.
    gf : bool
        Discriminate g-f (requires f-state data in file).
    ef : bool
        Discriminate e-f (requires f-state data in file).
    fidelity_method : str
        "method1" — manual IQ projection (used to build the 2D map).
        "method2" — plotting_helpers.hist() (used to build the 2D map).
        Note: method1 is significantly faster for the full sweep.
    hist_range : float
        Range parameter for method2.
    show_plot : bool
        If True, display the fidelity colormap figure.

    Returns
    -------
    fid_Array : 2D np.array, shape (gain_steps, freq_steps)
        Fidelity at each (gain, freq) point.
    gains_list : 1D np.array
        Gain values corresponding to rows.
    freqs_list : 1D np.array
        Frequency values corresponding to columns.
    fig : matplotlib Figure
        Handle to the fidelity colormap figure, for further modification.
    """
    data = _build_path(global_data_path, filename)
    print(f"Data path: {data}")

    with SlabFile(data, 'r') as f:
        I_g_array = array(f['I_g_array'])[0]
        Q_g_array = array(f['Q_g_array'])[0]
        I_e_array = array(f['I_e_array'])[0]
        Q_e_array = array(f['Q_e_array'])[0]
        if ef or gf:
            I_f_array = array(f['I_f_array'])[0]
            Q_f_array = array(f['Q_f_array'])[0]
        config = _load_config(f)

    # Build frequency and gain axes from config
    freq_start = config['freq_start']
    freq_steps = config['freq_steps']
    freq_step_size = config['freq_step_size']
    gain_start = config['gain_start']
    gain_steps = config['gain_steps']
    gain_step_size = config['gain_step_size']

    freqs_list = np.array([freq_start + i * freq_step_size for i in range(freq_steps)])
    gains_list = np.array([gain_start + i * gain_step_size for i in range(gain_steps)])

    fid_Array = np.zeros((gain_steps, freq_steps))

    for idx_f in range(freq_steps):
        for idx_g in range(gain_steps):
            if ge:
                I_g = I_g_array[idx_f][0][idx_g][0]
                Q_g = Q_g_array[idx_f][0][idx_g][0]
                I_e = I_e_array[idx_f][0][idx_g][0]
                Q_e = Q_e_array[idx_f][0][idx_g][0]

                if fidelity_method == "method2":
                    fid, _, _ = hist(data=[I_g, Q_g, I_e, Q_e],
                                     plot=False, ran=hist_range)
                    fid_Array[idx_g, idx_f] = fid
                else:
                    # method1: projection
                    vec_I = np.mean(I_e) - np.mean(I_g)
                    vec_Q = np.mean(Q_e) - np.mean(Q_g)
                    norm = abs(vec_I + 1j * vec_Q) ** 2
                    gstate = ((I_g - np.mean(I_g)) * vec_I
                              + (Q_g - np.mean(Q_g)) * vec_Q) / norm
                    estate = ((I_e - np.mean(I_g)) * vec_I
                              + (Q_e - np.mean(Q_g)) * vec_Q) / norm
                    th_list = np.linspace(min(gstate), max(estate), 1000)
                    fid_curve = [
                        (np.sum(gstate < th) + np.sum(estate > th))
                        / len(gstate) / 2
                        for th in th_list
                    ]
                    fid_Array[idx_g, idx_f] = max(fid_curve) * 2 - 1

            elif gf:
                I_g = I_g_array[idx_f][0][idx_g][0]
                Q_g = Q_g_array[idx_f][0][idx_g][0]
                I_f = I_f_array[idx_f][0][idx_g][0]
                Q_f = Q_f_array[idx_f][0][idx_g][0]
                vec_I = np.mean(I_f) - np.mean(I_g)
                vec_Q = np.mean(Q_f) - np.mean(Q_g)
                norm = abs(vec_I + 1j * vec_Q) ** 2
                gstate = ((I_g - np.mean(I_g)) * vec_I
                          + (Q_g - np.mean(Q_g)) * vec_Q) / norm
                fstate = ((I_f - np.mean(I_g)) * vec_I
                          + (Q_f - np.mean(Q_g)) * vec_Q) / norm
                th_list = np.linspace(min(gstate), max(fstate), 1000)
                fid_curve = [
                    (np.sum(gstate < th) + np.sum(fstate > th))
                    / len(gstate) / 2
                    for th in th_list
                ]
                fid_Array[idx_g, idx_f] = max(fid_curve) * 2 - 1

            elif ef:
                I_e = I_e_array[idx_f][0][idx_g][0]
                Q_e = Q_e_array[idx_f][0][idx_g][0]
                I_f = I_f_array[idx_f][0][idx_g][0]
                Q_f = Q_f_array[idx_f][0][idx_g][0]
                vec_I = np.mean(I_f) - np.mean(I_e)
                vec_Q = np.mean(Q_f) - np.mean(Q_e)
                norm = abs(vec_I + 1j * vec_Q) ** 2
                estate = ((I_e - np.mean(I_e)) * vec_I
                          + (Q_e - np.mean(Q_e)) * vec_Q) / norm
                fstate = ((I_f - np.mean(I_e)) * vec_I
                          + (Q_f - np.mean(Q_e)) * vec_Q) / norm
                th_list = np.linspace(min(estate), max(fstate), 1000)
                fid_curve = [
                    (np.sum(estate < th) + np.sum(fstate > th))
                    / len(estate) / 2
                    for th in th_list
                ]
                fid_Array[idx_g, idx_f] = max(fid_curve) * 2 - 1

    # 2D fidelity colormap
    fig = plt.figure(figsize=(24, 8), dpi=80)
    norm_color = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    figA = plt.pcolormesh(freqs_list, gains_list, fid_Array, norm=norm_color)
    figB = plt.contour(freqs_list, gains_list, fid_Array, 10, colors='w')
    plt.clabel(figB, figB.levels, inline=True, fontsize=20)
    plt.ylabel("Gain, DAC Unit")
    plt.xlabel("Frequency, (MHz)")
    if ge:
        plt.title("Fidelity, Discriminate g-e")
    elif gf:
        plt.title("Fidelity, Discriminate g-f")
    elif ef:
        plt.title("Fidelity, Discriminate e-f")
    plt.colorbar(figA)
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    # Print optimal point
    idx_amp_max, idx_freq_max = np.where(fid_Array == np.max(fid_Array))
    print(f"  Max fidelity:  {np.max(fid_Array):.4f}")
    print(f"  Optimal gain:  {gains_list[idx_amp_max[0]]:.4f}  (index {idx_amp_max[0]})")
    print(f"  Optimal freq:  {freqs_list[idx_freq_max[0]]:.4f} MHz  (index {idx_freq_max[0]})")

    return fid_Array, gains_list, freqs_list, fig


def inspect_readout_optimization_point(global_data_path, filename,
                                        gain_index, freq_index,
                                        ge=True, gf=False, ef=False,
                                        fidelity_method="method1",
                                        hist_range=50, bin_num=100,
                                        show_plot=True):
    """
    Inspect a specific (gain, freq) point from the readout optimisation sweep.

    Plots the IQ scatter and projected histogram for the chosen indices.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        Same file used in ``analyze_readout_optimization()``.
    gain_index : int
        Index into the gain axis.
    freq_index : int
        Index into the frequency axis.
    ge, gf, ef : bool
        Which pair of states to discriminate.
    fidelity_method : str
        "method1", "method2", or "both".
    hist_range : float
        Range parameter for method2.
    bin_num : int
        Number of histogram bins for method1.
    show_plot : bool
        If True, add a legend and display the IQ scatter figure.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the IQ scatter figure, for further modification.
    """
    data = _build_path(global_data_path, filename)

    with SlabFile(data, 'r') as f:
        I_g_array = array(f['I_g_array'])[0]
        Q_g_array = array(f['Q_g_array'])[0]
        I_e_array = array(f['I_e_array'])[0]
        Q_e_array = array(f['Q_e_array'])[0]
        if ef or gf:
            I_f_array = array(f['I_f_array'])[0]
            Q_f_array = array(f['Q_f_array'])[0]
        config = _load_config(f)

    # Rebuild axes for labelling
    freq_start = config['freq_start']
    freq_step_size = config['freq_step_size']
    gain_start = config['gain_start']
    gain_step_size = config['gain_step_size']
    freq_val = freq_start + freq_index * freq_step_size
    gain_val = gain_start + gain_index * gain_step_size

    I_g = I_g_array[freq_index][0][gain_index][0]
    Q_g = Q_g_array[freq_index][0][gain_index][0]
    I_e = I_e_array[freq_index][0][gain_index][0]
    Q_e = Q_e_array[freq_index][0][gain_index][0]

    # IQ scatter
    fig = plt.figure()
    plt.title(f"IQ Plot\nGain = {gain_val}, Frequency = {freq_val} MHz")
    plt.plot(I_g, Q_g, '.', label='g', zorder=1)
    plt.plot(I_e, Q_e, '.', label='e', zorder=2)
    plt.xlabel("I")
    plt.ylabel("Q")
    if show_plot:
        plt.legend()
    plt.axis('square')
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    print(f"  Gain:      {gain_val} (index {gain_index})")
    print(f"  Frequency: {freq_val} MHz (index {freq_index})")

    if ge:
        if fidelity_method in ("method1", "both"):
            _compute_fidelity_by_projection(I_g, Q_g, I_e, Q_e,
                                             bin_num=bin_num, plot=True)
        if fidelity_method in ("method2", "both"):
            _compute_fidelity_by_hist(I_g, Q_g, I_e, Q_e,
                                       plot=True, ran=hist_range)
    elif gf:
        I_f = I_f_array[freq_index][0][gain_index][0]
        Q_f = Q_f_array[freq_index][0][gain_index][0]
        _compute_fidelity_gf_projection(I_g, Q_g, I_f, Q_f,
                                         bin_num=bin_num, plot=True)
    elif ef:
        I_f = I_f_array[freq_index][0][gain_index][0]
        Q_f = Q_f_array[freq_index][0][gain_index][0]
        _compute_fidelity_ef_projection(I_e, Q_e, I_f, Q_f,
                                         bin_num=bin_num, plot=True)

    return fig
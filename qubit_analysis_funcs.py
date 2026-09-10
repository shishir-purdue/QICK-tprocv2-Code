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

    # Example: analyze an overnight folder of qubit spectroscopy runs vs. time
    analyze_qubit_spec_temporal_scan(
        global_data_path=r"C:/Users/G41Lab/Dropbox/_Data/...",
        folder_name="2026-08-17overnight _qubit_spec_ge",
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


def _qubit_prefix(config):
    """
    "Q1 - " for a config that recorded its qubit index, "" otherwise.

    initialize_configs only stores 'qubit_index' for non-MUX experiments, and
    files taken before it did so have no such key, so this quietly returns an
    empty string rather than guessing.
    """
    qubit_index = config.get('qubit_index', None)
    return f"Q{qubit_index} - " if qubit_index is not None else ""


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
                                     title_prefix="",
                                     data_path=None):
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
    data_path : str or None  If given (and plot), stamp it at the bottom
        of the figure.

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
        if data_path is not None:
            _stamp_data_path(data_path)
        plt.show()

    print(f"  Method 1 fidelity = {best_fid:.4f}")
    print(f"  sigma_g = {sigma_g:.4f}")
    print(f"  sigma_e = {sigma_e:.4f}")

    return best_fid, sigma_g, sigma_e


def _compute_fidelity_by_hist(I_g, Q_g, I_e, Q_e,
                               plot=True, ran=50, data_path=None):
    """
    Fidelity estimate — Method 2 (using the ``hist`` helper from
    plotting_helpers, which performs an optimal-angle rotation).

    Parameters
    ----------
    I_g, Q_g : array   Single-shot I/Q for |g>.
    I_e, Q_e : array   Single-shot I/Q for |e>.
    plot : bool         Show the rotated histogram.
    ran : float         Histogram range parameter passed to ``hist()``.
    data_path : str or None  If given (and plot), stamp it at the bottom
        of the figure.

    Returns
    -------
    fid : float         Fidelity after rotation.
    threshold : float   Optimal discrimination threshold.
    angle : float       Rotation angle used.
    """
    fid, threshold, angle = hist(
        data=[I_g, Q_g, I_e, Q_e], plot=plot, ran=ran
    )
    if plot and data_path is not None:
        _stamp_data_path(data_path)
    print(f"  Method 2 fidelity = {fid:.3f}")
    print(f"  threshold = {threshold:.4f},  angle = {angle:.4f}")
    return fid, threshold, angle


def _compute_fidelity_gf_projection(I_g, Q_g, I_f, Q_f,
                                     bin_num=100,
                                     n_threshold_pts=1000,
                                     plot=True,
                                     data_path=None):
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
        if data_path is not None:
            _stamp_data_path(data_path)
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
                                     plot=True,
                                     data_path=None):
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
        if data_path is not None:
            _stamp_data_path(data_path)
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
    plt.title(_qubit_prefix(config) + "Time of Flight Experiment")
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_resonator_spectroscopy(global_data_path, filename,
                                    fit_resonator=True, fit_method='circle',
                                    logscale=False, show_plot=True,
                                    perform_fit=True):
    """
    Single-resonator spectroscopy (amplitude + phase).

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filename : str
        HDF5 filename, e.g. "00000_res_spec_ge_Q1_2026-06-06.h5".
    perform_fit : bool
        If True (default), fit the resonator and draw the fit. If False, only
        the data is plotted - same scaling, labels and stamp, no fit.
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
    plt.title(_qubit_prefix(config) + f"Resonator Spectroscopy, Gain = {config['res_gain_ge']}")

    if perform_fit and fit_resonator:
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
    plt.title(_qubit_prefix(config) + "Resonator Spectroscopy Phase")
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig_amp, fig_phase


def analyze_qubit_spectroscopy(global_data_path, filename,
                                SS=False, ef=False, dac=0.0,
                                plot_cal=False, show_plot=True,
                                perform_fit=True):
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
    perform_fit : bool
        If True (default), fit the data and draw the fit. If False, only the
        data is plotted - same scaling, labels and stamp, no fit.

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
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix=_qubit_prefix(config) + "Qubit Spec ")
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
    plt.title(_qubit_prefix(config) + f"Qubit Spectroscopy:  Gain = {config['qubit_gain_ge']}, "
              f"Sigma (us) = {config['sigma_ge']}, DAC (mA) = {dac}")

    if perform_fit:
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


def analyze_qubit_spec_temporal_scan(global_data_path, folder_name,
                                      time_spacing_min=5.0, show_plot=True,
                                      perform_fit=True):
    """
    Qubit spectroscopy temporal scan (e.g. a folder of overnight qubit
    spectroscopy runs), plotted as a frequency-vs-time colormap.

    Loads every qubit spectroscopy data file in a folder and plots a 2D
    colormap: x = elapsed time, y = frequency, z = spectroscopy
    amplitude (magnitude). The peak frequency of each individual spectrum
    (the frequency of its maximum amplitude) is overlaid as a line, and
    the qubit frequency is also fit for each file. Measurements are
    assumed to be evenly spaced ``time_spacing_min`` minutes apart
    (matching e.g. the night_wait interval used by
    003a_overnight_qubit_spec_ge.py), in the order the files sort
    alphabetically (i.e. by their sequential file number).

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    folder_name : str
        Name of the subfolder (under global_data_path) containing the
        qubit spectroscopy data files, e.g.
        "2026-08-17overnight _qubit_spec_ge".
    time_spacing_min : float
        Assumed time in minutes between consecutive measurements.
    show_plot : bool
        If True, add a legend and display the figure.

    perform_fit : bool
        If True (default), fit the data and draw the fit. If False, only the
        data is plotted - same scaling, labels and stamp, no fit.
    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    freq_values : 1D np.array
        Fitted qubit frequency (MHz) for each file, in the order plotted.
    peaks : 1D np.array
        Frequency (MHz) of the maximum amplitude point of each individual
        spectrum -- this is the line overlaid on the colormap.
    """
    folder_path = os.path.join(global_data_path, folder_name)
    print(f"Data folder: {folder_path}")

    files = sorted(f for f in os.listdir(folder_path) if f.endswith('.h5'))
    if not files:
        raise ValueError(f"No .h5 files found in {folder_path}")

    freq_values = []
    peaks = []
    amps_list = []
    freqs_ref = None
    config = {}
    for fname in files:
        with SlabFile(os.path.join(folder_path, fname), 'r') as f:
            freqs = array(f['fpts'])[0]
            amps = array(f['amps'])[0]
            if not config:
                config = _load_config(f)

        if freqs_ref is None:
            freqs_ref = freqs
        elif len(freqs) != len(freqs_ref):
            raise ValueError(
                f"{fname} has a different number of frequency points "
                f"({len(freqs)}) than earlier files ({len(freqs_ref)}); "
                "can't build a shared colormap frequency axis.")
        amps_list.append(amps)

        if perform_fit:
            fit = Fit()
            fit_results = fit.transmission_resonator_spectroscopy(freqs, amps,
                                                                  plot=False)
            freq_values.append(fit_results['f'][0])
        else:
            freq_values.append(np.nan)
        peaks.append(freqs[np.argmax(amps)])

    freq_values = np.array(freq_values)
    peaks = np.array(peaks)
    time_axis = np.arange(len(files)) * time_spacing_min
    z_data = np.array(amps_list).T  # shape (n_freqs, n_files)

    # earliest file's timestamp, for the start-time stamp
    mtimes = [os.path.getmtime(os.path.join(folder_path, fname)) for fname in files]
    start_time = datetime.datetime.fromtimestamp(min(mtimes))

    fig = plt.figure(figsize=(12, 6))
    mesh = plt.pcolormesh(time_axis, freqs_ref, z_data, shading='auto')
    plt.colorbar(mesh, label='Amplitude (a.u.)')
    plt.plot(time_axis, peaks, marker='o', linestyle='-', c='red', label='peak')
    plt.xlabel("Elapsed Time (min)")
    plt.ylabel("Qubit Frequency (MHz)")
    plt.title(_qubit_prefix(config) + f"Qubit Spectroscopy Temporal Scan ({len(files)} measurements, "
              f"{time_spacing_min:.1f} min spacing)")
    if show_plot:
        plt.legend()
    _stamp_data_path(
        f"{folder_path}\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        fig)
    if show_plot:
        plt.show()

    return fig, freq_values, peaks


def analyze_length_rabi(global_data_path, filename,
                         SS=False, ef=False,
                         plot_cal=False, show_plot=True,
                         perform_fit=True):
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

    perform_fit : bool
        If True (default), fit the data and draw the fit. If False, only the
        data is plotted - same scaling, labels and stamp, no fit.
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
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix=_qubit_prefix(config) + "Length Rabi ")
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
    plt.title(_qubit_prefix(config) + f"Length Rabi Oscillations (Qubit gain = {config['qubit_gain_ge']:.2f})")

    if perform_fit:
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
                            full_pi_offset=0.0, half_pi_offset=0.0, title_append="",
                            perform_fit=True):
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

    perform_fit : bool
        If True (default), fit the data and draw the fit. If False, only the
        data is plotted - same scaling, labels and stamp, no fit.
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
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix=_qubit_prefix(config) + "Amp Rabi ")
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

    if perform_fit:
        # Fit within the specified gain window
        cutoff = (gains >= gain_cutoff_low) & (gains <= gain_cutoff_high)
        fit_results = fitdecaysin(gains[cutoff], ydata[cutoff], showfit=True)
        print("  fitdecaysin results:")
        pp.pprint(fit_results)

        peak = 0.5 / fit_results[1]
        half_pi = peak / 2.0
        peak += full_pi_offset
        half_pi += half_pi_offset
        print(f"  Pi-pulse gain:    {peak:.5f} DAC units")
        print(f"  Pi/2-pulse gain:  {half_pi:.5f} DAC units")
        print(f"  Max population:   {ydata[np.argmax(ydata)]:.5f}")
        print(f"  Phase offset (deg from fit): "
              f"{(fit_results[2] * 180 / np.pi) / (2 * np.pi * fit_results[1]):.5f}")

        plt.axvline(peak, c='red', label=f'pi pulse: {peak:.3f} DAC')
        plt.axvline(half_pi, c='red', linestyle='--',
                    label=f'pi/2 pulse: {half_pi:.3f} DAC')
    else:
        # fitdecaysin draws the trace as well as the fit, so with the fit off
        # the data has to be plotted here or the axes come out empty.
        cutoff = (gains >= gain_cutoff_low) & (gains <= gain_cutoff_high)
        plt.plot(gains[cutoff], ydata[cutoff], marker='o', label='data')
    plt.title(_qubit_prefix(config) + f"Amplitude Rabi Oscillations (Sigma = {config['sigma_ge']:.5f})\n" + title_append)
    if show_plot:
        plt.legend()
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_repeat_X90_pulse(global_data_path, filenames,
                              SS=False, ef=False,
                              gain_cutoff_low=0.0, gain_cutoff_high=1.0,
                              extremum='auto', fit_window=3,
                              plot_cal=False, show_plot=True,
                              gain_offset=0.0, title_append="", vline = None,
                              perform_fit=True):
    """
    Repeated-X90 (error amplification) pulse calibration.

    The experiment repeats the X90 pulse before each readout while sweeping
    its gain, so a fractional amplitude error shows up multiplied by the
    number of pulses instead of on its own.

    Pass a list of filenames to overlay several N on one axes - each trace
    gets its own colour and is labelled by its 2N+1 pulse count, so the
    sharpening of the feature with N is visible directly.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filenames : list of str
        HDF5 filenames to overlay on one plot, e.g.
        ["00000_repeat_X90_pulse_ge_Q1_2026-08-30.h5", ...]. Always a list -
        wrap a single file in one.
    SS : bool
        True if single-shot normalisation data was collected. Scales the y
        axis to qubit population instead of raw amplitude.
    ef : bool
        True if this is an e-f experiment.
    gain_cutoff_low : float
        Lower bound of the gain range to include.
    gain_cutoff_high : float
        Upper bound of the gain range to include.
    extremum : {'auto', 'max', 'min'}
        Unused while the fit is commented out.
    fit_window : int
        Unused while the fit is commented out.
    plot_cal : bool
        If True (and SS=True), plot the g/e calibration IQ blobs and fidelity
        for each file.
    show_plot : bool
        If True, add a legend and display the figure.
    gain_offset : float
        Unused while the fit is commented out.
    title_append : str
        Extra text appended to the plot title.
    vline : float or None
        If given, draw a red dashed vertical line at this gain. Tune it by eye
        to read off where the traces cross.

    perform_fit : bool
        No effect while the fit is commented out; kept so every analysis
        function takes the same switch.
    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    """
    if isinstance(filenames, str):
        raise TypeError(
            "filenames must be a list, not a single string - wrap it in one: "
            f'["{filenames}"]')

    # Explicit axes, so the figures _plot_ge_calibration opens do not steal
    # the traces from underneath us.
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.arange(len(filenames)) % 10)
    stamp_lines = []

    for idx, fname in enumerate(filenames):
        data = _build_path(global_data_path, fname)
        print(f"Data path: {data}")

        with SlabFile(data, 'r') as f:
            gains = np.atleast_1d(array(f['gains'])[0])
            I = np.atleast_1d(array(f['avgi'])[0])
            Q = np.atleast_1d(array(f['avgq'])[0])
            amps = np.atleast_1d(array(f['amps'])[0])
            config = _load_config(f)

            if SS:
                Ig = array(f['I_g'])[0]
                Qg = array(f['Q_g'])[0]
                Ie = array(f['I_e'])[0]
                Qe = array(f['Q_e'])[0]
                If_data = array(f['I_f'])[0] if ef else None
                Qf_data = array(f['Q_f'])[0] if ef else None

        # A gain axis shorter than the measured sweep means the gain never
        # became a QickSweep1D, so every point was taken at the same gain.
        if gains.size != amps.size:
            raise ValueError(
                f"{fname}: 'gains' has {gains.size} point(s) but 'amps' has "
                f"{amps.size}. The gain did not sweep - check that "
                "build_task.py turns qubit_gain_ge into a QickSweep1D for "
                "this experiment.")

        N = config.get('N', None)
        n_pulses = 2 * N + 1 if N is not None else None

        if SS:
            if plot_cal:
                _plot_ge_calibration(Ig, Qg, Ie, Qe,
                                     title_prefix=_qubit_prefix(config) + f"Repeat X90 (N = {N}) ")
            ydata = _normalize_population(I, Q, Ig, Qg, Ie, Qe,
                                           ef=ef, If=If_data, Qf=Qf_data)
        else:
            ydata = amps

        # Restrict to the requested gain window
        cutoff = (gains >= gain_cutoff_low) & (gains <= gain_cutoff_high)
        x = gains[cutoff]
        y = ydata[cutoff]

        label = f"N = {n_pulses}" if n_pulses is not None else fname
        ax.plot(x, y, 'o-', color=colors[idx], markersize=5, label=label)

        stamp_lines.append(f"{data}  (N = {n_pulses})")

        step = np.mean(np.diff(x)) if len(x) > 1 else float('nan')
        print(f"  Pulses per point: {n_pulses}  (N = {N})")
        print(f"  Sweep window:     {x[0]:.6f} to {x[-1]:.6f} DAC units "
              f"({len(x)} pts, step {step:.6f})")

        # ----- fit commented out for now -----
        #
        # # Local quadratic fit around the turning point
        # if extremum == 'auto':
        #     if SS:
        #         use_max = True
        #     else:
        #         i_hi, i_lo = int(np.argmax(y)), int(np.argmin(y))
        #         dist_from_edge = lambda i: min(i, len(y) - 1 - i)
        #         use_max = dist_from_edge(i_hi) >= dist_from_edge(i_lo)
        # else:
        #     use_max = (extremum == 'max')
        # i_ext = int(np.argmax(y)) if use_max else int(np.argmin(y))
        #
        # lo = max(i_ext - fit_window, 0)
        # hi = min(i_ext + fit_window + 1, len(x))
        # coeffs = None
        # if hi - lo >= 3:
        #     coeffs = np.polyfit(x[lo:hi], y[lo:hi], 2)
        #     best_gain = (-coeffs[1] / (2 * coeffs[0])
        #                  if coeffs[0] != 0 else x[i_ext])
        #     # A vertex outside the fitted points means the parabola is
        #     # extrapolating, so fall back to the measured extremum.
        #     if not (x[lo] <= best_gain <= x[hi - 1]):
        #         print("  Warning: quadratic vertex fell outside the fit "
        #               "window; using the raw extremum instead.")
        #         best_gain, coeffs = x[i_ext], None
        # else:
        #     print("  Warning: fewer than 3 points available; "
        #           "using the raw extremum instead of a fit.")
        #     best_gain = x[i_ext]
        #
        # best_gain += gain_offset
        #
        # if i_ext in (0, len(x) - 1):
        #     print("  Warning: the turning point is at the edge of the "
        #           "sweep - re-centre start/stop and re-run.")
        #
        # if coeffs is not None:
        #     x_fine = np.linspace(x[lo], x[hi - 1], 200)
        #     ax.plot(x_fine, np.polyval(coeffs, x_fine), '-', linewidth=1.5,
        #             color=colors[idx])
        # ax.axvline(best_gain, color=colors[idx], linestyle='--')
        # print(f"  Calibrated gain:  {best_gain:.6f} DAC units")

    if SS:
        ax.set_ylabel("Qubit Population")
        ax.axhline(1, color='black', linestyle='--')
        ax.axhline(0, color='black', linestyle='--')
        ax.axhline(0.5, color='black', linestyle='--')
    else:
        ax.set_ylabel("a.u.")
    ax.set_xlabel("Pulse Gain (DAC units)")
    ax.set_title(_qubit_prefix(config) + "Repeated X90 Pulse Calibration\n" + title_append)

    if vline is not None:
        ax.axvline(vline, color='red', linestyle='--',
                   label=f'gain = {vline:.5f}')

    if show_plot:
        ax.legend()

    # One data path per line, each tagged with its pulse count
    _stamp_data_path(chr(10).join(stamp_lines), fig)

    if show_plot:
        plt.show()

    return fig


# =============================================================================
#  Parked: X90 calibration-point selection (target / 'half' crossing)
#
#  Not used by analyze_repeat_X90_pulse above, which just finds the turning
#  point of the signal. Kept here because it may matter later: with n_pulses
#  X90 pulses the sequence has rotated n_pulses * pi/2 at the correct gain, so
#  where it lands depends on the pulse count - a multiple of 4 returns to |g>
#  (a minimum), 2 mod 4 lands on |e> (a maximum), and an odd count stops on the
#  equator, where the population sits at 0.5 and there is no extremum to find.
#
#  To bring it back: add target='auto' and n_pulses=None to the signature and
#  drop these two blocks in where the comments say.
# =============================================================================
#
# --- goes where the extremum direction is chosen ---
#
# if target == 'auto' and n_pulses is not None:
#     target = {0: 'min', 2: 'max'}.get(n_pulses % 4, 'half')
#     print(f"  Target from n_pulses = {n_pulses}: {target}")
#     if target == 'half' and not SS:
#         print("  Note: an odd pulse count lands on the equator, which "
#               "cannot be located without a normalised population axis. "
#               "Re-run the analysis with SS=True; falling back to the "
#               "nearest extremum for now.")
#         target = 'auto'
#
# if target == 'auto':
#     # No usable pulse count - read the target off the data instead. Keep
#     # whichever extremum sits further from the sweep edges, since one on an
#     # edge is usually the sweep running out of range rather than a real
#     # turning point. 'half' is not identifiable this way.
#     target = ('max' if _dist_from_edge(int(np.argmax(y)))
#                        >= _dist_from_edge(int(np.argmin(y))) else 'min')
#     print(f"  Target (auto-detected from data): {target}")
#
# if target == 'half' and not SS:
#     raise ValueError(
#         "target='half' needs a normalised population axis to locate the "
#         "0.5 crossing - re-analyse with SS=True.")
#
# --- replaces the quadratic fit when target == 'half' ---
#
# # Calibration point is where the population crosses 0.5.
# sign_change = np.where(np.diff(np.sign(y - 0.5)) != 0)[0]
# if sign_change.size == 0:
#     raise ValueError(
#         "the population never crosses 0.5 inside the gain window - "
#         "re-centre start/stop on the X90 gain and re-run.")
# # Prefer the crossing nearest the middle of the sweep.
# i_c = int(sign_change[np.argmin(np.abs(sign_change - (len(y) - 1) / 2.0))])
# lo = max(i_c - fit_window + 1, 0)
# hi = min(i_c + fit_window + 1, len(x))
# if hi - lo >= 2:
#     coeffs = np.polyfit(x[lo:hi], y[lo:hi], 1)
# if coeffs is not None and coeffs[0] != 0:
#     best_gain = (0.5 - coeffs[1]) / coeffs[0]
#     if not (x[lo] <= best_gain <= x[hi - 1]):
#         print("  Warning: the linear fit put the crossing outside the "
#               "fit window; interpolating between neighbours instead.")
#         coeffs = None
# if coeffs is None:
#     # Straight interpolation between the two bracketing points.
#     x0, x1, y0, y1 = x[i_c], x[i_c + 1], y[i_c], y[i_c + 1]
#     best_gain = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
# i_mark = i_c
#
# --- and the readable guard for SS=True on a file with no single-shot data ---
#
# if SS:
#     needed = ['I_g', 'Q_g', 'I_e', 'Q_e'] + (['I_f', 'Q_f'] if ef else [])
#     missing = [k for k in needed if k not in list(f.keys())]
#     if missing:
#         raise KeyError(
#             f"{filename} has no {', '.join(missing)} - it was taken "
#             "with SS = 'False' in system_config.json. Re-run the "
#             "experiment with SS = 'True', or analyse with SS=False.")
# =============================================================================


def analyze_repeat_X180_pulse(global_data_path, filenames,
                              SS=False, ef=False,
                              gain_cutoff_low=0.0, gain_cutoff_high=1.0,
                              extremum='auto', fit_window=3,
                              plot_cal=False, show_plot=True,
                              gain_offset=0.0, title_append="", vline = None,
                              perform_fit=True):
    """
    Repeated-X180 (error amplification) pulse calibration.

    The experiment repeats the X180 pulse before each readout while sweeping
    its gain, so a fractional amplitude error shows up multiplied by the
    number of pulses instead of on its own.

    Pass a list of filenames to overlay several N on one axes - each trace
    gets its own colour and is labelled by its 2N+1 pulse count, so the
    sharpening of the feature with N is visible directly.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filenames : list of str
        HDF5 filenames to overlay on one plot, e.g.
        ["00000_repeat_X180_pulse_ge_Q1_2026-09-02.h5", ...]. Always a list -
        wrap a single file in one.
    SS : bool
        True if single-shot normalisation data was collected. Scales the y
        axis to qubit population instead of raw amplitude.
    ef : bool
        True if this is an e-f experiment.
    gain_cutoff_low : float
        Lower bound of the gain range to include.
    gain_cutoff_high : float
        Upper bound of the gain range to include.
    extremum : {'auto', 'max', 'min'}
        Unused while the fit is commented out.
    fit_window : int
        Unused while the fit is commented out.
    plot_cal : bool
        If True (and SS=True), plot the g/e calibration IQ blobs and fidelity
        for each file.
    show_plot : bool
        If True, add a legend and display the figure.
    gain_offset : float
        Unused while the fit is commented out.
    title_append : str
        Extra text appended to the plot title.
    vline : float or None
        If given, draw a red dashed vertical line at this gain. Tune it by eye
        to read off where the traces cross.

    perform_fit : bool
        No effect while the fit is commented out; kept so every analysis
        function takes the same switch.
    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    """
    if isinstance(filenames, str):
        raise TypeError(
            "filenames must be a list, not a single string - wrap it in one: "
            f'["{filenames}"]')

    # Explicit axes, so the figures _plot_ge_calibration opens do not steal
    # the traces from underneath us.
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.arange(len(filenames)) % 10)
    stamp_lines = []

    for idx, fname in enumerate(filenames):
        data = _build_path(global_data_path, fname)
        print(f"Data path: {data}")

        with SlabFile(data, 'r') as f:
            gains = np.atleast_1d(array(f['gains'])[0])
            I = np.atleast_1d(array(f['avgi'])[0])
            Q = np.atleast_1d(array(f['avgq'])[0])
            amps = np.atleast_1d(array(f['amps'])[0])
            config = _load_config(f)

            if SS:
                Ig = array(f['I_g'])[0]
                Qg = array(f['Q_g'])[0]
                Ie = array(f['I_e'])[0]
                Qe = array(f['Q_e'])[0]
                If_data = array(f['I_f'])[0] if ef else None
                Qf_data = array(f['Q_f'])[0] if ef else None

        # A gain axis shorter than the measured sweep means the gain never
        # became a QickSweep1D, so every point was taken at the same gain.
        if gains.size != amps.size:
            raise ValueError(
                f"{fname}: 'gains' has {gains.size} point(s) but 'amps' has "
                f"{amps.size}. The gain did not sweep - check that "
                "build_task.py turns qubit_gain_ge into a QickSweep1D for "
                "this experiment.")

        N = config.get('N', None)
        n_pulses = N  if N is not None else None

        if SS:
            if plot_cal:
                _plot_ge_calibration(Ig, Qg, Ie, Qe,
                                     title_prefix=_qubit_prefix(config) + f"Repeat X180 (N = {N}) ")
            ydata = _normalize_population(I, Q, Ig, Qg, Ie, Qe,
                                           ef=ef, If=If_data, Qf=Qf_data)
        else:
            ydata = amps

        # Restrict to the requested gain window
        cutoff = (gains >= gain_cutoff_low) & (gains <= gain_cutoff_high)
        x = gains[cutoff]
        y = ydata[cutoff]

        label = f"N = {n_pulses}" if n_pulses is not None else fname
        ax.plot(x, y, 'o-', color=colors[idx], markersize=5, label=label)

        stamp_lines.append(f"{data}  (N = {n_pulses})")

        step = np.mean(np.diff(x)) if len(x) > 1 else float('nan')
        print(f"  Pulses per point: {n_pulses}  (N = {N})")
        print(f"  Sweep window:     {x[0]:.6f} to {x[-1]:.6f} DAC units "
              f"({len(x)} pts, step {step:.6f})")

        # ----- fit commented out for now -----
        #
        # # Local quadratic fit around the turning point
        # if extremum == 'auto':
        #     if SS:
        #         use_max = True
        #     else:
        #         i_hi, i_lo = int(np.argmax(y)), int(np.argmin(y))
        #         dist_from_edge = lambda i: min(i, len(y) - 1 - i)
        #         use_max = dist_from_edge(i_hi) >= dist_from_edge(i_lo)
        # else:
        #     use_max = (extremum == 'max')
        # i_ext = int(np.argmax(y)) if use_max else int(np.argmin(y))
        #
        # lo = max(i_ext - fit_window, 0)
        # hi = min(i_ext + fit_window + 1, len(x))
        # coeffs = None
        # if hi - lo >= 3:
        #     coeffs = np.polyfit(x[lo:hi], y[lo:hi], 2)
        #     best_gain = (-coeffs[1] / (2 * coeffs[0])
        #                  if coeffs[0] != 0 else x[i_ext])
        #     # A vertex outside the fitted points means the parabola is
        #     # extrapolating, so fall back to the measured extremum.
        #     if not (x[lo] <= best_gain <= x[hi - 1]):
        #         print("  Warning: quadratic vertex fell outside the fit "
        #               "window; using the raw extremum instead.")
        #         best_gain, coeffs = x[i_ext], None
        # else:
        #     print("  Warning: fewer than 3 points available; "
        #           "using the raw extremum instead of a fit.")
        #     best_gain = x[i_ext]
        #
        # best_gain += gain_offset
        #
        # if i_ext in (0, len(x) - 1):
        #     print("  Warning: the turning point is at the edge of the "
        #           "sweep - re-centre start/stop and re-run.")
        #
        # if coeffs is not None:
        #     x_fine = np.linspace(x[lo], x[hi - 1], 200)
        #     ax.plot(x_fine, np.polyval(coeffs, x_fine), '-', linewidth=1.5,
        #             color=colors[idx])
        # ax.axvline(best_gain, color=colors[idx], linestyle='--')
        # print(f"  Calibrated gain:  {best_gain:.6f} DAC units")

    if SS:
        ax.set_ylabel("Qubit Population")
        ax.axhline(1, color='black', linestyle='--')
        ax.axhline(0, color='black', linestyle='--')
        ax.axhline(0.5, color='black', linestyle='--')
    else:
        ax.set_ylabel("a.u.")
    ax.set_xlabel("Pulse Gain (DAC units)")
    ax.set_title(_qubit_prefix(config) + "Repeated X180 Pulse Calibration\n" + title_append)

    if vline is not None:
        ax.axvline(vline, color='red', linestyle='--',
                   label=f'gain = {vline:.5f}')

    if show_plot:
        ax.legend()

    # One data path per line, each tagged with its pulse count
    _stamp_data_path(chr(10).join(stamp_lines), fig)

    if show_plot:
        plt.show()

    return fig


def analyze_t2_ramsey(global_data_path, filename, my_offset=0.0,
                       SS=False, ef=False,
                       plot_cal=False, show_plot=True,
                       perform_fit=True):
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

    perform_fit : bool
        If True (default), fit the data and draw the fit. If False, only the
        data is plotted - same scaling, labels and stamp, no fit.
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
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix=_qubit_prefix(config) + "Ramsey ")
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
    plt.title(_qubit_prefix(config) + f"Ramsey Fringe Experiment, (Ramsey frequency = {config['ramsey_freq']:.2f} MHz)"
              f"\n$\\sigma$ = {config['sigma_ge']:.2f} $\\mu$s"
              f", $\\pi / 2$ Gain = {config.get('half_pi_gain_ge', config.get('half_pi_gain', 'n/a'))}")

    if perform_fit:
        fit = Fit()
        fit_results = fit.ramsey(x_ns, ydata, plot=show_plot)
        print("  Ramsey fit results:")
        pp.pprint(fit_results)

        # Decay envelope. The fringe does not ride on a constant offset: in
        # the qualang_tools model the baseline relaxes towards final_offset as
        # exp(-t/T2), so an envelope drawn about a constant sits wrong exactly
        # where the fringe is biggest. Recover the baseline by taking the
        # oscillation back out of the fitted curve - that needs no assumption
        # about how the fit scales its initial_offset, which is not returned
        # in the same units as amp. fit_func is only valid on the points it
        # was fitted on, which is what x_ns is.
        T2 = fit_results['T2'][0]
        amp = fit_results['amp'][0]
        final_offset = fit_results['final_offset'][0]
        half_amp = (amp / 2) * np.exp(-x_ns / T2)
        baseline = fit_results['fit_func'](x_ns) - half_amp * np.cos(
            2 * np.pi * fit_results['f'][0] * x_ns + fit_results['phase'][0])
        baseline = baseline + my_offset

        exp_lower = baseline - half_amp
        exp_upper = baseline + half_amp
        plt.plot(x_ns, exp_lower, label='fit lower bound')
        plt.plot(x_ns, exp_upper, label='fit upper bound')
        plt.plot(x_ns, baseline, color='black', linestyle='--',
                 label=f"offset (-> {final_offset + my_offset:.3f})")
    if show_plot:
        plt.legend()
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    if perform_fit:
        # Corrected qubit frequency
        new_freq_ge = (config['qubit_freq_ge']
                       - (fit_results['f'][0] * 1e3 - config['ramsey_freq']))
        print(f"  New Qubit frequency: {new_freq_ge:.6f} MHz")

    return fig


def analyze_ramsey_freq_calibration(global_data_path, filenames,
                                     SS=False, ef=False,
                                     plot_each=False, plot_cal=False,
                                     plot_phase=False,
                                     show_plot=True, title_append="",
                                     perform_fit=True):
    """
    Ramsey frequency calibration.

    Runs the Ramsey fit on each file of a ramsey_freq_calibration_ge sweep and
    plots the fitted fringe frequency against the Ramsey frequency that was
    applied. The second Ramsey pulse has its phase advanced by
    wait_time * 360 * ramsey_freq, so a perfectly tuned qubit gives back
    exactly the frequency that was asked for, and a detuned one gives that
    plus a constant:

        measured = applied + b

    The slope is fixed at 1 rather than fitted - it is set by the phase
    advance, not by the qubit - so b is just the mean of (measured - applied),
    and that is the detuning offset to correct out. A slope that visibly is
    not 1 means something other than a constant detuning is going on, so the
    fitted slope is reported alongside for a sanity check.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    filenames : list of str
        HDF5 filenames from the calibration sweep, one per applied Ramsey
        frequency, e.g. ["00000_ramsey_freq_calibration_ge_Q1_...h5", ...].
    SS : bool
        True if single-shot normalisation data was collected.
    ef : bool
        True if these are e-f Ramsey experiments.
    plot_each : bool
        If True, also run analyze_t2_ramsey on every file, so each individual
        fringe and its fit can be eyeballed.
    plot_cal : bool
        Passed through to analyze_t2_ramsey when plot_each is True.
    plot_phase : bool
        If True, also make a second figure of the fitted Ramsey phase against
        the applied frequency. The fit reports the phase in radians; it is
        converted to degrees and 360 subtracted, which puts a phase sitting
        just short of a full turn onto a small signed offset around zero.
    show_plot : bool
        If True, add a legend and display the figure.
    title_append : str
        Extra text appended to the plot title.

    perform_fit : bool
        Accepted for consistency with the other analysis functions, but
        must stay True here: the plotted quantity is itself a fit result.
    Returns
    -------
    fig : matplotlib Figure
        Handle to the summary figure.
    fig_phase : matplotlib Figure or None
        Handle to the phase figure, or None when plot_phase is False.
    applied : 1D np.array
        Applied Ramsey frequency (MHz) for each file, in the order plotted.
    measured : 1D np.array
        Fitted fringe frequency (MHz) for each file, in the order plotted.
    phases : 1D np.array
        Fitted phase (degrees, less 360) for each file, in the order plotted.
    """
    if isinstance(filenames, str):
        raise TypeError(
            "filenames must be a list, not a single string - wrap it in one: "
            f'["{filenames}"]')

    if not perform_fit:
        raise ValueError(
            "perform_fit=False has nothing to plot here - the y axis of this "
            "calibration is the fitted fringe frequency of each file, not raw "
            "data. Use analyze_t2_ramsey with perform_fit=False to look at an "
            "individual fringe.")

    applied, measured, phases, stamp_lines = [], [], [], []
    config = {}

    for fname in filenames:
        data = _build_path(global_data_path, fname)
        print(f"Data path: {data}")

        with SlabFile(data, 'r') as f:
            delay_times = array(f['delay_times'])[0]
            delay_times = delay_times - delay_times[0]
            I = array(f['avgi'])[0]
            Q = array(f['avgq'])[0]
            amps = array(f['amps'])[0]
            cfg = _load_config(f)

            if SS:
                Ig = array(f['I_g'])[0]
                Qg = array(f['Q_g'])[0]
                Ie = array(f['I_e'])[0]
                Qe = array(f['Q_e'])[0]
                If_data = array(f['I_f'])[0] if ef else None
                Qf_data = array(f['Q_f'])[0] if ef else None

        if not config:
            config = cfg

        ydata = (_normalize_population(I, Q, Ig, Qg, Ie, Qe,
                                        ef=ef, If=If_data, Qf=Qf_data)
                 if SS else amps)

        # Fit the fringe. x in ns puts fit_results['f'] in GHz, as
        # analyze_t2_ramsey assumes when it corrects the qubit frequency.
        x_ns = delay_times * 1000
        fit_results = Fit().ramsey(x_ns, ydata, plot=False)
        f_meas = fit_results['f'][0] * 1e3   # GHz -> MHz
        f_app = cfg['ramsey_freq']
        # The fit reports the phase in radians, near a full turn; degrees
        # less 360 puts it on a small signed offset about zero.
        phase_deg = np.degrees(fit_results['phase'][0]) - 360.0

        applied.append(f_app)
        measured.append(f_meas)
        phases.append(phase_deg)
        stamp_lines.append(f"{data}  (applied = {f_app} MHz)")
        print(f"  Applied {f_app} MHz -> measured {f_meas:.4f} MHz "
              f"(difference {f_meas - f_app:+.4f} MHz), "
              f"phase {phase_deg:+.2f} deg")

        if plot_each:
            analyze_t2_ramsey(global_data_path, fname, SS=SS, ef=ef,
                              plot_cal=plot_cal, show_plot=show_plot)

    applied = np.array(applied, dtype=float)
    measured = np.array(measured, dtype=float)
    phases = np.array(phases, dtype=float)

    order = np.argsort(applied)
    applied, measured = applied[order], measured[order]
    phases = phases[order]
    stamp_lines = [stamp_lines[i] for i in order]

    # Slope is fixed at 1, so the intercept is just the mean difference.
    offset = float(np.mean(measured - applied))
    spread = (float(np.std(measured - applied, ddof=1))
              if len(applied) > 1 else float('nan'))

    # Reported only as a sanity check on the fixed-slope assumption.
    if len(applied) > 1:
        free_slope, free_intercept = np.polyfit(applied, measured, 1)
    else:
        free_slope = free_intercept = float('nan')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(applied, measured, 'o', color='tab:blue', markersize=7,
            label='measured')

    x_line = np.linspace(applied.min(), applied.max(), 100)
    ax.plot(x_line, x_line + offset, 'r-', linewidth=1.5,
            label=f'y = x + b,  b = {offset:+.4f} MHz')
    ax.plot(x_line, x_line, color='gray', linestyle='--', linewidth=1,
            label='y = x (no detuning)')

    ax.set_xlabel("Applied Ramsey Frequency (MHz)")
    ax.set_ylabel("Measured Fringe Frequency (MHz)")
    ax.set_title(_qubit_prefix(config) + "Ramsey Frequency Calibration"
                 + chr(10) + title_append)

    print("")
    print(f"  Offset b:            {offset:+.6f} MHz  "
          f"(subtract from the qubit frequency)")
    print(f"  Scatter about y=x+b: {spread:.6f} MHz")
    print(f"  Free-fit slope:      {free_slope:.4f} "
          f"(sanity check, should be near 1)")
    if 'qubit_freq_ge' in config:
        print(f"  Corrected qubit frequency: "
              f"{config['qubit_freq_ge'] - offset:.6f} MHz")

    if show_plot:
        ax.legend()
    _stamp_data_path(chr(10).join(stamp_lines), fig)
    if show_plot:
        plt.show()

    # ----- Optional second figure: fitted phase ----- #
    fig_phase = None
    if plot_phase:
        fig_phase, ax_p = plt.subplots(figsize=(12, 6))
        ax_p.plot(applied, phases, 'o-', color='tab:green', markersize=7,
                  label='fitted phase')
        ax_p.axhline(0, color='black', linestyle='--', linewidth=1)
        ax_p.set_xlabel("Applied Ramsey Frequency (MHz)")
        ax_p.set_ylabel("Fitted Phase - 360 (degrees)")
        ax_p.set_title(_qubit_prefix(config) + "Ramsey Fitted Phase"
                       + chr(10) + title_append)

        print("")
        print(f"  Mean phase:    {np.mean(phases):+.3f} deg")
        if len(phases) > 1:
            print(f"  Phase spread:  {np.std(phases, ddof=1):.3f} deg")

        if show_plot:
            ax_p.legend()
        _stamp_data_path(chr(10).join(stamp_lines), fig_phase)
        if show_plot:
            plt.show()

    return fig, fig_phase, applied, measured, phases


def analyze_ramsey_ro_length(global_data_path, folder_name,
                              SS=False, ef=False,
                              plot_each=False, plot_cal=False,
                              show_plot=True, title_append="",
                              perform_fit=True):
    """
    Ramsey fitted phase against readout duration.

    Reads a folder written by 006b_ramsey_ro_length_ge.py - one file per
    readout length, all at the same Ramsey frequency - fits each fringe and
    plots the fitted phase against the ro_length that produced it.

    The fit reports the phase in radians sitting just short of a full turn, so
    it is converted to degrees and 360 subtracted, the same convention
    analyze_ramsey_freq_calibration uses, which puts it on a small signed
    offset about zero.

    Parameters
    ----------
    global_data_path : str
        Root directory for all data files.
    folder_name : str
        Subfolder holding the scan, e.g.
        "00000_ramsey_ro_length_ge_Q1_2026-09-10". Every .h5 in it is used.
    SS : bool
        True if single-shot normalisation data was collected.
    ef : bool
        True if these are e-f Ramsey experiments.
    plot_each : bool
        If True, also run analyze_t2_ramsey on every file, so each individual
        fringe and its fit can be eyeballed.
    plot_cal : bool
        Passed through to analyze_t2_ramsey when plot_each is True.
    show_plot : bool
        If True, add a legend and display the figure.
    title_append : str
        Extra text appended to the plot title.
    perform_fit : bool
        Accepted for consistency with the other analysis functions, but must
        stay True here: the plotted quantity is itself a fit result.

    Returns
    -------
    fig : matplotlib Figure
        Handle to the phase figure.
    ro_lengths : 1D np.array
        Readout duration (us) for each file, in the order plotted.
    phases : 1D np.array
        Fitted phase (degrees, less 360) for each file, in the order plotted.
    freqs : 1D np.array
        Fitted fringe frequency (MHz) for each file, in the order plotted.
    """
    if not perform_fit:
        raise ValueError(
            "perform_fit=False has nothing to plot here - the y axis of this "
            "scan is the fitted phase of each file, not raw data. Use "
            "analyze_t2_ramsey with perform_fit=False to look at an "
            "individual fringe.")

    folder_path = os.path.join(global_data_path, folder_name)
    print(f"Data folder: {folder_path}")

    files = sorted(f for f in os.listdir(folder_path) if f.endswith('.h5'))
    if not files:
        raise ValueError(f"No .h5 files found in {folder_path}")

    ro_lengths, phases, freqs = [], [], []
    config = {}

    for fname in files:
        data = os.path.join(folder_path, fname)

        with SlabFile(data, 'r') as f:
            delay_times = array(f['delay_times'])[0]
            delay_times = delay_times - delay_times[0]
            I = array(f['avgi'])[0]
            Q = array(f['avgq'])[0]
            amps = array(f['amps'])[0]
            cfg = _load_config(f)

            if SS:
                Ig = array(f['I_g'])[0]
                Qg = array(f['Q_g'])[0]
                Ie = array(f['I_e'])[0]
                Qe = array(f['Q_e'])[0]
                If_data = array(f['I_f'])[0] if ef else None
                Qf_data = array(f['Q_f'])[0] if ef else None

        if not config:
            config = cfg

        ydata = (_normalize_population(I, Q, Ig, Qg, Ie, Qe,
                                        ef=ef, If=If_data, Qf=Qf_data)
                 if SS else amps)

        # x in ns puts fit_results['f'] in GHz, as analyze_t2_ramsey assumes.
        x_ns = delay_times * 1000
        fit_results = Fit().ramsey(x_ns, ydata, plot=False)
        phase_deg = np.degrees(fit_results['phase'][0]) - 360.0
        f_meas = fit_results['f'][0] * 1e3   # GHz -> MHz
        ro = cfg['ro_length']

        ro_lengths.append(ro)
        phases.append(phase_deg)
        freqs.append(f_meas)
        print(f"  ro_length {ro} us -> phase {phase_deg:+.2f} deg, "
              f"fringe {f_meas:.4f} MHz")

        if plot_each:
            analyze_t2_ramsey(folder_path, fname, SS=SS, ef=ef,
                              plot_cal=plot_cal, show_plot=show_plot)

    ro_lengths = np.array(ro_lengths, dtype=float)
    phases = np.array(phases, dtype=float)
    freqs = np.array(freqs, dtype=float)

    order = np.argsort(ro_lengths)
    ro_lengths, phases, freqs = ro_lengths[order], phases[order], freqs[order]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ro_lengths, phases, 'o-', color='tab:green', markersize=7,
            label='fitted phase')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Readout Pulse Length (us)")
    ax.set_ylabel("Fitted Phase - 360 (degrees)")
    ramsey_freq = config.get('ramsey_freq', None)
    freq_note = (f" (Ramsey frequency = {ramsey_freq} MHz)"
                 if ramsey_freq is not None else "")
    ax.set_title(_qubit_prefix(config) + "Ramsey Phase vs Readout Duration"
                 + freq_note + chr(10) + title_append)

    print("")
    print(f"  Files:          {len(files)}")
    print(f"  ro_length:      {ro_lengths[0]:.3f} to {ro_lengths[-1]:.3f} us")
    print(f"  Phase span:     {np.max(phases) - np.min(phases):.3f} deg")
    print(f"  Mean phase:     {np.mean(phases):+.3f} deg")
    if len(phases) > 1:
        print(f"  Phase spread:   {np.std(phases, ddof=1):.3f} deg")
        print(f"  Fringe spread:  {np.std(freqs, ddof=1):.5f} MHz "
              f"(should be flat - the qubit does not know about ro_length)")

    if show_plot:
        ax.legend()
    _stamp_data_path(folder_path, fig)
    if show_plot:
        plt.show()

    return fig, ro_lengths, phases, freqs


def analyze_t2_spin_echo(global_data_path, filename,
                          SS=False, plot_cal=False, show_plot=True,
                          perform_fit=True):
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

    perform_fit : bool
        If True (default), fit the data and draw the fit. If False, only the
        data is plotted - same scaling, labels and stamp, no fit.
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
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix=_qubit_prefix(config) + "Spin Echo ")
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
    plt.title(_qubit_prefix(config) + "Spin Echo Experiment")
    if show_plot:
        plt.legend()

    if perform_fit:
        fit = Fit()
        fit_results = fit.ramsey(delay_times, ydata, plot=show_plot)
        print("  Spin echo fit results:")
        pp.pprint(fit_results)
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_t1(global_data_path, filename,
               SS=False, plot_cal=False, show_plot=True,
               perform_fit=True):
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

    perform_fit : bool
        If True (default), fit the data and draw the fit. If False, only the
        data is plotted - same scaling, labels and stamp, no fit.
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
            _plot_ge_calibration(Ig, Qg, Ie, Qe, title_prefix=_qubit_prefix(config) + "T1 ")
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
    plt.title(_qubit_prefix(config) + f"T1 Experiment: Gain = {config['qubit_gain_ge']}, "
              f"Sigma (us) = {config['sigma_ge']}")
    if show_plot:
        plt.legend()

    if perform_fit:
        fit = Fit()
        fit_results = fit.T1(x_ns, ydata, plot=show_plot)
        print("  T1 fit results:")
        pp.pprint(fit_results)
    _stamp_data_path(data)
    if show_plot:
        plt.show()

    return fig


def analyze_T1_temporal_scan(global_data_path, folder_name,
                              error_bars=True,
                              time_spacing_min=5.0, show_plot=True,
                              perform_fit=True):
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
    error_bars : bool
        If True (default), draw whisker-style error bars (in black) on
        each point using the standard error the T1 fit itself reports.
    time_spacing_min : float
        Assumed time in minutes between consecutive measurements.
    show_plot : bool
        If True, add a legend and display the figure.

    perform_fit : bool
        Accepted for consistency with the other analysis functions, but
        must stay True here: the plotted quantity is itself a fit result.
    Returns
    -------
    fig : matplotlib Figure
        Handle to the plotted figure, for further modification.
    T1_values : 1D np.array
        Fitted T1 (us) for each file, in the order plotted.
    T1_errors : 1D np.array
        Fit standard error on T1 (us) for each file, in the order plotted.
    """
    if not perform_fit:
        raise ValueError(
            "perform_fit=False has nothing to plot here - the y axis of this "
            "scan is the fitted T1 of each file, not raw data. Use analyze_t1 "
            "with perform_fit=False to look at an individual trace.")

    folder_path = os.path.join(global_data_path, folder_name)
    print(f"Data folder: {folder_path}")

    files = sorted(f for f in os.listdir(folder_path) if f.endswith('.h5'))
    if not files:
        raise ValueError(f"No .h5 files found in {folder_path}")

    T1_values = []
    T1_errors = []
    config = {}
    for fname in files:
        with SlabFile(os.path.join(folder_path, fname), 'r') as f:
            delay_times = array(f['delay_times'])[0]
            amps = array(f['amps'])[0]
            if not config:
                config = _load_config(f)

        x_ns = delay_times * 1000  # us -> ns

        fit = Fit()
        fit_results = fit.T1(x_ns, amps, plot=False)
        T1_values.append(fit_results['T1'][0])
        T1_errors.append(fit_results['T1'][1])

    T1_values = np.array(T1_values)
    T1_errors = np.array(T1_errors)
    time_axis = np.arange(len(files)) * time_spacing_min

    # earliest file's timestamp, for the start-time stamp
    mtimes = [os.path.getmtime(os.path.join(folder_path, fname)) for fname in files]
    start_time = datetime.datetime.fromtimestamp(min(mtimes))

    fig = plt.figure(figsize=(12, 6))
    if error_bars:
        plt.errorbar(time_axis, T1_values/1e3, yerr=T1_errors/1e3,
                     marker='o', linestyle='-', label='T1',
                     ecolor='black', elinewidth=1, capsize=4, capthick=1)
    else:
        plt.plot(time_axis, T1_values/1e3, marker='o', linestyle='-', label='T1')
    plt.xlabel("Elapsed Time (min)")
    plt.ylabel(r"T1 ($\mu s$)")
    plt.title(_qubit_prefix(config) + f"T1 Temporal Scan ({len(files)} measurements, "
              f"{time_spacing_min:.1f} min spacing)")
    if show_plot:
        plt.legend()
    _stamp_data_path(
        f"{folder_path}\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        fig)
    if show_plot:
        plt.show()

    return fig, T1_values, T1_errors


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
    plt.title(_qubit_prefix(config) + "IQ Plot")
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
                                         bin_num=bin_num, plot=True,
                                         data_path=data)
    if fidelity_method in ("method2", "both"):
        _compute_fidelity_by_hist(I_g, Q_g, I_e, Q_e,
                                   plot=True, ran=hist_range,
                                   data_path=data)

    # g-f and e-f fidelity if ef data present
    if ef:
        print("--- g-f fidelity ---")
        _compute_fidelity_gf_projection(I_g, Q_g, I_f, Q_f,
                                         bin_num=bin_num, plot=True,
                                         data_path=data)
        print("--- e-f fidelity ---")
        _compute_fidelity_ef_projection(I_e, Q_e, I_f, Q_f,
                                         bin_num=bin_num, plot=True,
                                         data_path=data)

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
        plt.title(_qubit_prefix(config) + "Fidelity, Discriminate g-e")
    elif gf:
        plt.title(_qubit_prefix(config) + "Fidelity, Discriminate g-f")
    elif ef:
        plt.title(_qubit_prefix(config) + "Fidelity, Discriminate e-f")
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
    plt.title(_qubit_prefix(config) + f"IQ Plot\nGain = {gain_val}, Frequency = {freq_val} MHz")
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
                                             bin_num=bin_num, plot=True,
                                             data_path=data)
        if fidelity_method in ("method2", "both"):
            _compute_fidelity_by_hist(I_g, Q_g, I_e, Q_e,
                                       plot=True, ran=hist_range,
                                       data_path=data)
    elif gf:
        I_f = I_f_array[freq_index][0][gain_index][0]
        Q_f = Q_f_array[freq_index][0][gain_index][0]
        _compute_fidelity_gf_projection(I_g, Q_g, I_f, Q_f,
                                         bin_num=bin_num, plot=True,
                                         data_path=data)
    elif ef:
        I_f = I_f_array[freq_index][0][gain_index][0]
        Q_f = Q_f_array[freq_index][0][gain_index][0]
        _compute_fidelity_ef_projection(I_e, Q_e, I_f, Q_f,
                                         bin_num=bin_num, plot=True,
                                         data_path=data)

    return fig
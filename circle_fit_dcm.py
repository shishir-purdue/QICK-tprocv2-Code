"""
circle_fit_dcm.py
==================
Notch-type resonator circle fit using the Diameter Correction Method (DCM),
the standard technique from:

    S. Probst, F. B. Song, P. A. Bushev, A. V. Ustinov, M. Weides,
    "Efficient and robust analysis of complex scattering data under noise
    in microwave resonators," Rev. Sci. Instrum. 86, 024706 (2015).

Given complex S21(f) data, this:
  1. corrects for cable/electronics delay by minimizing how far the delay-
     corrected data deviates from a perfect circle,
  2. algebraically fits a circle to the delay-corrected data,
  3. fits the phase around that circle to a standard arctan lineshape to
     get the loaded quality factor Ql and resonance frequency fr,
  4. normalizes by the off-resonant point on the circle to extract the
     diameter D and impedance-mismatch angle phi0, from which Qc and the
     diameter-corrected Qi follow.

Usage
-----
    from circle_fit_dcm import notch_circlefit, model_S21

    result = notch_circlefit(freqs, avgi + 1j * avgq)
    fit_curve = model_S21(freqs, result['fr'], result['Ql'], result['Qc'],
                           result['phi0'], result['P_off'], result['delay'])
"""

import numpy as np
from scipy import optimize


def _algebraic_circle_fit(z):
    """Least-squares algebraic circle fit to complex points z. Returns (xc, yc, r0)."""
    x = z.real
    y = z.imag
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc = sol[0] / 2.0
    yc = sol[1] / 2.0
    r0 = np.sqrt(sol[2] + xc ** 2 + yc ** 2)
    return xc, yc, r0


def _wrap_pi(theta):
    """Wrap an angle (or array of angles) into [-pi, pi]."""
    return np.mod(theta + np.pi, 2.0 * np.pi) - np.pi


def _phase_model(f, theta0, Ql, fr):
    return theta0 + 2.0 * np.arctan(2.0 * Ql * (1.0 - f / fr))


def _fit_phase(freqs, z_centered):
    """Fit the unwrapped phase of the centered circle data to the standard
    notch-resonator phase lineshape. Returns (theta0, Ql, fr, fit_result)."""
    theta = np.unwrap(np.angle(z_centered))

    fr0 = freqs[np.argmax(np.abs(np.diff(theta)))] if len(freqs) > 1 else freqs[len(freqs) // 2]
    theta0_0 = np.median(theta)
    slope = np.polyfit(freqs, theta, 1)[0]
    Ql0 = max(abs(slope) * fr0 / 4.0, 1.0)

    def resid(p):
        theta0, Ql, fr = p
        return _wrap_pi(theta - _phase_model(freqs, theta0, Ql, fr))

    p0 = [theta0_0, Ql0, fr0]
    result = optimize.least_squares(resid, p0, method='lm', max_nfev=20000)
    theta0, Ql, fr = result.x
    return theta0, abs(Ql), fr, result


def _delay_residual(delay, freqs, z_data):
    z_corr = z_data * np.exp(2j * np.pi * freqs * delay)
    xc, yc, r0 = _algebraic_circle_fit(z_corr)
    dist = np.abs(z_corr - (xc + 1j * yc))
    return np.sum((dist - r0) ** 2)


def fit_delay(freqs, z_data, search_range=1.0, n_coarse=801):
    """
    Estimate the cable/electronics delay by finding the value that makes
    z_data * exp(2j*pi*freqs*delay) trace out the most circular arc.

    ``search_range``/``delay`` are in units of 1/[freqs] (e.g. freqs in MHz
    means delay is in microseconds). A coarse grid search finds the basin,
    then a local minimization refines it.
    """
    delays = np.linspace(-search_range, search_range, n_coarse)
    residuals = [_delay_residual(d, freqs, z_data) for d in delays]
    i0 = int(np.argmin(residuals))
    d0 = delays[i0]
    try:
        lo = delays[max(i0 - 1, 0)]
        hi = delays[min(i0 + 1, n_coarse - 1)]
        res = optimize.minimize_scalar(_delay_residual, args=(freqs, z_data),
                                        bracket=(lo, d0, hi))
        return res.x
    except Exception:
        return d0


def notch_circlefit(freqs, z_data, correct_delay=True):
    """
    Fit notch-type resonator S21 data using the Diameter Correction Method.

    Parameters
    ----------
    freqs : array
        Frequency points (any consistent unit, e.g. MHz).
    z_data : array of complex
        Complex S21 data (e.g. avgi + 1j*avgq).
    correct_delay : bool
        If True (recommended), estimate and remove cable/electronics delay
        before circle fitting. Skipping this typically biases the circle
        center and can even produce unphysical (negative) Qi.

    Returns
    -------
    dict with keys: fr, Ql, Qc, Qi, phi0, theta0, xc, yc, r0, delay, P_off
        fr  : resonance frequency
        Ql  : loaded (total) quality factor
        Qc  : coupling quality factor (magnitude)
        Qi  : internal quality factor (diameter-corrected:
              1/Qi = 1/Ql - cos(phi0)/Qc)
        phi0: impedance-mismatch (Fano asymmetry) angle, radians
        P_off: complex off-resonant calibration point on the raw circle
    """
    freqs = np.asarray(freqs, dtype=float)
    z_data = np.asarray(z_data, dtype=complex)

    delay = 0.0
    if correct_delay:
        delay = fit_delay(freqs, z_data)
        z_data = z_data * np.exp(2j * np.pi * freqs * delay)

    xc, yc, r0 = _algebraic_circle_fit(z_data)
    z_centered = z_data - complex(xc, yc)

    theta0, Ql, fr, _fitres = _fit_phase(freqs, z_centered)

    beta = _wrap_pi(theta0 + np.pi)
    P_off = complex(xc, yc) + r0 * np.exp(1j * beta)

    z_res = complex(xc, yc) + r0 * np.exp(1j * theta0)
    z_res_norm = z_res / P_off

    D_vec = 1.0 - z_res_norm
    D = np.abs(D_vec)
    phi0 = np.angle(D_vec)

    Qc = Ql / D
    Qi = 1.0 / (1.0 / Ql - np.cos(phi0) / Qc)

    return {
        'fr': fr, 'Ql': Ql, 'Qc': Qc, 'Qi': Qi, 'phi0': phi0,
        'theta0': theta0, 'xc': xc, 'yc': yc, 'r0': r0,
        'delay': delay, 'P_off': P_off,
    }


def model_S21(freqs, fr, Ql, Qc, phi0, P_off, delay):
    """Evaluate the fitted complex S21(freqs) curve from a notch_circlefit result."""
    s21_norm = 1.0 - (Ql / Qc) * np.exp(1j * phi0) / (1.0 + 2j * Ql * (freqs / fr - 1.0))
    return P_off * s21_norm * np.exp(-2j * np.pi * freqs * delay)

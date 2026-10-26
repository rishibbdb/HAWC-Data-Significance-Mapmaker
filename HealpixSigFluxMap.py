#!/usr/bin/env python
"""
HealpixSigFluxMap.py - standalone (AERIE-free) HAWC likelihood significance and
flux maps. Port of aerie/src/aerie-apps/analysis/map-making/HealpixSigFluxMap.cc
for the threeML/HAL environment.

For each sky pixel, a point-like (or disk-like, with --extension) source with a
fixed spectral shape is placed at the pixel center. Its flux norm is the only
free parameter of a Poisson likelihood ratio against the background over all
analysis bins within a ROI of radius --padding (+ extension). The response
(expected counts and PSF) is interpolated linearly in declination between the
two nearest simulated declinations.

Run (same options as aerie-apps-HealpixSigFluxMap):

    conda activate astro_hal
    python HealpixSigFluxMap.py -i /path/to/nncoutmaps -b B4C0Ef B5C0Ef B6C0Ef \
        -d detRes-nn-pass5f_ML-refit.root --index 2.63 --pivot 7 \
        --extension 0.5 --window 83.63 22.01 3 3 --negFlux --negSignif \
        -o map.fits.gz

Or from Python / threeML, with this file in the same directory or on sys.path:

    from HealpixSigFluxMap import make_sig_flux_map
    res = make_sig_flux_map(inputs=..., bins=[...], response=..., index=2.5,
                            pivot=7.0, extension=0.5, window=(ra, dec, 6, 6),
                            allow_negative_ts=True, allow_negative_flux=True,
                            output="map.fits.gz")

-i takes per-bin FITS files (data in column 1, background in column 2), a
directory with them plus -b (files matched as bin<NAME>[-_.]...), or a HAL map
tree (.root/.hd5). The response can be the liff .root file (read with uproot)
or a HAL .hd5 file. The spectrum can be --index/--cutoff, an AERIE --spectrum
string, or (from Python) any astromodels function or callable f(E_TeV).

Output: same layout as AERIE (STARTMJD/STOPMJD/TOTDUR/EPOCH/COORDSYS/PIVOT in
the primary header; RING HEALPix table with significance, flux, flux_error, or
flux_lower_bound/flux_upper_bound with --fc). Uncomputed pixels are UNSEEN.

Differences from the C++ code:
  * only input pixels within the ROI of computed pixels are loaded, as float32
    by default (--float64 to keep double precision)
  * all input maps must share one nside (the output nside may differ)
  * the PSF is the double-Gaussian TF1 of the .root response with analytic
    integrals; for HAL .hd5 responses it is re-fitted from the tabulated PSF
  * --window accepts negative Dec directly (Dec+360 still accepted)
  * --nparts: the last part also takes the leftover pixels
  * --fullTransit uses the standard HAWC site coordinates (see below)
  * FC pseudo-experiments use a fixed seed (reproducible, numbers differ from AERIE)
  * no built-in EBL: pass a spectrum that includes it
"""

import argparse
import collections
import logging
import math
import os
import re
import sys
import time

import healpy as hp
import numpy as np
from astropy.io import fits
from numba import get_num_threads, njit, prange, set_num_threads

log = logging.getLogger("HealpixSigFluxMap")

# ============================================================================
# spectrum
# ============================================================================

class PowerLaw(object):
    """
    (E / pivot)^-index * exp(-E / cutoff), E in TeV.

    Same convention as liff MakeSpectrum: index is positive (e.g. 2.63) and
    cutoff=None means no cutoff.
    """

    def __init__(self, index, pivot=1.0, cutoff=None):
        self.index = float(index)
        self.pivot = float(pivot)
        self.cutoff = None if cutoff is None else float(cutoff)

    def __call__(self, energy_tev):
        e = np.asarray(energy_tev, dtype=float)
        out = (e / self.pivot) ** (-self.index)
        if self.cutoff is not None:
            out = out * np.exp(-e / self.cutoff)
        return out

    def __repr__(self):
        return "PowerLaw(index=%g, pivot=%g TeV, cutoff=%s TeV)" % (
            self.index, self.pivot, self.cutoff)


def parse_spectrum_string(spectrum, pivot=1.0):
    """
    Parse the AERIE --spectrum format, e.g. 'SimplePowerLaw,3.5e-11,2.63' or
    'CutOffPowerLaw,3.76e-11,2.39,14.3'. The amplitude is ignored (the norm is
    the free parameter), as in HealpixSigFluxMap.
    """
    parts = [p.strip() for p in spectrum.split(",")]
    function = parts[0]
    if function == "SimplePowerLaw":
        if len(parts) < 3:
            raise ValueError("Could not parse spectrum string '%s'" % spectrum)
        # liff uses a 1e6 TeV cutoff for a simple power law
        return PowerLaw(float(parts[2]), pivot=pivot, cutoff=1e6)
    if function in ("CutOffPowerLaw", "CutoffPowerLaw"):
        if len(parts) < 4:
            raise ValueError("Could not parse spectrum string '%s'" % spectrum)
        return PowerLaw(float(parts[2]), pivot=pivot, cutoff=float(parts[3]))
    raise ValueError("Unknown function '%s'; only SimplePowerLaw or CutOffPowerLaw "
                     "are understood. Use an astromodels function for other shapes."
                     % function)


class AstromodelsSpectrum(object):
    """
    Wrap an astromodels Function1D (e.g. Powerlaw, Cutoff_powerlaw,
    Log_parabola, or a composite such as Powerlaw() * EBLattenuation()).

    The function is evaluated with the normalization parameter as currently set
    and divided by it, so the fitted map value is that parameter expressed in
    TeV^-1 cm^-2 s^-1. Remember astromodels' convention: Powerlaw.index is
    negative (e.g. -2.5) and piv must be set to your pivot energy.

    :param function: astromodels function
    :param norm_parameter: name of the normalization parameter ('K' for most
        functions, 'K_1' for a composite whose first component is a power law)
    """

    def __init__(self, function, norm_parameter="K"):
        self.function = function
        self.norm_parameter = norm_parameter
        self._norm = function.parameters[norm_parameter]

    def __call__(self, energy_tev):
        e_kev = np.asarray(energy_tev, dtype=float) * 1e9
        # astromodels internal units: keV in, 1 / (keV cm2 s) out. The ratio to
        # the norm value (same internal units) is unit-free.
        return np.asarray(self.function(e_kev), dtype=float) / self._norm.value

    def __repr__(self):
        return "AstromodelsSpectrum(%s, norm=%s)" % (self.function.name, self.norm_parameter)


def as_spectrum(spectrum, pivot=1.0, index=None, cutoff=None):
    """Build a unit-normalized spectral shape from the supported inputs."""
    if spectrum is None:
        if index is None:
            raise ValueError("Provide either a spectrum or a spectral index")
        return PowerLaw(index, pivot=pivot, cutoff=cutoff)
    if isinstance(spectrum, str):
        return parse_spectrum_string(spectrum, pivot=pivot)
    if hasattr(spectrum, "parameters") and hasattr(spectrum, "evaluate_at"):
        return AstromodelsSpectrum(spectrum)
    if callable(spectrum):
        return spectrum
    raise TypeError("Unsupported spectrum %r" % (spectrum,))


# ============================================================================
# likelihood
# ============================================================================

# Defaults from NormLikelihood's constructor
NORM_FRAC_ERR = 0.005
ZERO_TS_TOL = 1e-5
MAX_BIG_STEPS = 10


@njit(cache=True)
def llr(data, bkg, unit_excess, norm):
    """Log-likelihood ratio TS(norm) (NormLikelihood::LLR)."""
    out = 0.0
    for i in range(bkg.shape[0]):
        excess = unit_excess[i] * norm
        if data[i] == 0:
            out += -excess
        elif bkg[i] > 0:
            out += data[i] * math.log((bkg[i] + excess) / bkg[i]) - excess
    return 2.0 * out


@njit(cache=True)
def dllr(data, bkg, unit_excess, norm, n):
    """n-th analytic derivative of the LLR with respect to norm (NormLikelihood::DLLR)."""
    out = 0.0
    for i in range(bkg.shape[0]):
        if data[i] > 0 and bkg[i] > 0:
            out += data[i] * (unit_excess[i] / (bkg[i] + unit_excess[i] * norm)) ** n

    if n == 1:
        for i in range(bkg.shape[0]):
            out -= unit_excess[i]

    nm1fact = 1
    for i in range(1, n):
        nm1fact *= i

    sign = 1.0 if (n - 1) % 2 == 0 else -1.0
    return out * sign * 2.0 * nm1fact


@njit(cache=True)
def solve(data, bkg, unit_excess, allow_neg_ts, allow_neg_flux, max_iter):
    """
    Maximize the likelihood in the norm.

    Returns (failed, ts, norm, norm_err) where norm_err is the Delta TS = 1
    parabolic error (NormLikelihood::Solve(ts, norm, normErr)).
    """
    dllr0 = dllr(data, bkg, unit_excess, 0.0, 1)

    if dllr0 <= 0:
        # Underfluctuation. A negative flux is not well defined with a Poisson
        # likelihood; when allowed we assume the Gaussian regime.
        ddllr0 = dllr(data, bkg, unit_excess, 0.0, 2)

        if allow_neg_ts:
            ts = dllr0 * dllr0 / 2.0 / ddllr0  # includes the negative sign
        else:
            ts = 0.0

        if allow_neg_flux:
            norm = -dllr0 / ddllr0
            norm_err = math.sqrt(-2.0 / ddllr0)
        else:
            norm = 0.0
            # Delta TS = 1 with respect to TS(norm = 0)
            if ddllr0 == 0:
                norm_err = -1.0 / dllr0
            else:
                norm_err = -(math.sqrt(dllr0 * dllr0 - 2.0 * ddllr0) + dllr0) / ddllr0

        return False, ts, norm, norm_err

    # Newton's method
    conv = False
    norm = 0.0
    iteration = 0
    while not conv and iteration < max_iter:
        step = -dllr(data, bkg, unit_excess, norm, 1) / dllr(data, bkg, unit_excess, norm, 2)
        norm += step
        # Same (signed) stopping criterion as the C++ implementation
        if step / norm < NORM_FRAC_ERR:
            conv = True
        iteration += 1

    # One extra step using Halley's method
    f = dllr(data, bkg, unit_excess, norm, 1)
    fp = dllr(data, bkg, unit_excess, norm, 2)
    fpp = dllr(data, bkg, unit_excess, norm, 3)
    norm -= 2.0 * f * fp / (2.0 * fp * fp - f * fpp)

    ts = llr(data, bkg, unit_excess, norm)
    norm_err = math.sqrt(-2.0 / dllr(data, bkg, unit_excess, norm, 2))

    failed = False
    if norm < 0 or iteration == max_iter:
        failed = True

    if ts < -ZERO_TS_TOL:
        failed = True
    elif ts < 0:
        ts = 0.0

    return failed, ts, norm, norm_err


@njit(cache=True)
def _normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@njit(cache=True)
def _chi2_1dof_pdf(x):
    return math.exp(-x / 2.0) / math.sqrt(2.0 * math.pi * x)


@njit(cache=True)
def _ts_crit(bkg, unit_excess, norm, pquant, nsample, max_iter, seed):
    """
    Critical Delta TS from Poisson pseudo-experiments (NormLikelihood::GetTScrit).

    As in AERIE the generator is re-seeded with the same seed on every call, so
    results are deterministic.
    """
    np.random.seed(seed)

    n = bkg.shape[0]
    mean = bkg + norm * unit_excess
    dts = np.empty(nsample)
    pseudo = np.empty(n)

    j = 0
    attempts = 0
    while j < nsample:
        for d in range(n):
            pseudo[d] = np.random.poisson(mean[d])
        failed, pts, _, _ = solve(pseudo, bkg, unit_excess, False, False, max_iter)
        attempts += 1
        if failed and attempts < 100 * nsample:
            continue  # discard failed pseudo-experiment
        dts[j] = pts - llr(pseudo, bkg, unit_excess, norm)
        j += 1

    dts.sort()
    k = int(math.ceil(pquant * nsample))
    if k > nsample - 1:
        k = nsample - 1
    return dts[k]


@njit(cache=True)
def solve_fc(data, bkg, unit_excess, sigmas, bounds_frac_err, allow_neg_ts,
             max_iter, seed):
    """
    Feldman-Cousins (unified approach) interval on the norm.

    Returns (failed, ts, norm, norm_lower, norm_upper). Port of
    NormLikelihood::Solve(ts, norm, normLB, normUB, sigmas). Negative fluxes
    are not allowed here.
    """
    failed, ts, norm, norm_err = solve(data, bkg, unit_excess, allow_neg_ts, False, max_iter)

    non_neg_ts = max(ts, 0.0)

    norm_lb = max(0.0, norm - sigmas * norm_err)
    norm_ub = norm + sigmas * norm_err

    # Number of pseudo-experiments needed assuming flux ~ sqrt(ts)
    ts_step = sigmas * sigmas * bounds_frac_err * (2.0 + bounds_frac_err)
    pquant = 1.0 - 2.0 * (1.0 - _normal_cdf(sigmas))
    nsample = int(math.ceil(pquant * (1.0 - pquant) / ts_step / ts_step /
                            _chi2_1dof_pdf(sigmas * sigmas) ** 2))

    # === Lower bound ===
    if norm == 0:
        norm_lb = 0.0
    else:
        dts = non_neg_ts - llr(data, bkg, unit_excess, norm_lb)
        dts_crit = _ts_crit(bkg, unit_excess, norm_lb, pquant, nsample, max_iter, seed)
        in_ci = dts < dts_crit

        # Big steps; may overshoot the solution
        n_big = 0
        last_in_ci = in_ci
        last_dts_crit = dts_crit
        in_ci_change = False
        while (not in_ci_change
               and abs(dts_crit - dts) > 3 * ts_step
               and (norm_lb > 0 or dts_crit < dts)
               and (norm_lb < norm or dts_crit > dts)):

            in_ci_change = last_in_ci != in_ci
            if in_ci_change:
                dts_goal = (dts_crit + last_dts_crit) / 2.0
            else:
                dts_goal = dts_crit

            norm_lb -= (norm - norm_lb) * ((math.sqrt(dts_goal) - math.sqrt(dts)) / math.sqrt(dts))
            norm_lb = min(norm, max(0.0, norm_lb))

            last_in_ci = in_ci
            last_dts_crit = dts_crit

            dts = non_neg_ts - llr(data, bkg, unit_excess, norm_lb)
            dts_crit = _ts_crit(bkg, unit_excess, norm_lb, pquant, nsample, max_iter, seed)
            in_ci = dts < dts_crit

            n_big += 1
            if n_big > MAX_BIG_STEPS:
                break

        # Fine grid
        if in_ci:
            last_lb = 0.0
            while in_ci and norm_lb > 0:
                last_lb = norm_lb
                norm_lb = max(0.0, norm_lb - ((math.sqrt(dts + ts_step) - math.sqrt(dts)) *
                                              (norm - norm_lb) / math.sqrt(dts)))
                dts = non_neg_ts - llr(data, bkg, unit_excess, norm_lb)
                dts_crit = _ts_crit(bkg, unit_excess, norm_lb, pquant, nsample, max_iter, seed)
                in_ci = dts < dts_crit
            if norm_lb > 0:
                norm_lb = (norm_lb + last_lb) / 2.0
        else:
            last_lb = norm_lb
            while not in_ci:
                last_lb = norm_lb
                norm_lb = norm_lb + ((math.sqrt(dts + ts_step) - math.sqrt(dts)) *
                                     (norm - norm_lb) / math.sqrt(dts))
                dts = non_neg_ts - llr(data, bkg, unit_excess, norm_lb)
                dts_crit = _ts_crit(bkg, unit_excess, norm_lb, pquant, nsample, max_iter, seed)
                in_ci = dts < dts_crit
            norm_lb = (norm_lb + last_lb) / 2.0

    # === Upper bound ===
    dts = non_neg_ts - llr(data, bkg, unit_excess, norm_ub)
    dts_crit = _ts_crit(bkg, unit_excess, norm_ub, pquant, nsample, max_iter, seed)
    in_ci = dts < dts_crit

    n_big = 0
    while abs(dts_crit - dts) > 3 * ts_step and (norm_ub > norm or dts_crit > dts):
        norm_ub += (norm_ub - norm) * ((math.sqrt(dts_crit) - math.sqrt(dts)) / math.sqrt(dts))
        norm_ub = max(norm, norm_ub)

        dts = non_neg_ts - llr(data, bkg, unit_excess, norm_ub)
        dts_crit = _ts_crit(bkg, unit_excess, norm_ub, pquant, nsample, max_iter, seed)
        in_ci = dts < dts_crit

        n_big += 1
        if n_big > MAX_BIG_STEPS:
            break

    last_ub = norm_ub
    if in_ci:
        while in_ci:
            last_ub = norm_ub
            norm_ub = norm_ub + ((math.sqrt(dts + ts_step) - math.sqrt(dts)) *
                                 (norm_ub - norm) / math.sqrt(dts))
            dts = non_neg_ts - llr(data, bkg, unit_excess, norm_ub)
            dts_crit = _ts_crit(bkg, unit_excess, norm_ub, pquant, nsample, max_iter, seed)
            in_ci = dts < dts_crit
    else:
        while not in_ci:
            last_ub = norm_ub
            norm_ub = norm_ub - ((math.sqrt(dts + ts_step) - math.sqrt(dts)) *
                                 (norm_ub - norm) / math.sqrt(dts))
            dts = non_neg_ts - llr(data, bkg, unit_excess, norm_ub)
            dts_crit = _ts_crit(bkg, unit_excess, norm_ub, pquant, nsample, max_iter, seed)
            in_ci = dts < dts_crit
    norm_ub = (norm_ub + last_ub) / 2.0

    return failed, ts, norm, norm_lb, norm_ub


class NormLikelihood(object):
    """
    Thin Python wrapper with the same interface as ZEBRA::NormLikelihood,
    convenient for single fits (e.g. at a source position) from threeML scripts.
    """

    def __init__(self, data=None, bkg=None, unit_excess=None, max_iter=1000):
        self.allow_neg_ts = False
        self.allow_neg_flux = False
        self.max_iter = max_iter
        self.bounds_frac_err = 0.01
        if bkg is not None:
            self.set_model(bkg, unit_excess)
        if data is not None:
            self.set_data(data)

    def allow_negative_ts(self, allow=True):
        self.allow_neg_ts = allow

    def allow_negative_flux(self, allow=True):
        self.allow_neg_flux = allow
        self.allow_neg_ts = self.allow_neg_ts or allow

    def set_model(self, bkg, unit_excess):
        self.bkg = np.ascontiguousarray(np.ravel(bkg), dtype=np.float64)
        self.unit_excess = np.ascontiguousarray(np.ravel(unit_excess), dtype=np.float64)
        if self.bkg.shape != self.unit_excess.shape:
            raise ValueError("bkg must have the same size as unit_excess")

    def set_data(self, data):
        self.data = np.ascontiguousarray(np.ravel(data), dtype=np.float64)
        if self.data.shape != self.bkg.shape:
            raise ValueError("data is not the same size as bkg")

    def ts(self, norm):
        return llr(self.data, self.bkg, self.unit_excess, float(norm))

    def solve(self):
        """Returns (failed, ts, norm, norm_err)."""
        return solve(self.data, self.bkg, self.unit_excess,
                     self.allow_neg_ts, self.allow_neg_flux, self.max_iter)

    def solve_fc(self, sigmas=2.0, seed=4357):
        """Returns (failed, ts, norm, norm_lower, norm_upper)."""
        if self.allow_neg_flux:
            raise ValueError("FC bounds are incompatible with allowing negative fluxes.")
        if sigmas <= 0:
            raise ValueError("sigmas must be a positive number")
        return solve_fc(self.data, self.bkg, self.unit_excess, float(sigmas),
                        self.bounds_frac_err, self.allow_neg_ts, self.max_iter, seed)


# ============================================================================
# response
# ============================================================================

PSF_LIM = 10.0  # [deg] upper limit of the PSF/DSF ("infinity"), as in liff


class ResponseBin(object):
    """Response of one analysis bin in one declination band."""

    def __init__(self, name, log_energy_centers, signal_events, sim_fluxes, psf_params):
        self.name = name
        self.log_energy_centers = np.asarray(log_energy_centers, dtype=float)  # log10(E/TeV)
        self.signal_events = np.asarray(signal_events, dtype=float)
        self.sim_fluxes = np.asarray(sim_fluxes, dtype=float)  # TeV^-1 cm^-2 s^-1
        self.psf_params = np.asarray(psf_params, dtype=float)  # A, f, var1, var2 [deg^2]

    def expected_signal(self, spectrum):
        """Expected signal events per transit for a unit-norm spectrum shape."""
        energies = 10.0 ** self.log_energy_centers
        good = self.sim_fluxes > 0
        weights = np.zeros_like(energies)
        weights[good] = spectrum(energies[good]) / self.sim_fluxes[good]
        return float(np.sum(self.signal_events * weights))


class DetectorResponse(object):
    """
    :param dec_bins: list of (lower_edge, sim_dec, upper_edge) in degrees,
        ascending in declination
    :param response_bins: OrderedDict {dec_index: {bin_name: ResponseBin}}
    """

    def __init__(self, filename, dec_bins, response_bins):
        self.filename = filename
        self.dec_bins = dec_bins
        self.response_bins = response_bins

    @property
    def n_dec_bins(self):
        return len(self.dec_bins)

    @property
    def sim_decs(self):
        return np.array([d[1] for d in self.dec_bins])

    @classmethod
    def open(cls, filename, bins=None):
        """Read a .root (liff) or .hd5/.hdf5 (HAL) response, optionally only some bins."""
        ext = os.path.splitext(filename)[-1].lower()
        if ext == ".root":
            return cls.from_root(filename, bins)
        if ext in (".hd5", ".hdf5", ".hdf", ".h5"):
            return cls.from_hdf5(filename, bins)
        raise ValueError("Unknown response file extension '%s'" % ext)

    @classmethod
    def from_root(cls, filename, bins=None):
        import uproot

        with uproot.open(filename) as f:
            loglog = f["LogLogSpectrum"]
            loglog_shape = str(loglog.member("fTitle"))
            loglog_params = np.asarray(loglog.member("fParams"), dtype=float)

            dec_tree = f["DecBins"].arrays(library="np")
            dec_bins = list(zip(dec_tree["lowerEdge"].astype(float),
                                dec_tree["simdec"].astype(float),
                                dec_tree["upperEdge"].astype(float)))

            ab = f["AnalysisBins"]
            if "name" in ab.keys():
                file_bins = [str(n) for n in ab["name"].array(library="np")]
            else:
                file_bins = [str(n) for n in ab["id"].array(library="np")]

            if bins is None:
                bins = file_bins
            missing = [b for b in bins if b not in file_bins]
            if missing:
                raise KeyError("Bins %s not found in response %s" % (missing, filename))

            response_bins = collections.OrderedDict()
            for d in range(len(dec_bins)):
                these = collections.OrderedDict()
                for name in bins:
                    prefix = "dec_%02d/nh_%s" % (d, name)
                    if prefix not in f:
                        prefix = "dec_%02d/nh_0%s" % (d, name)
                    suffix = "_dec%d_nh%s" % (d, name)

                    counts, edges = f["%s/EnSig%s" % (prefix, suffix)].to_numpy()
                    log_centers = 0.5 * (edges[:-1] + edges[1:])
                    sim_fluxes = 10.0 ** _log_log_spectrum(log_centers, loglog_shape,
                                                           loglog_params)

                    psf_tf1 = f["%s/PSF%s_fit" % (prefix, suffix)]
                    psf_params = np.asarray(psf_tf1.member("fParams"), dtype=float)
                    if psf_params.size != 4:
                        raise ValueError("PSF TF1 %s is not a double Gaussian" % prefix)

                    these[name] = ResponseBin(name, log_centers, counts, sim_fluxes,
                                              psf_params)
                response_bins[d] = these

        return cls(filename, dec_bins, response_bins)

    @classmethod
    def from_hdf5(cls, filename, bins=None):
        """
        HAL HDF5 response. The PSF is tabulated there, so the double-Gaussian
        parameters are re-fitted from the table.
        """
        from hawc_hal.response.response import HAWCResponse

        hal = HAWCResponse.from_hdf5(filename)
        dec_bins = [tuple(map(float, d)) for d in hal.dec_bins]

        response_bins = collections.OrderedDict()
        for d, dec_center in enumerate(hal.response_bins):
            these = collections.OrderedDict()
            hal_bins = hal.response_bins[dec_center]
            names = list(hal_bins.keys()) if bins is None else bins
            for name in names:
                rb = hal_bins[name]
                psf_params = fit_double_gaussian(rb.psf.xs, rb.psf.ys)
                these[name] = ResponseBin(name,
                                          np.log10(rb.sim_energy_bin_centers),
                                          rb.sim_signal_events_per_bin,
                                          rb.sim_differential_photon_fluxes,
                                          psf_params)
            response_bins[d] = these

        return cls(filename, dec_bins, response_bins)

    def expected_signal(self, spectrum, dec_index, bin_name):
        return self.response_bins[dec_index][bin_name].expected_signal(spectrum)

    def pixelated_psf(self, dec_index, bin_name, pixel_area, extension=0.0):
        """
        Relative signal density per pixel vs. angular distance.

        :param pixel_area: pixel area [deg^2]
        :param extension: disk radius [deg]; 0 for a point source
        :return: (bin_low_edges [deg], values) on a uniform grid on [0, PSF_LIM]
        """
        params = self.response_bins[dec_index][bin_name].psf_params
        if extension > 0:
            return pixelated_dsf(params, extension, pixel_area)
        return pixelated_psf(params, pixel_area)


def _log_log_spectrum(log_energy, shape, params):
    """log10 of the simulated differential flux (TeV^-1 cm^-2 s^-1)."""
    if shape == "SimplePowerLaw":
        return np.log10(params[0]) - params[1] * log_energy
    if shape in ("CutOffPowerLaw", "CutoffPowerLaw"):
        return (np.log10(params[0]) - params[1] * log_energy
                - np.log10(np.e) * 10.0 ** (log_energy - np.log10(params[2])))
    raise ValueError("Unknown simulated spectrum shape '%s'" % shape)


def _psf_cumulative(r, params):
    """Integral of the double-Gaussian dN/dr from 0 to r."""
    a, f, v1, v2 = params
    return a * (f * v1 * (1.0 - np.exp(-r * r / 2.0 / v1)) +
                (1.0 - f) * v2 * (1.0 - np.exp(-r * r / 2.0 / v2)))


def pixelated_psf(params, pixel_area):
    """Port of PointSourceDetectorResponse::CalculatePixelatedPsf (analytic integrals)."""
    pixel_size = math.sqrt(pixel_area)

    # liff finds where the PSF TF1 reaches zero; for a double Gaussian that is
    # effectively PSF_LIM.
    radius_limit = PSF_LIM
    total = _psf_cumulative(radius_limit, params)

    n_bins = int(math.ceil(5.0 * PSF_LIM / pixel_size))
    width = PSF_LIM / n_bins
    low_edges = np.arange(n_bins) * width
    centers = low_edges + width / 2.0

    rmin = np.clip(centers - pixel_size / 2.0, 0.0, None)
    rmax = np.minimum(centers + pixel_size / 2.0, radius_limit)

    ring_counts = _psf_cumulative(rmax, params) - _psf_cumulative(rmin, params)
    ring_area = np.pi * (rmax * rmax - rmin * rmin)
    values = ring_counts * pixel_area / ring_area / total
    values = np.minimum(values, 1.0)
    values[rmin >= radius_limit] = 0.0

    return low_edges, values


# Gauss-Legendre nodes for the DSF integrals
_GL100_X, _GL100_W = np.polynomial.legendre.leggauss(100)
_GL20_X, _GL20_W = np.polynomial.legendre.leggauss(20)


@njit(cache=True)
def _i0e(x):
    """Exponentially scaled modified Bessel function I0(x) exp(-x), x >= 0 (A&S 9.8.1-2)."""
    if x <= 3.75:
        t = (x / 3.75) ** 2
        i0 = 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (
            0.2659732 + t * (0.0360768 + t * 0.0045813)))))
        return i0 * math.exp(-x)
    t = 3.75 / x
    p = 0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (
        0.00916281 + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))))
    return p / math.sqrt(x)


@njit(cache=True)
def _disk_gauss_integral(r, var, disk_r, gl_x, gl_w):
    """
    int_0^R exp(-(r^2 + r'^2) / 2 var) I0(r r' / var) r' dr'
    (the theta integral in PointSourceDetectorResponse::ComputeDSF).
    """
    if r * r / var > 100:  # asymptotic behaviour, as in liff
        return var * math.erfc((r - disk_r) / math.sqrt(2 * var)) / 2
    half = disk_r / 2.0
    out = 0.0
    for k in range(gl_x.shape[0]):
        rp = half * (gl_x[k] + 1.0)
        # exp(-(r^2+rp^2)/2v) I0(r rp/v) = exp(-(r-rp)^2/2v) * i0e(r rp/v)
        out += gl_w[k] * math.exp(-(r - rp) ** 2 / 2.0 / var) * _i0e(r * rp / var) * rp
    return out * half


@njit(cache=True)
def _dsf(r, frac1, var1, var2, disk_r, gl_x, gl_w):
    i1 = 0.0 if frac1 == 0 else _disk_gauss_integral(r, var1, disk_r, gl_x, gl_w)
    i2 = 0.0 if frac1 == 1 else _disk_gauss_integral(r, var2, disk_r, gl_x, gl_w)
    return (frac1 * i1 + (1.0 - frac1) * i2) * r


@njit(cache=True)
def _dsf_integral(a, b, frac1, var1, var2, disk_r, gl_x, gl_w, ring_x, ring_w):
    half = (b - a) / 2.0
    mid = (a + b) / 2.0
    out = 0.0
    for k in range(ring_x.shape[0]):
        out += ring_w[k] * _dsf(mid + half * ring_x[k], frac1, var1, var2, disk_r, gl_x, gl_w)
    return out * half


@njit(cache=True)
def _pixelated_dsf(frac1, var1, var2, disk_r, pixel_area, gl_x, gl_w, ring_x, ring_w):
    pixel_size = math.sqrt(pixel_area)

    # Total counts on [0, PSF_LIM], integrated piecewise
    n_seg = 2000
    seg = PSF_LIM / n_seg
    total = 0.0
    for s in range(n_seg):
        total += _dsf_integral(s * seg, (s + 1) * seg, frac1, var1, var2, disk_r,
                               gl_x, gl_w, ring_x, ring_w)

    n_bins = int(math.ceil(5.0 * PSF_LIM / pixel_size))
    width = PSF_LIM / n_bins
    low_edges = np.empty(n_bins)
    values = np.empty(n_bins)
    for k in range(n_bins):
        low_edges[k] = k * width
        center = low_edges[k] + width / 2.0
        rmin = max(0.0, center - pixel_size / 2.0)
        rmax = min(PSF_LIM, center + pixel_size / 2.0)
        ring = _dsf_integral(rmin, rmax, frac1, var1, var2, disk_r, gl_x, gl_w, ring_x, ring_w)
        values[k] = ring * pixel_area / (math.pi * (rmax * rmax - rmin * rmin)) / total
    return low_edges, values


def pixelated_dsf(params, disk_radius, pixel_area):
    """Port of PointSourceDetectorResponse::CalculatePixelatedDsf (PSF convolved with a disk)."""
    _, frac1, var1, var2 = params
    return _pixelated_dsf(float(frac1), float(var1), float(var2), float(disk_radius),
                          float(pixel_area), _GL100_X, _GL100_W, _GL20_X, _GL20_W)


def fit_double_gaussian(xs, ys):
    """Fit A, f, var1, var2 of the double-Gaussian dN/dr to a tabulated PSF."""
    from scipy.optimize import least_squares

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    norm = np.trapz(ys, xs)
    y = ys / norm

    var0 = np.trapz(xs ** 2 * y, xs) / 2.0  # <r^2> = 2 var for a 2D Gaussian

    def model(p):
        a, f, lv1, lv2 = p
        v1, v2 = np.exp(lv1), np.exp(lv2)
        return a * xs * (f * np.exp(-xs ** 2 / 2 / v1) + (1 - f) * np.exp(-xs ** 2 / 2 / v2))

    p0 = [1.0 / var0, 0.7, np.log(var0 / 2), np.log(var0 * 3)]
    res = least_squares(lambda p: model(p) - y, p0,
                        bounds=([0, 0, -20, -20], [np.inf, 1, 10, 10]))
    a, f, lv1, lv2 = res.x
    return np.array([a * norm, f, np.exp(lv1), np.exp(lv2)])


# ============================================================================
# healpix_tools
# ============================================================================

UNSEEN = hp.UNSEEN  # -1.6375e30

# HAWC site (WGS84). AERIE takes it from the config directory; these are the
# standard HAWC coordinates.
HAWC_LATITUDE = 18.99456   # deg N
HAWC_LONGITUDE = -97.30861  # deg E
HAWC_HEIGHT = 4100.0        # m


class RingTable(object):
    """Per-ring geometry of a RING-ordered HEALPix map, as plain arrays for numba."""

    def __init__(self, nside):
        self.nside = nside
        rings = np.arange(1, 4 * nside)
        startpix, ringpix, costheta, _, _ = hp.ringinfo(nside, rings)
        self.start = startpix.astype(np.int64)
        self.npix = ringpix.astype(np.int64)
        self.z = costheta.astype(np.float64)
        self.theta = np.arccos(self.z)
        self.phi0 = hp.pix2ang(nside, self.start)[1].astype(np.float64)

    def max_disc_size(self, radius):
        """Upper bound on the number of pixels returned by disc_query."""
        pixrad = hp.max_pixrad(self.nside)
        frac = (1.0 - math.cos(min(math.pi, radius + 2 * pixrad))) / 2.0
        n_rings = int(2 * (radius + 2 * pixrad) / (math.pi / (4 * self.nside))) + 4
        return int(min(12 * self.nside ** 2, 1.5 * frac * 12 * self.nside ** 2 + 4 * n_rings))


@njit(cache=True)
def disc_query(theta0, phi0, radius, ring_theta, ring_z, ring_start, ring_npix, ring_phi0,
               out_pix, out_cos):
    """
    Pixels whose centers are within `radius` of (theta0, phi0), i.e.
    healpy.query_disc(inclusive=False). Fills out_pix with pixel ids and out_cos
    with the cosine of the angular distance; returns the number of pixels.
    """
    cos_r = math.cos(radius)
    z0 = math.cos(theta0)
    s0 = math.sin(theta0)
    x0 = s0 * math.cos(phi0)
    y0 = s0 * math.sin(phi0)

    i_lo = np.searchsorted(ring_theta, theta0 - radius - 1e-10)
    i_hi = np.searchsorted(ring_theta, theta0 + radius + 1e-10, side="right")

    n = 0
    for i in range(i_lo, i_hi):
        z = ring_z[i]
        st = math.sqrt(max(0.0, 1.0 - z * z))
        npix = ring_npix[i]
        step = 2.0 * math.pi / npix

        denom = st * s0
        full = False
        dphi = 0.0
        if denom <= 1e-15:
            full = True
        else:
            c = (cos_r - z * z0) / denom
            if c >= 1.0 + 1e-12:
                continue
            if c <= -1.0:
                full = True
            else:
                dphi = math.acos(min(1.0, c))

        if full:
            j_lo = 0
            j_hi = npix - 1
        else:
            # One extra pixel on each side; the exact test below decides
            j_lo = int(math.floor((phi0 - dphi - ring_phi0[i]) / step)) - 1
            j_hi = int(math.ceil((phi0 + dphi - ring_phi0[i]) / step)) + 1
            if j_hi - j_lo + 1 >= npix:
                j_lo = 0
                j_hi = npix - 1

        for j in range(j_lo, j_hi + 1):
            jj = j % npix
            phi = ring_phi0[i] + jj * step
            cosr = x0 * st * math.cos(phi) + y0 * st * math.sin(phi) + z0 * z
            if cosr >= cos_r:
                if n >= out_pix.shape[0]:
                    raise ValueError("disc_query buffer overflow")
                out_pix[n] = ring_start[i] + jj
                out_cos[n] = cosr
                n += 1
    return n


@njit(cache=True, parallel=False)
def disc_union_mask(thetas, phis, radius, ring_theta, ring_z, ring_start, ring_npix,
                    ring_phi0, mask, buffer_size):
    """Set mask[p] = True for every pixel within `radius` of any of the given centers."""
    out_pix = np.empty(buffer_size, dtype=np.int64)
    out_cos = np.empty(buffer_size, dtype=np.float64)
    for k in range(thetas.shape[0]):
        n = disc_query(thetas[k], phis[k], radius, ring_theta, ring_z, ring_start,
                       ring_npix, ring_phi0, out_pix, out_cos)
        for m in range(n):
            mask[out_pix[m]] = True


def needed_input_pixels(nside_in, out_nside, out_pixels, radius, precessed=None):
    """
    Input-map pixels needed to evaluate the given output pixels with ROI
    `radius` [rad]. Works on a coarse grid so it is cheap even for the full sky.

    :param precessed: optional (theta, phi) of the output pixels if they will be
        queried at precessed positions
    """
    ring = RingTable(nside_in)
    mask = np.zeros(hp.nside2npix(nside_in), dtype=np.bool_)

    coarse_nside = min(128, out_nside)
    if precessed is None:
        theta, phi = hp.pix2ang(out_nside, out_pixels)
    else:
        theta, phi = precessed
    coarse = np.unique(hp.ang2pix(coarse_nside, theta, phi))
    ctheta, cphi = hp.pix2ang(coarse_nside, coarse)

    pad = radius + hp.max_pixrad(coarse_nside) + 2 * hp.max_pixrad(nside_in)
    if precessed is not None:
        pad += np.radians(1.0)  # precession moves positions by < 1 deg over decades

    disc_union_mask(ctheta.astype(np.float64), cphi.astype(np.float64), float(pad),
                    ring.theta, ring.z, ring.start, ring.npix, ring.phi0, mask,
                    ring.max_disc_size(pad))
    return np.flatnonzero(mask)


def dec_band_map(nside, dec_bins):
    """
    Declination-band index of every pixel (-1 outside the response).

    Bands are assumed contiguous and ascending, as in the C++ code; inclusive
    strip queries mean edge pixels belong to the upper band.
    """
    band = np.full(hp.nside2npix(nside), -1, dtype=np.int64)
    for d, (lower, _, upper) in enumerate(dec_bins):
        pix = hp.query_strip(nside, np.radians(90.0 - upper), np.radians(90.0 - lower),
                             inclusive=True)
        band[pix] = d
    return band


def window_pixels(nside, ra, dec, delta_ra, delta_dec):
    """--window RA Dec Delta_RA Delta_Dec (deg): polygon with these corners."""
    phi0 = np.radians(ra)
    theta0 = np.radians(90.0 - dec)
    dphi = np.radians(delta_ra) / 2.0
    dtheta = np.radians(delta_dec) / 2.0

    thetas = [theta0 - dtheta, theta0 + dtheta, theta0 + dtheta, theta0 - dtheta]
    phis = [phi0 - dphi, phi0 - dphi, phi0 + dphi, phi0 + dphi]
    vertices = np.array(hp.ang2vec(np.array(thetas), np.array(phis)))
    return hp.query_polygon(nside, vertices, inclusive=False)


def dec_range_pixels(nside, dec_min, dec_max):
    """--decRange min max (deg)."""
    if dec_min >= dec_max:
        raise ValueError("Provide minimum and maximum declination, in order.")
    return hp.query_strip(nside, np.radians(90.0 - dec_max), np.radians(90.0 - dec_min),
                          inclusive=False)


def full_transit_pixels(nside, start_mjd, stop_mjd, fov,
                        latitude=HAWC_LATITUDE, longitude=HAWC_LONGITUDE,
                        height=HAWC_HEIGHT):
    """
    --fullTransit: pixels that completed at least one full transit through a
    field of view of radius `fov` [rad] between start_mjd and stop_mjd.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, FK5, SkyCoord
    from astropy.time import Time

    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=height * u.m)

    def zenith_j2000(mjd):
        t = Time(mjd, format="mjd", scale="utc")
        z = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=AltAz(obstime=t, location=location))
        z = z.transform_to(FK5(equinox="J2000"))
        return z.ra.rad, z.dec.rad

    start_ra, start_dec = zenith_j2000(start_mjd)
    stop_ra, _ = zenith_j2000(stop_mjd)

    det_theta = np.pi / 2 - start_dec
    start_phi, stop_phi = start_ra, stop_ra
    while start_phi > stop_phi:
        start_phi -= 2 * np.pi
    if stop_mjd - start_mjd > 0.99 and (stop_phi - start_phi) / (2 * np.pi) < 0.1:
        start_phi = stop_phi - 2 * np.pi  # numerical errors for ~1 day maps

    rings = np.arange(1, 4 * nside)
    startpix, ringpix, costheta, sintheta, _ = hp.ringinfo(nside, rings)

    selected = []
    for r in range(rings.size):
        theta = np.arccos(costheta[r])
        if abs(theta - det_theta) > fov:
            continue
        cos_dphi = ((np.cos(fov) - np.cos(det_theta) * costheta[r]) /
                    (np.sin(det_theta) * sintheta[r]))
        if abs(cos_dphi) > 1:
            continue
        dphi = np.arccos(cos_dphi)
        if stop_phi - start_phi < 2 * dphi:
            continue

        p_start = int(hp.ang2pix(nside, theta, np.mod(start_phi + dphi, 2 * np.pi)))
        p_stop = int(hp.ang2pix(nside, theta, np.mod(stop_phi - dphi, 2 * np.pi)))
        if p_start == p_stop:
            continue
        if p_start < p_stop:
            selected.append(np.arange(p_start + 1, p_stop))
        else:
            selected.append(np.arange(p_start + 1, startpix[r] + ringpix[r]))
            selected.append(np.arange(startpix[r], p_stop))

    if not selected:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(selected))


# ============================================================================
# maps
# ============================================================================

_FITS_BIN_RE = re.compile(r"bin_?([A-Za-z0-9]+?)(?:[-_.]|$)")


class MapInfo(object):
    """Header information HealpixSigFluxMap takes from the first input map."""

    def __init__(self, start_mjd=np.nan, stop_mjd=np.nan, tot_dur=np.nan, epoch="unknown",
                 n_transits=None):
        self.start_mjd = start_mjd
        self.stop_mjd = stop_mjd
        self.tot_dur = tot_dur  # hours
        self.epoch = epoch
        self._n_transits = n_transits

    @property
    def n_transits(self):
        if self._n_transits is not None:
            return self._n_transits
        if np.isfinite(self.tot_dur):
            return max(self.tot_dur / 24.0, 1.0)
        raise ValueError("Number of transits unknown; pass n_transits")


class InputMaps(object):
    """
    Data and background of several analysis bins, restricted to the pixels that
    are needed (all bins must share one nside).

    data/bkg have shape (n_bins, n_loaded); lookup[pixel] gives the column of a
    pixel or -1 if it was not loaded.
    """

    def __init__(self, bins, nside, loaded_pixels, data, bkg, info):
        self.bins = list(bins)
        self.nside = nside
        self.loaded_pixels = loaded_pixels
        self.data = data
        self.bkg = bkg
        self.info = info
        self.lookup = np.full(hp.nside2npix(nside), -1, dtype=np.int64)
        self.lookup[loaded_pixels] = np.arange(loaded_pixels.size)


def bin_name_from_filename(path):
    """'binB7C0Ej-rejiggered.fits.gz' -> 'B7C0Ej'; 'residual_binB7C0Ej.fits.gz' -> 'B7C0Ej'."""
    m = _FITS_BIN_RE.search(os.path.basename(path))
    if not m:
        raise ValueError("Cannot infer the bin name of %s; pass bins explicitly" % path)
    return m.group(1)


def find_fits_files(directory, bins):
    """Find the FITS file of each bin in a directory (same matching as pipeline-map-maker.py)."""
    files = []
    for name in bins:
        pattern = re.compile(r"bin_?%s(?:[-_.])" % re.escape(name))
        matches = sorted(f for f in os.listdir(directory)
                         if pattern.search(f) and ".fits" in f)
        if not matches:
            raise IOError("No FITS file found for bin %s in %s" % (name, directory))
        if len(matches) > 1:
            log.warning("Several files for bin %s, using %s", name, matches[0])
        files.append(os.path.join(directory, matches[0]))
    return files


def read_fits_info(path):
    """STARTMJD/STOPMJD/TOTDUR/EPOCH from the primary (or first extension) header."""
    with fits.open(path) as hdul:
        headers = [hdul[0].header] + ([hdul[1].header] if len(hdul) > 1 else [])
        nside = None
        for h in headers:
            if "NSIDE" in h:
                nside = int(h["NSIDE"])

        def get(key, default=None):
            for h in headers:
                if key in h:
                    return h[key]
            return default

        start, stop, tot = get("STARTMJD"), get("STOPMJD"), get("TOTDUR")
        if start is None or stop is None or tot is None:
            raise KeyError("The header keys STARTMJD, STOPMJD and TOTDUR are needed in "
                           "the first map (%s)" % path)
        info = MapInfo(float(start), float(stop), float(tot), str(get("EPOCH", "unknown")).strip())
    return info, nside


def fits_nside(path):
    with fits.open(path) as hdul:
        for h in hdul:
            if "NSIDE" in h.header:
                return int(h.header["NSIDE"])
    return None


def load_fits_maps(files, bins, needed_pixels_fn, dtype=np.float32):
    """
    :param needed_pixels_fn: callable nside -> array of pixels to keep
    """
    info, _ = read_fits_info(files[0])
    nsides = [fits_nside(f) for f in files]
    if len(set(nsides)) != 1 or nsides[0] is None:
        raise ValueError("All input maps must have the same nside (got %s)" % nsides)
    nside = nsides[0]

    pixels = needed_pixels_fn(nside)
    data = np.empty((len(files), pixels.size), dtype=dtype)
    bkg = np.empty((len(files), pixels.size), dtype=dtype)
    for i, f in enumerate(files):
        log.info("Reading bin %s from %s", bins[i], f)
        d, b = hp.read_map(f, field=(0, 1), dtype=np.float64)
        data[i] = d[pixels]
        bkg[i] = b[pixels]
        del d, b
    return InputMaps(bins, nside, pixels, data, bkg, info)


def load_maptree(maptree_file, bins, needed_pixels_fn, n_transits=None, dtype=np.float32):
    """HAL map tree (.root or .hd5), full sky read through hawc_hal."""
    from hawc_hal.maptree import map_tree_factory

    tree = map_tree_factory(maptree_file, None, n_transits)
    if bins is None:
        bins = list(tree.analysis_bins_labels)
    nside = tree[bins[0]].nside
    pixels = needed_pixels_fn(nside)

    data = np.empty((len(bins), pixels.size), dtype=dtype)
    bkg = np.empty((len(bins), pixels.size), dtype=dtype)
    for i, name in enumerate(bins):
        b = tree[name]
        if b.nside != nside:
            raise ValueError("All bins must have the same nside")
        data[i] = b.observation_map.as_dense()[pixels]
        bkg[i] = b.background_map.as_dense()[pixels]

    info = MapInfo(n_transits=tree.n_transits, epoch="J2000")
    return InputMaps(bins, nside, pixels, data, bkg, info)


def write_output(path, nside, columns, info, pivot, per_sr, overwrite=False):
    """
    Write the significance/flux maps in the same layout as HealpixSigFluxMap:
    header keys in the primary HDU, a RING HEALPix binary table with one column
    per map.

    :param columns: list of (name, full-sky array)
    """
    if os.path.exists(path) and not overwrite:
        raise IOError("%s already exists" % path)

    primary = fits.PrimaryHDU()
    h = primary.header
    h["STARTMJD"] = (info.start_mjd, "MJD of first event")
    h["STOPMJD"] = (info.stop_mjd, "MJD of last event")
    h["TOTDUR"] = (info.tot_dur, "Total integration time [hours]")
    h["EPOCH"] = ("J2000", "e.g. J2000, current, J2016, B1950, etc.")
    h["COORDSYS"] = ("C", "Coordinate system")
    units = "TeV^-1 cm^-2 s^-1 sr^-1" if per_sr else "TeV^-1 cm^-2 s^-1"
    h["PIVOT"] = (pivot, "Pivot energy [TeV]")
    h["FLUXUNIT"] = (units, "Units of the flux columns")

    npix = hp.nside2npix(nside)
    per_row = 1024 if npix % 1024 == 0 else 1
    cols = [fits.Column(name=name, format="%dD" % per_row,
                        array=np.asarray(values, dtype=np.float64).reshape(-1, per_row))
            for name, values in columns]
    table = fits.BinTableHDU.from_columns(cols)
    t = table.header
    t["PIXTYPE"] = ("HEALPIX", "HEALPIX pixelisation")
    t["ORDERING"] = ("RING", "Pixel ordering scheme, either RING or NESTED")
    t["NSIDE"] = (nside, "Resolution parameter for HEALPIX")
    t["FIRSTPIX"] = (0, "First pixel # (0 based)")
    t["LASTPIX"] = (npix - 1, "Last pixel # (0 based)")
    t["INDXSCHM"] = ("IMPLICIT", "Indexing: IMPLICIT or EXPLICIT")
    t["OBJECT"] = ("FULLSKY", "Sky coverage, either FULLSKY or PARTIAL")
    t["COORDSYS"] = ("C", "Coordinate system")

    fits.HDUList([primary, table]).writeto(path, overwrite=overwrite)


# ============================================================================
# sigfluxmap
# ============================================================================

FC_SEED = 4357  # fixed seed, so FC intervals are reproducible (as in AERIE)


class SigFluxMapResult(object):
    """Full-sky RING maps (UNSEEN where not computed) plus bookkeeping."""

    def __init__(self, nside, significance, flux, flux_error=None, flux_lower=None,
                 flux_upper=None, computed_pixels=None, failed_pixels=0, info=None,
                 pivot=1.0, per_sr=False):
        self.nside = nside
        self.significance = significance
        self.flux = flux
        self.flux_error = flux_error
        self.flux_lower = flux_lower
        self.flux_upper = flux_upper
        self.computed_pixels = computed_pixels
        self.failed_pixels = failed_pixels
        self.info = info
        self.pivot = pivot
        self.per_sr = per_sr

    def columns(self):
        cols = [("significance", self.significance), ("flux", self.flux)]
        if self.flux_lower is not None:
            cols += [("flux_lower_bound", self.flux_lower), ("flux_upper_bound", self.flux_upper)]
        else:
            cols += [("flux_error", self.flux_error)]
        return cols

    def write(self, path, overwrite=False):
        write_output(path, self.nside, self.columns(), self.info, self.pivot, self.per_sr,
                     overwrite=overwrite)

    def value_at(self, ra, dec, column="significance"):
        """Value of a map at (RA, Dec) in degrees."""
        pix = hp.ang2pix(self.nside, ra, dec, lonlat=True)
        return dict(self.columns())[column][pix]


@njit(parallel=True, cache=True)
def _compute_pixels(theta, phi, dec_bin1, sim_theta,
                    ring_theta, ring_z, ring_start, ring_npix, ring_phi0,
                    lookup, data_maps, bkg_maps, padding, share_roi, buffer_size,
                    psf_values, psf_nbins, psf_width, unit_tot_excess,
                    allow_neg_ts, allow_neg_flux, max_iter,
                    get_fc, sigmas_fc, bounds_frac_err, steradians,
                    out_sig, out_flux, out_a, out_b, out_status):
    """
    out_status: 0 = computed, 1 = skipped (UNSEEN/non-positive bkg), 2 = failed fit.
    out_a/out_b: flux error (and unused), or FC lower/upper bounds.
    """
    n_pix = theta.shape[0]
    n_bins = data_maps.shape[0]
    n_dec = sim_theta.shape[0]

    for i in prange(n_pix):
        # --- declination interpolation weights (same logic as the C++ code)
        d1 = dec_bin1[i]
        st1 = sim_theta[d1]
        th = theta[i]

        d2 = d1 - 1
        if d1 == 0 or (d1 + 1 < n_dec and th < st1):
            d2 = d1 + 1
        if n_dec == 1:
            d2 = 0
        st2 = sim_theta[d2]

        if n_dec == 1:
            w1 = 1.0
            w2 = 0.0
        else:
            w1 = (st2 - th) / (st2 - st1)
            w2 = (st1 - th) / (st1 - st2)
            if w1 > 1:
                w1 = 1.0
                w2 = 0.0
            elif w2 > 1:
                w2 = 1.0
                w1 = 0.0

        # --- ROI pixels for every bin
        n_rows = 1 if share_roi else n_bins
        roi_pix = np.empty((n_rows, buffer_size), dtype=np.int64)
        roi_cos = np.empty((n_rows, buffer_size), dtype=np.float64)
        roi_n = np.zeros(n_bins, dtype=np.int64)
        total = 0
        for b in range(n_bins):
            if b > 0 and share_roi:
                roi_n[b] = roi_n[0]
            else:
                roi_n[b] = disc_query(th, phi[i], padding[b], ring_theta, ring_z,
                                          ring_start, ring_npix, ring_phi0,
                                          roi_pix[b], roi_cos[b])
            total += roi_n[b]

        data = np.empty(total)
        bkg = np.empty(total)
        excess = np.empty(total)

        bad = False
        n = 0
        for b in range(n_bins):
            src = 0 if (b > 0 and share_roi) else b
            norm_excess = w1 * unit_tot_excess[b, d1] + w2 * unit_tot_excess[b, d2]
            for k in range(roi_n[b]):
                c = roi_cos[src, k]
                if c > 1.0:
                    c = 1.0
                psf_bin = int(math.acos(c) / psf_width[b])
                if psf_bin > psf_nbins[b] - 1:
                    psf_bin = psf_nbins[b] - 1

                col = lookup[roi_pix[src, k]]
                if col < 0:
                    bad = True
                    break
                dv = data_maps[b, col]
                bv = bkg_maps[b, col]
                if dv < -1e30 or bv < -1e30 or bv <= 0:
                    bad = True
                    break

                data[n] = dv
                bkg[n] = bv
                excess[n] = ((w1 * psf_values[b, d1, psf_bin] + w2 * psf_values[b, d2, psf_bin])
                             * norm_excess)
                n += 1
            if bad:
                break

        if bad:
            out_status[i] = 1
            continue

        if get_fc:
            failed, ts, norm, lo, hi = solve_fc(data, bkg, excess, sigmas_fc, bounds_frac_err,
                                                allow_neg_ts, max_iter, 4357)
        else:
            failed, ts, norm, err = solve(data, bkg, excess, allow_neg_ts, allow_neg_flux,
                                          max_iter)
            lo = err
            hi = 0.0

        if failed:
            out_status[i] = 2
            continue

        out_status[i] = 0
        out_flux[i] = norm / steradians
        if ts >= 0:
            out_sig[i] = math.sqrt(ts)
        else:
            out_sig[i] = -math.sqrt(-ts)
        out_a[i] = lo / steradians
        out_b[i] = hi / steradians


def _precess_to_epoch(theta, phi, mjd):
    """J2000 -> mean equinox of `mjd` (for maps in 'current' epoch)."""
    import astropy.units as u
    from astropy.coordinates import FK5, SkyCoord
    from astropy.time import Time

    c = SkyCoord(ra=phi * u.rad, dec=(np.pi / 2 - theta) * u.rad, frame=FK5(equinox="J2000"))
    c = c.transform_to(FK5(equinox=Time(mjd, format="mjd")))
    return np.pi / 2 - c.dec.rad, c.ra.wrap_at(360 * u.deg).rad


def make_sig_flux_map(inputs, response, bins=None, output=None, overwrite=False,
                      spectrum=None, index=None, cutoff=None, pivot=1.0, extension=0.0,
                      padding=2.0, nside=None, n_transits=None, per_sr=False,
                      allow_negative_ts=False, allow_negative_flux=False,
                      fc_sigmas=None, fc_precision=5.0, max_iter=1000,
                      window=None, dec_range=None, full_transit_fov=None,
                      n_threads=None, n_parts=1, part=1, map_dtype=np.float32,
                      progress_chunks=50):
    """
    Compute significance and flux maps (HealpixSigFluxMap).

    :param inputs: list of per-bin FITS files (data in column 1, background in
        column 2), a directory containing them (needs `bins`), or a HAL map tree
        (.root/.hd5)
    :param response: detector response (.root from liff, or HAL .hd5)
    :param bins: analysis bin names matching the inputs. For a list of FITS files
        they default to the names in the file names.
    :param spectrum: spectral shape: an astromodels function, an AERIE string
        ('SimplePowerLaw,3.5e-11,2.63' / 'CutOffPowerLaw,3.76e-11,2.39,14.3') or
        any callable f(E_TeV). Supersedes index/cutoff.
    :param index: power-law index, positive convention (e.g. 2.5)
    :param cutoff: exponential cutoff [TeV]
    :param pivot: pivot energy [TeV]
    :param extension: disk radius [deg] convolved with the PSF (0: point source)
    :param padding: ROI radius [deg], one value or one per bin; extension is added
    :param nside: output nside (default: input nside)
    :param n_transits: override the number of transits (default max(TOTDUR/24, 1))
    :param per_sr: report flux per steradian for extended searches
    :param allow_negative_ts: report underfluctuations with negative significance
        (--negSignif)
    :param allow_negative_flux: also let the flux go negative (--negFlux)
    :param fc_sigmas: compute Feldman-Cousins intervals for this many sigmas (--fc)
    :param fc_precision: precision of FC limits [%] (--fcPres)
    :param window: (RA, Dec, Delta_RA, Delta_Dec) in degrees; Dec may be negative
    :param dec_range: (dec_min, dec_max) in degrees
    :param full_transit_fov: only pixels with a full transit within this FOV [rad]
    :param n_parts, part: split the pixels in n_parts and compute part (1-based)
    :param map_dtype: storage type of the input maps (float32 halves memory)
    :return: SigFluxMapResult
    """
    t0 = time.time()

    if n_threads is not None:
        set_num_threads(int(n_threads))
    log.info("Using %d threads", get_num_threads())

    extension = max(float(extension), 0.0)
    steradians = 2 * np.pi * (1 - np.cos(np.radians(extension))) if per_sr and extension > 0 else 1.0

    get_fc = fc_sigmas is not None
    if get_fc and allow_negative_flux:
        raise ValueError("fc_sigmas incompatible with allow_negative_flux")
    if allow_negative_flux:
        allow_negative_ts = True

    if output is not None and os.path.exists(output) and not overwrite:
        raise IOError("%s already exists" % output)

    if window is not None and window[1] > 90:
        # AERIE CLI workaround: negative Dec given as Dec + 360
        window = (window[0], window[1] - 360.0, window[2], window[3])

    # ===== Resolve inputs
    maptree = None
    if isinstance(inputs, str) and os.path.isdir(inputs):
        if bins is None:
            raise ValueError("bins are needed when inputs is a directory")
        files = find_fits_files(inputs, bins)
    elif isinstance(inputs, str) and os.path.splitext(inputs)[-1] in (".root", ".hd5", ".hdf5"):
        maptree = inputs
        files = None
    else:
        files = [inputs] if isinstance(inputs, str) else list(inputs)
        if bins is None:
            bins = [bin_name_from_filename(f) for f in files]
        if len(bins) != len(files):
            raise ValueError("Size of inputs and bins should be the same")

    # ===== Response
    log.info("Reading detector response %s", response)
    dr = DetectorResponse.open(response, bins)
    if bins is None:
        bins = list(dr.response_bins[0].keys())
    n_bins = len(bins)
    n_dec = dr.n_dec_bins
    if n_dec == 1:
        log.warning("Response has a single declination band at Dec = %g deg", dr.dec_bins[0][1])

    pad = np.atleast_1d(np.asarray(padding, dtype=float))
    if pad.size == 1:
        pad = np.full(n_bins, pad[0])
    elif pad.size != n_bins:
        raise ValueError("padding size must be either 1 or equal to the number of bins")
    pad_rad = np.radians(pad + extension)
    for b, name in enumerate(bins):
        log.debug("Bin %s maximum padding radius = %g deg", name, np.degrees(pad_rad[b]))
    share_roi = bool(np.all(pad_rad == pad_rad[0]))

    # ===== Pixels to compute. The output nside is only known once the input
    # nside is, so selection happens inside the loader callback.
    state = {}

    def select_pixels(nside_in):
        out_nside = nside or nside_in
        state["out_nside"] = out_nside
        band = dec_band_map(out_nside, dr.dec_bins)
        state["band"] = band
        computed = band >= 0
        if window is not None:
            log.info("Setting window %s", window)
            sel = np.zeros_like(computed)
            sel[window_pixels(out_nside, *window)] = True
            computed &= sel
        if dec_range is not None:
            sel = np.zeros_like(computed)
            sel[dec_range_pixels(out_nside, *dec_range)] = True
            computed &= sel
        state["computed"] = computed
        return np.flatnonzero(computed)

    def needed(nside_in):
        pixels = select_pixels(nside_in)
        # full transit needs the map times; it is applied afterwards on top
        if state.get("precess"):
            th, ph = hp.pix2ang(state["out_nside"], pixels)
            prec = _precess_to_epoch(th, ph, state["mean_mjd"])
        else:
            prec = None
        log.info("Selecting input pixels within %.2f deg of %d output pixels",
                 np.degrees(pad_rad.max()), pixels.size)
        return needed_input_pixels(nside_in, state["out_nside"], pixels, pad_rad.max(), prec)

    # Epoch must be known before selecting input pixels
    if files is not None:
        info0, _ = read_fits_info(files[0])
        state["precess"] = info0.epoch == "current"
        state["mean_mjd"] = (info0.start_mjd + info0.stop_mjd) / 2.0
        if state["precess"]:
            log.warning("Map is in current epoch, will precess to J2000")
        log.info("Opening maps...")
        maps = load_fits_maps(files, bins, needed, dtype=map_dtype)
    else:
        log.info("Opening map tree %s", maptree)
        maps = load_maptree(maptree, bins, needed, n_transits=n_transits, dtype=map_dtype)

    info = maps.info
    ntr = float(n_transits) if n_transits is not None else info.n_transits
    log.info("Number of transits: %g", ntr)

    out_nside = state["out_nside"]
    computed = state["computed"]

    if full_transit_fov is not None:
        if not np.isfinite(info.start_mjd):
            raise ValueError("full_transit_fov needs STARTMJD/STOPMJD in the input maps")
        if info.stop_mjd - info.start_mjd >= 1:
            log.warning("full transit selection used with a map that covers one or more "
                        "sidereal days.")
        log.info("Finding pixels with at least one full transit...")
        sel = np.zeros_like(computed)
        sel[full_transit_pixels(out_nside, info.start_mjd, info.stop_mjd,
                                    full_transit_fov)] = True
        computed &= sel

    all_pixels = np.flatnonzero(computed)

    # nparts / part (the last part takes the remainder)
    if not 1 <= part <= n_parts:
        raise ValueError("part must be between 1 and n_parts")
    per_part = all_pixels.size // n_parts
    lo = (part - 1) * per_part
    hi = all_pixels.size if part == n_parts else part * per_part
    pixels = all_pixels[lo:hi]
    log.info("Computing %d pixels (nside %d)", pixels.size, out_nside)

    # ===== Response tables
    log.info("Processing detector response...")
    shape = as_spectrum(spectrum, pivot=pivot, index=index, cutoff=cutoff)
    log.info("Spectrum: %r", shape)

    sim_theta = np.radians(90.0 - dr.sim_decs)
    pixel_area = 4 * np.pi / hp.nside2npix(maps.nside) / np.radians(1.0) ** 2  # deg^2

    unit_tot_excess = np.empty((n_bins, n_dec))
    psf_tables = []
    for d in range(n_dec):
        if extension > 0:
            log.info("Declination band %d/%d", d + 1, n_dec)
        for b, name in enumerate(bins):
            unit_tot_excess[b, d] = dr.expected_signal(shape, d, name) * ntr
            psf_tables.append(dr.pixelated_psf(d, name, pixel_area, extension))

    low_edges = psf_tables[0][0]
    n_psf = low_edges.size
    psf_values = np.zeros((n_bins, n_dec, n_psf))
    for d in range(n_dec):
        for b in range(n_bins):
            edges, values = psf_tables[d * n_bins + b]
            psf_values[b, d] = values
    psf_width = np.full(n_bins, np.radians(low_edges[1] - low_edges[0]))

    # Keep PSF bins up to the first one beyond the ROI
    psf_nbins = np.empty(n_bins, dtype=np.int64)
    low_rad = np.radians(low_edges)
    for b in range(n_bins):
        beyond = np.flatnonzero(pad_rad[b] < low_rad)
        psf_nbins[b] = beyond[0] + 1 if beyond.size else n_psf

    # ===== Pixel positions
    theta, phi = hp.pix2ang(out_nside, pixels)
    if state.get("precess"):
        theta, phi = _precess_to_epoch(theta, phi, state["mean_mjd"])
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    phi = np.ascontiguousarray(phi, dtype=np.float64)
    dec_bin1 = state["band"][pixels].astype(np.int64)

    ring = RingTable(maps.nside)
    buffer_size = ring.max_disc_size(pad_rad.max())

    # ===== Fit
    log.info("Computing significance...")
    n = pixels.size
    sig = np.full(n, UNSEEN)
    flux = np.full(n, UNSEEN)
    a = np.full(n, UNSEEN)
    bb = np.full(n, UNSEEN)
    status = np.full(n, 1, dtype=np.int64)

    bounds_frac_err = fc_precision / 100.0
    chunks = np.array_split(np.arange(n), max(1, min(progress_chunks, n)))
    t_fit = time.time()
    for k, idx in enumerate(chunks):
        if idx.size == 0:
            continue
        s = slice(idx[0], idx[-1] + 1)
        _compute_pixels(theta[s], phi[s], dec_bin1[s], sim_theta,
                        ring.theta, ring.z, ring.start, ring.npix, ring.phi0,
                        maps.lookup, maps.data, maps.bkg, pad_rad, share_roi, buffer_size,
                        psf_values, psf_nbins, psf_width, unit_tot_excess,
                        bool(allow_negative_ts), bool(allow_negative_flux), int(max_iter),
                        get_fc, float(fc_sigmas or 0.0), bounds_frac_err, float(steradians),
                        sig[s], flux[s], a[s], bb[s], status[s])
        log.info("%d%% (%.0f s)", int(100 * (k + 1) / len(chunks)), time.time() - t_fit)

    failed = int(np.sum(status == 2))
    if failed:
        log.warning("The minimization failed or reached the maximum number of iterations "
                    "allowed in %d pixels, which were set to UNSEEN", failed)

    def full(values):
        m = np.full(hp.nside2npix(out_nside), UNSEEN)
        m[pixels] = values
        return m

    result = SigFluxMapResult(out_nside, full(sig), full(flux),
                              flux_error=None if get_fc else full(a),
                              flux_lower=full(a) if get_fc else None,
                              flux_upper=full(bb) if get_fc else None,
                              computed_pixels=pixels, failed_pixels=failed, info=info,
                              pivot=pivot, per_sr=per_sr and extension > 0)

    if output is not None:
        log.info("Writing file %s to disk...", output)
        result.write(output, overwrite=overwrite)

    log.info("Elapsed time: %.2f min", (time.time() - t0) / 60.0)
    return result


# ============================================================================
# cli
# ============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Computes significances using the maximum likelihood ratio with the "
                    "common norm as the only free parameter (standalone HealpixSigFluxMap).")

    io = p.add_argument_group("Input/output")
    io.add_argument("-i", "--input", nargs="+", required=True,
                    help="FITS files with data and background (one per bin), a directory "
                         "with them (requires --bins), or a HAL map tree (.root/.hd5)")
    io.add_argument("-b", "--bins", nargs="+",
                    help="Bin names corresponding to the inputs. Default: from the file names")
    io.add_argument("-d", "--detfile", required=True, help="Detector response (.root or .hd5)")
    io.add_argument("-o", "--output", required=True, help="Output map file name")
    io.add_argument("--overwrite", action="store_true", help="Overwrite the output file")

    m = p.add_argument_group("Model controls")
    m.add_argument("--index", type=float, help="Spectral index (e.g. 2.7)")
    m.add_argument("--cutoff", type=float, help="Cutoff [TeV]")
    m.add_argument("-s", "--spectrum",
                   help="Source spectrum e.g. 'SimplePowerLaw,3.5e-11,2.63' or "
                        "'CutOffPowerLaw,3.76e-11,2.39,14.3'. Supersedes index/cutoff")
    m.add_argument("--extension", type=float, default=0.0,
                   help="Convolve PSF with a disc of this radius [deg]")

    c = p.add_argument_group("Minimizer controls")
    c.add_argument("--negSignif", action="store_true", help="Allow negative significance.")
    c.add_argument("--negFlux", action="store_true", help="Allow also negative fluxes.")
    c.add_argument("--fc", type=float,
                   help="Get Feldman-Cousins confidence interval for a given number of "
                        "standard deviations. Slower.")
    c.add_argument("--fcPres", type=float, default=5.0, help="Precision of FC limits [%%]")
    c.add_argument("--maxIter", type=int, default=1000, help="Maximum number of iterations")

    s = p.add_argument_group("Only compute a portion of the sky")
    s.add_argument("--window", nargs=4, type=float, metavar=("RA", "DEC", "DRA", "DDEC"),
                   help="Compute only for a region: RA Dec Delta_RA Delta_Dec [deg]. "
                        "Negative Dec can be given directly (Dec+360 also accepted)")
    s.add_argument("--fullTransit", type=float,
                   help="Only pixels that completed at least a full transit. FOV in radians")
    s.add_argument("--decRange", nargs=2, type=float, help="Declination range [deg]")

    par = p.add_argument_group("Parallelize")
    par.add_argument("--nthreads", type=int, help="Number of threads (default: all cores)")
    par.add_argument("--nparts", type=int, default=1, help="Divide pixels in nparts")
    par.add_argument("--part", type=int, default=1, help="Do a part between 1 and --nparts")

    o = p.add_argument_group("Other optional arguments")
    o.add_argument("--nside", type=int, help="Nside of the output map. Default: input nside")
    o.add_argument("--padding", nargs="+", type=float, default=[2.0],
                   help="Padding radius (ROI) [deg]; extension is added. One value or one per bin")
    o.add_argument("--pivot", type=float, default=1.0, help="Pivot energy [TeV]")
    o.add_argument("--ntransits", type=float,
                   help="Number of transits, otherwise max(TOTDUR/24, 1)")
    o.add_argument("--perSr", action="store_true",
                   help="Report flux per steradian when doing an extended search")
    o.add_argument("--float64", action="store_true",
                   help="Keep input maps in double precision (twice the memory)")
    o.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.spectrum is None and args.index is None:
        print("Either --spectrum or --index is required", file=sys.stderr)
        return 1

    inputs = args.input[0] if len(args.input) == 1 else args.input

    import numpy as np
    result = make_sig_flux_map(
        inputs=inputs, response=args.detfile, bins=args.bins, output=args.output,
        overwrite=args.overwrite, spectrum=args.spectrum, index=args.index,
        cutoff=args.cutoff, pivot=args.pivot, extension=args.extension,
        padding=args.padding, nside=args.nside, n_transits=args.ntransits,
        per_sr=args.perSr, allow_negative_ts=args.negSignif,
        allow_negative_flux=args.negFlux, fc_sigmas=args.fc, fc_precision=args.fcPres,
        max_iter=args.maxIter, window=args.window, dec_range=args.decRange,
        full_transit_fov=args.fullTransit, n_threads=args.nthreads,
        n_parts=args.nparts, part=args.part,
        map_dtype=np.float64 if args.float64 else np.float32)
    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())

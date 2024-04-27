from pathlib import Path
from typing import Any, Iterable

from . import paths

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import QTable
from synphot import SpectralElement


def load_vampires_filter(name: str, csv_path=paths.datadir / "vampires_filters.csv"):
    return SpectralElement.from_file(
        str(csv_path.absolute()), wave_unit="nm", include_names=["wave", name]
    )


def get_filter_info(filt: SpectralElement) -> dict[str, Any]:
    waves = filt.waveset
    through = filt.model.lookup_table
    above_50 = np.nonzero(through >= 0.5 * np.nanmax(through))[0]
    waveset = waves[above_50]
    filt_info = dict(
        lam_min=waveset[0].to(u.nm),
        lam_max=waveset[-1].to(u.nm),
        lam_ave=filt.avgwave(waveset).to(u.nm),
    )
    filt_info["width"] = filt_info["lam_max"] - filt_info["lam_min"]
    filt_info["dlam/lam"] = filt_info["width"] / filt_info["lam_ave"]
    filt_info["ext"] = get_average_extinction(filt, waveset)
    filt_info["qe"] = get_average_qe(filt, waveset)
    return filt_info


def create_filter_table(filters: Iterable[SpectralElement]) -> QTable:
    return QTable(list(map(get_filter_info, filters)))


def create_vampires_filter_table(filters: Iterable[str]) -> QTable:
    _filters = list(sorted(filters))
    tbl = create_filter_table(load_vampires_filter(f) for f in _filters)
    tbl["filter"] = _filters
    zp_info = load_vampires_zeropoints()
    zp_e = np.array([zp_info[f] for f in tbl["filter"]])
    tbl["zp_mag"] = 2.5 * np.log10(zp_e)
    tbl["zp_e"] = zp_e * u.electron / u.s
    return tbl


def load_snf_ext(fits_path=paths.datadir / "SNFext.fits"):
    return fits.getdata(fits_path)


def get_average_extinction(filt: SpectralElement, waveset=None, extcurve=None):
    if extcurve is None:
        extcurve = load_snf_ext()
    if waveset is None:
        waveset = filt.waveset
    # SNF extinction data is in Angstrom
    waves = waveset.to("Angstrom").value
    extinction = np.interp(waves, extcurve["LAMBDA"], extcurve["EXT"], left=0, right=0)
    # calculate average extinction
    return np.trapz(extinction, x=waves) / np.ptp(waves)


def load_vampires_qe(csv_path: Path = paths.datadir / "vampires_qe.csv"):
    return pd.read_csv(csv_path)


def get_average_qe(filt: SpectralElement, waveset=None, qe=None):
    if qe is None:
        qe = load_vampires_qe()
    if waveset is None:
        waveset = filt.waveset
    # QE data is in nanometers
    waves = waveset.to("nm").value
    qe = np.interp(waves, qe["wave (nm)"], qe["QE (%)"] / 100, left=0, right=0)
    # calculate average extinction
    return np.trapz(qe, x=waves) / np.ptp(waves)


def load_vampires_zeropoints(datadir: Path = paths.datadir):
    tbl = pd.read_csv(datadir / "vampires_zeropoints.csv")
    filts = tbl["filter"]
    zps = tbl["zp (e-/s)"]
    return dict(zip(filts, zps))


VAMPIRES_STD_FILTERS = {"Open", "625-50", "675-50", "725-50", "750-50", "775-50"}
VAMPIRES_MBI_FILTERS = {"F610", "F670", "F720", "F760"}
VAMPIRES_NB_FILTERS = {"Halpha", "Ha-Cont", "SII", "SII-Cont"}
VAMPIRES_FILTERS = VAMPIRES_STD_FILTERS | VAMPIRES_MBI_FILTERS | VAMPIRES_NB_FILTERS

FILTERS = {
    "U": SpectralElement.from_filter("johnson_u"),
    "B": SpectralElement.from_filter("johnson_b"),
    "V": SpectralElement.from_filter("johnson_v"),
    "R": SpectralElement.from_filter("johnson_r"),
    "I": SpectralElement.from_filter("johnson_i"),
    # "G": SpectralElement.from_filter("johnson_v"),
    # "G_BP": SpectralElement.from_filter("johnson_v"),
    # "G_RP": SpectralElement.from_filter("johnson_v"),
    "J": SpectralElement.from_filter("johnson_j"),
    "H": SpectralElement.from_filter("bessel_h"),
    "K": SpectralElement.from_filter("johnson_k"),
}

VAMP_FILTERS = {f: load_vampires_filter(f) for f in VAMPIRES_FILTERS}
VAMP_ZEROPOINTS = load_vampires_zeropoints()

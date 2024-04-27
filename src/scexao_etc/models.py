from pathlib import Path

import astropy.units as u
from astropy.io import fits
from synphot import Empirical1D, Observation, SourceSpectrum, SpectralElement
from synphot.units import VEGAMAG

BASE_DIR = Path("/Users/mileslucas/dev/python/scexao_etc/data/pickles_uvk")
TYPES = ("V", "IV", "III", "II", "I")


def prepare_pickles_dict(base_dir=BASE_DIR):
    tbl = fits.getdata(BASE_DIR / "pickles_uk.fits")
    fnames = (base_dir / f"{fname}.fits" for fname in tbl["FILENAME"])
    return dict(zip(tbl["SPTYPE"], fnames))


PICKLES_MAP = prepare_pickles_dict(BASE_DIR)
VEGASPEC = SourceSpectrum.from_vega()


def load_pickles(spec_type):
    filename = PICKLES_MAP[spec_type]
    tbl = fits.getdata(filename)
    sp = SourceSpectrum(
        Empirical1D,
        points=tbl["WAVELENGTH"] * u.angstrom,
        lookup_table=tbl["FLUX"] * u.erg / u.s / u.cm**2 / u.angstrom,
    )
    return sp


def color_correction(model, filt1: SpectralElement, filt2: SpectralElement):
    obs1 = Observation(model, filt1)
    obs2 = Observation(model, filt2)
    return obs1.effstim(VEGAMAG, vegaspec=VEGASPEC) - obs2.effstim(VEGAMAG, vegaspec=VEGASPEC)

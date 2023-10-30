import hcipy as hp
import numpy as np
from pathlib import Path
from astropy.io import fits
from synphot.models import get_waveset
from synphot.utils import generate_wavelengths
from astropy.convolution import Gaussian2DKernel, convolve_fft

BASE_DIR = Path("/Users/mileslucas/dev/python/scexao_etc/data")
PUPIL_DATA = fits.getdata(BASE_DIR / "scexao_pupil.fits")


def get_psf(spelem, counts, exptime, rn=0, dc=0):
    pup_grid = hp.make_pupil_grid(PUPIL_DATA.shape, diameter=7.82)
    pupil = hp.Field(PUPIL_DATA.ravel(), pup_grid)
    plx = np.deg2rad(6e-3 / 3600)
    Npix = 536//4
    foc_grid = hp.make_uniform_grid((Npix, Npix), (plx * Npix, plx * Npix))
    prop = hp.FraunhoferPropagator(pup_grid, foc_grid)
    det = hp.NoiselessDetector(foc_grid)
    noisy_det = hp.NoisyDetector(foc_grid, dark_current_rate=dc, read_noise=rn)

    wlave = spelem.avgwave().value
    width = spelem.rectwidth().value
    waves = generate_wavelengths(wlave - width / 2, wlave + width / 2, num=3)[0].to("m").value
    psf_ave = 0.0
    noisy_psf_ave = 0.0
    weights = spelem(waves).value
    if np.all(weights == 0):
        weights = [0.33, 0.33, 0.33]
    else:
        weights = weights / weights.sum()
    for wl, f in zip(waves, weights):
        wavefront = hp.Wavefront(pupil, wavelength=wl)
        weight = spelem(wl).value
        wavefront.total_power = counts * f
        focal_plane = prop.forward(wavefront)

        # noiseless PSF
        det.integrate(focal_plane, exptime)
        image = det.read_out()
        psf = image.shaped
        psf_ave += psf

        # noiseless PSF
        noisy_det.integrate(focal_plane, exptime)
        noisy_image = noisy_det.read_out()
        noisy_psf = noisy_image.shaped
        noisy_psf_ave += noisy_psf

        # focal_plane_ave += focal_plane.intensity
        # corr_fact += weight

    # focal_plane_ave /= corr_fact
    # focal_plane_ave.total_power = counts

    return np.ascontiguousarray(psf_ave), np.ascontiguousarray(noisy_psf_ave)

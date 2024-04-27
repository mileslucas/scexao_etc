import hcipy as hp
import numpy as np
from synphot.utils import generate_wavelengths

## constants
PUPIL_DIAMETER = 7.95  # m
OBSTRUCTION_DIAMETER = 2.3397  # m
INNER_RATIO = OBSTRUCTION_DIAMETER / PUPIL_DIAMETER
SPIDER_WIDTH = 0.1735  # m
SPIDER_OFFSET = 0.639  # m, spider intersection offset
SPIDER_ANGLE = 51.75  # deg
ACTUATOR_SPIDER_WIDTH = 0.089  # m
ACTUATOR_SPIDER_OFFSET = (0.521, -1.045)
ACTUATOR_DIAMETER = 0.632  # m
ACTUATOR_OFFSET = ((1.765, 1.431), (-0.498, -2.331))  # (x, y), m
PUPIL_OFFSET = -41  # deg
PIXEL_SCALE = 6  # mas / pix

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def field_combine(field1, field2):
    return lambda grid: field1(grid) * field2(grid)


def generate_pupil_field(
    n: int = 256,
    outer: float = 1,
    inner: float = INNER_RATIO,
    scale: float = 1,
    angle: float = 0,
    oversample: int = 8,
    spiders: bool = True,
    actuators: bool = True,
):
    pupil_diameter = PUPIL_DIAMETER * outer
    # make grid over full diameter so undersized pupils look undersized
    max_diam = PUPIL_DIAMETER if outer <= 1 else pupil_diameter
    grid = hp.make_pupil_grid(n, diameter=max_diam)

    # This sets us up with M1+M2, just need to add spiders and DM masks
    # fix ratio
    inner_val = inner * PUPIL_DIAMETER
    inner_fixed = inner_val / pupil_diameter
    pupil_field = hp.make_obstructed_circular_aperture(pupil_diameter, inner_fixed)

    # add spiders to field generator
    if spiders:
        spider_width = SPIDER_WIDTH * scale
        sint = np.sin(np.deg2rad(SPIDER_ANGLE))
        cost = np.cos(np.deg2rad(SPIDER_ANGLE))

        # spider in quadrant 1
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET, sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 2
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET, sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 3
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET, -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 4
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET, -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )

    # add actuator masks to field generator
    if actuators:
        # circular masks
        actuator_diameter = ACTUATOR_DIAMETER * scale
        actuator_mask_1 = hp.make_obstruction(
            hp.circular_aperture(diameter=actuator_diameter, center=ACTUATOR_OFFSET[0])
        )
        pupil_field = field_combine(pupil_field, actuator_mask_1)

        actuator_mask_2 = hp.make_obstruction(
            hp.circular_aperture(diameter=actuator_diameter, center=ACTUATOR_OFFSET[1])
        )
        pupil_field = field_combine(pupil_field, actuator_mask_2)

        # spider
        sint = np.sin(np.deg2rad(SPIDER_ANGLE))
        cost = np.cos(np.deg2rad(SPIDER_ANGLE))
        actuator_spider_width = ACTUATOR_SPIDER_WIDTH * scale
        actuator_spider = hp.make_spider(
            ACTUATOR_SPIDER_OFFSET,
            (
                ACTUATOR_SPIDER_OFFSET[0] - cost * pupil_diameter,
                ACTUATOR_SPIDER_OFFSET[1] - sint * pupil_diameter,
            ),
            spider_width=actuator_spider_width,
        )
        pupil_field = field_combine(pupil_field, actuator_spider)

    rotated_pupil_field = hp.make_rotated_aperture(pupil_field, np.deg2rad(angle))

    return hp.evaluate_supersampled(rotated_pupil_field, grid, oversample)


def generate_pupil(*args, **kwargs):
    pupil = generate_pupil_field(*args, **kwargs)
    return pupil.shaped


def get_psf(spelem, counts, exptime, rn=0, dc=0):
    pupil = generate_pupil_field()
    pupil_grid = pupil.grid
    plx = np.deg2rad(PIXEL_SCALE * 1e-3 / 3600)
    Npix = 536 // 4
    foc_grid = hp.make_uniform_grid((Npix, Npix), (plx * Npix, plx * Npix))
    prop = hp.FraunhoferPropagator(pupil_grid, foc_grid)
    det = hp.NoiselessDetector(foc_grid)
    noisy_det = hp.NoisyDetector(foc_grid, dark_current_rate=dc, read_noise=rn)

    wlave = spelem.avgwave().value
    width = spelem.rectwidth().value
    waves = generate_wavelengths(wlave - width / 2, wlave + width / 2, num=3)[0].to("m").value
    psf_ave = 0.0
    noisy_psf_ave = 0.0
    weights = spelem(waves).value
    weights = [0.33, 0.33, 0.33] if np.all(weights == 0) else weights / weights.sum()
    for wl, f in zip(waves, weights):
        wavefront = hp.Wavefront(pupil, wavelength=wl)
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

    return np.ascontiguousarray(psf_ave), np.ascontiguousarray(noisy_psf_ave)

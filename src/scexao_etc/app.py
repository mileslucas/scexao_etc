from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

from .instruments import INSTRUMENTS
from .models import PICKLES_MAP, VEGASPEC, load_pickles, color_correction, TYPES
from .filters import FILTERS, get_average_extinction
from .psf import get_psf

import re

from synphot import Observation
from synphot.models import get_waveset
from synphot.units import VEGAMAG
from synphot.utils import generate_wavelengths
import astropy.units as u
import numpy as np
import sep

app = Dash("scexao_etc")

app.layout = html.Div(
    [
        html.H1(children="SCExAO Exposure Time Calculator", style={"textAlign": "center"}),
        html.Hr(),
        html.H2(children="Instrument"),
        dcc.RadioItems(list(INSTRUMENTS.keys()), "VAMPIRES", id="instrument"),
        html.Hr(),
        html.H2(children="Target Information"),
        html.Div(
            children=dcc.RadioItems(
                ["Point source", "Extended object"],
                "Point source",
                id="source-type",
            )
        ),
        html.Div(
            children=[
                "Object class",
                dcc.RadioItems(list(TYPES), "V", id="star-type"),
            ]
        ),
        html.Div(id="spectral-type-dropdown"),
        html.Div(
            children=[
                "Input filter:",
                dcc.Dropdown(
                    list(FILTERS.keys()),
                    "V",
                    id="source-filter",
                ),
            ]
        ),
        html.Div(id="magnitude-setup"),
        html.Div(
            children=[
                "Airmass: ",
                dcc.Input(
                    value=1,
                    type="number",
                    id="airmass",
                ),
            ]
        ),
        html.Div(
            children=[
                "Seeing: ",
                dcc.Input(
                    value=0.02,
                    type="number",
                    id="seeing",
                ),
                ' "',
            ]
        ),
        html.Hr(),
        html.H2(children="Instrument Setup"),
        html.Div(id="instrument-setup"),
        html.Hr(),
        html.H2(children="Exposure time calculations"),
        html.Div(id="results"),
        html.Hr(),
        html.H2(children="Plots"),
        dcc.Graph(id="psf-plot", style={"width": "50vh"}),
        dcc.Graph(id="spectral-plot", style={"width": "50vh"}),
        html.Div(children=[]),
    ]
)


@callback(Output("spectral-type-dropdown", "children"), Input("star-type", "value"))
def get_spectral_options(startype):
    re_match = re.compile(f"[0-9]{startype}")
    sublist = list(filter(lambda f: re_match.search(f), PICKLES_MAP.keys()))
    default = sublist[0]
    return [
        "Spectral type (uses Pickles stellar models):",
        dcc.Dropdown(sublist, default, id="spectral-type"),
    ]


@callback(Output("instrument-setup", "children"), Input("instrument", "value"))
def get_instrument_options(instrument):
    return INSTRUMENTS[instrument].get_options()


@callback(Output("magnitude-setup", "children"), Input("source-type", "value"))
def get_magnitude_options(source_type):
    default = [
        html.Div(
            children=[
                "Brightness: ",
                dcc.Input(placeholder="magnitude", type="number", id="source-mag", value=0),
                " mag",
            ]
        ),
    ]
    if source_type.lower() == "point source":
        return default
    elif source_type.lower() == "extended object":
        default.extend(
            [
                html.Div(
                    children=[
                        "Surface Brightness: ",
                        dcc.Input(
                            placeholder="surface brightness",
                            type="number",
                            id="surf-mag",
                            value=0,
                        ),
                        " mag / sq. arcsec",
                    ]
                ),
                html.Div(
                    children=[
                        "Extent: ",
                        dcc.Input(placeholder="extent", type="number", id="surface-area", value=1),
                        " sq. arcsec",
                    ]
                ),
            ]
        )
        return default


@callback(
    Output("spectral-plot", "figure"),
    Input("spectral-type", "value"),
    Input("source-mag", "value"),
    Input("source-filter", "value"),
    Input("instrument", "value"),
    Input("inst-filter", "value"),
)
def update_spectrum_plot(sptype, mag, srcfilt, inst, instfilt):
    src_elem = FILTERS[srcfilt]
    inst = INSTRUMENTS[inst]
    inst_elem = inst.FILTERS[instfilt]
    sp = load_pickles(sptype)
    sp_norm = sp.normalize(mag * VEGAMAG, src_elem, vegaspec=VEGASPEC, force="extrap")
    waves = get_waveset(sp_norm.model)
    obs_source = Observation(sp_norm, src_elem, waves, force="extrap").as_spectrum()
    obs_inst = Observation(sp_norm, inst_elem, waves, force="extrap").as_spectrum()
    waves_nm = waves / 10

    filt1waves = get_waveset(src_elem.model)
    filt2waves = get_waveset(inst_elem.model)
    min_wave = min(filt1waves.min(), filt2waves.min()) / 10 - 100
    max_wave = max(filt1waves.max(), filt2waves.max()) / 10 + 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=waves_nm, y=sp_norm(waves, u.Jy).value, mode="lines", name=f"{sptype} spectrum"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=waves_nm, y=obs_source(waves, u.Jy).value, mode="lines", name=f"{srcfilt} (target)"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=waves_nm, y=obs_inst(waves, u.Jy).value, mode="lines", name=f"{instfilt} (inst)"
        )
    )

    fig.update_layout(
        legend_orientation="v",
    )
    fig.update_xaxes(title_text="wavelength (nm)", range=[min_wave, max_wave])
    fig.update_yaxes(title_text="flux (Jy)")

    return fig


@callback(
    Output("results", "children"),
    Output("psf-plot", "figure"),
    Input("spectral-type", "value"),
    Input("source-type", "value"),
    Input("source-mag", "value"),
    Input("source-filter", "value"),
    Input("instrument", "value"),
    Input("inst-readout-mode", "value"),
    Input("inst-filter", "value"),
    Input("inst-exptime", "value"),
    Input("inst-bs", "value"),
    Input("airmass", "value"),
    Input("seeing", "value"),
)
def calculate_results(
    sptype, sourcetype, mag, srcfilt, inst, readmode, instfilt, exptime, bs, airmass, seeing
):
    src_elem = FILTERS[srcfilt]
    inst = INSTRUMENTS[inst]
    inst_elem = inst.FILTERS[instfilt]
    sp = load_pickles(sptype)
    ## step 1 get color correction
    color_corr = color_correction(sp, src_elem, inst_elem).value
    ## extinction
    ext_coeff = get_average_extinction(inst_elem)
    total_ext = ext_coeff * airmass

    ## don't use color correction because we're still using source filter
    # as reference here
    input_mag = mag + total_ext.value
    sp_norm = sp.normalize(input_mag * VEGAMAG, src_elem, vegaspec=VEGASPEC, force="extrap")
    waves = get_waveset(sp_norm.model)
    obs_inst = Observation(sp_norm, inst_elem, waves, force="extrap")
    obs_mag = obs_inst.effstim(VEGAMAG, vegaspec=VEGASPEC).value

    ## apply zeropoint
    zp = 2.5 * np.log10(inst.ZEROPOINTS[instfilt])
    inst_mag = obs_mag - zp
    inst_counts = 10 ** (-0.4 * inst_mag)
    if bs.upper() in ("PBS", "NPBS"):
        inst_counts *= 0.5

    ## synphot counts
    area = 40.3 * u.m**2
    synphot_counts = obs_inst.countrate(area=area).value

    rn = inst.rn[readmode]
    total_noise = np.sqrt(rn**2 + inst.dt * exptime)
    total_signal = inst_counts * exptime

    psf, noisy_psf = get_psf(inst_elem, inst_counts, exptime, rn=rn, dc=inst.dt, seeing=seeing)
    # psf *= strehl
    # noisy_psf *= strehl

    peak_signal = noisy_psf.max()
    strehl = peak_signal / psf.max()
    # aper_rad = 0.5
    # aper_rad_px = 0.5 * 1e3 / inst.pxscale
    aper_rad_px, _ = sep.flux_radius(noisy_psf, [267.5], [267.5], [267.5], frac=0.95)
    best_rad = 0
    best_snr = -1
    for r in np.arange(0.05, 1.5, 0.05):
        r_px = r * 1e3 / inst.pxscale
        aper_sum, aper_sum_err, _ = sep.sum_circle(
            noisy_psf, [267.5], [267.5], r_px, err=total_noise, gain=1
        )
        snr = aper_sum / aper_sum_err
        if snr > best_snr:
            best_snr = snr
            best_rad = r

    aper_rad = best_rad
    aper_rad_px = best_rad * 1e3 / inst.pxscale
    aper_sum, aper_sum_err, _ = sep.sum_circle(
        noisy_psf, [267.5], [267.5], aper_rad_px, err=total_noise, gain=1
    )
    snr = aper_sum[0] / aper_sum_err[0] / 0.95

    ## go as fast as possible while having at least ~2000 adu
    ratio = peak_signal / exptime
    suggested_exptime = 400 / ratio
    max_time = 6100 / ratio
    if readmode.upper() == "FAST":
        suggested_exptime = np.clip(suggested_exptime, 1 / 254, max_time)
    elif readmode.upper() == "SLOW":
        suggested_exptime = np.clip(suggested_exptime, 1 / 10, max_time)

    keys = [
        "Source mag",
        "Color correction",
        "Extinction coeff",
        "Total extinction",
        "Observed mag",
        "Zero point",
        "Aperture area",
        "Counts at telescope",
        "System transmission",
        "Counts at detector",
        "RMS Read noise",
        "Dark current",
        "Total integrated signal",
        "Total noise",
        "Peak signal",
        "Strehl",
        "Aperture radius",
        "Aperture sum",
        "Aperture noise",
        "Aperture S/N",
        "Suggested exptime",
    ]
    vals = [
        f"{mag:4.02f}",
        f"{color_corr:4.02f}",
        f"{ext_coeff:4.02f}",
        f"{total_ext:4.02f}",
        f"{obs_mag:4.02f}",
        f"{zp:4.02f}",
        f"{area.to(u.m**2).value:4.02f}",
        f"{synphot_counts:4.02e}",
        f"{inst_counts / synphot_counts:4.03f}",
        f"{inst_counts:4.02e}",
        f"{rn:4.02f}",
        f"{inst.dt:3.01e}",
        f"{total_signal:3.01e}",
        f"{total_noise:3.01e}",
        f"{peak_signal:3.01e}",
        f"{np.clip(strehl, 0, 1):3.02f}",
        f'{aper_rad:5.01f}" ({aper_rad_px:3.0f} px)',
        f"{aper_sum[0] / 0.95:3.01e}",
        f"{aper_sum_err[0]:3.01e}",
        f"{snr:6.0f}",
        f"{suggested_exptime:3.01e}",
    ]
    units = [
        "mag",
        f"mag ({instfilt} - {srcfilt})",
        "mag / airmass",
        "mag",
        "mag",
        "mag",
        "m^2",
        "photon / s",
        " (estimated)",
        "e- / s",
        "e- / px",
        "e- / s / px",
        "e-",
        "e- / px",
        f"e-{'' if peak_signal < 6870 else ' SATURATING'}",
        "",
        "",
        "e-",
        "e-",
        "",
        "s",
    ]
    pretty_keys = [f"{k} {'.'*(50 - len(k))}" for k in keys]
    lines = [f"{k} {v} {u}" for k, v, u in zip(pretty_keys, vals, units)]
    results = [html.Div(children=l) for l in lines]

    noisy_psf_adu = (noisy_psf / inst.gain[readmode] + 200).astype("uint16")

    fig = px.imshow(np.log10(noisy_psf_adu), origin="lower", color_continuous_scale="inferno")

    return results, fig


def main():
    app.run(debug=True, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

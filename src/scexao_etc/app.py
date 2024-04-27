import re
import sys
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np

# import streamlit_pydantic as sp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sep
import streamlit as st
import streamlit.web.cli as stcli
from pydantic import BaseModel
from synphot import Observation
from synphot.models import get_waveset
from synphot.units import VEGAMAG

from scexao_etc.filters import FILTERS, get_average_extinction
from scexao_etc.instruments import INSTRUMENTS
from scexao_etc.models import PICKLES_MAP, TYPES, VEGASPEC, load_pickles
from scexao_etc.psf import get_psf


class TargetOptions(BaseModel):
    sptype: str
    filter: str
    mag: float
    mag_sys: Literal["Vega", "AB"]
    airmass: float = 1


def update_spectrum_plot(sptype, mag, magsys, srcfilt, inst, inst_filts):
    src_elem = FILTERS[srcfilt]
    inst = INSTRUMENTS[inst]
    sp = load_pickles(sptype)
    if magsys == "Vega":
        unit = VEGAMAG
    elif magsys == "AB":
        unit = u.ABmag
    sp_norm = sp.normalize(mag * unit, src_elem, vegaspec=VEGASPEC, force="extrap")
    waves = get_waveset(sp_norm.model)
    obs_source = Observation(sp_norm, src_elem, waves, force="extrap").as_spectrum()
    waves_nm = waves / 10

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

    filt1waves = get_waveset(src_elem.model)
    min_wave = np.inf
    max_wave = -np.inf
    for inst_filt in inst_filts:
        inst_elem = inst.FILTERS[inst_filt]
        filt2waves = get_waveset(inst_elem.model)
        min_wave = min(min(filt1waves.min(), filt2waves.min()) / 10 - 100, min_wave)
        max_wave = max(max(filt1waves.max(), filt2waves.max()) / 10 + 100, max_wave)
        obs_inst = Observation(sp_norm, inst_elem, waves, force="extrap").as_spectrum()

        fig.add_trace(
            go.Scatter(
                x=waves_nm, y=obs_inst(waves, u.Jy).value, mode="lines", name=f"{inst_filt} (inst)"
            )
        )

    fig.update_layout(
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="center",
        legend_x=0.5,
        # autosize=True,
        # width=500
    )
    fig.update_xaxes(title_text="wavelength (nm)", range=[min_wave, max_wave])
    fig.update_yaxes(title_text="flux (Jy)")

    return fig


# def get_counts(input_spec, src_elem, inst, inst_filt):
#     inst_elem = inst.FILTERS[inst_filt]
#     ## step 1 get color correction
#     color_corr = color_correction(input_spec, src_elem, inst_elem).value
#     ## extinction
#     ext_coeff = get_average_extinction(inst_elem)
#     total_ext = ext_coeff * airmass

#     ## don't use color correction because we're still using source filter
#     # as reference here
#     input_mag = mag + total_ext.value
#     sp_norm = input_spec.normalize(input_mag * unit, src_elem, vegaspec=VEGASPEC, force="extrap")
#     waves = get_waveset(sp_norm.model)
#     obs_inst = Observation(sp_norm, inst_elem, waves, force="extrap")
#     obs_mag = obs_inst.effstim(unit, vegaspec=VEGASPEC).value

#     ## apply zeropoint
#     zp_mag = 2.5 * np.log10(inst.ZEROPOINTS[inst_filt])
#     inst_mag = obs_mag - zp_mag - 2.5 * np.log10(strehl)
#     inst_counts = 10 ** (-0.4 * inst_mag)
#     if use_bs:
#         inst_counts *= 0.5

#     ## synphot counts
#     area = 40.3 * u.m**2
#     synphot_counts = obs_inst.countrate(area=area).value


def calculate_results(
    sptype, sourcetype, mag, magsys, srcfilt, inst, exptime, airmass, strehl, area=None
):
    src_elem = FILTERS[srcfilt]
    if sourcetype == "Extended Object":
        px_size = 2.5 * np.log10(inst.pxarea)
        px_size = 2.5 * np.log10(area)
        mag = mag - px_size
    sp = load_pickles(sptype)
    tbls = []
    figs = []
    if magsys == "Vega":
        unit = VEGAMAG
    elif magsys == "AB":
        unit = u.ABmag
    for inst_filt, inst_elem in inst.filters.items():
        ## step 1 get color correction
        # color_corr = color_correction(sp, inst_elem, src_elem).value
        ## extinction
        ext_coeff = get_average_extinction(inst_elem)
        total_ext = ext_coeff * airmass

        ## don't applly color correction here because we're still using source filter
        # as reference here
        input_mag = mag + total_ext
        sp_norm = sp.normalize(input_mag * unit, src_elem, vegaspec=VEGASPEC, force="extrap")
        waves = get_waveset(sp_norm.model)
        obs_inst = Observation(sp_norm, inst_elem, waves, force="extrap")
        obs_mag = obs_inst.effstim(unit, vegaspec=VEGASPEC).value
        color_corr = obs_mag - input_mag

        ## apply zeropoint
        zp = 2.5 * np.log10(inst.ZEROPOINTS[inst_filt])
        inst_mag = obs_mag - zp
        inst_counts = 10 ** (-0.4 * inst_mag) * inst.throughput

        ## synphot counts
        area = 40.3 * u.m**2
        synphot_counts = obs_inst.countrate(area=area).value

        rn = inst.readnoise
        dc = inst.dark_current
        total_noise = np.sqrt(rn**2 + dc * exptime)
        total_signal = inst_counts * exptime
        psf, noisy_psf = get_psf(inst_elem, inst_counts, exptime, rn=rn, dc=dc)
        # psf *= strehl
        # noisy_psf *= strehl

        peak_signal = np.nanmax(noisy_psf) * strehl
        # strehl = peak_signal / np.nanmax(psf)

        # lam = inst_elem.avgwave().to(u.nm).value
        # r0 = fried_param(lam, seeing=0.6, airmass=airmass)
        # L0 = 46
        # fwhm = Fiq(lam=lam, seeing=seeing, airmass=airmass, L0=L0)

        # strehl
        # aper_rad = 0.5
        aper_rad_px = 4
        ctr = np.array(noisy_psf.shape[-2:]) / 2 - 0.5
        # aper_rad_px, _ = sep.flux_radius(noisy_psf, [ctr[0]], [ctr[1]], [np.min(ctr)], frac=0.95)
        # best_rad = 0
        # best_snr = -1
        # for r in np.arange(0.05, 1.5, 0.05):
        #     r_px = r / inst.pxscale
        #     aper_sum, aper_sum_err, _ = sep.sum_circle(
        #         noisy_psf, [ctr[0]], [ctr[1]], r_px, err=total_noise, gain=1
        #     )
        #     snr = aper_sum / aper_sum_err
        #     if snr > best_snr:
        #         best_snr = snr
        #         best_rad = r

        # aper_rad = best_rad
        # aper_rad_px = best_rad / inst.pxscale
        aper_sum, aper_sum_err, _ = sep.sum_circle(
            noisy_psf, [ctr[0]], [ctr[1]], aper_rad_px, err=total_noise, gain=1
        )
        snr = aper_sum[0] / aper_sum_err[0]

        ## go as fast as possible while having at least ~2000 adu
        ratio = peak_signal / exptime
        suggested_exptime = 400 / ratio
        max_time = 6100 / ratio
        suggested_exptime = np.clip(suggested_exptime, 1 / 254, max_time)

        saturating = peak_signal > 6870
        if saturating:
            st.toast(f":red[Peak counts are saturating for {inst_filt} filter]")

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
            # "Seeing",
            # "r0",
            # "L0",
            # "FWHM",
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
            f"{dc:3.01e}",
            f"{total_signal:3.01e}",
            f"{total_noise:3.01e}",
            f"{peak_signal:3.01e}",
            f"{np.clip(strehl, 0, 1):3.02f}",
            # f"{seeing:3.02f}",
            # f"{r0 * 1e2:3.01f}",
            # f"{L0:3.01f}",
            # f"{fwhm * 1e3:3.02f}",
            f"{aper_rad_px:3.0f}",
            f"{aper_sum[0] / 0.95:3.01e}",
            f"{aper_sum_err[0]:3.01e}",
            f"{snr:6.0f}",
            f"{suggested_exptime:3.01e}",
        ]
        units = [
            "mag",
            f"mag ({inst_filt} - {srcfilt})",
            "mag / airmass",
            "mag",
            "mag",
            "mag",
            "m^2",
            "photon / s",
            "e- / photon",
            "e- / s",
            "e- / px",
            "e- / s / px",
            "e-",
            "e- / px",
            f"e-{' SATURATING!' if saturating else ''}",
            "",
            # '"',
            # "cm",
            # "m",
            # "mas",
            "px",
            "e-",
            "e-",
            "",
            "s",
        ]
        tbl = pd.DataFrame(dict(Name=keys, Value=vals, Unit=units))

        noisy_psf_adu = inst.convert_data(noisy_psf)

        fig = px.imshow(
            np.log10(noisy_psf_adu),
            origin="lower",
            aspect="equal",
            color_continuous_scale="cividis",
        )
        fig.update_layout(coloraxis_colorbar=dict(title=dict(text="log10(flux)", side="right")))
        tbls.append(tbl)
        figs.append(fig)

    return tbls, figs


def fried_param(lam, seeing, airmass):
    return 0.1 / seeing * (lam / 500) ** 1.2 * airmass ** (-0.6)


def Fkolb(diam=7.92, L0=46):
    return 1 / (1 + 300 * diam / L0) - 1


def Fatm(lam, seeing, airmass, L0=46):
    fkolb = Fkolb(L0=L0)
    r0 = fried_param(lam, seeing, airmass)
    radicand = 1 + fkolb * 2.183 * (r0 / L0) ** 0.356
    if radicand <= 0:
        return 0
    else:
        return seeing * airmass**0.6 * (lam / 500) ** (-0.2) * np.sqrt(radicand)


def Fiq(lam, seeing, airmass, L0=46):
    fatm = Fatm(lam, seeing, airmass, L0=L0)
    ftel = 1.028 * np.rad2deg(lam / 7.95e9) * 3600
    return np.sqrt(fatm**2 + ftel**2)


def app():
    ## Preamble

    st.set_page_config(page_title="SCExAO ETC", page_icon=":star:", layout="wide")

    ## Body

    area = None

    st.title("SCExAO Exposure Time Calculator")

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Target Parameters")

        src_type = st.radio("Object type", ("Point Source", "Extended Object"), horizontal=True)
        ll_col, lr_col = st.columns([0.3, 0.7])
        obj_type = lr_col.radio("Class", TYPES, horizontal=True)
        re_match = re.compile(f"[0-9]{obj_type}$")
        sublist = list(filter(lambda f: re_match.search(f), PICKLES_MAP.keys()))
        sp_type = ll_col.selectbox("Spectral Type", sublist)
        match src_type:
            case "Point Source":
                cols = st.columns((0.2, 0.2, 0.6))
                mag = cols[1].number_input("Mag", value=6)
            case "Extended Object":
                cols = st.columns((0.2, 0.2, 0.2, 0.4))
                mag = cols[1].number_input("Mag / sq. arcsec", value=10)
                area = cols[2].number_input("Area", value=1)
        src_filt = cols[0].selectbox("Filter", FILTERS.keys(), list(FILTERS.keys()).index("V"))
        mag_sys = cols[-1].radio("System", ("Vega", "AB"), horizontal=True)
        airmass = st.slider("Airmass", 1.0, 3.0, step=0.1)
        # seeing = st.slider("Seeing", 0.0, 3.0, 0.6, step=0.1)
        strehl = st.slider("Strehl", 0.0, 1.0, 0.4, step=0.1)

    with right_col:
        st.subheader("Instrument Parameters")
        inst = st.radio("Instrument", INSTRUMENTS.keys())
        inst_obj = INSTRUMENTS[inst].input_options()
        exptime = st.number_input("Exposure Time", value=0.1, min_value=7.2e-6, max_value=1800.0)
    st.header("Results")

    inst_filt = inst_obj.filters.keys()
    tbls, psf_figs = calculate_results(
        sptype=sp_type,
        sourcetype=src_type,
        mag=mag,
        magsys=mag_sys,
        srcfilt=src_filt,
        inst=inst_obj,
        exptime=exptime,
        airmass=airmass,
        # seeing=seeing,
        strehl=strehl,
        area=area,
    )

    left_col, right_col = st.columns([0.5, 0.5])
    with left_col:
        tabs = st.tabs(inst_filt)
        for tab, tbl in zip(tabs, tbls):
            tab.dataframe(tbl, use_container_width=True, hide_index=True, height=773)
    with right_col:
        # st.write(fig)
        # with psf_tab:
        tabs = st.tabs(inst_filt)
        for tab, fig in zip(tabs, psf_figs):
            tab.plotly_chart(fig, use_container_width=True)

        fig = update_spectrum_plot(sp_type, mag, mag_sys, src_filt, inst, inst_filt)
        st.plotly_chart(fig, use_container_width=False)


def main():
    if st.runtime.exists():
        app()
    else:
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())


if __name__ == "__main__":
    main()

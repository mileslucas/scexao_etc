from pydantic import BaseModel, Field
from typing import Literal, Annotated, ClassVar, Final, Optional
from annotated_types import Interval, Gt, Ge, Lt, Le
from dash import Dash, html, dcc, callback, Output, Input
from pathlib import Path
import pandas as pd
import streamlit as st

from .filters import VAMP_FILTERS, VAMP_ZEROPOINTS
import numpy as np

BASE_DIR = Path("/Users/mileslucas/dev/websites/scexao_etc/data")

class Instrument(BaseModel):
    readnoise: Annotated[float, Ge(0)]
    gain: Annotated[float, Gt(0)]
    dark_current: Annotated[float, Ge(0)]
    filters: dict
    pxscale: float
    bias: float = 0
    factor: float = 1


    @property
    def pxarea(self):
        """pixel area in sq. arcsec"""
        return (self.pxscale)**2
    
    def convert_data(self, data, clip=True):
        data_adu = (data / self.gain + self.bias)
        return data_adu
    
class VAMPIRES(Instrument):
    nd_filt: Optional[Literal["ND10", "ND25"]]
    dark_current: float = 3.6e-3 # e- / px / s 
    pxscale: float = 6e-3 # arcsec / px
    bias: float = 200 # adu

    READNOISE: ClassVar[dict[str, float]] = {"FAST": 0.4, "SLOW": 0.2}
    GAIN: ClassVar[dict[str, float]] = {"FAST": 0.103, "SLOW": 0.105}
    FILTERS: ClassVar[list] = VAMP_FILTERS
    ZEROPOINTS: ClassVar[list] = VAMP_ZEROPOINTS

    @classmethod
    def input_options(cls):
        mbi_filts = set(filter(lambda f: f.startswith("F"), cls.FILTERS.keys()))
        ha_filts = set(filter(lambda f: f.startswith("H"), cls.FILTERS.keys()))
        sii_filts = set(filter(lambda f: f.startswith("S"), cls.FILTERS.keys()))
        std_filts = set(cls.FILTERS.keys()) - mbi_filts - ha_filts - sii_filts
        match st.radio("Mode", ("Standard", "Multiband", "Narrowband"), horizontal=True):
            case "Standard":
                inst_filt = [st.selectbox("Filter", sorted(std_filts))]
            case "Multiband":
                inst_filt = mbi_filts
            case "Narrowband":
                match st.selectbox("Filter Set", ("H-alpha", "SII")):
                    case "H-alpha":
                        inst_filt = ha_filts
                    case "SII":
                        inst_filt = sii_filts
        readout_mode = st.radio("Readout Mode", ("FAST", "SLOW"), horizontal=True)
        use_bs = st.toggle("Use beamsplitter", True)
        match st.selectbox("ND Filter", ("None", "OD 1.0", "OD 2.5")):
            case "None":
                nd_filt = None
            case "OD 1.0":
                nd_filt = "ND10"
            case "OD 2.5":
                nd_filt = "ND25"
        filts = {k: cls.FILTERS[k] for k in sorted(inst_filt)}
        return cls(readnoise=cls.READNOISE[readout_mode], gain=cls.GAIN[readout_mode], filters=filts, nd_filt=nd_filt, factor=0.5 if use_bs else 1)

    def convert_data(self, data, clip=True):
        data_adu = super().convert_data(data)
        if clip:
            return np.where(data_adu > 2**16 - 1, np.nan, data_adu).astype("f4")
    
        return data_adu


INSTRUMENTS = {"VAMPIRES": VAMPIRES}

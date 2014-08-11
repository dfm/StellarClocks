#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import kplr
import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import lombscargle

from data import LightCurve

client = kplr.API()

koi = client.koi("326.01")
star = koi.star
# kicid = 5515314
# star = client.star(kicid)

lcs = star.get_light_curves()

datasets = []
for lc in lcs:
    data = lc.read()
    t, f, fe, q = [data[k] for k in ["TIME", "SAP_FLUX", "SAP_FLUX_ERR",
                                     "SAP_QUALITY"]]
    datasets += LightCurve(t.astype("float64"), f.astype("float64"),
                           fe.astype("float64"), quality=(q == 0)) \
        .autosplit(0.1)

data = np.concatenate([(d.time, d.flux - 1.0) for d in datasets], axis=1)

pl.clf()
[pl.plot(d.time, d.flux, ".") for d in datasets]
pl.savefig("data.png")

# BOOM.
freq_muH = np.exp(np.linspace(np.log(25), np.log(1000), 50000))
omega = freq_muH * 1e-6 * 86400. * (2 * np.pi)
amp = lombscargle(data[0], data[1], omega)

pl.clf()
pl.plot(freq_muH, amp, "k")
pl.savefig("dude.png")

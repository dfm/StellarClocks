#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import lombscargle

import george
from george import kernels


def muhz_to_days(freq):
    return 1.0 / (freq * 1e-6 * 86400.)


def envelope(mu):
    return a0 * np.exp(-0.5 * (mu-mu_mx) ** 2 / 100.)


mu_mx = 135.
dmu = 10.0
a0 = 1.0  # 1e-5

kernel = None
# for n in range(-3, 3+1):
for n in range(-1, 1+1):
    mu = mu_mx + n * dmu
    print(mu, envelope(mu))
    k = envelope(mu) * kernels.ExpSine2Kernel(1, muhz_to_days(mu))
    kernel = k if kernel is None else kernel + k

# kernel *= 1e-8 * kernels.ExpSquaredKernel(1000.0)

# kernel += 1e-5 * kernels.ExpSquaredKernel(muhz_to_days(50.))
# kernel += kernels.WhiteKernel(1e-3)

gp = george.GP(kernel)

x = np.arange(0, 60, 0.5 / 24.)
y = gp.sample(x)

pl.plot(x, y, ".k")
pl.savefig("test_data.png")

freq_muH = np.exp(np.linspace(np.log(120), np.log(150), 10000))
omega = freq_muH * 1e-6 * 86400. * (2 * np.pi)
amp = lombscargle(x, y, omega)

pl.clf()
pl.plot(freq_muH, amp, "k")
pl.savefig("test_periodogram.png")

"""
This file is part of the StellarClocks project.
Copyright 2014 David W. Hogg (NYU).
"""

import numpy as np
import matplotlib.pylab as plt
import emcee

def trapezoid(times, period, offset, depth, duration, gress):
    fractions = np.zeros_like(times)
    ts = np.mod((times - offset), period)
    ts[ts > 0.5 * period] -= period
    ts = np.abs(ts)
    inside = ts < 0.5 * (duration + gress)
    fractions[inside] = (depth / gress) * (0.5 * (duration + gress) - ts[inside])
    fractions[ts < 0.5 * (duration - gress)] = depth
    return fractions

def get_fractions(times, period, offset, depth, duration, gress):
    return trapezoid(times, period, offset, depth, duration, gress)

def integrate_fractions(times, exptime, period, offset, depth, duration, gress, K):
    delta_times = np.arange(-0.5 * exptime + 0.5 * exptime / K, 0.5 * exptime, exptime / K)
    bigtimes = times[:, None] + delta_times[None, :]
    bigfracs = get_fractions(bigtimes, period, offset, depth, duration, gress)
    return np.mean(bigfracs, axis=1)

def observe_star(times, exptime, sigma, period, offset, depth, duration, gress, K=21):
    fluxes = np.ones_like(times)
    fluxes *= (1. - integrate_fractions(times, exptime, period, offset, depth, duration, gress, K))
    fluxes += sigma * np.random.normal(size=fluxes.shape)
    return fluxes

def ln_like(data, pars):
    times, fluxes, ivars = data
    period, offset, depth, duration, gress = pars
    fracs = integrate_fractions(times, exptime, period, offset, depth, duration, gress, 5)
    return -0.5 * np.sum(ivars * (fluxes - (1. - fracs)) ** 2)

def ln_prior(pars):
    return 0.

def lnprob(pars, data):
    lp = ln_prior(pars)
    if not np.isfinite(lp):
        return -np.Inf
    return lp + ln_like(data, pars)

if __name__ == "__main__":
    times = np.arange(0., 90., 1.0 / 48.) # 30-min cadence in d
    exptime = ((1.0 / 24.) / 60.) * 27. # 27 min in d
    sigma = 1.e-5
    truepars = np.array([6.5534, 31.55, 0.005235, 0.32322, 0.05232])
    fluxes = observe_star(times, exptime, sigma, *truepars)
    plt.plot(times, fluxes, ".")
    plt.savefig("hotcold_data.png")
    ivars = np.zeros_like(fluxes) + 1. / (sigma ** 2)
    data = np.array([times, fluxes, ivars])
    initpars = truepars
    ndim, nwalkers = len(initpars), 128
    pos = [initpars + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,))
    sampler.run_mcmc(pos, 512)
    

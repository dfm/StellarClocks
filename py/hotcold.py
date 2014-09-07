"""
This file is part of the StellarClocks project.
Copyright 2014 David W. Hogg (NYU).

# to-do items
- fork the plotting
- figure out real-data pre-processing or update likelihood function noise model
- run on real data
"""

import numpy as np
import matplotlib.pylab as plt
import emcee
import triangle

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

def distort_times(times, ln_period, Aamp, Bamp):
    thetas = 2 * np.pi * times / np.exp(ln_period)
    return times - Aamp * np.cos(thetas) - Bamp * np.sin(thetas)

def integrate_fractions(times, exptime, hotperiod, offset, depth, duration, gress, ln_coldperiod, Aamp, Bamp, K):
    delta_times = np.arange(-0.5 * exptime + 0.5 * exptime / K, 0.5 * exptime, exptime / K)
    bigtimes = times[:, None] + delta_times[None, :]
    bigtimes = distort_times(bigtimes, ln_coldperiod, Aamp, Bamp)
    bigfracs = get_fractions(bigtimes, hotperiod, offset, depth, duration, gress)
    return np.mean(bigfracs, axis=1)

def observe_star(times, exptime, sigma, hotperiod, offset, depth, duration, gress, ln_coldperiod, Aamp, Bamp, K=21): # MAGIC
    fluxes = np.ones_like(times)
    fluxes *= (1. - integrate_fractions(times, exptime, hotperiod, offset, depth, duration, gress, ln_coldperiod, Aamp, Bamp, K))
    fluxes += sigma * np.random.normal(size=fluxes.shape)
    return fluxes

def ln_like(data, pars):
    times, fluxes, ivars = data
    hotperiod, offset, depth, duration, gress, ln_coldperiod, Aamp, Bamp = pars
    fracs = integrate_fractions(times, exptime, hotperiod, offset, depth, duration, gress, ln_coldperiod, Aamp, Bamp, 5) # MAGIC
    return -0.5 * np.sum(ivars * (fluxes - (1. - fracs)) ** 2)

def ln_prior(pars):
    hotperiod, offset, depth, duration, gress, ln_coldperiod, Aamp, Bamp = pars
    if ln_coldperiod < 5. or ln_coldperiod > 10.:
        return -np.Inf
    return 0.

def ln_posterior(pars, data):
    lp = ln_prior(pars)
    if not np.isfinite(lp):
        return -np.Inf
    return lp + ln_like(data, pars)

if __name__ == "__main__":
    np.random.seed(42)
    times = np.arange(0., 4.1 * 365, 1.0 / 48.) # 30-min cadence in d
    times -= np.median(times) # put zero in the center of the plot
    exptime = ((1.0 / 24.) / 60.) * 27. # 27 min in d
    sigma = 1.e-5
    hotperiod = 6.5534 # MAGIC
    ln_coldperiod = np.log(365.25 * 11.8618) # MAGIC (Jupiter's period in days)
    # http://www.google.com/search?q=((1+Jupiter+mass)+%2F+(1+Solar+mass))+*+((5.204267+AU)+%2F+c)
    coldamp = 2.478 / 86400. # MAGIC (Jupiter's amplitude in days)
    coldphase = 0.1 # radian MAGIC
    Aamp = coldamp * np.cos(coldphase)
    Bamp = coldamp * np.sin(coldphase)
    truepars = np.array([hotperiod, 2.55, 0.005235, 0.32322, 0.05232, ln_coldperiod, Aamp, Bamp]) # MAGIC
    true_offsets = times - distort_times(times, *(truepars[5:]))
    fluxes = observe_star(times, exptime, sigma, *truepars)
    fig1 = plt.figure(1)
    plt.clf()
    plt.plot(times, fluxes, ".")
    plt.xlabel("time (d)")
    plt.ylabel("flux")
    fig1.savefig("hotcold_data.png")
    plt.close("all")
    ivars = np.zeros_like(fluxes) + 1. / (sigma ** 2)
    data = np.array([times, fluxes, ivars])
    initpars = 1. * truepars
    initpars[6:] = [0., 0.] # zero out amplitudes
    ndim, nwalkers, nlinks = len(initpars), 16, 512
    pos = initpars[None, :] + 1e-6 * np.random.normal(size=(nwalkers, ndim))
    pos[:, 5] += -0.5 + np.random.uniform(size=nwalkers) # more scatter for ln_coldperiod
    nburn = 100
    for burn in range(nburn):
        print "burning %d, ndim %d, nwalkers %d, nlinks %d" % (burn, ndim, nwalkers, nlinks)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, args=(data,), threads=17)
        sampler.run_mcmc(pos, nlinks)
        chain = sampler.flatchain
        low = 3 * len(chain) / 4
        if nwalkers < 128:
            nwalkers *= 2
        if nlinks > 64:
            nlinks /= 2
        pos = chain[np.random.randint(low, high=len(chain), size=nwalkers)]

        # plot samples
        fig2 = plt.figure(2)
        plt.clf()
        plt.plot(times, 86400. * true_offsets, "b-")
        for ii in np.random.randint(len(sampler.flatchain), size=16):
            # HACK to plot offsets in a clean way, removing hotperiod and offset dependencies
            # NOTE dependence on `truepars`
            offsets = (sampler.flatchain[ii, 1] - truepars[1]) \
                + times - distort_times(times, *(sampler.flatchain[ii, 5:])) \
                + (times - sampler.flatchain[ii, 1]) * (1. - truepars[0] / sampler.flatchain[ii, 0])
            plt.plot(times, 86400. * offsets, "k-", alpha=0.25)
        plt.xlabel("time (d)")
        plt.ylabel("offsets (s)")
        fig2.savefig("hotcold_time_delays.png")

        # triangle-plot samples
        resids = (sampler.flatchain - truepars[None, :])
        resids[:, [0, 1, 3, 4, 6, 7]] *= 86400.
        resids[:, 2] *= 1.e6
        fig = triangle.corner(resids,
                              labels=["hot period resid (s)", "offset resid (s)", 
                                      "depth resid (ppm)", "duration resid (s)", 
                                      "gress resid (s)", "ln cold period resid",
                                      "A amplitude resid (s)", "B amplitude resid (s)"],
                              truths=(truepars - truepars))
        fig.savefig("hotcold_triangle.png")
        plt.close("all")

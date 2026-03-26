"""
Project D: Varying the Equation of State.

Extends the unit 5 analysis by freeing the dark energy equation of state w0.
The integrand becomes:
  [Omega_m*(1+z)^3 + Omega_k*(1+z)^2 + Omega_Lambda*(1+z)^(3*(1+w0))]^(-1/2)
where w0=-1 recovers the cosmological constant.

Runs a Metropolis MCMC over [Omega_m, Omega_Lambda, H0, w0] and produces
trace plots, 1D histograms, and 2D joint posterior plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from unit4_copy import Likelihood
from Unit5 import Metropolis


# =====================================================================
# SETUP
# =====================================================================

# Initialise likelihood with Pantheon data
likelihood = Likelihood('pantheon_data.txt')

# Wrap so that Metropolis (which calls likelihood(theta, n=...)) uses model_type='w0'
def likelihood_w0(theta, n=200):
    return likelihood(theta, n=n, model_type='w0')


# Parameter names for all plots
param_names = [r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$ (km/s/Mpc)', r'$w_0$']

# Starting point: cosmological constant model (w0=-1)
theta_init = [0.3, 0.7, 70.0, -1.0]

# Proposal step sizes: sigma_w0=0.2 as specified in the project brief
sigma = [0.07, 0.1, 0.3, 0.2]


# =====================================================================
# RUN METROPOLIS
# =====================================================================

sampler = Metropolis(likelihood_w0, theta_init=theta_init, sigma=sigma, n_integration=200)
sampler.run(n_steps=50500, burn_in=500)


# =====================================================================
# TRACE AND 1D HISTOGRAMS
# These methods auto-detect the number of parameters from chain shape
# =====================================================================

sampler.plot_trace(param_names=param_names)
sampler.plot_1d_histograms(param_names=param_names)


# =====================================================================
# 2D JOINT POSTERIORS — all 6 unique pairs for 4 parameters
# =====================================================================

def plot_2d_histograms_4param(sampler, burn_in=None, param_names=None, bins=40):
    """
    Plot all 6 unique 2D joint posterior distributions for a 4-parameter chain.
    """
    chain = sampler.get_chain(burn_in)
    if param_names is None:
        param_names = [f'param {i}' for i in range(chain.shape[1])]

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, (xi, yi) in zip(axes, pairs):
        h, xedges, yedges = np.histogram2d(chain[:, xi], chain[:, yi], bins=bins, density=True)
        im = ax.imshow(h.T, origin='lower', aspect='auto',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        ax.set_xlabel(param_names[xi])
        ax.set_ylabel(param_names[yi])
        ax.set_title(f'{param_names[xi]} vs {param_names[yi]}')
        fig.colorbar(im, ax=ax, label='Probability density')

    fig.suptitle('Project D: 2D Joint Posteriors (w0 model)', fontsize=13)
    plt.tight_layout()
    plt.show()


plot_2d_histograms_4param(sampler, param_names=param_names)


# =====================================================================
# PRINT SUMMARY STATISTICS
# =====================================================================

chain = sampler.get_chain()
print("\n" + "="*60)
print("PARAMETER ESTIMATES (median +/- 1-sigma)")
print("="*60)
for i, name in enumerate(param_names):
    lo, med, hi = np.percentile(chain[:, i], [16, 50, 84])
    print(f"  {name:35s}  {med:.4f}  +{hi-med:.4f} / -{med-lo:.4f}")
print(f"\nAcceptance rate: {sampler.acceptance_rate():.2%}")
print(f"Chain length (post burn-in): {len(chain)}")

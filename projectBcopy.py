"""
Unit 6 - Project B: Redshift Range Analysis

Repeats the Unit 5 Metropolis MCMC analysis on three different redshift ranges:
- Short range: 0 < z < z_mid1
- Medium range: 0 < z < z_mid2
- Long range: 0 < z < z_max

This tests how cosmological parameter constraints evolve with redshift coverage.
"""

import numpy as np
import matplotlib.pyplot as plt
from unit4_copy import Likelihood
import time
import os


class FilteredLikelihood:
    """
    Wrapper around the Likelihood class that filters data by redshift range.
    """
    def __init__(self, likelihood, z_min, z_max):
        """
        Initialize with redshift filtering bounds.
        
        Parameters
        ----------
        likelihood : Likelihood
            The base Likelihood object
        z_min : float
            Minimum redshift (typically 0)
        z_max : float
            Maximum redshift for this range
        """
        self.likelihood = likelihood
        self.z_min = z_min
        self.z_max = z_max
        
        # Create a filtered dataset
        self._filter_data()
        
    def _filter_data(self):
        """Filter the likelihood's data to the specified redshift range."""
        # Get original data
        z_data = self.likelihood.z_data
        mu_obs = self.likelihood.mu_obs
        sigma_mu = self.likelihood.sigma_mu
        
        # Create mask for redshift range
        mask = (z_data >= self.z_min) & (z_data <= self.z_max)
        
        # Store filtered data temporarily
        self.z_data_filtered = z_data[mask]
        self.mu_obs_filtered = mu_obs[mask]
        self.sigma_mu_filtered = sigma_mu[mask]
        self.N_data_filtered = len(self.z_data_filtered)
        
        print(f"Z range [{self.z_min:.2f}, {self.z_max:.2f}]: "
              f"{self.N_data_filtered} data points selected (from {len(z_data)} total)")
    
    def __call__(self, theta, n=1000):
        """
        Compute log-likelihood using only the filtered redshift range.
        
        Parameters
        ----------
        theta : array_like
            Cosmological parameters [Omega_m, Omega_Lambda, H0]
        n : int
            Number of integration points
        
        Returns
        -------
        log_likelihood : float
        """
        # Use the base likelihood's model function
        mu_model = self.likelihood.model(self.z_data_filtered, theta, n=n)
        
        # Calculate residuals for filtered data
        residuals = self.mu_obs_filtered - mu_model
        
        # Compute chi-squared
        chi_squared = np.sum((residuals / self.sigma_mu_filtered)**2)
        
        # Return log-likelihood
        log_likelihood = -0.5 * chi_squared
        
        return log_likelihood


# =====================================================================
# METROPOLIS MCMC — for individual redshift ranges
# =====================================================================

class Metropolis:
    """
    Generic Metropolis MCMC sampler for any log-likelihood function.
    """

    def __init__(self, likelihood, theta_init, sigma, n_integration=200):
        """
        Initialise the sampler.

        Parameters
        ----------
        likelihood : callable
            Log-likelihood function with signature likelihood(theta, n=...) -> float
        theta_init : array-like
            Starting parameter vector [Omega_m, Omega_Lambda, H0]
        sigma : array-like
            Proposal step sizes for each parameter
        n_integration : int, optional
            Number of integration points passed to the likelihood (default 200)
        """

        self.likelihood = likelihood
        self.theta = np.array(theta_init, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self.n_integration = n_integration

        # Storage for the chain
        self.chain = []
        self.log_likelihoods = []

        # Track acceptance statistics
        self.n_accepted = 0
        self.n_total = 0

        # Evaluate log-likelihood at the starting point
        self.log_L_current = self.likelihood(self.theta, n=self.n_integration)

    def step(self):
        """
        Perform a single Metropolis step.
        """
        # Propose new parameters by perturbing each dimension independently
        theta_proposal = self.theta + self.sigma * np.random.randn(len(self.theta))

        # Evaluate log-likelihood at the proposed point
        log_L_proposal = self.likelihood(theta_proposal, n=self.n_integration)

        # Metropolis acceptance criterion
        log_ratio = log_L_proposal - self.log_L_current
        if log_ratio > 0 or np.log(np.random.uniform()) < log_ratio:
            self.theta = theta_proposal
            self.log_L_current = log_L_proposal
            self.n_accepted += 1

        # Record current state regardless of whether we moved
        self.chain.append(self.theta.copy())
        self.log_likelihoods.append(self.log_L_current)
        self.n_total += 1

    def run(self, n_steps, burn_in=500):
        """
        Run the sampler for a given number of steps.
        """
        self.burn_in = burn_in

        t_start = time.time()
        for i in range(n_steps):
            self.step()
        t_end = time.time()

        print(f"Metropolis: {n_steps} steps in {t_end - t_start:.1f} s  "
              f"| acceptance rate: {self.acceptance_rate():.2%}")
        
    def acceptance_rate(self):
        """Fraction of proposed steps that were accepted."""
        return self.n_accepted / self.n_total if self.n_total > 0 else 0.0
    
    def get_chain(self, burn_in=None):
        """Return the post-burn-in chain as a numpy array."""
        if burn_in is None:
            burn_in = getattr(self, 'burn_in', 0)
        return np.array(self.chain[burn_in:])
    
    def get_log_likelihoods(self, burn_in=None):
        """Return post-burn-in log-likelihood values."""
        if burn_in is None:
            burn_in = getattr(self, 'burn_in', 0)
        return np.array(self.log_likelihoods[burn_in:])

    def plot_1d_histograms(self, burn_in=None, param_names=None, bins=50):
        """
        Plot 1D marginalised posterior histograms for each parameter.
        """
        chain = self.get_chain(burn_in)
        if param_names is None:
            param_names = [r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$ (km/s/Mpc)']

        n_params = chain.shape[1]
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))

        for idx in range(n_params):
            axes[idx].hist(chain[:, idx], bins=bins, density=True,
                           color='steelblue', edgecolor='white', linewidth=0.4)
            axes[idx].set_xlabel(param_names[idx])
            axes[idx].set_ylabel('Probability density')

            lo, med, hi = np.percentile(chain[:, idx], [16, 50, 84])
            axes[idx].axvline(med, color='red', lw=1.5, label=f'median = {med:.3g}')
            axes[idx].axvline(lo, color='red', lw=1, linestyle='--')
            axes[idx].axvline(hi, color='red', lw=1, linestyle='--',
                              label=f'+{hi-med:.2g} / -{med-lo:.2g}')
            axes[idx].legend(fontsize=8)

        return fig, axes

    def plot_2d_histograms(self, burn_in=None, param_names=None, bins=40):
        """
        Plot 2D joint posterior distributions as filled 2D histograms.
        """
        chain = self.get_chain(burn_in)
        if param_names is None:
            param_names = [r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$ (km/s/Mpc)']

        # All three unique parameter pairs for a 3-parameter model
        pairs = [(0, 1), (0, 2), (1, 2)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (xi, yi) in zip(axes, pairs):
            h, xedges, yedges = np.histogram2d(chain[:, xi], chain[:, yi], bins=bins, density=True)
            im = ax.imshow(h.T, origin='lower', aspect='auto',
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            ax.set_xlabel(param_names[xi])
            ax.set_ylabel(param_names[yi])
            ax.set_title(f'{param_names[xi]} vs {param_names[yi]}')
            fig.colorbar(im, ax=ax, label='Probability density')

        return fig, axes


def print_parameter_summary(sampler, label, param_names=None):
    """
    Print a summary table of median and 68% credible intervals for each parameter.
    """
    chain = sampler.get_chain()
    if param_names is None:
        param_names = ['Omega_m', 'Omega_Lambda', 'H0']

    print(f"\n{label}:")
    print(f"{'Parameter':<15} {'Median':>10} {'−1σ':>10} {'+1σ':>10}")
    print("-" * 48)
    for idx, name in enumerate(param_names):
        lo, med, hi = np.percentile(chain[:, idx], [16, 50, 84])
        print(f"{name:<15} {med:>10.4f} {med-lo:>10.4f} {hi-med:>10.4f}")


# =====================================================================
# MAIN ANALYSIS
# =====================================================================

# Initialise the Likelihood object with Pantheon supernova data
script_dir = os.path.dirname(os.path.abspath(__file__))
likelihood = Likelihood(os.path.join(script_dir, 'pantheon_data.txt'))

# Define the three redshift ranges
z_max_data = np.max(likelihood.z_data)
z_short = z_max_data / 3          # 0 to short z
z_medium = 2 * z_max_data / 3     # 0 to medium z
z_long = z_max_data               # 0 to long z (full range)

print(f"Maximum redshift in data: {z_max_data:.3f}")
print(f"Redshift ranges: [0, {z_short:.3f}], [0, {z_medium:.3f}], [0, {z_long:.3f}]")

# Initial parameters for all MCMC runs
theta_init = [0.3, 0.7, 70.0]   # [Omega_m, Omega_Lambda, H0]
sigma = [0.07, 0.1, 0.3]        # proposal step sizes

# Store all samplers for comparison
samplers = []
labels = [f'Short (z < {z_short:.2f})', 
          f'Medium (z < {z_medium:.2f})', 
          f'Long (z < {z_long:.2f})']
z_max_values = [z_short, z_medium, z_long]

# Run MCMC for each redshift range
for i, (z_max, label) in enumerate(zip(z_max_values, labels)):
    print(f"\n{'='*60}")
    print(f"Running MCMC for {label}")
    print(f"{'='*60}")
    
    # Create filtered likelihood for this redshift range
    filtered_likelihood = FilteredLikelihood(likelihood, z_min=0.0, z_max=z_max)
    
    # Run MCMC
    sampler = Metropolis(filtered_likelihood, theta_init=theta_init, sigma=sigma, n_integration=200)
    sampler.run(n_steps=5500, burn_in=500)
    
    # Print summary
    print_parameter_summary(sampler, label)
    
    samplers.append(sampler)


# =====================================================================
# COMPARISON PLOTS
# =====================================================================

# Plot 1: Overlay 1D posterior distributions for comparison
param_names = [r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$ (km/s/Mpc)']
param_ranges = [
    (0.0, 0.6),      # Omega_m range
    (0.2, 1.2),      # Omega_Lambda range
    (60, 80)         # H0 range
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (ax, param_name, (xmin, xmax)) in enumerate(zip(axes, param_names, param_ranges)):
    for sampler, label, color in zip(samplers, labels, ['blue', 'green', 'red']):
        chain = sampler.get_chain()
        ax.hist(chain[:, idx], bins=50, density=True, alpha=0.5, 
               label=label, color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title(f'Posterior distributions: {param_name}', fontsize=12)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle('Project B: Comparison across redshift ranges', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('project_B_comparison_1d.png', dpi=100, bbox_inches='tight')
print("\nComparison plot saved to project_B_comparison_1d.png")
plt.show()


# Plot 2: 2D constraints for each range (Omega_m vs H0)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, sampler, label in zip(axes, samplers, labels):
    chain = sampler.get_chain()
    
    # Create 2D histogram for Omega_m vs H0
    h, xedges, yedges = np.histogram2d(chain[:, 0], chain[:, 2], bins=40, density=True)
    im = ax.imshow(h.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax.set_xlabel(r'$\Omega_m$', fontsize=11)
    ax.set_ylabel(r'$H_0$ (km/s/Mpc)', fontsize=11)
    ax.set_title(label, fontsize=11)
    fig.colorbar(im, ax=ax, label='Probability density')

fig.suptitle('Project B: 2D constraints (Ωₘ vs H₀) across redshift ranges', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('project_B_comparison_2d.png', dpi=100, bbox_inches='tight')
print("2D comparison plot saved to project_B_comparison_2d.png")
plt.show()


# Plot 3: Print constraint widths for comparison
print("\n" + "="*60)
print("CONSTRAINT WIDTHS (68% credible intervals)")
print("="*60)

for sampler, label in zip(samplers, labels):
    chain = sampler.get_chain()
    print(f"\n{label}:")
    print(f"{'Parameter':<15} {'σ(68% CI)':>12}")
    print("-" * 30)
    for idx, name in enumerate(['Omega_m', 'Omega_Lambda', 'H0']):
        lo, med, hi = np.percentile(chain[:, idx], [16, 50, 84])
        width = (hi - lo) / 2  # Standard deviation equivalent
        print(f"{name:<15} {width:>12.4f}")

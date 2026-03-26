"""
Unit 5: Likelihood grid and Metropolis MCMC analysis of Pantheon supernova data.

Builds a 3D log-likelihood grid over (Omega_m, Omega_Lambda, H0), marginalizes
to produce 2D and 1D probability distributions, then runs a Metropolis MCMC
sampler to explore the same posterior more efficiently.
"""

import numpy as np
import matplotlib.pyplot as plt
from unit4_copy import Likelihood
import time
import os


def create_likelihood_grid(likelihood, omega_m_values, omega_lambda_values, H0_values):
    """
    Create a 3D grid of log-likelihood values for all combinations
    of the given cosmological parameters.

    The grid G[i, j, k] = log L(Omega_m[i], Omega_Lambda[j], H0[k])
    is built by evaluating the likelihood at every point in parameter space.

    Parameters
    ----------
    likelihood : Likelihood
        Callable likelihood object from unit4
    omega_m_values : ndarray
        Array of Omega_m values to evaluate
    omega_lambda_values : ndarray
        Array of Omega_Lambda values to evaluate
    H0_values : ndarray
        Array of H0values to evaluate

    Returns
    -------
    grid : ndarray
        3D array of log-likelihood values, shape (n, n, n)
        where grid[i, j, k] = log L(Omega_m[i], Omega_Lambda[j], H0[k])
    """
    n = len(omega_m_values)

    # Initialise empty 3D array to store log-likelihood at each grid point
    grid = np.zeros((n, n, n))

    # Loop over all combinations of the three parameters
    # i indexes Omega_m, j indexes Omega_Lambda, k indexes H0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Construct the parameter vector for this grid point
                theta = [omega_m_values[i], omega_lambda_values[j], H0_values[k]]

                # Evaluate log-likelihood and store in grid
                # n=100 integration points balances speed and accuracy
                grid[i, j, k] = likelihood(theta, n=100)

    return grid

# Initialise the Likelihood object with Pantheon supernova data
# Note: use of lowercase variable name to avoid overwriting the class itself
script_dir = os.path.dirname(os.path.abspath(__file__))
likelihood = Likelihood(os.path.join(script_dir, 'pantheon_data.txt'))

# Define linearly spaced arrays for each cosmological parameter
# These ranges can be adjusted to zoom in on the interesting structure
omega_m_values = np.linspace(0.2, 0.5, 5)      
omega_lambda_values = np.linspace(0.5, 1.1, 5)  
H0_values = np.linspace(68, 74, 5)             

# Build the 3D likelihood grid over all parameter combinations
t_start = time.time()
grid = create_likelihood_grid(likelihood, omega_m_values, omega_lambda_values, H0_values)
t_end = time.time()
print(f"Grid computation took {t_end - t_start:.1f} seconds")

def marginalize_2d(grid, axis):
    """
    Marginalize a 3D log-likelihood grid over one parameter to produce a 2D probability grid.

    Implements: P[j,k] = Σᵢ exp(G[i,j,k] - G_max)
    Subtracting G_max before exponentiating prevents numerical underflow.

    Parameters
    ----------
    grid : ndarray
        3D log-likelihood grid of shape (n, n, n)
    axis : int
        Axis to marginalize (sum) over:
        0 = Omega_m, 1 = Omega_Lambda, 2 = H0

    Returns
    -------
    prob_2d : ndarray
        2D probability grid with the chosen axis summed out
    """
    # Convert log-likelihoods to likelihoods, subtracting the max first to prevent underflow
    prob_grid = np.exp(grid - np.max(grid))

    # Sum (integrate) over the chosen axis to marginalise out that parameter
    prob_2d = np.sum(prob_grid, axis=axis)

    # Normalise to [0, 1] so the result is a relative probability
    return prob_2d / np.max(prob_2d)

def marginalize_1d(grid, axes):
    """
    Marginalize a 3D log-likelihood grid over two parameters to produce a 1D probability distribution.

    Implements: P[i] = Σⱼ Σₖ exp(G[i,j,k] - G_max)

    Parameters
    ----------
    grid : ndarray
        3D log-likelihood grid of shape (n, n, n)
    axes : tuple of int
        Two axes to marginalize over:
        (1, 2) -> keeps Omega_m
        (0, 2) -> keeps Omega_Lambda
        (0, 1) -> keeps H0

    Returns
    -------
    prob_1d : ndarray
        1D probability distribution for the remaining parameter
    """
    # Convert log-likelihoods to likelihoods, subtracting the max first to prevent underflow
    prob_grid = np.exp(grid - np.max(grid))

    # Sum over both chosen axes to marginalise out two parameters, leaving a 1D distribution
    prob_1d = np.sum(prob_grid, axis=axes)

    # Normalise to [0, 1] so the result is a relative probability
    return prob_1d / np.max(prob_1d)


# =====================================================================
# MARGINALIZATION
# =====================================================================

# Marginalize over each parameter to get three 2D grids
grid_2d_no_Omega_m  = marginalize_2d(grid, axis=0)  # Sums over Omega_m     -> Omega_Lambda vs H0
grid_2d_no_Omega_lambda = marginalize_2d(grid, axis=1)  # Sums over Omega_Lambda -> Omega_m vs H0
grid_2d_no_H0 = marginalize_2d(grid, axis=2)  # Sums over H0         -> Omega_m vs Omega_Lambda

# Marginalize over two parameters to get three 1D distributions
prob_1d_Omega_m  = marginalize_1d(grid, axes=(1, 2))  # P(Omega_m)
prob_1d_Omega_lambda = marginalize_1d(grid, axes=(0, 2))  # P(Omega_Lambda)
prob_1d_H0 = marginalize_1d(grid, axes=(0, 1))  # P(H0)

def plot_2d_grids(grid_2d_no_Omega_m, grid_2d_no_Omega_lambda, grid_2d_no_H0,
                  omega_m_values, omega_lambda_values, H0_values):
    """
    Plot the three 2D marginalized probability grids as heatmaps using imshow.

    Note: imshow plots the first array axis along y and second along x,
    so the extent is set accordingly as [xmin, xmax, ymin, ymax].

    Parameters
    ----------
    grid_2d_no_Omega_m : ndarray
        2D grid marginalized over Omega_m, shape (n, n) -> Omega_Lambda vs H0
    grid_2d_no_Omega_lambda : ndarray
        2D grid marginalized over Omega_Lambda, shape (n, n) -> Omega_m vs H0
    grid_2d_no_H0 : ndarray
        2D grid marginalized over H0, shape (n, n) -> Omega_m vs Omega_Lambda
    omega_m_values : ndarray
        Array of Omega_m values
    omega_lambda_values : ndarray
        Array of Omega_Lambda values
    H0_values : ndarray
        Array of H0 values
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Omega_Lambda vs H0 (marginalized over Omega_m)
    # Array axes are [Omega_Lambda, H0] so y=Omega_Lambda, x=H0
    # Grids are already normalised to [0, 1] by marginalize_2d
    im0 = axes[0].imshow(grid_2d_no_Omega_m, origin='lower', aspect='auto',
                         extent=[H0_values[0], H0_values[-1],
                                 omega_lambda_values[0], omega_lambda_values[-1]])
    axes[0].set_xlabel('H0 (km/s/Mpc)')
    axes[0].set_ylabel('Omega_Lambda')
    axes[0].set_title('Marginalized over Omega_m')
    fig.colorbar(im0, ax=axes[0], label='Relative probability')

    # Plot 2: Omega_m vs H0 (marginalized over Omega_Lambda)
    # Array axes are [Omega_m, H0] so y=Omega_m, x=H0
    im1 = axes[1].imshow(grid_2d_no_Omega_lambda, origin='lower', aspect='auto',
                         extent=[H0_values[0], H0_values[-1],
                                 omega_m_values[0], omega_m_values[-1]])
    axes[1].set_xlabel('H0 (km/s/Mpc)')
    axes[1].set_ylabel('Omega_m')
    axes[1].set_title('Marginalized over Omega_Lambda')
    fig.colorbar(im1, ax=axes[1], label='Relative probability')

    # Plot 3: Omega_m vs Omega_Lambda (marginalized over H0)
    # Array axes are [Omega_m, Omega_Lambda] so y=Omega_m, x=Omega_Lambda
    im2 = axes[2].imshow(grid_2d_no_H0, origin='lower', aspect='auto',
                         extent=[omega_lambda_values[0], omega_lambda_values[-1],
                                 omega_m_values[0], omega_m_values[-1]])
    axes[2].set_xlabel('Omega_Lambda')
    axes[2].set_ylabel('Omega_m')
    axes[2].set_title('Marginalized over H0')
    fig.colorbar(im2, ax=axes[2], label='Relative probability')

    plt.tight_layout()
    plt.show()

def plot_1d_distributions(prob_1d_Omega_m, prob_1d_Omega_lambda, prob_1d_H0,
                           omega_m_values, omega_lambda_values, H0_values):
    """
    Plot the three 1D marginalized probability distributions as line curves.

    Parameters
    ----------
    prob_1d_Omega_m : ndarray
        1D probability distribution for Omega_m
    prob_1d_Omega_lambda : ndarray
        1D probability distribution for Omega_Lambda
    prob_1d_H0 : ndarray
        1D probability distribution for H0
    omega_m_values : ndarray
        Array of Omega_m values
    omega_lambda_values : ndarray
        Array of Omega_Lambda values
    H0_values : ndarray
        Array of H0 values
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot each parameter's 1D probability distribution as a line curve
    axes[0].plot(omega_m_values, prob_1d_Omega_m)
    axes[0].set_xlabel('Omega_m')
    axes[0].set_ylabel('Relative probability')
    axes[0].set_title('P(Omega_m)')

    axes[1].plot(omega_lambda_values, prob_1d_Omega_lambda)
    axes[1].set_xlabel('Omega_Lambda')
    axes[1].set_ylabel('Relative probability')
    axes[1].set_title('P(Omega_Lambda)')

    axes[2].plot(H0_values, prob_1d_H0)
    axes[2].set_xlabel('H0 (km/s/Mpc)')
    axes[2].set_ylabel('Relative probability')
    axes[2].set_title('P(H0)')

    plt.tight_layout()
    plt.show()


# =====================================================================
# PLOTTING
# =====================================================================

# Plot the three 2D marginalized grids as heatmaps
plot_2d_grids(grid_2d_no_Omega_m, grid_2d_no_Omega_lambda, grid_2d_no_H0, omega_m_values, omega_lambda_values, H0_values)

# Plot the three 1D probability distributions as line curves
plot_1d_distributions(prob_1d_Omega_m, prob_1d_Omega_lambda, prob_1d_H0, omega_m_values, omega_lambda_values, H0_values)

# =====================================================================
# METROPOLIS MCMC
# =====================================================================

class Metropolis:
    """
    Generic Metropolis MCMC sampler for any log-likelihood function.

    The algorithm proposes a new point in parameter space by adding
    Gaussian noise scaled by sigma to the current position. The proposal
    is accepted with probability min(1, exp(log_L_proposal - log_L_current)),
    which ensures the chain converges to the posterior distribution.

    Attributes
    ----------
    chain : list of ndarray
        All recorded parameter vectors (including burn-in)
    log_likelihoods : list of float
        Log-likelihood at each recorded step
    n_accepted : int
        Number of accepted proposals
    n_total : int
        Total number of steps taken
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
            Proposal step sizes for each parameter [sigma_Omega_m, sigma_Omega_Lambda, sigma_H0]
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

        Proposes a new point by adding Gaussian noise, then accepts or rejects
        it according to the Metropolis criterion. Records the current point
        (whether it moved or stayed) in the chain.
        """

        # Propose new parameters by perturbing each dimension independently
        theta_proposal = self.theta + self.sigma * np.random.randn(len(self.theta))

        # Evaluate log-likelihood at the proposed point
        log_L_proposal = self.likelihood(theta_proposal, n=self.n_integration)

        # Metropolis acceptance criterion: always accept uphill moves;
        # accept downhill moves with probability exp(log_L_proposal - log_L_current)
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

        Parameters
        ----------
        n_steps : int
            Total number of steps to take (including burn-in)
        burn_in : int, optional
            Number of initial steps to label as burn-in (default 500).
            Stored on the instance so get_chain() can discard them automatically.
        """

        # Store burn_in on the instance so get_chain() can discard it automatically
        self.burn_in = burn_in

        # Time the run so we can report how long it took
        t_start = time.time()
        for i in range(n_steps):
            self.step()
        t_end = time.time()

        # Print a summary of the run
        print(f"Metropolis: {n_steps} steps in {t_end - t_start:.1f} s  "
              f"| acceptance rate: {self.acceptance_rate():.2%}")
            
    def acceptance_rate(self):
        """Fraction of proposed steps that were accepted."""
        return self.n_accepted / self.n_total if self.n_total > 0 else 0.0

    def get_chain(self, burn_in=None):
        """
        Return the post-burn-in chain as a numpy array.

        Parameters
        ----------
        burn_in : int, optional
            Number of initial steps to discard. Defaults to self.burn_in set by run().

        Returns
        -------
        chain : ndarray, shape (n_samples, n_params)
        """
        # Fall back to the burn_in set by run(), or 0 if run() hasn't been called
        if burn_in is None:
            burn_in = getattr(self, 'burn_in', 0)

        # Slice off the burn-in steps and convert to a numpy array
        return np.array(self.chain[burn_in:])

    def get_log_likelihoods(self, burn_in=None):
        """
        Return post-burn-in log-likelihood values.

        Parameters
        ----------
        burn_in : int, optional
            Number of initial steps to discard.

        Returns
        -------
        log_ls : ndarray, shape (n_samples,)
        """
        # Fall back to the burn_in set by run(), or 0 if run() hasn't been called
        if burn_in is None:
            burn_in = getattr(self, 'burn_in', 0)

        # Slice off the burn-in steps and convert to a numpy array
        return np.array(self.log_likelihoods[burn_in:])

    def plot_trace(self, burn_in=None, param_names=None):
        """
        Plot the trace of each parameter and the log-likelihood vs step number.

        This is useful for diagnosing burn-in: if the trace is still drifting,
        the chain has not yet converged.
        """
        # Use burn_in from run() if not explicitly provided
        if burn_in is None:
            burn_in = getattr(self, 'burn_in', 0)
        if param_names is None:
            param_names = [r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$']

        # Use the full chain including burn-in so the shading makes sense visually
        full_chain = np.array(self.chain)
        full_log_ls = np.array(self.log_likelihoods)
        steps = np.arange(len(full_chain))

        # One subplot per parameter plus one for log L, all sharing the x-axis
        n_params = full_chain.shape[1]
        fig, axes = plt.subplots(n_params + 1, 1, figsize=(10, 3 * (n_params + 1)), sharex=True)

        # Plot each parameter trace with burn-in region shaded salmon
        for idx in range(n_params):
            axes[idx].plot(steps, full_chain[:, idx], lw=0.5, color='steelblue')
            axes[idx].axvspan(0, burn_in, color='salmon', alpha=0.3, label='burn-in')
            axes[idx].set_ylabel(param_names[idx])
            axes[idx].legend(loc='upper right', fontsize=8)

        # Plot log-likelihood trace on the bottom panel
        axes[-1].plot(steps, full_log_ls, lw=0.5, color='darkorange')
        axes[-1].axvspan(0, burn_in, color='salmon', alpha=0.3, label='burn-in')
        axes[-1].set_ylabel('log L')
        axes[-1].set_xlabel('Step')
        axes[-1].legend(loc='upper right', fontsize=8)

        fig.suptitle('Metropolis Trace Plots', fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_1d_histograms(self, burn_in=None, param_names=None, bins=50):
        """
        Plot 1D marginalised posterior histograms for each parameter.
        """
        # Discard burn-in before plotting
        chain = self.get_chain(burn_in)
        if param_names is None:
            param_names = [r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$ (km/s/Mpc)']

        n_params = chain.shape[1]
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))

        for idx in range(n_params):
            # Normalised histogram so the y-axis is probability density
            axes[idx].hist(chain[:, idx], bins=bins, density=True,
                           color='steelblue', edgecolor='white', linewidth=0.4)
            axes[idx].set_xlabel(param_names[idx])
            axes[idx].set_ylabel('Probability density')

            # 16th/50th/84th percentiles give the median and ±1σ credible interval
            lo, med, hi = np.percentile(chain[:, idx], [16, 50, 84])
            axes[idx].axvline(med, color='red', lw=1.5, label=f'median = {med:.3g}')
            axes[idx].axvline(lo, color='red', lw=1, linestyle='--')
            axes[idx].axvline(hi, color='red', lw=1, linestyle='--',
                              label=f'+{hi-med:.2g} / -{med-lo:.2g}')
            axes[idx].legend(fontsize=8)

        fig.suptitle('Metropolis 1D Marginalised Posteriors', fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_2d_histograms(self, burn_in=None, param_names=None, bins=40):
        """
        Plot 2D joint posterior distributions as filled 2D histograms.
        """
        # Discard burn-in before plotting
        chain = self.get_chain(burn_in)
        if param_names is None:
            param_names = [r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$ (km/s/Mpc)']

        # All three unique parameter pairs for a 3-parameter model
        pairs = [(0, 1), (0, 2), (1, 2)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (xi, yi) in zip(axes, pairs):
            # Build a 2D histogram; .T is needed because imshow expects [y, x] not [x, y]
            h, xedges, yedges = np.histogram2d(chain[:, xi], chain[:, yi], bins=bins, density=True)
            im = ax.imshow(h.T, origin='lower', aspect='auto',
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            ax.set_xlabel(param_names[xi])
            ax.set_ylabel(param_names[yi])
            ax.set_title(f'{param_names[xi]} vs {param_names[yi]}')
            fig.colorbar(im, ax=ax, label='Probability density')

        fig.suptitle('Metropolis 2D Joint Posteriors', fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_3d_scatter(self, burn_in=None, param_names=None, max_points=5000):
        """
        3D scatter plot of the chain, coloured by log-likelihood value.
        """
        # Discard burn-in before plotting
        chain = self.get_chain(burn_in)
        log_ls = self.get_log_likelihoods(burn_in)
        if param_names is None:
            param_names = [r'$\Omega_m$', r'$\Omega_\Lambda$', r'$H_0$']

        # Downsample if necessary — plotting tens of thousands of points is very slow
        if len(chain) > max_points:
            idx = np.random.choice(len(chain), max_points, replace=False)
            chain = chain[idx]
            log_ls = log_ls[idx]

        # Create a 3D axes and scatter the chain, colouring each point by log-likelihood
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(chain[:, 0], chain[:, 1], chain[:, 2],
                        c=log_ls, cmap='viridis', s=2, alpha=0.5)
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        ax.set_zlabel(param_names[2])
        ax.set_title('Metropolis chain — colour = log L')
        fig.colorbar(sc, ax=ax, shrink=0.6, label='log L')
        plt.tight_layout()
        plt.show()


# =====================================================================
# RUN METROPOLIS
# =====================================================================

theta_init = [0.3, 0.7, 70.0]   # [Omega_m, Omega_Lambda, H0]
sigma      = [0.07, 0.1, 0.3]   # proposal step sizes

sampler = Metropolis(likelihood, theta_init=theta_init, sigma=sigma, n_integration=200)
sampler.run(n_steps=5500, burn_in=500)

sampler.plot_trace()
sampler.plot_1d_histograms()
sampler.plot_2d_histograms()
sampler.plot_3d_scatter()
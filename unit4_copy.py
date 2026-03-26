"""
Unit 4: Likelihood analysis for Pantheon supernova data.
Testing cosmology class and implementing Gaussian likelihood calculation.
"""

import numpy as np
from Cosmology import Cosmology
from scipy.optimize import minimize

class Likelihood:
    """
    Likelihood class for computing the Gaussian log-likelihood of 
    cosmological parameters given Pantheon supernova data.
    """

    def __init__(self, data_file='pantheon_data.txt'):
        """
        Initialize the Likelihood class by loading Pantheon data.
        
        Parameters
        ----------
        data_file : str
            Path to the Pantheon data file (default: 'pantheon_data.txt')
        """ 

        # Load the Pantheon supernova data
        # Columns: redshift (z), distance modulus (mu), uncertainty (sigma_mu)
        data = np.loadtxt(data_file)

        # Store redshift values from first column
        self.z_data = data[:, 0]

        # Store observed distance modulus from second column
        self.mu_obs = data[:, 1]

        # Store uncertainties in distance modulus from third column
        self.sigma_mu = data[:, 2]

        # Number of data points
        self.N_data = len(self.z_data)


    def model(self, z, theta, n=1000):
        """
        Compute theoretical distance modulus m(z) for given cosmological parameters.
        
        Parameters
        ----------
        z : array_like
            Redshift values where model should be evaluated
        theta : array_like
            Cosmological parameters [Omega_m, Omega_Lambda, H0]
        n : int
            Number of integration points for distance calculation
        
        Returns
        -------
        mu_model : ndarray
            Theoretical distance modulus values at each redshift
        """
        # Extract cosmological parameters from theta vector
        omega_m = theta[0]      # Matter density parameter
        omega_lambda = theta[1]  # Dark energy density parameter
        H0 = theta[2]           # Hubble constant in km/s/Mpc

        # Create cosmology instance with given parameters
        cosmo = Cosmology(H0, omega_m, omega_lambda)

        # Compute theoretical distance modulus using the cosmology class
        # mu_integrand returns (mu_array, z_array)
        mu_model, _ = cosmo.mu_integrand(z, n=n)

        M_abs = -19.3 # Absolute magnitude of Type Ia supernovae
        mu_model = mu_model + M_abs

        return mu_model
    
    def __call__(self, theta, n=1000, model_type='standard'):
        """
        Compute log-likelihood for given cosmological parameters.
        
        This method allows the Likelihood object to be called like a function:
        likelihood = Likelihood('pantheon_data.txt')
        log_L = likelihood([0.3, 0.7, 70.0])
        
        Parameters
        ----------
        theta : array_like
            Cosmological parameters [Omega_m, Omega_Lambda, H0]
            If model_type='standard': [Omega_m, Omega_Lambda, H0]
            If model_type='no_lambda': [Omega_m, H0] (ΩΛ fixed to 0)
        n : int
            Number of integration points for distance calculation
         model_type : str
            'standard' - varies Ωm, ΩΛ, H0 (3 parameters)
            'no_lambda' - varies Ωm, H0 with ΩΛ=0 (2 parameters)
        
        Returns
        -------
        log_likelihood : float
            Natural logarithm of the Gaussian likelihood
        """

        # Construct full parameter vector based on model type
        if model_type == 'no_lambda':
            # theta = [Omega_m, H0] 
            # Set ΩΛ = 0
            theta_full = np.array([theta[0], 0.0, theta[1]])
        
        elif model_type == 'standard':
            # theta = [Omega_m, Omega_Lambda, H0]
            theta_full = theta
        
        else:
            raise ValueError("Invalid model_type. Choose 'standard' or 'no_lambda'.")

        # Compute theoretical model predictions for our data redshifts
        mu_model = self.model(self.z_data, theta_full, n)

        # Calculate residuals (difference between observed and predicted)
        residuals = self.mu_obs - mu_model

        # Compute chi-squared statistic
        # Sum of squared residuals weighted by uncertainties
        chi_squared = np.sum((residuals / self.sigma_mu)**2)

        # Calculate Gaussian log-likelihood
        # log(L) = -0.5 * chi^2 
        log_likelihood = -0.5 * chi_squared 


        return log_likelihood
    

    def test_convergence(self, theta, n_values=None):
        """
        Test convergence of log-likelihood with respect to number of integration points.
        
        Parameters
        ----------
        theta : array_like
            Cosmological parameters [Omega_m, Omega_Lambda, H0]
        n_values : array_like, optional
            Array of n values to test. If None, uses default range.
        
        Returns
        -------
        n_values : ndarray
            Array of n values tested
        log_L_values : ndarray
            Log-likelihood values for each n
        differences : ndarray
            Absolute differences between consecutive log-likelihood values
        """
        if n_values is None:
            # Default n values to test
            n_values = np.array([10, 50, 100, 200, 500, 1000, 2000, 5000])

        # Store log-likelihood values for each n
        log_L_values = np.zeros(len(n_values))

        print("Convergence Test:")
        print(f"{'n':<10} {'log(L)':<15} {'|Δ log(L)|':<15}")
        print("-" * 40)

        for i, n in enumerate(n_values):
            # Compute log-likelihood for current n
            log_L_values[i] = self(theta, n=n)

            # Compute difference from previous value 
            if i > 0:
                log_diff = abs(log_L_values[i] - log_L_values[i-1])
                print(f"{n:<10} {log_L_values[i]:<15.6f} {log_diff:<15.6f}")
            else:
                print(f"{n:<10} {log_L_values[i]:<15.6f} {'N/A':<15}")

        # Calculate absolute differences between consecutive log-likelihoods
        differences = np.abs(np.diff(log_L_values))

        return n_values, log_L_values, differences

# Commented out to prevent execution on import
# if __name__ == "__main__":
#     # Run only when unit4_copy.py is executed directly, not when imported by Unit5.py
#     likelihood = Likelihood('pantheon_data.txt')
#     theta_test = np.array([0.3, 0.7, 70.0])
#     n_values, log_L_values, differences = likelihood.test_convergence(theta_test)

                
    



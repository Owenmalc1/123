
"""
This script computes cosmological distances and distance moduli using various
numerical integration methods. It demonstrates convergence testing, cumulative
integration, interpolation, and parameter exploration for a given cosmology.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.interpolate import interp1d

class Cosmology:
    """
    Cosmology class for computing cosmological distances and distance modulus
    using various numerical integration methods.
    """

    def __init__(self, H0, omega_m, omega_lambda):
        """
        Initialize the cosmology with Hubble constant and density parameters.

        Parameters
        ----------
            H0 : (float) 
                 Hubble constant in km/s/Mpc
           
            omega_m : (float)
                      Matter density parameter
            
            omega_lambda : (float)
                           Dark energy density parameter
        """
        # Set attributes for class
        self.H0 = H0 
        self.omega_m = omega_m
        self.omega_lambda = omega_lambda
        self.omega_k = 1.0 - omega_m - omega_lambda

    def integrand(self, z):
        """
        The integrand for the cosmological distance integral.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Value of the integrand at redshift z.
        """
        # Calculate the integrand for the cosmological distance integral
        return (1.0 / np.sqrt(self.omega_m * (1 + z)**3 + self.omega_k * (1 + z)**2 + self.omega_lambda))

    def distance_rectangle(self, z, n):
        """
        Compute distance using the rectangle rule.

        Parameters
        ----------
        z : float
            Maximum redshift.
        n : int
            Number of steps.

        Returns
        -------
        float
            Distance in Mpc.
        """
        
        # Calculate distance using the rectangle rule
        
        # speed of light in km/s
        c = constants.speed_of_light / 1000.0 
        
        # step size
        dz = z / n
        
        # initial integral value
        integral = 0.0
       
        # Loop over each step to compute the integral as a summation
        for i in range(n):
            z_i = i * dz
            integral += self.integrand(z_i) * dz
        D = (c / self.H0) * integral
        return D

    def distance_trapezoid(self, z, n):
        """
        Compute distance using the trapezoid rule.

        Parameters
        ----------
        z : float
            Maximum redshift.
        n : int
            Number of steps.

        Returns
        -------
        float
            Distance in Mpc.
        """
        
        # speed of light in km/s
        c = constants.speed_of_light / 1000.0  
        
        # step size 
        dz = z / (n - 1)
        
        # set initial integral value
        integral = self.integrand(0) + self.integrand(z)
        
        #  Loop over each step to compute the integral as a summation
        for i in range(1, n - 1):
            z_i = i * dz
            integral += 2 * self.integrand(z_i)
        
        
        integral *= dz / 2
        D = (c / self.H0) * integral
        return D

    def Simpson(self, z, n):
        """
        Compute distance using Simpson's rule.

        Parameters
        ----------
        z : float
            Maximum redshift.
        n : int
            Number of steps (should be even).

        Returns
        -------
        float
            Distance in Mpc.
        """
        # speed of light in km/s
        c = constants.speed_of_light / 1000.0  
        
        # step size
        dz = (z / (2 * n))
        
        # initial integral value
        integral = self.integrand(0) + self.integrand(z)
        
        # Loop over each step to compute the integral as a summation
        
        # Odd indices
        for i in range(1, 2 * n, 2):
            z_i = i * dz
            integral += 4 * self.integrand(z_i)
        
        # Even indices
        for i in range(2, 2 * n - 1, 2):
            z_i = i * dz
            integral += 2 * self.integrand(z_i)
        
        integral *= dz / 3
        D = (c / self.H0) * integral
        return D

    def GraphDistanceError(self, z):
        """
        Plot the relative error of each integration method as a function of n.

        Parameters
        ----------
        z : float
            Redshift to compute distance to.

        Returns
        -------
        None
        """
        # n_values on a log scale 
        # Distance_Simpson as reference distance
        n_values = np.logspace(1, 4, num=20, dtype=int)
        Distance_Simpson = self.Simpson(z, 10000)  


        # Initialize error and evaluation count lists
        error_rect, error_trap, error_simp = [], [], []
        eval_rect, eval_trap, eval_simp = [], [], []

        # Loop over n_values to compute errors for each method
        for n in n_values:
            D_rect = self.distance_rectangle(z, n)
            error_rect.append(abs(D_rect - Distance_Simpson) / Distance_Simpson)
            eval_rect.append(n)
        for n in n_values:
            D_trap = self.distance_trapezoid(z, n)
            error_trap.append(abs(D_trap - Distance_Simpson) / Distance_Simpson)
            eval_trap.append(n - 1)
        for n in n_values:
            D_simp = self.Simpson(z, n)
            error_simp.append(abs(D_simp - Distance_Simpson) / Distance_Simpson)
            eval_simp.append(2 * n)

        # Plotting the errors on a log-log scale
        plt.loglog(eval_rect, error_rect, marker='o', markersize=5, label='Rectangle Rule')
        plt.loglog(eval_trap, error_trap, marker='s', markersize=5, label='Trapezoid Rule')
        plt.loglog(eval_simp, error_simp, marker='^', markersize=5, label="Simpson's Rule")
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Relative Error')
        plt.title('Error in Distance Calculation Methods')
        plt.legend()
        
        # Plot target accuracy line at 10^-5
        target_error = 1e-5
        
        # Add horizontal line for target accuracy
        plt.axhline(target_error, color='red', linestyle='--', label='Target Accuracy ($10^{-5}$)')
        plt.show()

    def cumulative_trapezoid(self, z, n):
        """
        Compute cumulative distance for an array of redshifts using the trapezoid rule.

        Parameters
        ----------
        z : float
            Maximum redshift.
        n : int
            Number of steps.

        Returns
        -------
        z_values : ndarray
            Array of redshift sample points.
        Y : ndarray
            Array of cumulative distances at each z.
        """
        
        # speed of light in km/s
        c = constants.speed_of_light / 1000.0 
        
        # Step size
        dz = z / (n - 1)
       
        # Sample points
        z_values = np.linspace(0, z, n)
        
        # Function values at sample points
        f_values = np.array([self.integrand(zi) for zi in z_values])
        
        # Initialize cumulative integral array
        Y = np.zeros_like(z_values)
        
        # Loop to compute cumulative integral
        for i in range(1, n):
            Y[i] = Y[i - 1] + (dz / 2) * (f_values[i] + f_values[i - 1])
        Y *= (c / self.H0)
        return z_values, Y

    def interpolate_distance(self, z_array, n):
        """
        Interpolate distance for arbitrary redshift values.

        Parameters
        ----------
        z_array : array_like
            Array of redshift values.
        n : int
            Number of steps for integration.

        Returns
        -------
        ndarray
            Interpolated distances for z_array.
        """
        # Maximum redshift in input array
        zmax = np.max(z_array)
        
        # Compute cumulative distance using trapezoid rule
        z_grid, D_grid = self.cumulative_trapezoid(zmax, n)
        
        # Create interpolation function
        interp_func = interp1d(z_grid, D_grid, kind='cubic', fill_value="extrapolate")
        return interp_func(z_array)

    def mu_integrand(self, z_array, n=1000):
        """
        Compute the distance modulus μ(z) for an array of redshifts.

        Parameters
        ----------
        z_array : array_like
            Array of redshift values.

        Returns
        -------
        mu_array : ndarray
            Distance modulus values for each redshift.
        z_array : ndarray
            The input array of redshifts (for convenience).
        """

        # Initialize list to hold distance modulus values
        distance_array = []
        
        # speed of light in km/s
        c = constants.speed_of_light / 1000.0  # km/s
        
        # Loop over each redshift to compute distance modulus
        for z in z_array:
            
            # x is the argument for sinh or sin functions and x itself
            x = np.sqrt(abs(self.omega_k)) * self.distance_trapezoid(z, n) * self.H0 / c
            
            # Calculate luminosity distance based on curvature
           
            # open universe
            if self.omega_k > 0:  
                D_L = (1 + z) * (np.sinh(x) / np.sqrt(abs(self.omega_k))) * (c / self.H0)
            
            # closed universe
            elif self.omega_k < 0: 
                D_L = (1 + z) * (np.sin(x) / np.sqrt(abs(self.omega_k))) * (c / self.H0)
            
            # flat universe
            else:  
                D_L = (1 + z) * self.distance_trapezoid(z, n)
            
            # To avoid log of non-positive number which otherwise leads to infinite mu
            if D_L <= 0:
                mu = np.nan
            
            else:
                mu = 5 * np.log10(D_L) + 25
            distance_array.append(mu)
        return np.array(distance_array), z_array
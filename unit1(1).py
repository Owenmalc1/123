# This is a template file for Computer Modelling Unit 1.
# You will fill it in, and should update the comments accordingly.
import numpy as np
import matplotlib.pyplot as plt
#write completed class to refer to later

class Cosmology: 
    def __init__(self,H0, omega_m, omega_lambda):
        self.H0 = H0
        self.omega_m = omega_m
        self.omega_lambda = omega_lambda
        self.omega_k = 1.0 - omega_m - omega_lambda
# "__init__" is method that runs when you make a class and sets up the objects attributes
# "self"  allows you to access and edit the attributes and methods of the object

# compute integrand of distance formula
    def integrand(self, z):
        return 1.0 / np.sqrt(self.omega_m * (1 + z)**3 + self.omega_k * (1 + z)**2 + self.omega_lambda)
#return whether universe is flat by setting a tolerance near to zero giving an output of true or false for flat or not flat
    def is_flat(self):
        if self.omega_k<= 1e-3 and self.omega_k>= -1e-3:
            print("the universe is flat")
            return True
          
        else: 
            print("the universe is not flat")
            return False
# set omega_m constant and modify omega_lambda keeping curvature of universe the same
    def set_omega_m(self, new_omega_m, omega_lambda):
        self.omega_m = new_omega_m
        self.omega_lambda = 1.0 - self.omega_k - new_omega_m
        return self.omega_m, self.omega_lambda
    
#set omega_lambda constant and modify omega_m keeping curvature of universe the same
    def set_omega_lambda(self, omega_m, new_omega_lambda):
        self.omega_lambda = new_omega_lambda
        self.omega_m = 1.0 - self.omega_k - new_omega_lambda
        return self.omega_m, self.omega_lambda

# Return the quanity omega_m*h**2 where h=H0/100kms/s/Mpc
    def omega_m_h2(self):
        h = self.H0 / 100.0
        return self.omega_m * h**2

# Return a string describing the model 

    def __str__(self): 
       return f"Cosmology(H0={self.H0}, omega_m={self.omega_m}, omega_lambda={self.omega_lambda}, omega_k={self.omega_k})"f"Cosmology(H0={self.H0}, omega_m={self.omega_m}, omega_lambda={self.omega_lambda}, omega_k={self.omega_k})"
       
    
# a simple function that shows the use of these methods
def demo_cosmology():
    cosmo = Cosmology(70, 0.3, 0.7)
    print(cosmo)
    result = cosmo.integrand(z=2)
    print(f"Integrand at z=2: {result}")
    cosmo.is_flat()

def graph():
        cosmo = Cosmology(70, 0.3, 0.7)
        z_values = np.linspace(0,1,100)
        integrand_values = [cosmo.integrand(z) for z in z_values]
        plt.plot(z_values, integrand_values)
        plt.xlabel("z")
        plt.ylabel("Integrand")
        plt.title("Integrand vs z")
        plt.grid()
        plt.show()

# Plot with varying omega_m
def graph():
    cosmo = Cosmology(70, 0.3, 0.7)
    z_values = np.linspace(0, 1, 100)
    plt.figure()
    
    for omega_m in (0.2, 0.3, 0.4):
        cosmo.set_omega_m(omega_m, cosmo.omega_lambda)
        integrand_values = [cosmo.integrand(z) for z in z_values]
        plt.plot(z_values, integrand_values, label=f"omega_m={omega_m}, omega_lambda={cosmo.omega_lambda:.2f}")
    
    plt.xlabel("z")
    plt.ylabel("Integrand")
    plt.title("Integrand vs z for varying omega_m")
    plt.legend()
    plt.grid()
    plt.show()

# This is a special python idiom that
# allows the code to be run from the command line,
# but if you import this module in another script
# the code below will not be executed.

if __name__ == "__main__":
    demo_cosmology()
    graph()

#  a plot of the integrand as function of z with range 0 to 1 
    
   

    
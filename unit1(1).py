
import numpy as np
import matplotlib.pyplot as plt


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

#  a plot of the integrand as function of z with range 0 to 1
def graph1():
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
def graph2():
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


# graph with set omega_m and varied omega_lambda for one line and one line with set omega_lambda and varied omega_m

def graph3():
    cosmo = Cosmology(70, 0.3, 0.7)
    z_values = np.linspace(0, 1, 100)
    plt.figure()
    
    # Varying omega_lambda with fixed omega_m
    for omega_lambda in (0.6, 0.7, 0.8):
        cosmo.set_omega_lambda(cosmo.omega_m, omega_lambda)
        integrand_values = [cosmo.integrand(z) for z in z_values]
        plt.plot(z_values, integrand_values, label=f"omega_m={cosmo.omega_m:.2f}, omega_lambda={omega_lambda}")
    
    # the square brackets[] create a list using a for loop to calculate the integrand values for each z in z_values
    # "omega_m={cosmo.omega_m:.2f}" formats omega_m to 2 decimal places for clarity in the legend
    # "omega_lambda={omega_lambda}" shows the current value of omega_lambda in the legend
    
    # Varying omega_m with fixed omega_lambda
    for omega_m in (0.2, 0.3, 0.4):
        cosmo.set_omega_m(omega_m, cosmo.omega_lambda)
        integrand_values = [cosmo.integrand(z) for z in z_values]
        plt.plot(z_values, integrand_values, linestyle='--', label=f"omega_m={omega_m}, omega_lambda={cosmo.omega_lambda:.2f}")

    #linestyle='--' makes the second set of lines dashed to differentiate them
    
    plt.xlabel("z")
    plt.ylabel("Integrand")
    plt.title("Integrand vs z for varying omega_m and omega_lambda")
    plt.legend()
    plt.grid()
    plt.show()

def print_objects():

    cosmoA = Cosmology(70, 0.3, 0.7)
    cosmoB = Cosmology(67.67,0.25,0.75)
    cosmoC = Cosmology(100, 1.0, 0.0)
    cosmoD = Cosmology(20, 0.0, 1.0)

    print(cosmoA)
    print(cosmoB)
    print(cosmoC)
    print(cosmoD)

    print("Is cosmo1 flat?", cosmoA.is_flat())
    print("Is cosmo2 flat?", cosmoB.is_flat())
    print("Is cosmo3 flat?", cosmoC.is_flat())
    print("Is cosmo4 flat?", cosmoD.is_flat())
    print("integrand value at z=1 for cosmology A:", cosmoA.integrand(1))
    print("omega_m*h^2 for cosmology A:", cosmoA.omega_m_h2())

                

if __name__ == "__main__":
    demo_cosmology()
    graph1()
    graph2()
    graph3()
    print_objects()




 
   

    
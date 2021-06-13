from numpy import any, arange, exp, log, logical_or, pi, sqrt, nonzero, zeros
from numpy.random import uniform
import matplotlib.pyplot as plt

# Boolean values to control plotting

show_kinetic_energy_histories = True
show_phase_space_plots = True
show_velocity_histograms = True

# Declaring constants

dt = pi/20 # Time step
dts = [pi/30, pi/20, pi/10] # Different time steps
histogram_bin_width = 0.25
Ns = [128, 512, 2048] # Number of particles
particle_mass = 1 # For simplicity
vx_FWHM = 2 # Full width at half maximum of our Maxwellian distribution
x_domain_endpoints = [-2*pi, 2*pi] # Expected form: [low_endpoint, high_endpoint]
vx_domain_endpoints = [-5, 5] # Expected form: [low_endpoint, high_endpoint]

u = 0 # This is an assumption

x_domain_length = x_domain_endpoints[1] - x_domain_endpoints[0]
vx_domain_length = vx_domain_endpoints[1] - vx_domain_endpoints[0]

# Declaring functions

def maxwellian(kT_over_m, vx, u):
    """
    kT_over_m is expected to be a float representing Boltzmann's constant times the temperature of the particles in the Maxwellian velocity
    distribution over the mass of the particles in the Maxwellian velocity distribution
    vx is expected to be a float or vector of floats representing the velocity

    We use n=1 since the integral of the probability density over all space should be 1
    """
    return (1/(2*pi*kT_over_m))**(3/2)*exp(-(1/(2*kT_over_m))*(vx-u)**2)

def inverse_maxwellian(kT_over_m, f, u):
    """
    kT_over_m is expected to be a float representing Boltzmann's constant times the temperature of the particles in the Maxwellian velocity
    distribution over the mass of the particles in the Maxwellian velocity distribution
    f is expected to be a float or vector of floats representing the output of a Maxwellian velocity distribution

    We ignore the negative square root for our purposes (the slice sampling below)
    """
    return u+sqrt(-2*kT_over_m*log(f*(1/(2*pi*kT_over_m))**(-3/2)))

def maxwellian_slice_sample(kT_over_m, num_samples):
    vx = 0
    samples = zeros((num_samples, 1))
    for i in range(num_samples):
        f = uniform(low=0, high=maxwellian(kT_over_m, vx, u))
        vx = uniform(low=-inverse_maxwellian(kT_over_m, f, u), high=inverse_maxwellian(kT_over_m, f, u), size=1) 
        # This uses that the Maxwellian distribution function is even and monotonically decreases going away from vx=0
        samples[i] = vx
    return samples

# Finding the Maxwellian velocity distribution with the desired FWHM
kT_over_m = -vx_FWHM/(8*log(1/2)) # See report for derivation

# Generating initial conditions

particle_xs = [uniform(low=x_domain_endpoints[0], high=x_domain_endpoints[1], size=(N, 1)) for N in Ns]
particle_vxs = [maxwellian_slice_sample(kT_over_m, N) for N in Ns] 

# We now correct any points in particle_vxs that aren't in our domain
fmax = maxwellian(kT_over_m, u, u)
for particle_vx in particle_vxs:
    while True: # Loop to make sure resampled values aren't out of our domain themselves
        if any(logical_or(particle_vx<vx_domain_endpoints[0], particle_vx>vx_domain_endpoints[1])):
            bad_value_indices = nonzero(logical_or(particle_vx<vx_domain_endpoints[0], particle_vx>vx_domain_endpoints[1]))
            for index in bad_value_indices:
                particle_vx[index] = maxwellian_slice_sample(kT_over_m, 1) # Resample bad indices
        else:
            break

# Task 1: Generating histograms

if show_velocity_histograms:
    num_bins = int(vx_domain_length/histogram_bin_width)
    for i, N in enumerate(Ns):
        particle_vx = particle_vxs[i]
        plt.hist(particle_vx, bins=num_bins)
        plt.title(f'Particle Velocity Histogram N={N}')
        plt.xlabel('Particle Velocity $v_x$')
        plt.ylabel('Particle Count')
        plt.show()

# Task 2: Evolving the particles in time Until t=8*pi and Task 3: Plotting particle positions

def time_step(particle_state, dt):
    '''
    Takes a particle state as input and returns the particle state advanced in time by dt
    '''
    new_positions = particle_state[0] + dt*particle_state[1]
    # The below results from our periodic boundary conditions
    if any(logical_or(new_positions<x_domain_endpoints[0], new_positions>x_domain_endpoints[1])):
        too_low_indices = nonzero(new_positions<x_domain_endpoints[0])[0]
        too_high_indices = nonzero(new_positions>x_domain_endpoints[1])[0]
        for i in range(len(too_low_indices)):
            new_positions[too_low_indices[i]] += x_domain_length
        for i in range(len(too_high_indices)):
            new_positions[too_high_indices[i]] -= x_domain_length
    new_particle_state = [new_positions, particle_state[1]]
    return new_particle_state

t_vec = arange(0, 8*pi+dt, dt) # The extra dt accounts for arange not including the stop value
for i, N in enumerate(Ns):
    particle_state = [particle_xs[i], particle_vxs[i]]
    for t in t_vec:
        if t == 0:
            pass
        else:
            particle_state = time_step(particle_state, dt)
        if show_phase_space_plots:
            if N == 512 and (t==0 or t==2*pi or t==8*pi):
                plt.scatter(particle_state[0], particle_state[1])
                if t==0:
                    formatted_t_step = 0
                elif t==2*pi:
                    formatted_t_step = '$2\pi$'
                elif t==8*pi:
                    formatted_t_step = '$8\pi$'
                else:
                    formatted_t_step = dt
                plt.title(f'Particle Positions for N={N}, t={formatted_t_step}')
                plt.xlabel('Particle Position x')
                plt.ylabel('Particle Velocity $v_x$')
                plt.show()
            if N == 128 and t<=2*pi:
                fig = plt.figure('Trajectories')
                plt.scatter(particle_state[0], particle_state[1], c='#1f77b4', linewidths=1)  # This color is the matplotlib default
                if t==2*pi:
                    plt.title(f'Particle Trajectories for N={N}, t=2$\pi$')
                    plt.xlabel('Particle Position x')
                    plt.ylabel('Particle Velocity $v_x$')
                    plt.show()
            

# Task 4: Generating kinetic energy history plots
# We use N=128 and go from t=0 to t=2*pi for simplicity

if show_kinetic_energy_histories:
    N = 128
    t_vec = arange(0, 2*pi+dt, dt) # The extra dt accounts for arange not including the stop value
    for dt in dts:
        particle_state = [particle_xs[0], particle_vxs[0]]
        kinetic_energy_history = [particle_mass*sum(particle_state[1]**2)]
        for t in t_vec:
            if t == 0:
                continue
            particle_state = time_step(particle_state, dt)
            kinetic_energy_history.append(particle_mass*sum(particle_state[1]**2))
        if dt == pi/30:
            formatted_dt = '$\\frac{\pi}{30}$'
        elif dt == pi/20:
            formatted_dt = '$\\frac{\pi}{20}$'
        elif dt == pi/10:
            formatted_dt = '$\\frac{\pi}{10}$'
        else:
            formatted_dt = dt
        plt.title(f'Kinetic Energy History for N={N}, dt={formatted_dt}')
        plt.xlabel('Time')
        plt.ylabel('Kinetic Energy')
        plt.plot(t_vec, kinetic_energy_history)
        plt.show()
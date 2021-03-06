from numpy import abs, any, arange, argmax, array, cos, diag, linspace, logical_or, matmul, mean, nonzero, ones, pi, polyfit, sin, sqrt, trapz, zeros
from numpy.fft import fft, fftfreq
from numpy.linalg import inv
from numpy.random import uniform
import matplotlib.pyplot as plt

# Boolean values to control what is run and shown
run_task_1 = True
run_task_2 = False
run_task_3 = True
show_velocity_space_plots = True    

# Declaring constants

dt = pi/20 # Time step
epsilon0 = 1 # Chosen normalization
num_grid_points = 512
particle_mass = 1 # For simplicity
particle_charge = 1 # Choice
perturbation_amplitude = 0.1
simulation_time = 4*pi
vx_domain_endpoints = [-5, 5] # Expected form: [low_endpoint, high_endpoint]
vy_domain_endpoints = [-5, 5] # Expected form: [low_endpoint, high_endpoint]
weighting_order = 1
x_domain_endpoints = [-pi, pi] # Expected form: [low_endpoint, high_endpoint]

x_domain_length = x_domain_endpoints[1] - x_domain_endpoints[0]
vx_domain_length = vx_domain_endpoints[1] - vx_domain_endpoints[0]
vy_domain_length = vy_domain_endpoints[1] - vy_domain_endpoints[1]

# Declaring functions

def charge_weighting(grid, particle_state, weighting_order):
    '''
    Takes the spatial grid, particle state, and weighting order as input, then returns the charge weighted grid 
    generated by the particles.
    The returned charge_weighted_grid will be a column vector
    '''
    num_grid_points = len(grid)
    charge_weighted_grid = zeros((num_grid_points, 1))
    grid_spacing = grid[1]-grid[0]
    num_particles = len(particle_state[0])
    total_charge = num_particles*particle_charge
    if weighting_order == 0:
        for i in range(num_particles):
            nearest_grid_index = (abs(grid-particle_state[0][i])).argmin()
            charge_weighted_grid[nearest_grid_index] += particle_charge
            if nearest_grid_index == 0: 
                # Keeping the endpoints consistent. This doesn't double count charges since the grid points at either end are the same point in space
                # since we are using periodic boundary conditions
                charge_weighted_grid[-1] += particle_charge
            elif nearest_grid_index == num_grid_points:
                charge_weighted_grid[0] += particle_charge
    elif weighting_order == 1:
        for i in range(num_particles):
            grid_distance_array = abs(grid-particle_state[0][i])
            nearest_grid_distance = grid_distance_array.min()
            nearest_grid_index = grid_distance_array.argmin()
            if nearest_grid_distance == 0: # We here handle the case that the particle is exactly on a grid point
                charge_weighted_grid[nearest_grid_index] += particle_charge
                continue
            second_nearest_grid_distance = grid_spacing - nearest_grid_distance
            if nearest_grid_index == num_grid_points-1:
                second_nearest_grid_index = num_grid_points-2
                # Keeping the endpoints consistent. This doesn't double count charges since the grid points at either end are the same point in space
                # since we are using periodic boundary conditions
                charge_weighted_grid[0] += particle_charge*second_nearest_grid_distance/grid_spacing
            elif nearest_grid_index == 0:
                second_nearest_grid_index = 1
                charge_weighted_grid[-1] += particle_charge*second_nearest_grid_distance/grid_spacing
            elif particle_state[0][i] > grid[nearest_grid_index]:
                second_nearest_grid_index = nearest_grid_index + 1
            else:
                second_nearest_grid_index = nearest_grid_index - 1
            charge_weighted_grid[nearest_grid_index] += particle_charge*second_nearest_grid_distance/grid_spacing
            charge_weighted_grid[second_nearest_grid_index] += particle_charge*nearest_grid_distance/grid_spacing
            if second_nearest_grid_index == 0:
                charge_weighted_grid[-1] += particle_charge*nearest_grid_distance/grid_spacing
            elif second_nearest_grid_index == num_grid_points-1:
                charge_weighted_grid[0] += particle_charge*nearest_grid_distance/grid_spacing
    charge_weighted_grid -= total_charge/num_grid_points # This is the assumption of a uniform background
    return charge_weighted_grid

def compute_cyclotron_frequency(frequency_ratio, num_particles):
    '''
    Computes the cyclotron frequency from the frequency ratio and the number of particles
    '''
    plasma_frequency = compute_plasma_frequency(num_particles)
    cyclotron_frequency = sqrt(plasma_frequency/frequency_ratio)
    return cyclotron_frequency

def compute_electric_field_energy(electric_field, grid_spacing):
    '''
    Computes the energy stored in the provided electric field
    The provided electric field is expected to be in the units of the simulation
    ''' 
    return epsilon0*trapz(1/2*electric_field**2, axis=0, dx=grid_spacing)[0]   

def compute_kinetic_energy(particle_state):
    '''
    Computes the kinetic energy of the particles in the given state
    '''
    return particle_mass*sum(particle_state[1][0,:]**2+particle_state[1][1,:]**2)

def compute_plasma_frequency(num_particles):
    '''
    Computes the plasma frequency for the system given by the grid and number of particles
    '''
    number_density = num_particles/x_domain_length
    return sqrt(number_density*particle_charge**2/(particle_mass*epsilon0))

def compute_v0(cyclotron_frequency, normalized_wave_vector):
    '''
    Computes v0 from the normalized wave vector assuming the wave vector corresponds to the first mode
    '''
    kperp = 2*pi/x_domain_length
    v0 = normalized_wave_vector*cyclotron_frequency/kperp
    return v0

def field_solve(charge_weighted_grid, grid_spacing):
    '''
    Takes the charge weighted grid as input then returns the electric field generated
    '''

    num_grid_points = len(charge_weighted_grid)

    # Solving Poisson's equation for the potential

    A = diag(-2*ones(num_grid_points)) + diag(ones(num_grid_points-1), 1) + diag(ones(num_grid_points-1), -1)

    # We set the the potential in the middle of the grid to 0 to set our gauge
    middle_index = num_grid_points//2 
    middle_row = zeros(num_grid_points)
    middle_row[middle_index] = 1
    A[middle_index] = middle_row

    # Apply periodic boundary conditions
    A[0][-1] = 1
    A[-1][0] = 1

    A = (1/grid_spacing**2)*A
    potential = matmul(inv(A), -charge_weighted_grid/epsilon0)

    # Computing the derivative of the potential to get the electric field

    B = diag(-ones(num_grid_points-1),-1) + diag(ones(num_grid_points-1), 1)

    # Apply periodic boundary conditions
    B[0][-1] = -1
    B[-1][0] = 1

    B = -1/(2*grid_spacing)*B

    electric_field = matmul(B, potential)
    return electric_field

def force_weighting(electric_field, grid, particle_state, weighting_order):
    '''
    Takes the electric field, spatial grid, particle state, and weighting order as input, then returns 
    the forces generated by the electric field on the particles.
    '''
    num_particles = len(particle_state[0])
    forces = zeros((1, num_particles))
    grid_spacing = grid[1]-grid[0]
    num_grid_points = len(grid)
    if weighting_order == 0:
        for i in range(num_particles):
            nearest_grid_index = (abs(grid-particle_state[0][i])).argmin()
            forces[0, i] = particle_charge*electric_field[nearest_grid_index]
    elif weighting_order == 1:
        for i in range(num_particles):
            grid_distance_array = abs(grid-particle_state[0][i])
            nearest_grid_distance = grid_distance_array.min()
            nearest_grid_index = grid_distance_array.argmin()
            if nearest_grid_distance == 0: # We here handle the case that the particle is exactly on a grid point
                forces[0, i] = particle_charge*electric_field[nearest_grid_index]
                continue
            second_nearest_grid_distance = grid_spacing - nearest_grid_distance
            if nearest_grid_index == num_grid_points-1:
                second_nearest_grid_index = num_grid_points-2
            elif nearest_grid_index == 0:
                second_nearest_grid_index = 1
            elif particle_state[0][i] > grid[nearest_grid_index]:
                second_nearest_grid_index = nearest_grid_index + 1
            else:
                second_nearest_grid_index = nearest_grid_index - 1
            forces[0, i] = particle_charge*(electric_field[nearest_grid_index]*(second_nearest_grid_distance/grid_spacing) 
                                        + electric_field[second_nearest_grid_index]*(nearest_grid_distance/grid_spacing)) 
    return forces

def generate_initial_particle_state(num_particles, v0):
    '''
    Generates an initial particle state according to a cold ring distribution

    The particle state is a list where the first component holds the particle location and the second component
    holds the particle velocities as a matrix with two rows, the first row contains the x components and the
    second row contains the y components of the velocity. The columns correspond to particles.

    The ring distribution is sampled by uniformly sampling around the ring then mapping to x and y components.
    The particle's positions are evenly distributed along the domain.
    '''
    particle_positions = linspace(x_domain_endpoints[0], x_domain_endpoints[1], num_particles)

    # Particle position perturbation
    particle_positions += perturbation_amplitude*sin(2*pi*particle_positions/x_domain_length)
    # Enforcing our domain boundaries
    left_boundary = x_domain_endpoints[0]
    right_boundary = x_domain_endpoints[1]
    if any(logical_or(particle_positions<left_boundary, particle_positions>right_boundary)):
        too_low_indices = nonzero(particle_positions<left_boundary)[0]
        too_high_indices = nonzero(particle_positions>right_boundary)[0]
        for i in range(len(too_low_indices)):
            particle_positions[too_low_indices[i]] += x_domain_length
        for i in range(len(too_high_indices)):
            particle_positions[too_high_indices[i]] -= x_domain_length

    particle_velocities = zeros((2, num_particles))
    particle_velocity_samples = uniform(0, 2*pi*v0, num_particles)
    particle_velocities[0, :] = cos(particle_velocity_samples)
    particle_velocities[1, :] = sin(particle_velocity_samples)
    return [particle_positions, particle_velocities]

def generate_grid(num_grid_points, x_domain_endpoints):
    '''
    Generates a grid from the provided domain endpoints and number of grid points
    Note that the result includes both endpoints of the domain
    '''
    return linspace(x_domain_endpoints[0], x_domain_endpoints[1], num_grid_points)

def generate_time_vector(dt, simulation_time):
    '''
    Generates the time vector for a simulation with the given total time and time step
    '''
    return arange(0, simulation_time+dt, dt) # The extra dt accounts for arange not including the stop value

def measure_instability_growth_rate(energy_history, t_vec):
    '''
    Measures the growth rate of the instability assumed to be present in the provided energy history
    by fitting a line to the data and returning the slope
    '''
    coefficients = polyfit(t_vec, energy_history, 1)
    return coefficients[1]

def measure_oscillation_frequency(dt, vector):
    '''
    Measures the frequency of motion in the provided vector assuming it was generated during the simulation
    '''

    # Total Oscillations Approach
    mean_value = mean(vector)
    mean_crossings = 0
    last_element = 0
    for element in vector:
        if element == vector[0]:
            last_element = element
            continue
        elif element >= mean_value and last_element < mean_value or element < mean_value and last_element >= mean_value:
            mean_crossings += 1
            last_element = element
    frequency = 1/2*mean_crossings/simulation_time # The one half comes from there being two crossings for one oscillation

    # One Oscillation Approach
    # mean_value = mean(vector)
    # mean_crossings = 0
    # last_element = 0
    # for index, element in enumerate(vector):
    #     if element == vector[0]:
    #         last_element = element
    #         continue
    #     elif element >= mean_value and last_element < mean_value or element < mean_value and last_element >= mean_value:
    #         mean_crossings += 1
    #         last_element = element
    #     if mean_crossings == 1:
    #         first_crossing_index = index
    #     if mean_crossings == 3:
    #         period = (index-first_crossing_index)*dt
    #         frequency = 1/period
    #         break
    
    # FFT Approach
    # frequency_index = argmax(fft(vector)) 
    # frequency = fftfreq(len(vector), d=dt)[frequency_index]
    # This approach doesn't work since somehow the FFT is leaving the vector unchanged
    return frequency

def particle_push(cyclotron_frequency, dt, forces, grid, particle_state):
    '''
    Takes the time step, forces on particles, and particle state as input, and returns the particle state
    advanced in time by one time step.
    The velocities in particle_state are expected to be one half time step behind the positions.
    This is calculated using Strang splitting.
    '''
    left_boundary = grid[0]
    right_boundary = grid[-1]
    domain_length = right_boundary-left_boundary
    new_particle_state = particle_state.copy()
    # First half acceleration
    new_particle_state[1][0,:] = new_particle_state[1][0,:]+1/2*dt*(forces/particle_mass)
    # Full rotation
    rotation_angle = cyclotron_frequency*dt
    rotation_matrix = array([[cos(rotation_angle), sin(rotation_angle)], [-sin(rotation_angle), cos(rotation_angle)]])
    new_particle_state[1] = matmul(rotation_matrix, new_particle_state[1])
    # Second half acceleration
    new_particle_state[1][0,:] = new_particle_state[1][0,:]+1/2*dt*(forces/particle_mass)
    # Advancing Position
    new_particle_state[0] = new_particle_state[0] + new_particle_state[1][0,:]*dt
    # Applying periodic boundary conditions
    if any(logical_or(new_particle_state[0]<left_boundary, new_particle_state[0]>right_boundary)):
        too_low_indices = nonzero(new_particle_state[0]<left_boundary)[0]
        too_high_indices = nonzero(new_particle_state[0]>right_boundary)[0]
        for i in range(len(too_low_indices)):
            new_particle_state[0][too_low_indices[i]] += domain_length
        for i in range(len(too_high_indices)):
            new_particle_state[0][too_high_indices[i]] -= domain_length
    return new_particle_state

def plot_electric_field_energy_history(electric_field_energy_history, t_vec, plot_title='Electric Field Energy History'):
    '''
    Plots the provided electric field energy history
    '''
    plt.figure()
    plt.semilogy(t_vec, electric_field_energy_history)
    plt.title(plot_title)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.show()

def plot_energy_histories(electric_field_energy_history, kinetic_energy_history, t_vec, plot_title='Energy Histories'):
    '''
    Plots the electric field energy history, kinetic energy history, and the history of the sum of their energies.
    We ignore that the kinetic_energy_history is reported for times that are one half of a time step behind the electric field energy.
    '''
    plt.figure()
    plt.semilogy(t_vec, electric_field_energy_history)
    plt.semilogy(t_vec, kinetic_energy_history)
    plt.semilogy(t_vec, array(electric_field_energy_history)+array(kinetic_energy_history))
    plt.title(plot_title)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend(['Electric Field Energy History', 'Kinetic Field Energy History', 'Total Energy History'])
    plt.show()

def plot_energy_history(energy_history, t_vec, plot_title='Energy History'):
    '''
    Plots the provided energy history
    '''
    plt.figure()
    plt.semilogy(t_vec, energy_history)
    plt.title(plot_title)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.show()

def run_simulation(cyclotron_frequency, grid, initial_particle_state, t_vec, weighting_order, show_velocity_space_plot=False, simulation_name='1D Plasma Simulation'):
    '''
    Runs a simulation with the given parameters
    '''
    dt = t_vec[1]-t_vec[0]
    electric_field_energy_history = []
    grid_spacing = grid[1]-grid[0]
    kinetic_energy_history = []
    particle_states = []
    for t in t_vec:
        if t == 0: # Not integrating this step aligns time t with the particle state after the time step
            particle_state = initial_particle_state
            charge_weighted_grid = charge_weighting(grid, particle_state, weighting_order)
            electric_field = field_solve(charge_weighted_grid, grid_spacing)
            electric_field_energy_history.append(compute_electric_field_energy(electric_field, grid_spacing))
            kinetic_energy_history.append(compute_kinetic_energy(particle_state))
        else:
            [particle_state, electric_field] = time_step(cyclotron_frequency, dt, grid, grid_spacing, particle_state, weighting_order)
            electric_field_energy_history.append(compute_electric_field_energy(electric_field, grid_spacing))
            kinetic_energy_history.append(compute_kinetic_energy(particle_state))
        particle_states.append(particle_state)
        if show_velocity_space_plot:
            plt.figure('Velocity Space Plot')
            plt.scatter(particle_state[1][0,:], particle_state[1][1,:])
            plt.xlim([vx_domain_endpoints[0], vx_domain_endpoints[1]])
            plt.ylim([vy_domain_endpoints[0], vy_domain_endpoints[1]])
            plt.xlabel('$v_x$')
            plt.ylabel('$v_y$')
            plt.title(simulation_name)
            plt.show(block=False)
            plt.pause(0.05)
            if t == t_vec[-1]:
                plt.close()
            else:
                plt.clf()
    return [particle_states, electric_field_energy_history, kinetic_energy_history]

def time_step(cyclotron_frequency, dt, grid, grid_spacing, particle_state, weighting_order):
    '''
    Takes a particle state as input and returns the system state advanced in time by dt.
    The velocities in particle_state are assumed to be one half step behind the positions in particle_state.
    The electric field is passed along with the new particle state so it can be used in the energy history diagnostic.
    '''
    charge_weighted_grid = charge_weighting(grid, particle_state, weighting_order)
    electric_field = field_solve(charge_weighted_grid, grid_spacing)
    forces = force_weighting(electric_field, grid, particle_state, weighting_order)
    new_particle_state = particle_push(cyclotron_frequency, dt, forces, grid, particle_state)
    return [new_particle_state, electric_field]
            
# Task 1: Observe Cyclotron Motion

if run_task_1:

    initial_particle_state = [array([0]), array([[0], [1]])]
    grid = generate_grid(num_grid_points, x_domain_endpoints)
    num_particles = 1
    t_vec = generate_time_vector(dt, simulation_time)

    cyclotron_frequency = 1 # Arbitrary for this

    [particle_states, electric_field_energy_history, kinetic_energy_history] = run_simulation( 
                                                                                        cyclotron_frequency,
                                                                                        grid, 
                                                                                        initial_particle_state, 
                                                                                        t_vec, 
                                                                                        weighting_order, 
                                                                                        show_velocity_space_plot=show_velocity_space_plots, 
                                                                                        simulation_name='One Particle Simple Harmonic Motion'
                                                                                        )

    # Compiling the history of the particle's x component of velocity to measure the oscillation frequency
    particle_x_history = []
    for particle_state in particle_states:
        particle_x_history.append(particle_state[1][0])
    motion_frequency = measure_oscillation_frequency(dt, particle_x_history)
    print(f'For task 1 the measured frequency of motion was {motion_frequency}\nThe expected cyclotron frequency was {cyclotron_frequency}')

# Task 2: Observe Ring Distribution Evolution

if run_task_2:

    frequency_ratio = 10
    grid = generate_grid(num_grid_points, x_domain_endpoints)
    normalized_wave_vector = 4.5
    num_particles = 4096
    t_vec = generate_time_vector(dt, simulation_time)

    cyclotron_frequency = compute_cyclotron_frequency(frequency_ratio, num_particles)
    v0 = compute_v0(cyclotron_frequency, normalized_wave_vector)
    initial_particle_state = generate_initial_particle_state(num_particles, v0)

    [particle_states, electric_field_energy_history, kinetic_energy_history] = run_simulation( 
                                                                                    cyclotron_frequency,
                                                                                    grid, 
                                                                                    initial_particle_state, 
                                                                                    t_vec, 
                                                                                    weighting_order, 
                                                                                    show_velocity_space_plot=show_velocity_space_plots, 
                                                                                    simulation_name=f'Ring Distribution Simulation Wave Vector={normalized_wave_vector}'
                                                                                    )

    plot_energy_history(electric_field_energy_history, t_vec, plot_title=f'Electric Field History Normalized Wave Vector={normalized_wave_vector}')

    normalized_wave_vector = 5

    cyclotron_frequency = compute_cyclotron_frequency(frequency_ratio, num_particles)
    v0 = compute_v0(cyclotron_frequency, normalized_wave_vector)
    initial_particle_state = generate_initial_particle_state(num_particles, v0)

    [particle_states, electric_field_energy_history, kinetic_energy_history] = run_simulation( 
                                                                                    cyclotron_frequency,
                                                                                    grid, 
                                                                                    initial_particle_state, 
                                                                                    t_vec, 
                                                                                    weighting_order, 
                                                                                    show_velocity_space_plot=show_velocity_space_plots, 
                                                                                    simulation_name=f'Ring Distribution Simulation Wave Number={normalized_wave_vector}'
                                                                                    )

    plot_energy_history(electric_field_energy_history, t_vec, plot_title=f'Electric Field Energy History Normalized Wave Vector={normalized_wave_vector}')

# Task 3: Observe Ring Distribution Evolution for Multiple Normalized Wave Numbers

if run_task_3:

    frequency_ratio = 10
    grid = generate_grid(num_grid_points, x_domain_endpoints)
    normalized_wave_vectors = [4.1, 4.5, 5, 5.6, 6, 6.6]
    num_particles = 4096
    t_vec = generate_time_vector(dt, simulation_time)

    for normalized_wave_vector in normalized_wave_vectors:

        cyclotron_frequency = compute_cyclotron_frequency(frequency_ratio, num_particles)
        v0 = compute_v0(cyclotron_frequency, normalized_wave_vector)
        initial_particle_state = generate_initial_particle_state(num_particles, v0)

        [particle_states, electric_field_energy_history, kinetic_energy_history] = run_simulation( 
                                                                                cyclotron_frequency,
                                                                                grid, 
                                                                                initial_particle_state, 
                                                                                t_vec, 
                                                                                weighting_order, 
                                                                                show_velocity_space_plot=show_velocity_space_plots, 
                                                                                simulation_name=f'Ring Distribution Simulation Wave Number={normalized_wave_vector}'
                                                                                )

        oscillation_frequency = measure_oscillation_frequency(dt, kinetic_energy_history)
        instability_growth_rate = measure_instability_growth_rate(electric_field_energy_history, t_vec)
        print(f'The growth rate for a normalized wave vector of {normalized_wave_vector} was {instability_growth_rate}')
        print(f'The oscillation frequency for a normalized wave vector of {normalized_wave_vector} was {oscillation_frequency}')
import matplotlib.pyplot as plt
from numpy import arange, arctan2, array, cos, cross, linspace, meshgrid, pi, reshape, sin, sqrt, trapz, zeros
from scipy.special import jv

dt = 0.01 # Time step
equilibrium_charge_density = 1 # Assumption
grid_side_points = [21, 41, 81]
RL_ratios = [1, 2, 3, 4]
simulation_time = 5
z_side_length = 1

# These function are outside of alphabetical order for dependency reasons
def get_coordinates_from_grid_indices(grid, i, j):
    '''
    Returns the spatial coordinates corresponding to the provided indices
    '''
    r_coordinate = grid[0][i, j]
    z_coordinate = grid[1][i, j]
    return [r_coordinate, z_coordinate]

def cylindrical_to_cartesian_coordinates(r, theta, z):
    '''
    Converts the provided cylindrical coordinates to cartesian coordinates
    '''
    return array([r*cos(theta), r*sin(theta), z])

def calculate_equilibrium_velocity(r, z):
    '''
    Calculates the equilibrium velocity at the provided point.
    We currently assume this is 1 in all directions
    '''
    vr = 1
    vtheta = 1
    vz = 1
    return array([vr, vtheta, vz])

def calculate_equilibrium_magnetic_field(kr, kz, lambda0, r, z):
    '''
    Calculates the equilibrium magnetic field in a spheromak given parameters
    and the location
    '''
    Br = -kz*jv(1, kr*r)*cos(kz*z)
    Btheta = lambda0*jv(1, kr*r)*sin(kz*z)
    Bz = kr*jv(0, kr*r)*sin(kz*z)
    return array([Br, Btheta, Bz])

def calculate_equilbrium_magnetic_field_parameters(grid):
    '''
    Calculates the parameters of the equilibrium magnetic field given
    the spatial grid
    '''
    [grid_spacing, r_side_length, z_side_length] = get_grid_spacing_and_side_lengths(grid)
    kr = 3.832/r_side_length
    kz = pi/z_side_length
    lambda0 = sqrt(kr**2+kz**2)
    return [kr, kz, lambda0]

def calculate_perturbed_kinetic_energy(grid_spacing, state):
    '''
    Calculates the kinetic energy of the provided state
    '''
    r_len, z_len = state.shape[0], state.shape[1]
    perturbed_velocity_squared_array = zeros((r_len, z_len))
    for i in range(r_len):
        for j in range(z_len):
                v = state[i, j, 0:3]
                v2 = v[0]**2+v[2]**2 # The magnitude squared in cylindrical coordinates
                perturbed_velocity_squared_array[i, j] = v2
    return 1/2*trapz(trapz(perturbed_velocity_squared_array, dx=grid_spacing), dx=grid_spacing)

def cartesian_to_cylindrical_coordinates(x, y, z):
    '''
    Converts the provided cartesian coordinates to cylindrical coordinates
    '''
    return array([sqrt(x**2+y**2), arctan2(y, x), z])

def cartesian_to_polar_coordinates(x, y):
    '''
    Converts the provided cartesian coordinates to cylindrical coordinates
    '''
    return array([sqrt(x**2+y**2), arctan2(y, x)])

def compute_equilibrium_magnetic_field_array(grid, kr, kz, lambda0):
    '''
    Computes an array of values of the equilibrium magnetic field
    '''
    R = grid[0]
    B0_array = zeros((R.shape[0], R.shape[1], 3))
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            [r, z] = get_coordinates_from_grid_indices(grid, i, j)
            B0_array[i, j] = calculate_equilibrium_magnetic_field(kr, kz, lambda0, r, z)
    return B0_array

def compute_v1_cross_B0_array(grid, kr, kz, lambda0, state):
    '''
    Computes an array of values of v1 x B0
    '''
    v1_cross_b0_array = zeros((state.shape[0], state.shape[1], 3))
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            [r, z] = get_coordinates_from_grid_indices(grid, i, j)
            B0cylin = calculate_equilibrium_magnetic_field(kr, kz, lambda0, r, z)
            v1cylin = state[i, j, 0:3]
            B0 = cylindrical_to_cartesian_coordinates(*B0cylin)
            v1 = cylindrical_to_cartesian_coordinates(*v1cylin)
            cross_product = cross(v1, B0)
            v1_cross_b0_array[i, j, :] = cartesian_to_cylindrical_coordinates(*cross_product)
    return v1_cross_b0_array

def generate_initial_state(grid):
    '''
    Generates an initial state in the form of a 3D numpy array where the first two axes
    corresponds to spatial dimensions (r and z) and the third axis corresponds to the length
    of the state vector. The state vector consists of 6 quantities, the three components
    of perturbed velocity and the three components of the perturbed magnetic field
    (i.e. state = [vx, vy, vz, Bx, By, Bz]).

    The magnetic field state conditions correspond to the equilibrium of a spheromak
    and we assume the initial perturbed velocity is 0. The magnetic field is defined to be 
    half a time step behind the velocity.
    '''
    R = grid[0]
    initial_state = zeros((R.shape[0], R.shape[1], 6))
    [kr, kz, lambda0] = calculate_equilbrium_magnetic_field_parameters(grid)
    for i in range(initial_state.shape[0]):
        for j in range(initial_state.shape[1]):
            r, z = get_coordinates_from_grid_indices(grid, i, j)
            [vr, vtheta, vz] = calculate_equilibrium_velocity(r, z)
            [Br, Btheta, Bz] = calculate_equilibrium_magnetic_field(kr, kz, lambda0, r, z)
            if j == 0 or j == initial_state.shape[1]-1:
                Br = 0 # Applying our boundary conditions
            Bx, By, Bz = cylindrical_to_cartesian_coordinates(Br, Btheta, Bz)
            initial_state[i, j, :] = [vr, vtheta, vz, Bx, By, Bz]
    return initial_state

def generate_grid(grid_side_points, RL_ratio, z_side_length):

    '''
    Generates a rectangular 2D cartesian grid with the given uniform spacing and the provided side length

    The grid consists of 2 matrices where the first matrix contains the r component and the second matrix contains
    the z component. The first axis of each matrix corresponds to the r-direction and the second axis of each matrix 
    corresponds to the z-direction.

    I don't understand how the grid is supposed to be square while maintaining uniform grid spacing, as the
    figures in provided reference 2 seem to show. Accordingly I have chosen to eschew the square grid in favor
    of uniform grid spacing, taking the provided number of points for the shorter side (L since R/L is 
    greater than or equal to 1) 
    '''
    z_side = linspace(0, z_side_length, grid_side_points)
    grid_spacing = z_side[1]-z_side[0]
    r_side_length = RL_ratio*z_side_length
    r_side = arange(0, r_side_length+grid_spacing, grid_spacing) # The extra grid spacing accounts for arange not including the last point
    [R, Z] = meshgrid(r_side, z_side, indexing='ij')
    return [R, Z]

def generate_time_vector(dt, simulation_time):
    '''
    Generates the time vector for a simulation with the given time step and total time
    '''
    return arange(0, simulation_time+dt, dt) # The extra dt accounts for arange not including the stop value

def generate_scalar_vector(state, state_vector_component):
    '''
    Generates a vector representing the value of the selected component of the
    state vector at every point in space.

    state_vector component is exected to be an integer between 0 and 5, and 
    denotes the index corresponding to the desired state vector component
    '''
    scalar_array = state[:, :, state_vector_component]
    num_elements = scalar_array.shape[0]*scalar_array.shape[1]
    scalar_vector = reshape(scalar_array, (num_elements, 1))
    return scalar_vector

def generate_vector_arrays(quantity, state):
    '''
    Generates an array of the specified vector quantity from the state array
    '''
    r_len = state.shape[0]
    z_len = state.shape[1]
    r_component_array = zeros((r_len, z_len))
    theta_component_array = zeros((r_len, z_len))
    z_component_array = zeros((r_len, z_len))
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if quantity == 'Velocity':
                v = state[i, j, 0:3]
                r_component_array[i, j] = v[0]
                theta_component_array[i, j] = v[1]
                z_component_array[i, j] = v[2]
            elif quantity == 'Magnetic Field':
                B = state[i, j, 3:6]
                r_component_array[i, j] = B[0]
                theta_component_array[i, j] = B[1]
                z_component_array[i, j] = B[2]
    return [r_component_array, theta_component_array, z_component_array]

def get_grid_spacing_and_side_lengths(grid):
    '''
    Gets the grid spacing and side length from the provided grid
    '''
    R = grid[0]
    Z = grid[1]
    grid_spacing = R[1, 0] - R[0, 0]
    r_side_length = R[-1, 0] - R[0, 0]
    z_side_length = Z[0, -1] - Z[0, 0]
    return [grid_spacing, r_side_length, z_side_length]

def plot_quantity_over_time(plot_title, quantity, quantity_name, t_vec):
    '''
    Plots the provided quantity over time
    '''
    plt.figure()
    plt.semilogy(t_vec, quantity)
    plt.title(plot_title)
    plt.xlabel('Time')
    plt.ylabel(quantity_name)
    plt.show()

def polar_to_cartesian_coordinates(r, theta):
    '''
    Converts the provided cylindrical coordinates to cartesian coordinates
    '''
    return array([r*cos(theta), r*sin(theta)])

def time_step(dt, grid,  grid_spacing, kr, kz, lambda0, state):
    '''
    Computes a time step of the linearize ideal MHD equations using the leapfrog algorithm.

    state is expected to be a 3D array where each element is the state vector at that location.
    The magnetic field in the state vector is expected to be half a time step behind the velocity.
    '''
    derivative_matrix = state.copy()
    v1_cross_b0_array = compute_v1_cross_B0_array(grid, kr, kz, lambda0, state)
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):

            v1cylin = state[i, j, 0:3]
            [r, z] = get_coordinates_from_grid_indices(grid, i, j)
            J0cylin = equilibrium_charge_density*calculate_equilibrium_velocity(r, z)
            J1cylin = equilibrium_charge_density*v1cylin
            B1cylin = state[i, j, 3:6]
            B0cylin = calculate_equilibrium_magnetic_field(kr, kz, lambda0, r, z)
            J0 = cylindrical_to_cartesian_coordinates(*J0cylin)
            B0 = cylindrical_to_cartesian_coordinates(*B0cylin)
            J1 = cylindrical_to_cartesian_coordinates(*J1cylin)
            B1 = cylindrical_to_cartesian_coordinates(*B1cylin)
            dvdtcart = cross(J0, B1)+cross(J1, B0)
            dvdt = cartesian_to_cylindrical_coordinates(*dvdtcart)

            # These are 0 due to our assumed perturbation form
            dveczdtheta = 0
            dvecrdtheta = 0
            if i == 0 and j == 0:
                dvecthetadz = (-3*v1_cross_b0_array[i, j, 1]+4*v1_cross_b0_array[i, j+1, 1]-v1_cross_b0_array[i, j+2, 1])/(2*grid_spacing)
                dvecrdz = (-3*v1_cross_b0_array[i, j, 0]+4*v1_cross_b0_array[i, j+1, 0]-v1_cross_b0_array[i, j+2, 0])/(2*grid_spacing)
                dveczdr = 0 # Boundary condition
                drvecthetadr = 0 # Boundary condition
                # Desired boundary conditions 
                vr = 0
                Br = 0
                vz = 0
                Bz = 0
            elif i == 0 and j == state.shape[1]-1:
                dvecthetadz = (3*v1_cross_b0_array[i, j, 1]-4*v1_cross_b0_array[i, j-1, 1]+v1_cross_b0_array[i, j-2, 1])/(2*grid_spacing)
                dvecrdz = (3*v1_cross_b0_array[i, j, 0]-4*v1_cross_b0_array[i, j-1, 0]+v1_cross_b0_array[i, j-2, 0])/(2*grid_spacing)
                dveczdr = 0 # Boundary condition
                drvecthetadr = 0 # Boundary condition
                # Desired boundary conditions 
                vr = 0
                Br = 0
                vz = 0
                Bz = 0
            elif i == 0:
                dvecthetadz = (v1_cross_b0_array[i, j+1, 1]-v1_cross_b0_array[i, j-1, 1])/(2*grid_spacing)
                dvecrdz = (v1_cross_b0_array[i, j+1, 0]-v1_cross_b0_array[i, j-1, 0])/(2*grid_spacing)
                dveczdr = 0 # Boundary condition
                drvecthetadr = 0 # Boundary condition
                # Desired boundary conditions 
                vr = 0
                Br = 0
            elif i == state.shape[0]-1 and j == 0:
                dvecthetadz = (-3*v1_cross_b0_array[i, j, 1]+4*v1_cross_b0_array[i, j+1, 1]-v1_cross_b0_array[i, j+2, 1])/(2*grid_spacing)
                dvecrdz = (-3*v1_cross_b0_array[i, j, 0]+4*v1_cross_b0_array[i, j+1, 0]-v1_cross_b0_array[i, j+2, 0])/(2*grid_spacing)
                dveczdr = (3*v1_cross_b0_array[i, j, 2]-4*v1_cross_b0_array[i-1, j, 2]+v1_cross_b0_array[i-2, j, 2])/(2*grid_spacing)
                drvecthetadr = r*(3*v1_cross_b0_array[i, j, 1]-4*v1_cross_b0_array[i-1, j, 1]+v1_cross_b0_array[i-2, j, 1])/(2*grid_spacing)
                # Desired boundary conditions 
                vz = 0
                Bz = 0
            elif i == state.shape[0]-1 and j == state.shape[1]-1:
                dvecthetadz = (3*v1_cross_b0_array[i, j, 1]-4*v1_cross_b0_array[i, j-1, 1]+v1_cross_b0_array[i, j-2, 1])/(2*grid_spacing)
                dvecrdz = (3*v1_cross_b0_array[i, j, 0]-4*v1_cross_b0_array[i, j-1, 0]+v1_cross_b0_array[i, j-2, 0])/(2*grid_spacing)
                dveczdr = (3*v1_cross_b0_array[i, j, 2]-4*v1_cross_b0_array[i-1, j, 2]+v1_cross_b0_array[i-2, j, 2])/(2*grid_spacing)
                drvecthetadr = r*(3*v1_cross_b0_array[i, j, 1]-4*v1_cross_b0_array[i-1, j, 1]+v1_cross_b0_array[i-2, j, 1])/(2*grid_spacing)
                # Desired boundary conditions 
                vr = 0
                Br = 0
                vz = 0
                Bz = 0
            elif i == state.shape[0]-1:
                dvecthetadz = (v1_cross_b0_array[i, j+1, 1]-v1_cross_b0_array[i, j-1, 1])/(2*grid_spacing)
                dvecrdz = (v1_cross_b0_array[i, j+1, 0]-v1_cross_b0_array[i, j-1, 0])/(2*grid_spacing)
                dveczdr = (3*v1_cross_b0_array[i, j, 2]-4*v1_cross_b0_array[i-1, j, 2]+v1_cross_b0_array[i-2, j, 2])/(2*grid_spacing)
                drvecthetadr = r*(3*v1_cross_b0_array[i, j, 1]-4*v1_cross_b0_array[i-1, j, 1]+v1_cross_b0_array[i-2, j, 1])/(2*grid_spacing)
                # Desired boundary conditions 
                vr = 0
                Br = 0
            elif j == 0:
                dvecthetadz = (-3*v1_cross_b0_array[i, j, 1]+4*v1_cross_b0_array[i, j+1, 1]-v1_cross_b0_array[i, j+2, 1])/(2*grid_spacing)
                dvecrdz = (-3*v1_cross_b0_array[i, j, 0]+4*v1_cross_b0_array[i, j+1, 0]-v1_cross_b0_array[i, j+2, 0])/(2*grid_spacing)
                dveczdr = (v1_cross_b0_array[i+1, j, 2]-v1_cross_b0_array[i-1, j, 2])/(2*grid_spacing)
                drvecthetadr = r*(v1_cross_b0_array[i+1, j, 1]-v1_cross_b0_array[i-1, j, 1])/(2*grid_spacing)
                # Desired boundary conditions
                vz = 0
                Bz = 0
            elif j == state.shape[1]-1:
                dvecthetadz = (3*v1_cross_b0_array[i, j, 1]-4*v1_cross_b0_array[i, j-1, 1]+v1_cross_b0_array[i, j-2, 1])/(2*grid_spacing)
                dvecrdz = (3*v1_cross_b0_array[i, j, 0]-4*v1_cross_b0_array[i, j-1, 0]+v1_cross_b0_array[i, j-2, 0])/(2*grid_spacing)
                dveczdr = (v1_cross_b0_array[i+1, j, 2]-v1_cross_b0_array[i-1, j, 2])/(2*grid_spacing)
                drvecthetadr = r*(v1_cross_b0_array[i+1, j, 1]-v1_cross_b0_array[i-1, j, 1])/(2*grid_spacing)
                # Desired boundary conditions
                vz = 0
                Bz = 0
            else:
                dvecthetadz = (v1_cross_b0_array[i, j+1, 1]-v1_cross_b0_array[i, j-1, 1])/(2*grid_spacing)
                dvecrdz = (v1_cross_b0_array[i, j+1, 0]-v1_cross_b0_array[i, j-1, 0])/(2*grid_spacing)
                dveczdr = (v1_cross_b0_array[i+1, j, 2]-v1_cross_b0_array[i-1, j, 2])/(2*grid_spacing)
                drvecthetadr = r*(v1_cross_b0_array[i+1, j, 1]-v1_cross_b0_array[i-1, j, 1])/(2*grid_spacing)

            # These are the components of the derivative of perturbed magnetic field
            theta_component = dvecrdz-dveczdr
            if r == 0:
                # To avoid dividing by zero, we set quantities to 0 that would be divided by r to 0
                r_component = 0
                z_component = 0
            else:
                r_component = (1/r)*dveczdtheta-dvecthetadz
                z_component = (1/r)*(drvecthetadr-dvecrdtheta)

            dBdt = array([r_component, theta_component, z_component])

            derivative_matrix[i, j] = array([dvdt[0], dvdt[1], dvdt[2], dBdt[0], dBdt[1], dBdt[2]])

    new_state = state+dt*derivative_matrix

    return new_state

def run_simulation(grid, grid_spacing, initial_state, t_vec, show_plot=False, quantity_to_plot='Velocity'):
    '''
    Runs a simulation with the provided parameters
    '''
    dt = t_vec[1]-t_vec[0]
    kinetic_energy_history = []
    simulation_states = []
    [kr, kz, lambda0] = calculate_equilbrium_magnetic_field_parameters(grid)
    for t in t_vec:
        if t==0: # Not performing a time step here aligns the time vector with the state after the time step
            state = initial_state
            kinetic_energy_history.append(calculate_perturbed_kinetic_energy(grid_spacing, state))
        else:
            state = time_step(dt, grid, grid_spacing, kr, kz, lambda0, state)
            kinetic_energy_history.append(calculate_perturbed_kinetic_energy(grid_spacing, state))
        simulation_states.append(state)
        if show_plot:
            [r_component_array, theta_component_array, z_component_array] = generate_vector_arrays(quantity_to_plot, state)
            fig = plt.figure(f'{quantity_to_plot} Plot')
            plt.quiver(grid[0], grid[1], r_component_array, z_component_array)
            plt.title(f'{quantity_to_plot} Plot')
            plt.xlabel('r')
            plt.ylabel('z')
            plt.show(block=False)
            plt.pause(0.05)
            if t == t_vec[-1]:
                plt.close()
            else:
                plt.clf()
    return [kinetic_energy_history, simulation_states]

# Task: R/L Ratio Convergence Study
for RL_ratio in RL_ratios:
    for grid_side_point_num in grid_side_points:
        grid = generate_grid(grid_side_point_num, RL_ratio, z_side_length)
        grid_spacing = grid[0][1, 0] - grid[0][0, 0]
        initial_state = generate_initial_state(grid)
        t_vec = generate_time_vector(dt, simulation_time)
        [kinetic_energy_history, simulation_states] = run_simulation(grid, grid_spacing, initial_state, t_vec, show_plot=True, quantity_to_plot='Velocity')
        t_vec = generate_time_vector(dt, simulation_time)
        plot_quantity_over_time(f'Kinetic Energy History R/L Ratio={RL_ratio} Grid Size={grid_side_point_num}x{grid_side_point_num}', 
                                kinetic_energy_history, 
                                'Kinetic Energy',
                                t_vec)
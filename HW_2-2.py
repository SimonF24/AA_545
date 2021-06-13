from copy import deepcopy
import matplotlib.pyplot as plt
from numpy import abs, arange, arctan2, array, cos, linspace, meshgrid, pi, sin, sqrt, trapz, unique, zeros

dt = 0.1
initial_axial_magnetic_field_strength = 1
initial_density_in_pinch = 1
initial_density_outside_pinch = 0.01
initial_pressure_in_pinch = 1
initial_pressure_outside_pinch = 0.01
initial_momentum_outside_pinch = 0.01
gamma = 5/3 # This is the ratio of specific heats
grid_spacing = 0.1
peak_initial_momentum = 1
screw_radius = 0.5
simulation_time = 8*pi
xy_side_length = 1
z_side_length = 3
mu0 = 4*pi*10**-7

def calculate_kinetic_energy(state):
    '''
    Calculates the kinetic energy of the provided state
    '''
    velocity_squared_array = zeros((len(state), len(state[0]), len(state[0][0])))
    for i in range(len(state)):
        for j in range(len(state[0])):
            for k in range(len(state[1])):
                Q = state[i][j][k]
                rho = Q[0]
                J = Q[1:4]
                v = J/rho
                v2 = v[0]**2+v[1]**2+v[2]**2
                velocity_squared_array[i][j][k] = v2
    return 1/2*trapz(trapz(trapz(velocity_squared_array, dx=grid_spacing), dx=grid_spacing), dx=grid_spacing)

def calculate_safety_factor(grid, screw_radius, state):
    '''
    Calculates the average safety factor of the provided state inside the pinch
    
    To avoid NaN values we set the safety factor to 0 when there is no poloidal field
    '''
    z_side_length = grid[2][0][0][-1]-grid[2][0][0][0]
    safety_factor = 0
    element_counter = 0 # Counts the number of elements that have been used in our average
    for i in range(len(state)):
        for j in range(len(state[0])):
            for k in range(len(state[0][0])):
                x, y, z = get_coordinates_from_grid_indices(grid, i, j, k)
                r, theta, z = cartesian_to_cylindrical_coordinates(x, y, z)
                if r < screw_radius:
                    element_counter += 1
                    Q = state[i][j][k]
                    B = Q[4:7]
                    Br, Btheta, Bz = cartesian_to_cylindrical_coordinates(B[0], B[1], B[2])
                    if Btheta == 0:
                        local_safety_factor = 0
                    else:
                        local_safety_factor = abs(screw_radius*Bz/(z_side_length*Btheta))
                    safety_factor += (local_safety_factor-safety_factor)/element_counter               
    return safety_factor

def cartesian_to_cylindrical_coordinates(x, y, z):
    '''
    Converts the provided cartesian coordinates to cylindrical coordinates
    '''
    return array([sqrt(x**2+y**2), arctan2(y, x), z])

def cylindrical_to_cartesian_coordinates(r, theta, z):
    '''
    Converts the provided cylindrical coordinates to cartesian coordinates
    '''
    return array([r*cos(theta), r*sin(theta), z])

def initial_momentum_in_pinch(peak_initial_momentum, z, z_side_length):
    '''
    Returns the initial current density at point with the given x and y coordinates

    This is defined as a parabolic function of the distance from the origin with a maximum
    determined by the peak_current_density value set above and set to zero at the ends of the
    domain.
    '''
    a = -peak_initial_momentum/z_side_length**2 
    c = peak_initial_momentum
    return array([0, 0, a*z**2+c])

def generate_F_G_and_H_arrays(state):
    '''
    Generates arrays of F, G, and H values given the current state array
    '''
    F_array = deepcopy(state)
    G_array = deepcopy(state)
    H_array = deepcopy(state)
    for i in range(len(state)):
        for j in range(len(state[0])):
            for k in range(len(state[0][0])):
                Q = state[i][j][k]
                rho = Q[0]
                J = Q[1:4]
                B = Q[4:7]
                e = Q[7]
                v = J/rho
                v2 = v[0]**2+v[1]**2+v[2]**2
                B2over2mu0 = (B[0]**2+B[1]**2+B[2]**2)/(2*mu0)
                P = (gamma-1)*(e-rho*v2/2-B2over2mu0)
                Bdotvovermu0 = (v[0]*B[0]+v[1]*B[1]+v[2]*B[2])/mu0
                F_array[i][j][k] = array([rho*v[0],
                                    rho*v[0]**2-B[0]**2/mu0+P+B2over2mu0, 
                                    rho*v[0]*v[1]-B[0]*B[1]/mu0,
                                    rho*v[0]*v[1]-B[0]*B[2]/mu0,
                                    0,
                                    v[1]*B[0]-B[1]*B[0],
                                    v[2]*B[0]-B[2]*v[0],
                                    (e+P+B2over2mu0)*v[0]-Bdotvovermu0*B[0]])
                G_array[i][j][k] = array([rho*v[1], 
                                    rho*v[0]*v[1]-B[0]*B[1]/mu0,
                                    rho*v[1]**2-B[1]**2/mu0+P+B2over2mu0,
                                    rho*v[1]*v[2]-B[1]*B[2]/mu0,
                                    v[0]*B[1]-B[0]*v[1],
                                    0,
                                    v[2]*B[0]-B[2]*v[0],
                                    (e+P+B2over2mu0)*v[1]-Bdotvovermu0*B[1]])
                H_array[i][j][k] = array([rho*v[2],
                                    rho*v[0]*v[2]-B[0]*B[2]/mu0,
                                    rho*v[1]*v[2]-B[1]*B[2]/mu0,
                                    rho*v[2]**2-B[2]**2/mu0+P+B2over2mu0,
                                    v[0]*B[2]-B[0]*v[2],
                                    v[1]*B[2]-B[1]*v[2],
                                    0,
                                    (e+P+B2over2mu0)*v[2]-Bdotvovermu0*B[2]])
    return [F_array, G_array, H_array]

def generate_initial_state(initial_axial_magnetic_field_strength, initial_momentum_outside_pinch, initial_pressure_in_pinch, initial_pressure_outside_pinch, grid, peak_initial_momentum):
    '''
    Generates an initial state in the form of a 3D array where each element of the array is the 
    state vector at the corresponding point on the grid

    To preserve right handedness the first axis of the state array is the y-axis, the second axis
    is the x-axis, and the third axis is the z-axis. Accordingly, the state vector for a point with 
    indices given by (y, x, z) can be accessed through state[y][x][z]
    '''
    X = grid[0]
    z_side_length = X.shape[2]
    initial_state = X.tolist() # This is initially full of x coordinates, but will be overwritten
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                x, y, z = get_coordinates_from_grid_indices(grid, i, j, k)
                r, theta, z = cartesian_to_cylindrical_coordinates(x, y, z)
                if r>screw_radius:
                    rho = initial_density_outside_pinch
                    momentum = [0, 0, initial_momentum_outside_pinch]
                    v = [rhov/rho for rhov in momentum]
                    v2 = v[0]**2+v[1]**2+v[2]**2
                    B = [0, 0, initial_axial_magnetic_field_strength]
                    e = initial_pressure_outside_pinch/(gamma-1)+initial_density_outside_pinch*v2/2+initial_axial_magnetic_field_strength**2/(2*mu0)
                else:
                    rho = initial_density_in_pinch
                    momentum = initial_momentum_in_pinch(peak_initial_momentum, z, z_side_length)
                    v = [rhov/rho for rhov in momentum]
                    v2 = v[0]**2+v[1]**2+v[2]**2
                    B = [0, 0, initial_axial_magnetic_field_strength]
                    e = initial_pressure_in_pinch/(gamma-1)+initial_density_in_pinch*v2/2+initial_axial_magnetic_field_strength**2/(2*mu0)
                initial_state[i][j][k] = array([rho, momentum[0], momentum[1], momentum[2], B[0], B[1], B[2], e])
    return initial_state

def generate_grid(grid_spacing, xy_side_length, z_side_length):

    '''
    Generates a cubic 3D cartesian grid with the given uniform spacing and the provided side length

    The grid consists of 3 matrices where the first matrix contains the x component, the second matrix contains
    the y component and the third matrix contains the z component. Indices into these grids have the form [y, x, z] 
    (i.e. the first index denotes the y-component, the second index denotes the x-component, and the third component 
    denotes the z-component). Points on the grid can be accessed by indexing into each matrix with the same 3 indices 
    (i.e. (grid[0][1, 2, 3], grid[1][1, 2, 3], grid[2][1, 2, 3]) returns the coordinates of the second (zero-indexing) 
    point in the y-direction, third point in the x-direction, and fourth point in the z-direction. 
    '''
    xy_side = arange(-xy_side_length/2, xy_side_length/2+grid_spacing, grid_spacing)
    z_side = arange(0, z_side_length+grid_spacing, grid_spacing)
    [X, Y, Z] = meshgrid(xy_side, xy_side, z_side)
    return [X, Y, Z]

def generate_time_vector(dt, simulation_time):
    '''
    Generates the time vector for a simulation with the given time step and total time
    '''
    return arange(0, simulation_time+dt, dt) # The extra dt accounts for arange not including the stop value

def generate_vector_arrays(quantity, state):
    '''
    Generates an array of the specified vector quantity from the state array
    '''
    x_component_array = deepcopy(state)
    y_component_array = deepcopy(state)
    z_component_array = deepcopy(state)
    for i in range(len(state)):
        for j in range(len(state[0])):
            for k in range(len(state[0][0])):
                Q = state[i][j][k]
                if quantity == 'Momentum':
                    x_component_array[i][j][k] = Q[1]
                    y_component_array[i][j][k] = Q[2]
                    z_component_array[i][j][k] = Q[3]
                elif quantity == 'Velocity':
                    rho = Q[0]
                    x_component_array[i][j][k] = Q[1]/rho
                    y_component_array[i][j][k] = Q[2]/rho
                    z_component_array[i][j][k] = Q[3]/rho
                elif quantity == 'Magnetic Field':
                    x_component_array[i][j][k] = Q[4]
                    y_component_array[i][j][k] = Q[5]
                    z_component_array[i][j][k] = Q[6]
    return [x_component_array, y_component_array, z_component_array]

def get_coordinates_from_grid_indices(grid, i, j, k):
    '''
    Returns the spatial coordinates corresponding to the provided indices
    '''
    x_coordinate = grid[0][i, j, k]
    y_coordinate = grid[1][i, j, k]
    z_coordinate = grid[2][i, j, k]
    return [x_coordinate, y_coordinate, z_coordinate]

def get_screw_boundary_indices(grid, screw_radius):
    '''
    Returns the x and y indices of grid points to be used as the screw boundary as a list of [x,y] pairs
    
    Does so by sampling a number of points along the exact screw boundary then deciding
    a grid point will be used as the boundary if it is the closest grid point to the sampled point.
    '''
    theta_values = linspace(0, 2*pi, 1000)
    sampled_points = [[screw_radius*cos(theta), screw_radius*sin(theta)] for theta in theta_values]
    X_values = unique(grid[0])
    Y_values = unique(grid[1])
    boundary_indices = []
    for sampled_point in sampled_points:
        x_index = abs(X_values-sampled_point[0]).argmin()
        y_index = abs(Y_values-sampled_point[1]).argmin()
        boundary_point_indices = [x_index, y_index]
        if boundary_point_indices not in boundary_indices:
            boundary_indices.append(boundary_point_indices)
    return boundary_indices

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

def time_step(dt, grid_spacing, screw_boundary_indices, state):
    '''
    Computes a time step of the ideal MHD equations in conservative form using the MacCormack algorithm.

    state is expected to be a 3D array where each element is the state vector at that location
    '''
    dtoverdx = dt/grid_spacing
    new_state = state.copy()
    [F_array, G_array, H_array] = generate_F_G_and_H_arrays(state)
    for i in range(len(state)):
        for j in range(len(state[0])):
            for k in range(len(state[0][0])):
                # Applying periodic boundary conditions since we don't care about the edge region
                if i == 0:
                    y_backward_index = -1
                    y_forward_index = i+1
                elif i == len(state)-1:
                    y_backward_index = i-1
                    y_forward_index = 0
                else:
                    y_backward_index = i-1
                    y_forward_index = i+1
                if j == 0:
                    x_backward_index = -1
                    x_forward_index = j+1
                elif j == len(state[0])-1:
                    x_backward_index = j-1
                    x_forward_index = 0
                else:
                    x_backward_index = j-1
                    x_forward_index = j+1
                if k == 0:
                    z_backward_index = -1
                    z_forward_index = k+1
                elif k == len(state[0][0])-1:
                    z_backward_index = k-1
                    z_forward_index = 0
                else:
                    z_backward_index = k-1
                    z_forward_index = k+1
                Q = state[i][j][k]
                Qbar = (Q
                        -dtoverdx*(F_array[i][x_forward_index][k]-F_array[i][j][k])
                        -dtoverdx*(G_array[y_forward_index][j][k]-G_array[i][j][k])
                        -dtoverdx*(H_array[i][j][z_forward_index]-H_array[i][j][k]))
                Qdoublebar = (Q
                              -dtoverdx*(F_array[i][j][k]-F_array[i][x_backward_index][k])
                              -dtoverdx*(G_array[i][j][k]-G_array[y_backward_index][j][k])
                              -dtoverdx*(H_array[i][j][k]-H_array[i][j][z_backward_index]))
                Qnew = (Qbar+Qdoublebar)/2
                if [j, i] in screw_boundary_indices:
                    rho = Qnew[0]
                    J = Qnew[1:4]
                    Bnew = Qnew[4:7]
                    e = Qnew[7]
                    Bold = Q[4:7]
                    Jcylin = cartesian_to_cylindrical_coordinates(*J)
                    Jcylin[0] = 0 # Applying our wall boundary condition
                    J = cylindrical_to_cartesian_coordinates(*Jcylin)
                    Bnewcylin = cartesian_to_cylindrical_coordinates(*Bnew)
                    Boldcylin = cartesian_to_cylindrical_coordinates(*Bold)
                    Bnewcylin[0] = Boldcylin[0] # Applying our wall boundary condition
                    Bnew = cylindrical_to_cartesian_coordinates(*Bnewcylin)
                    Qnew = [rho, J[0], J[1], J[2], Bnew[0], Bnew[1], Bnew[2], e]
                new_state[i][j][k] = Qnew
    return new_state

def run_simulation(grid, grid_spacing, initial_state, screw_radius, t_vec, show_plot=False, quantity_to_plot='Velocity'):
    '''
    Runs a simulation with the provided parameters
    '''
    dt = t_vec[1]-t_vec[0]
    kinetic_energy_history = []
    safety_factor_history = []
    simulation_states = []
    for t in t_vec:
        if t==0: # Not performing a time step here aligns the time vector with the state after the time step
            state = initial_state
            kinetic_energy_history.append(calculate_kinetic_energy(state))
            safety_factor_history.append(calculate_safety_factor(grid, screw_radius, state))
        else:
            screw_boundary_indices = get_screw_boundary_indices(grid, screw_radius)
            state = time_step(dt, grid_spacing, screw_boundary_indices, state)
            kinetic_energy_history.append(calculate_kinetic_energy(state))
            safety_factor_history.append(calculate_safety_factor(grid, screw_radius, state))
        simulation_states.append(state)
        if show_plot:
            [x_component_array, y_component_array, z_component_array] = generate_vector_arrays(quantity_to_plot, state)
            fig = plt.figure(f'{quantity_to_plot} Plot')
            ax = fig.add_subplot(projection='3d')
            ax.quiver(grid[0], grid[1], grid[2], x_component_array, y_component_array, z_component_array, length=10**-4)
            plt.show(block=False)
            plt.pause(0.05)
    return [kinetic_energy_history, safety_factor_history, simulation_states]

# Task: Run Simulation
grid = generate_grid(grid_spacing, xy_side_length, z_side_length)
initial_state = generate_initial_state(initial_axial_magnetic_field_strength, 
                                       initial_momentum_outside_pinch, 
                                       initial_pressure_in_pinch, 
                                       initial_pressure_outside_pinch, 
                                       grid, 
                                       peak_initial_momentum)
t_vec = generate_time_vector(dt, simulation_time)
[kinetic_energy_history, safety_factor_history, simulation_states] = run_simulation(grid, grid_spacing, initial_state, screw_radius, t_vec, show_plot=False)
plot_quantity_over_time('Kinetic Energy History', kinetic_energy_history, 'Kinetic Energy', t_vec)
plot_quantity_over_time('Safety Factor History', safety_factor_history, 'Safety Factor', t_vec)
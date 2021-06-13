import matplotlib.pyplot as plt
from numpy import linspace, ones, polyfit, zeros

dt = 0.5
num_grid_points = 201
x_domain_endpoints = [0, 200]

def generate_initial_state(grid):
    '''
    Generates an initial state where y(x=10-20)=1 and y(x)=0 everywhere else for the provided grid
    '''
    state = zeros((len(grid), 1))
    state[11:21] = ones((10, 1))
    return state

def generate_grid(num_grid_points, x_domain_endpoints):
    '''
    Generates a grid from the provided domain endpoints and number of grid points
    Note that the result includes both endpoints of the domain
    '''
    return linspace(x_domain_endpoints[0], x_domain_endpoints[1], num_grid_points)

def run_simulation(dt, dx, initial_state, grid, measure_growth=False, scheme='ftcs', show_analytic_solution=False, simulation_name='Linear Wave Equation Simulation'):
    '''
    Runs a simulation of the linear wave equation with unit speed given
    an initial state, time step, grid spacing, and scheme type.

    scheme options are 'ftcs', 'upwind-difference', 'lax', or 'lax-wendroff'
    '''
    if measure_growth:
        max_values = [1]
        t_vec = [0]
    if show_analytic_solution:
        analytic_solution = initial_state
        update_analytic_solution = False
    state = initial_state
    while True:
        plt.plot(grid, state)
        if show_analytic_solution:
            plt.plot(grid, analytic_solution)
            plt.legend([f'{scheme} solution', 'analytic solution'])
            if update_analytic_solution:
                analytic_solution = time_step_analytic_solution(analytic_solution)
            update_analytic_solution = not update_analytic_solution
        plt.xlabel('Space')
        plt.ylabel('Solution')
        plt.title(simulation_name)
        plt.show(block=False)
        plt.pause(0.05)
        plt.clf()
        state = time_step(dt, dx, state, scheme)
        if measure_growth:
            t_vec.append(t_vec[-1]+dt)
            max_values.append(max(state)[0])
        if state[-2] != 0:
            # The solution ends when the pulse hits the right boundary
            break
    if measure_growth:
            coefficients = polyfit(t_vec, max_values, 1)
            print(f'The growth rate for the {scheme} scheme was {coefficients[0]}')
    return state

def time_step(dt, dx, state, scheme='ftcs'):
    '''
    Computes a time step of the linear wave equation with unit speed given a time step, grid spacing,
    and the current state of the solution according to the chosen scheme

    scheme options are 'ftcs', 'upwind-difference', 'lax', or 'lax-wendroff'
    '''
    solution = zeros((len(state), 1))
    for i in range(len(state)):
        if i==0:
            continue
        elif i == len(state)-1:
            continue
        else:
            if scheme == 'ftcs':
                solution[i] = state[i]-(dt/(2*dx))*(state[i+1]-state[i-1])
            elif scheme == 'upwind-difference':
                solution[i] = state[i]-(dt/dx)*(state[i]-state[i-1])
            elif scheme == 'lax':
                solution[i] = state[i]-(dt/(2*dx))*(state[i+1]-state[i-1])+(1/2)*(state[i+1]-2*state[i]+state[i-1])
            elif scheme == 'lax-wendroff':
                solution[i] = state[i]-(dt/(2*dx))*(state[i+1]-state[i-1])+(dt**2/(2*dx**2))*(state[i+1]-2*state[i]+state[i-1])
            else:
                solution[i] = state[i]
    return solution

def time_step_analytic_solution(analytic_solution):
    '''
    This time steps the provided analytic solution assuming a=1 and 1 unit of time has elapsed since the analytic
    solution was last updated
    '''
    solution = zeros((len(analytic_solution), 1))
    for i in range(len(analytic_solution)):
        if i == 0:
            continue
        elif i == len(analytic_solution)-1:
            continue
        else:
            solution[i] = analytic_solution[i-1]
    return solution

grid = generate_grid(201, x_domain_endpoints)
initial_state = generate_initial_state(grid)

dx = grid[1]-grid[0]

# Task 1: Simulation with FTCS and Upwind Differencing

final_state = run_simulation(dt, dx, initial_state, grid, measure_growth=True, scheme='ftcs', show_analytic_solution=True, simulation_name='FTCS Simulation')

final_state = run_simulation(dt, dx, initial_state, grid, scheme='upwind-difference', show_analytic_solution=True, simulation_name='Upwind Difference Simulation')

# Task 2: Simulation with Lax and Lax-Wendroff Methods

final_state = run_simulation(dt, dx, initial_state, grid, scheme='lax', show_analytic_solution=True, simulation_name='Lax Simulation')

final_state = run_simulation(dt, dx, initial_state, grid, scheme='lax-wendroff', show_analytic_solution=True, simulation_name='Lax-Wendroff Simulation')

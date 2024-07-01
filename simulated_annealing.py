import torch
import random
import math

def indicator(interval, yj):
    """
    Indicator Function
    """
    l, u = interval
    greater_than_lower = (yj > l)
    less_than_upper = (yj < u)
    return (greater_than_lower & less_than_upper) #.int()

def R(interval, yj):
    """
    Reflection Function
    """
    lower, upper = interval
    term1 = yj * indicator(interval, yj)
    term2 = (upper - abs(yj-upper)) * indicator((upper, float('inf')), yj)
    term3 = (lower - abs(lower-yj)) * indicator((float('-inf'), lower), yj)
    return term1 + term2 + term3

def simulated_annealing(model_, initial_solution, max_temperature, min_temperature, cooling_rate, num_iterations, interval, sigma=0.1):
    """
    Simulated Annealing
    """

    x = initial_solution
    i = 0
    temperature = max_temperature

    with torch.no_grad():
        best_value = model_(x).item()

    while temperature > min_temperature:
        for _ in range(num_iterations):

            neighbor_solution = x + torch.tensor([random.normalvariate(0, sigma) for _ in range(len(x))], dtype=torch.float32) # Generate
            lower, upper = interval
            newl = lower * 2
            newu = upper * 2

            condition = (neighbor_solution > newl) & (neighbor_solution < newu)

            while not torch.prod(condition).item():
                neighbor_solution = x + torch.tensor([random.normalvariate(0, sigma) for _ in range(len(x))], dtype=torch.float32) # Generate
                condition = (neighbor_solution > newl) & (neighbor_solution < newu)

            neighbor_solution = R(interval, neighbor_solution)

            with torch.no_grad():
                f_x = model_(x).item()
                neighbor_value = model_(neighbor_solution).item()

            # Calculate the acceptance probability
            delta_f = neighbor_value - f_x
            if delta_f < 0:
                q_xy = 1.0 
            else:
                q_xy = math.exp((f_x-neighbor_value) / temperature)
                
            if random.random() < q_xy:
                x = neighbor_solution
                f_x = neighbor_value
        
                if f_x < best_value:
                    best_solution = x
                    best_value = f_x

        # Cool down the temperature
        temperature *= cooling_rate

        # Print the progress
        print(f"Iteration {i+1}: Current Value = {f_x:.10f}, Current Solution = {x}, Temperature = {temperature}") #x.numpy()
        i += 1

    return best_solution, best_value

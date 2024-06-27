import os
import numpy as np
import pandas as pd

def calculate_reward(pump_schedule, state):
    reservoir_capacity = 1000 * 1000  # in liters
    rewards = 0
    current_reservoir_level = state['reservoir_level_percentage_mean'] * reservoir_capacity * 0.01

    for hour, pump_status in enumerate(pump_schedule):
        outflow_rate = state[f'forecast_output_flow_rate_{hour + 1}h'] * 3600  # Convert L/S to L/H
        inflow_rate = state['input_flow_rate_mean'] * 3600  # Convert L/S to L/H

        # Update the reservoir level based on inflow and outflow
        new_reservoir_level = current_reservoir_level + inflow_rate - (outflow_rate * pump_status)

        # Penalize if the reservoir goes below 20% or above 95%
        if new_reservoir_level < 0.2 * reservoir_capacity:
            rewards -= 50
        elif new_reservoir_level > 0.95 * reservoir_capacity:
            rewards -= 50
        else:
            # Reward for keeping the reservoir within the desired range
            rewards += 50

        # Penalize for using the pump during peak hours (18h to 21h)
        if 18 <= (state['hour'] + hour) % 24 <= 21 and pump_status == 1:
            rewards -= 20
        # Reward for using the pump during non-peak hours (0h to 5h)
        elif 0 <= (state['hour'] + hour) % 24 <= 5 and pump_status == 1:
            rewards += 20

        # Reward for maintaining a good time_to_depletion
        if state['time_to_depletion'] < 2:
            rewards -= 20
        elif state['time_to_depletion'] > 10:
            rewards += 100
        elif state['time_to_depletion'] > 6:
            rewards += 50
        elif state['time_to_depletion'] > 4:
            rewards += 20

        # Update the reservoir level
        current_reservoir_level = new_reservoir_level

    return rewards

# Initialize the population with random pump schedules
def initialize_population(size):
    return [np.random.choice([0, 1], size=24) for _ in range(size)]

# Select parents for crossover
def select_parents(population, rewards):
    min_reward = np.min(rewards)
    shifted_rewards = rewards - min_reward + 1  # Shift rewards to make them non-negative
    probabilities = shifted_rewards / np.sum(shifted_rewards)
    parents_indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[parents_indices[0]], population[parents_indices[1]]

# Perform crossover between two parents
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, 23)
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

# Mutate a child's pump schedule
def mutate(child):
    MUTATION_RATE = 0.1
    for i in range(len(child)):
        if np.random.rand() < MUTATION_RATE:
            child[i] = 1 - child[i]
    return child

def find_best_pump_schedule(year, month, day, hour):
    POPULATION_SIZE = 100
    NUM_GENERATIONS = 100

    # Load and preprocess the data
    df = pd.read_parquet(os.path.join(os.path.dirname(__file__),"../../data/silver/states_dataset.parquet"))
    df['pump_active'] = ((df['pump_1_active']) | (df['pump_2_active'])).astype(int)
    df = df.drop(columns=['pump_1_active', 'pump_2_active', 'pump_1_duration_sum', 'pump_2_duration_sum', 'timestamp'])
    df = df.iloc[72:]
    
    timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    day_of_week = timestamp.dayofweek
    week_of_year = timestamp.isocalendar().week
    initial_state = df[(df['hour'] == hour) & (df['day_of_week'] == day_of_week) & (df['week_of_year'] == week_of_year)].iloc[0].to_dict()

    # Run the genetic algorithm
    population = initialize_population(POPULATION_SIZE)
    best_schedule = None
    best_reward = -np.inf

    for generation in range(NUM_GENERATIONS):
        rewards = np.array([calculate_reward(individual, initial_state) for individual in population])
        
        if rewards.max() > best_reward:
            best_reward = rewards.max()
            best_schedule = population[rewards.argmax()]
        
        new_population = []
        
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, rewards)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        
        population = new_population

    return best_schedule, best_reward

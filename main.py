import neuroptica as neu
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from typing import List, Tuple, Dict
import logging
import json
from scipy.stats import unitary_group

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
NUM_WAVEGUIDES = 8
TARGET_MATRIX = unitary_group.rvs(NUM_WAVEGUIDES)  # Example target matrix

class OptimizedMZIMesh:
    def __init__(self, num_waveguides: int):
        self.num_waveguides = num_waveguides

        # Create MZI components for the mesh
        mzis = []
        for i in range(num_waveguides - 1):
            for j in range(num_waveguides - 1 - i):
                mzis.append(neu.MZI(j, j + 1))  # Specify the waveguides connected by this MZI

        # Initialize the MZILayer with the created MZIs
        self.mesh = neu.MZILayer(num_waveguides, mzis=mzis)

    def set_phases(self, phases: np.ndarray) -> None:
        if len(phases) != self.get_num_params():
            raise ValueError("The number of phases provided does not match the number of tunable parameters.")

        index = 0
        for mzi in self.mesh.mzis:
            mzi.theta = phases[index]
            mzi.phi = phases[index + 1]
            index += 2

    def get_transfer_matrix(self) -> np.ndarray:
        return self.mesh.get_transfer_matrix()

    def calculate_fidelity(self, target_matrix: np.ndarray) -> float:
        actual_matrix = self.get_transfer_matrix()
        fidelity = np.abs(np.trace(np.dot(target_matrix.conj().T, actual_matrix))) / self.num_waveguides
        logger.debug(f"Calculated fidelity: {fidelity}")
        return fidelity

    def get_num_params(self) -> int:
        return 2 * len(self.mesh.mzis)

def calculate_rvd(ideal_matrix: np.ndarray, actual_matrix: np.ndarray) -> float:
    rvd = np.linalg.norm(ideal_matrix - actual_matrix, ord='fro') / np.linalg.norm(ideal_matrix, ord='fro')
    logger.debug(f"Calculated RVD: {rvd}")
    return rvd

# Q-learning with epochs
def q_learning_optimize_mesh(mesh: OptimizedMZIMesh, target_matrix: np.ndarray, epochs: int = 10, episodes: int = 1000, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1) -> np.ndarray:
    num_params = mesh.get_num_params()
    current_phases = np.random.uniform(low=-np.pi, high=np.pi, size=num_params)
    best_fidelity = 0
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}")
        print(f"Starting epoch {epoch + 1}")
        q_table = np.zeros((num_params, 2))  # Reset Q-table for each epoch
        mesh.set_phases(current_phases)
        best_fidelity = mesh.calculate_fidelity(target_matrix)

        for episode in range(episodes):
            for i in range(num_params):
                if np.random.rand() < epsilon:
                    action = np.random.choice([0, 1])  # Explore
                else:
                    action = np.argmax(q_table[i])  # Exploit

                # Apply action
                delta = 0.1 if action == 0 else -0.1
                new_phases = np.copy(current_phases)
                new_phases[i] += delta
                mesh.set_phases(new_phases)
                fidelity = mesh.calculate_fidelity(target_matrix)

                # Calculate reward
                reward = fidelity - best_fidelity

                # Update Q-table
                q_table[i, action] = q_table[i, action] + alpha * (reward + gamma * np.max(q_table[i]) - q_table[i, action])

                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    current_phases = new_phases
                    logger.info(f"Epoch {epoch + 1}, Episode {episode}: Improved fidelity to {best_fidelity:.4f}")
                    print(f"Epoch {epoch + 1}, Episode {episode}: Improved fidelity to {best_fidelity:.4f}")

        logger.info(f"End of epoch {epoch + 1}: Best fidelity so far {best_fidelity:.4f}")
        print(f"End of epoch {epoch + 1}: Best fidelity so far {best_fidelity:.4f}")
    
    return current_phases

# Genetic algorithm
def advanced_optimize_mesh(mesh: OptimizedMZIMesh, target_matrix: np.ndarray, generations: int = 500, population_size: int = 100) -> List[float]:
    if 'FitnessMulti' not in creator.__dict__:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if 'Individual' not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -np.pi, np.pi)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=mesh.get_num_params())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual: List[float]) -> Tuple[float, float]:
        mesh.set_phases(np.array(individual))
        fidelity = mesh.calculate_fidelity(target_matrix)
        rvd = calculate_rvd(target_matrix, mesh.get_transfer_matrix())
        return fidelity, rvd

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-np.pi, up=np.pi, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=-np.pi, up=np.pi, eta=20.0, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate the initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Begin the generational process
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        population = toolbox.select(population + offspring, k=population_size)

        # Compile statistics about the new population
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        logger.info(logbook.stream)
        print(logbook.stream)

    best_ind = tools.selBest(population, k=1)[0]
    logger.info(f"Optimization completed. Best individual: {best_ind}")
    print(f"Optimization completed. Best individual: {best_ind}")

    return best_ind

# Optional: Gradient Descent
def gradient_descent_optimize_mesh(mesh: OptimizedMZIMesh, target_matrix: np.ndarray, learning_rate: float = 0.01, epochs: int = 1000) -> np.ndarray:
    num_params = mesh.get_num_params()
    phases = np.random.uniform(low=-np.pi, high=np.pi, size=num_params)
    best_fidelity = mesh.calculate_fidelity(target_matrix)

    for epoch in range(epochs):
        grad = np.zeros(num_params)
        for i in range(num_params):
            delta = 1e-5
            phases[i] += delta
            mesh.set_phases(phases)
            fidelity_plus = mesh.calculate_fidelity(target_matrix)

            phases[i] -= 2 * delta
            mesh.set_phases(phases)
            fidelity_minus = mesh.calculate_fidelity(target_matrix)

            phases[i] += delta
            grad[i] = (fidelity_plus - fidelity_minus) / (2 * delta)  # Central difference approximation

        # Update phases using gradient
        phases += learning_rate * grad
        mesh.set_phases(phases)
        fidelity = mesh.calculate_fidelity(target_matrix)

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            logger.info(f"Epoch {epoch + 1}: Improved fidelity to {best_fidelity:.4f}")
            print(f"Epoch {epoch + 1}: Improved fidelity to {best_fidelity:.4f}")
    
    return phases

# Visualization and reporting functions
def plot_mesh_architecture(mesh: OptimizedMZIMesh) -> None:
    plt.figure(figsize=(12, 8))
    for layer in range(mesh.num_waveguides - 1):
        for index in range(mesh.num_waveguides - layer - 1):
            plt.plot([layer, layer], [index, index + 1], 'r-', linewidth=2)
    plt.title("MZI Mesh Architecture")
    plt.xlabel("Layer")
    plt.ylabel("Waveguide")
    plt.grid(True)
    plt.savefig('mzi_mesh_architecture.png')
    plt.close()
    logger.info("Mesh architecture plot saved as 'mzi_mesh_architecture.png'")
    print("Mesh architecture plot saved as 'mzi_mesh_architecture.png'")

def plot_performance_comparison(initial_performance: Dict[str, float], optimized_performance: Dict[str, float]) -> None:
    labels = ['Initial', 'Optimized']
    fidelity = [initial_performance['fidelity'], optimized_performance['fidelity']]
    rvd = [initial_performance['rvd'], optimized_performance['rvd']]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width / 2, fidelity, width, label='Fidelity', color='b', alpha=0.7)
    rects2 = ax2.bar(x + width / 2, rvd, width, label='RVD', color='r', alpha=0.7)

    ax1.set_ylabel('Fidelity')
    ax2.set_ylabel('RVD')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()
    logger.info("Performance comparison plot saved as 'performance_comparison.png'")
    print("Performance comparison plot saved as 'performance_comparison.png'")

def generate_report(mesh: OptimizedMZIMesh, initial_performance: Dict[str, float],
                    optimized_performance: Dict[str, float]) -> None:
    report = {
        "mesh_info": {
            "num_waveguides": mesh.num_waveguides,
            "num_phase_shifters": mesh.get_num_params(),
        },
        "initial_performance": initial_performance,
        "optimized_performance": optimized_performance,
        "improvement": {
            "fidelity": (optimized_performance['fidelity'] - initial_performance['fidelity']) / initial_performance['fidelity'] * 100,
            "rvd": (initial_performance['rvd'] - optimized_performance['rvd']) / initial_performance['rvd'] * 100
        }
    }

    with open('mzi_mesh_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    logger.info("Report generated and saved as 'mzi_mesh_report.json'")
    print("Report generated and saved as 'mzi_mesh_report.json'")

def main():
    print("Starting MZI mesh optimization")
    logger.info("Starting MZI mesh optimization")

    mesh = OptimizedMZIMesh(NUM_WAVEGUIDES)

    # Initial performance
    initial_phases = np.random.uniform(low=-np.pi, high=np.pi, size=mesh.get_num_params())
    mesh.set_phases(initial_phases)
    initial_fidelity = mesh.calculate_fidelity(TARGET_MATRIX)
    initial_rvd = calculate_rvd(TARGET_MATRIX, mesh.get_transfer_matrix())
    initial_performance = {'fidelity': initial_fidelity, 'rvd': initial_rvd}
    logger.info(f"Initial performance - Fidelity: {initial_fidelity:.4f}, RVD: {initial_rvd:.4f}")
    print(f"Initial performance - Fidelity: {initial_fidelity:.4f}, RVD: {initial_rvd:.4f}")

    # Optimize using genetic algorithm
    optimized_ga_result = advanced_optimize_mesh(mesh, TARGET_MATRIX, generations=500, population_size=100)

    # Optimize using Q-learning with multiple epochs
    optimized_rl_result = q_learning_optimize_mesh(mesh, TARGET_MATRIX, epochs=10)

    # Optimize using gradient descent
    optimized_gd_result = gradient_descent_optimize_mesh(mesh, TARGET_MATRIX)

    # Apply optimized result from the best method
    mesh.set_phases(optimized_rl_result)  # You can switch this to the best result from any method

    # Final performance
    final_fidelity = mesh.calculate_fidelity(TARGET_MATRIX)
    final_rvd = calculate_rvd(TARGET_MATRIX, mesh.get_transfer_matrix())
    optimized_performance = {'fidelity': final_fidelity, 'rvd': final_rvd}
    logger.info(f"Optimized performance - Fidelity: {final_fidelity:.4f}, RVD: {final_rvd:.4f}")
    print(f"Optimized performance - Fidelity: {final_fidelity:.4f}, RVD: {final_rvd:.4f}")

    # Generate reports and plots
    plot_mesh_architecture(mesh)
    plot_performance_comparison(initial_performance, optimized_performance)
    generate_report(mesh, initial_performance, optimized_performance)

if __name__ == "__main__":
    main()

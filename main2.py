!pip install neuroptica 
!pip install deap
import neuroptica as neu
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import logging
import json
from scipy.stats import unitary_group
from scipy.optimize import differential_evolution

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NUM_WAVEGUIDES = 4
TARGET_MATRIX = unitary_group.rvs(NUM_WAVEGUIDES)  

class OptimizedMZIMesh:
    def __init__(self, num_waveguides: int):
        self.num_waveguides = num_waveguides
        mzis = []
        for i in range(num_waveguides - 1):
            for j in range(num_waveguides - 1 - i):
                mzis.append(neu.MZI(j, j + 1))
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

def apply_uncertainties(phases: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    noise = np.random.normal(0, sigma, size=phases.shape)
    return phases * (1 + noise) 

def calculate_rvd(ideal_matrix: np.ndarray, actual_matrix: np.ndarray) -> float:
    numerator = np.linalg.norm(ideal_matrix - actual_matrix, ord='fro')
    denominator = np.linalg.norm(ideal_matrix, ord='fro')
    rvd = numerator / denominator
    logger.debug(f"Calculated RVD: {rvd}")
    return rvd

#for ideal fidelity improvement we want num generations between 1000-3000
def de_optimize_mesh(mesh: OptimizedMZIMesh, target_matrix: np.ndarray, generations: int = 1200, episodes: int = 3000, population_size: int = 150, mutation: float = 0.8, recombination: float = 0.7) -> np.ndarray:
    """
    Optimize the MZI mesh using Differential Evolution.
    """
    def evaluate(phases: np.ndarray) -> float:
        mesh.set_phases(phases)
        fidelity = mesh.calculate_fidelity(target_matrix)
        rvd = calculate_rvd(target_matrix, mesh.get_transfer_matrix())
        return -(fidelity - rvd)  

    bounds = [(-np.pi, np.pi) for _ in range(mesh.get_num_params())]

    result = differential_evolution(evaluate, bounds, maxiter=generations, popsize=population_size, mutation=mutation, recombination=recombination)

    best_solution = result.x
    logger.info(f"DE optimization completed. Best individual: {best_solution}")
    
    return best_solution

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

def generate_report(mesh: OptimizedMZIMesh, initial_performance: Dict[str, float], optimized_performance: Dict[str, float]) -> None:
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

def main():
    logger.info("Starting MZI mesh optimization")
    mesh = OptimizedMZIMesh(NUM_WAVEGUIDES)

    initial_phases = np.random.uniform(low=-np.pi, high=np.pi, size=mesh.get_num_params())
    mesh.set_phases(initial_phases)
    initial_fidelity = mesh.calculate_fidelity(TARGET_MATRIX)
    initial_rvd = calculate_rvd(TARGET_MATRIX, mesh.get_transfer_matrix())
    initial_performance = {'fidelity': initial_fidelity, 'rvd': initial_rvd}
    logger.info(f"Initial performance - Fidelity: {initial_fidelity:.4f}, RVD: {initial_rvd:.4f}")

    optimized_de_result = de_optimize_mesh(mesh, TARGET_MATRIX)

    mesh.set_phases(np.array(optimized_de_result))
    de_fidelity = mesh.calculate_fidelity(TARGET_MATRIX)
    de_rvd = calculate_rvd(TARGET_MATRIX, mesh.get_transfer_matrix())
    logger.info(f"DE optimization result - Fidelity: {de_fidelity:.4f}, RVD: {de_rvd:.4f}")

    final_fidelity = de_fidelity
    final_rvd = de_rvd
    optimized_performance = {'fidelity': final_fidelity, 'rvd': final_rvd}
    logger.info(f"Final optimized performance - Fidelity: {final_fidelity:.4f}, RVD: {final_rvd:.4f}")

    plot_mesh_architecture(mesh)
    plot_performance_comparison(initial_performance, optimized_performance)
    generate_report(mesh, initial_performance, optimized_performance)

if __name__ == "__main__":
    main()

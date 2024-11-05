**Optimizing MZI Mesh Fidelity with Q-Learning, Genetic Algorithms, and Gradient Descent**

This project implements and compares three optimization techniques—Q-Learning, Genetic Algorithms, and Gradient Descent—for tuning the phases of a Mach-Zehnder Interferometer (MZI) mesh. The goal is to maximize the fidelity between the transfer matrix of the MZI mesh and a randomly generated unitary target matrix.

Features
Q-Learning Optimization: Reinforcement learning-based approach to iteratively improve mesh fidelity over multiple epochs.
Genetic Algorithm Optimization: Evolutionary algorithm to search for the optimal set of phases.
Gradient Descent Optimization: Gradient-based optimization for fine-tuning mesh parameters.
Visualization: Generates plots showing the architecture of the MZI mesh and performance comparison before and after optimization.
Reporting: Produces a JSON report that summarizes the optimization results, including fidelity and RVD improvements.
Requirements
Ensure you have the following libraries installed:

bash
Copy code
pip install neuroptica matplotlib deap scipy numpy
How to Run
Clone the repository to your local machine.
Install the required dependencies as listed in the requirements.txt or manually using the above command.
Run the optimization by executing:
bash
Copy code
python main.py
Optimization Techniques
Q-Learning: A reinforcement learning algorithm that explores and exploits phase configurations to improve fidelity over time.
Genetic Algorithm: Uses evolutionary concepts like selection, crossover, and mutation to search for the optimal configuration.
Gradient Descent: A traditional gradient-based method used to adjust the MZI mesh's parameters to achieve higher fidelity.
Visualization and Results
The script saves the following visual outputs:
mzi_mesh_architecture.png: A plot illustrating the MZI mesh's architecture.
performance_comparison.png: A comparison of fidelity and RVD between the initial and optimized mesh configurations.
A detailed JSON report (mzi_mesh_report.json) summarizing the optimization results is generated.
Logging
The script uses Python's logging module to track progress and results throughout the optimization process. Log output is displayed both in the terminal and saved in the log file.

Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you have ideas for improvements.

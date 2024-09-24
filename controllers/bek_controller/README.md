
# Context-Dependent Spatial Representations in the Hippocampus using Place Cell Dendritic Computing

## Citation
A. Alabi, D. Vanderelst and A. A. Minai, "Context-Dependent Spatial Representations in the Hippocampus using Place Cell Dendritic Computation," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892401.

## Overview
This project explores the role of place cells in the hippocampus of rodents, focusing on how they encode physical space and exhibit context-dependent spatial representations. The project includes simulations that model place cells using dendritic computation to demonstrate how they can reuse spatial representations across different contexts while maintaining uncorrelated place fields.

## Usage
1. **Initialize the Driver**: The `Driver` class controls the robot and its interactions with the environment. It handles the loading or initialization of neural networks (place cell network and reward cell network) and manages the robot's sensors and movements.

2. **Startup**: To start the simulation, use the `initialization` method of the `Driver` class.

    Example:
    ```python
    from driver import Driver

    driver = Driver(num_place_cells=1000, num_reward_cells=10, num_head_directions=8, run_time_hours=2, timestep=96)
    driver.initialization(context=0, mode="explore", randomize=True)
    ```

3. **Running the Simulation**: After initializing, use the `run` method to start the simulation in the specified mode.

    Example:
    ```python
    driver.run(mode="explore")
    ```

4. **Customization**: Modify the `Driver` class parameters to suit your simulation needs, such as the number of place cells, reward cells, or the duration of the simulation.

## File Structure
- `driver.py`: Contains the `Driver` class which manages the simulation.
- `networks.py`: Contains neural network definitions used by the driver.
- `requirements.txt`: Lists the dependencies required to run the project.

## Setting Up the Environment

1. **Clone the Git repository**:
   - First, you clone the repository with the following command:
     ```bash
     git clone <repository_url>
     ```
   - Replace `<repository_url>` with the URL of the Git repository you want to clone.

2. **Navigate into the cloned repository's directory**:
   - Once the repository is cloned, navigate into the directory:
     ```bash
     cd <repository_directory>
     ```
   - Replace `<repository_directory>` with the name of the cloned repository.

3. **Create a virtual environment**:
   - Inside the cloned repository directory, create a virtual environment:
     ```bash
     python -m venv venv
     ```
   - This will create a `venv` directory containing the virtual environment.

4. **Activate the virtual environment**:
   - Activate the virtual environment:
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```
     - On Windows:
       ```bash
       .\venv\Scripts\activate
       ```

5. **Install the dependencies from `requirements.txt`**:
   - Once the virtual environment is activated, install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - This command reads the `requirements.txt` file and installs all the Python packages listed in it into the virtual environment.

By following these steps, you ensure that all dependencies are installed in an isolated environment, which helps avoid conflicts with other projects and keeps your development environment clean.

## License
MIT License

Copyright (c) 2024 University of Cincinnati


# Self Organizing Place Cell Navigation Model

## Citations
A. Alabi, D. Vanderelst and A. A. Minai, "Context-Dependent Spatial Representations in the Hippocampus using Place Cell Dendritic Computation," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892401.
B. FILL IN ANY MORE

## Overview
This project explores the role of place cells in the hippocampus of rodents, focusing on how they encode physical space. This model is based off of research that studies place cells in rat hippocampuses. This model aims to further the research of place cell based navigational models.

## Usage
1. **Initialize the Driver**: The `Driver` class controls the robot and its interactions with the environment. It handles the loading or initialization of neural networks (place cell network and reward cell network) and manages the robot's sensors and movements.

2. **Startup**: To start the simulation, use the `initialization` method of the `Driver` class.

    Example:
    ```python
    from driver import Driver
    from enums import RobotMode

    driver = Driver(num_place_cells=200, num_reward_cells=10, num_head_directions=8, run_time_hours=2, timestep=96)
    driver.initialization(mode=RobotMode.LEARN_OJAS, randomize_start_loc=True, run_time_hours=1)
    ```

3. **Running the Simulation**: After initializing, use the `run` method to start the simulation in the mode set in initialization.

    Example:
    ```python
    driver.run()
    ```

4. **Customization**: Modify the `Driver` class parameters to suit your simulation needs, such as the number of place cells, reward cells, or the duration of the simulation.

## File Structure
- `driver.py`: Contains the `Driver` class which manages the simulation.
- `networks/`: Directory that contains neural network definitions used by the driver.
  - `boundary_vector_cell_layer`: Layer that activates cells based on obstacles and their placements in the environment relative to the agent
  - `head_direction_layer`: Layer that activates cells based on the robot's position relative to its starting orientation
  - `place_cell_layer`: Layer that activates cells based on the robot's learned representation of location in the environment
  - `reward_cell_layer`: Layer that associates a set of place cell activations to a reward value
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

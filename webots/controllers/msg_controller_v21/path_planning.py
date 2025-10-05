"""
Path Planning Module for EXPLOIT_LOCATIONS_RANDOM Mode

This module handles:
1. World environment definitions with obstacles
2. Random spawn location generation with constraints
3. A* pathfinding on discretized grids
4. Visualization for validation

Author: Claude Code
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional, Any
import random
import heapq
import os
from pathlib import Path
import math

# Configuration Parameters
GRID_CELL_SIZE = 0.1  # meters per grid cell 
MAX_SPAWN_ATTEMPTS = 1000  # Maximum attempts to find valid spawn location

# World Configuration Definitions
WORLD_CONFIGS = {
    "20x20_multi_goal": {
        "size": [20.0, 20.0],  # [width, height] in meters
        "obstacles": []  # Open environment - no obstacles
    },

    "20x20_cross_multi_goal": {
        "size": [20.0, 20.0],
        "obstacles": [
            {
                "type": "rectangle",
                "name": "horizontal_wall",
                # Original wall: [-6, 0] to [6, 0] with thickness 0.3
                # Bounds: [-6, -0.15] to [6, 0.15]
                "bounds": [[-6.0, -0.15], [6.0, 0.15]]
            },
            {
                "type": "rectangle",
                "name": "vertical_wall",
                # Original wall: [0, -6] to [0, 6] with thickness 0.3
                # Bounds: [-0.15, -6] to [0.15, 6]
                "bounds": [[-0.15, -6.0], [0.15, 6.0]]
            }
        ]
    },

    "20x20_maze_multi_goal": {
        "size": [20.0, 20.0],
        "obstacles": [
            # Internal maze walls extracted from 20x20_maze_multi_goal.wbt
            {
                "type": "rectangle",
                "name": "MazeMid_HorizLeft",
                # Wall at translation -5, z=3, size 10x0.3 (horizontal)
                # Bounds: x=[-10, 0], z=[2.85, 3.15]
                "bounds": [[-10.0, 2.85], [0.0, 3.15]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_HorizRight",
                # Wall at translation 5, z=-3, size 10x0.3 (horizontal)
                # Bounds: x=[0, 10], z=[-3.15, -2.85]
                "bounds": [[0.0, -3.15], [10.0, -2.85]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertLeft",
                # Wall at translation -4, z=-0.5, size 7x0.3 (vertical, rotated 90°)
                # Bounds: x=[-4.15, -3.85], z=[-4.0, 3.0]
                "bounds": [[-4.15, -4.0], [-3.85, 3.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertRight",
                # Wall at translation 3, z=0.5, size 7x0.3 (vertical, rotated 90°)
                # Bounds: x=[2.85, 3.15], z=[-3.0, 4.0]
                "bounds": [[2.85, -3.0], [3.15, 4.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeBottom_VertCenter",
                # Wall at translation 0, z=-8.5, size 3x0.3 (vertical, rotated 90°)
                # Bounds: x=[-0.15, 0.15], z=[-10.0, -7.0]
                "bounds": [[-0.15, -10.0], [0.15, -7.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeTop_VertCenter",
                # Wall at translation -0.5, z=8.5, size 3x0.3 (vertical, rotated 90°)
                # Bounds: x=[-0.65, -0.35], z=[7.0, 10.0]
                "bounds": [[-0.65, 7.0], [-0.35, 10.0]]
            }
        ]
    }
}

class PathPlanner:
    """Main class for handling path planning operations"""

    def __init__(self, world_name: str, wall_clearance: float = 0.2):
        """
        Initialize path planner for specific world

        Args:
            world_name: Name of the world configuration to use
            wall_clearance: Additional clearance around obstacles (meters)
        """
        if world_name not in WORLD_CONFIGS:
            raise ValueError(f"Unknown world: {world_name}. Available: {list(WORLD_CONFIGS.keys())}")

        self.world_name = world_name
        self.world_config = WORLD_CONFIGS[world_name].copy()
        self.wall_clearance = wall_clearance
        self.grid_cell_size = GRID_CELL_SIZE

        # Calculate grid dimensions
        world_size = self.world_config["size"]
        self.grid_width = int(world_size[0] / self.grid_cell_size)
        self.grid_height = int(world_size[1] / self.grid_cell_size)

        # World bounds with clearance applied
        self.usable_bounds = self._calculate_usable_bounds()

        # Initialize grid for pathfinding
        self.grid = self._create_obstacle_grid()

        print(f"[PATH_PLANNER] Initialized for world: {world_name}")
        print(f"[PATH_PLANNER] Grid size: {self.grid_width}x{self.grid_height}")
        print(f"[PATH_PLANNER] Usable bounds: {self.usable_bounds}")
        print(f"[PATH_PLANNER] Wall clearance: {wall_clearance}m")

    def _calculate_usable_bounds(self) -> List[List[float]]:
        """
        Calculate usable world bounds accounting for wall clearance

        Returns:
            [[min_x, min_y], [max_x, max_y]] - usable world bounds
        """
        world_size = self.world_config["size"]
        half_width = world_size[0] / 2
        half_height = world_size[1] / 2

        # Apply wall clearance to environment boundaries
        # World extends from [-half_width, -half_height] to [half_width, half_height]
        usable_bounds = [
            [-half_width + self.wall_clearance, -half_height + self.wall_clearance],
            [half_width - self.wall_clearance, half_height - self.wall_clearance]
        ]

        return usable_bounds

    def _create_obstacle_grid(self) -> np.ndarray:
        """
        Create binary grid marking obstacle cells (True = obstacle, False = free)

        Returns:
            Binary numpy array representing obstacle grid
        """
        # Initialize all cells as free
        grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)

        # Mark obstacle cells
        for obstacle in self.world_config["obstacles"]:
            if obstacle["type"] == "rectangle":
                self._mark_rectangle_obstacle(grid, obstacle)

        return grid

    def _mark_rectangle_obstacle(self, grid: np.ndarray, obstacle: Dict[str, Any]):
        """
        Mark rectangular obstacle cells in grid with clearance applied

        Args:
            grid: Binary grid to modify
            obstacle: Obstacle definition with bounds
        """
        bounds = obstacle["bounds"]
        min_x, min_y = bounds[0]
        max_x, max_y = bounds[1]

        # Apply wall clearance to obstacle bounds
        min_x -= self.wall_clearance
        min_y -= self.wall_clearance
        max_x += self.wall_clearance
        max_y += self.wall_clearance

        # Convert world coordinates to grid coordinates
        world_size = self.world_config["size"]
        half_width = world_size[0] / 2
        half_height = world_size[1] / 2

        # Grid coordinates (0,0) is top-left, world (0,0) is center
        grid_min_x = int((min_x + half_width) / self.grid_cell_size)
        grid_max_x = int((max_x + half_width) / self.grid_cell_size)
        grid_min_y = int((-max_y + half_height) / self.grid_cell_size)  # Y inverted for grid
        grid_max_y = int((-min_y + half_height) / self.grid_cell_size)

        # Clamp to grid bounds
        grid_min_x = max(0, grid_min_x)
        grid_max_x = min(self.grid_width - 1, grid_max_x)
        grid_min_y = max(0, grid_min_y)
        grid_max_y = min(self.grid_height - 1, grid_max_y)

        # Mark obstacle cells
        grid[grid_min_y:grid_max_y+1, grid_min_x:grid_max_x+1] = True

        print(f"[PATH_PLANNER] Marked obstacle '{obstacle.get('name', 'unnamed')}' "
              f"at grid [{grid_min_x}:{grid_max_x}, {grid_min_y}:{grid_max_y}]")

def world_to_grid_coords(world_pos: List[float], world_size: List[float],
                        grid_cell_size: float) -> Tuple[int, int]:
    """
    Convert world coordinates to grid coordinates

    Args:
        world_pos: [x, y] in world coordinates
        world_size: [width, height] of world
        grid_cell_size: Size of each grid cell

    Returns:
        (grid_x, grid_y) tuple
    """
    world_x, world_y = world_pos
    half_width, half_height = world_size[0] / 2, world_size[1] / 2

    grid_x = int((world_x + half_width) / grid_cell_size)
    grid_y = int((-world_y + half_height) / grid_cell_size)  # Y inverted

    return grid_x, grid_y

def grid_to_world_coords(grid_pos: Tuple[int, int], world_size: List[float],
                        grid_cell_size: float) -> List[float]:
    """
    Convert grid coordinates to world coordinates (center of cell)

    Args:
        grid_pos: (grid_x, grid_y) tuple
        world_size: [width, height] of world
        grid_cell_size: Size of each grid cell

    Returns:
        [x, y] in world coordinates
    """
    grid_x, grid_y = grid_pos
    half_width, half_height = world_size[0] / 2, world_size[1] / 2

    world_x = grid_x * grid_cell_size - half_width + grid_cell_size / 2
    world_y = -(grid_y * grid_cell_size - half_height + grid_cell_size / 2)  # Y inverted

    return [world_x, world_y]

def is_within_bounds(pos: List[float], bounds: List[List[float]]) -> bool:
    """
    Check if position is within specified bounds

    Args:
        pos: [x, y] position to check
        bounds: [[min_x, min_y], [max_x, max_y]] bounds

    Returns:
        True if position is within bounds
    """
    x, y = pos
    min_bounds, max_bounds = bounds
    min_x, min_y = min_bounds
    max_x, max_y = max_bounds

    return min_x <= x <= max_x and min_y <= y <= max_y

def is_within_obstacle(pos: List[float], obstacle: Dict[str, Any], wall_clearance: float) -> bool:
    """
    Check if position is within obstacle bounds (including clearance)

    Args:
        pos: [x, y] position to check
        obstacle: Obstacle definition
        wall_clearance: Additional clearance around obstacle

    Returns:
        True if position intersects with obstacle
    """
    if obstacle["type"] == "rectangle":
        bounds = obstacle["bounds"]
        min_x, min_y = bounds[0]
        max_x, max_y = bounds[1]

        # Apply wall clearance
        min_x -= wall_clearance
        min_y -= wall_clearance
        max_x += wall_clearance
        max_y += wall_clearance

        expanded_bounds = [[min_x, min_y], [max_x, max_y]]
        return is_within_bounds(pos, expanded_bounds)

    return False

def is_too_close_to_goal(pos: List[float], goal_location: List[float], min_distance: float) -> bool:
    """
    Check if position is too close to goal location

    Args:
        pos: [x, y] position to check
        goal_location: [x, y] goal position
        min_distance: Minimum allowed distance

    Returns:
        True if position is too close to goal
    """
    distance = math.sqrt((pos[0] - goal_location[0])**2 + (pos[1] - goal_location[1])**2)
    return distance < min_distance

def snap_to_grid(pos: List[float], grid_cell_size: float, world_size: List[float]) -> List[float]:
    """
    Snap position to nearest grid cell center

    Args:
        pos: [x, y] position to snap
        grid_cell_size: Size of grid cells
        world_size: [width, height] of world

    Returns:
        [x, y] position snapped to grid cell center
    """
    # Convert to grid coordinates and back to get cell center
    grid_x, grid_y = world_to_grid_coords(pos, world_size, grid_cell_size)
    return grid_to_world_coords((grid_x, grid_y), world_size, grid_cell_size)

def is_valid_spawn_location(pos: List[float], world_config: Dict[str, Any], goal_location: List[float],
                          min_spawn_distance: float, wall_clearance: float) -> bool:
    """
    Check if position is valid for spawning

    Args:
        pos: [x, y] position to validate
        world_config: World configuration with size and obstacles
        goal_location: [x, y] goal position
        min_spawn_distance: Minimum distance from goal
        wall_clearance: Wall clearance factor

    Returns:
        True if position is valid for spawning
    """
    # Calculate usable world bounds
    world_size = world_config["size"]
    half_width = world_size[0] / 2
    half_height = world_size[1] / 2

    usable_bounds = [
        [-half_width + wall_clearance, -half_height + wall_clearance],
        [half_width - wall_clearance, half_height - wall_clearance]
    ]

    # Check 1: Within usable world bounds
    if not is_within_bounds(pos, usable_bounds):
        return False

    # Check 2: Not within any obstacle
    for obstacle in world_config["obstacles"]:
        if is_within_obstacle(pos, obstacle, wall_clearance):
            return False

    # Check 3: Not too close to goal
    if is_too_close_to_goal(pos, goal_location, min_spawn_distance):
        return False

    return True

def generate_random_position(world_bounds: List[List[float]]) -> List[float]:
    """
    Generate random position within specified bounds

    Args:
        world_bounds: [[min_x, min_y], [max_x, max_y]] bounds

    Returns:
        [x, y] random position within bounds
    """
    min_bounds, max_bounds = world_bounds
    min_x, min_y = min_bounds
    max_x, max_y = max_bounds

    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)

    return [x, y]

def generate_spawn_locations(world_name: str, goals: List[Dict], trials_per_goal: int,
                           min_spawn_distance: float, wall_clearance: float) -> Dict[str, List[List[float]]]:
    """
    Generate random valid spawn locations for each goal

    Args:
        world_name: Name of world configuration
        goals: List of goal definitions with name, location, radius
        trials_per_goal: Number of spawn locations to generate per goal
        min_spawn_distance: Minimum distance from goal (meters)
        wall_clearance: Wall clearance factor (meters)

    Returns:
        Dictionary mapping goal names to lists of spawn locations
    """
    if world_name not in WORLD_CONFIGS:
        raise ValueError(f"Unknown world: {world_name}")

    world_config = WORLD_CONFIGS[world_name]
    world_size = world_config["size"]

    # Calculate bounds for random generation
    half_width = world_size[0] / 2
    half_height = world_size[1] / 2
    generation_bounds = [
        [-half_width + wall_clearance, -half_height + wall_clearance],
        [half_width - wall_clearance, half_height - wall_clearance]
    ]

    spawn_locations = {}

    print(f"[SPAWN_GEN] Generating {trials_per_goal} spawn locations per goal for world: {world_name}")

    for goal in goals:
        goal_name = goal["name"]
        goal_location = goal["location"]
        goal_spawn_locations = []

        print(f"[SPAWN_GEN] Generating spawns for goal '{goal_name}' at {goal_location}")

        attempts = 0
        while len(goal_spawn_locations) < trials_per_goal and attempts < MAX_SPAWN_ATTEMPTS:
            attempts += 1

            # Generate random candidate position
            candidate_pos = generate_random_position(generation_bounds)

            # Validate the candidate position
            if is_valid_spawn_location(candidate_pos, world_config, goal_location,
                                     min_spawn_distance, wall_clearance):
                # Snap to grid and add to list
                snapped_pos = snap_to_grid(candidate_pos, GRID_CELL_SIZE, world_size)
                goal_spawn_locations.append(snapped_pos)

                if len(goal_spawn_locations) % 5 == 0:  # Progress updates
                    print(f"[SPAWN_GEN]   Generated {len(goal_spawn_locations)}/{trials_per_goal} for '{goal_name}'")

        if len(goal_spawn_locations) < trials_per_goal:
            print(f"[SPAWN_GEN] WARNING: Only generated {len(goal_spawn_locations)}/{trials_per_goal} "
                  f"valid spawns for goal '{goal_name}' after {attempts} attempts")

        spawn_locations[goal_name] = goal_spawn_locations

    total_spawns = sum(len(spawns) for spawns in spawn_locations.values())
    print(f"[SPAWN_GEN] Generated {total_spawns} total spawn locations")

    return spawn_locations

class AStarNode:
    """Node for A* pathfinding algorithm"""

    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0, parent=None):
        self.position = position  # (grid_x, grid_y)
        self.g_cost = g_cost      # Cost from start
        self.h_cost = h_cost      # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent      # Parent node for path reconstruction

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two grid positions"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_neighbors(position: Tuple[int, int], grid_width: int, grid_height: int) -> List[Tuple[Tuple[int, int], float]]:
    """
    Get valid neighboring positions with movement costs (8-directional)

    Args:
        position: (grid_x, grid_y) current position
        grid_width: Width of the grid
        grid_height: Height of the grid

    Returns:
        List of ((neighbor_x, neighbor_y), movement_cost) tuples
    """
    x, y = position
    neighbors = []

    # 8-directional movement (including diagonals)
    directions = [
        (0, 1, 1.0),    # North
        (1, 1, 1.414),  # Northeast (sqrt(2))
        (1, 0, 1.0),    # East
        (1, -1, 1.414), # Southeast
        (0, -1, 1.0),   # South
        (-1, -1, 1.414), # Southwest
        (-1, 0, 1.0),   # West
        (-1, 1, 1.414)  # Northwest
    ]

    for dx, dy, cost in directions:
        new_x, new_y = x + dx, y + dy

        # Check bounds
        if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
            neighbors.append(((new_x, new_y), cost))

    return neighbors

def get_goal_cells(goal_location: List[float], goal_radius: float, world_size: List[float],
                  grid_cell_size: float) -> List[Tuple[int, int]]:
    """
    Get all grid cells within goal radius

    Args:
        goal_location: [x, y] goal center
        goal_radius: Goal radius in meters
        world_size: [width, height] of world
        grid_cell_size: Size of grid cells

    Returns:
        List of (grid_x, grid_y) positions within goal radius
    """
    goal_cells = []
    center_grid = world_to_grid_coords(goal_location, world_size, grid_cell_size)
    center_x, center_y = center_grid

    # Calculate search radius in grid cells (add 1 to ensure we don't miss edge cases)
    search_radius = int(goal_radius / grid_cell_size) + 1

    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            candidate_grid = (center_x + dx, center_y + dy)

            # Convert back to world coordinates to check actual distance
            candidate_world = grid_to_world_coords(candidate_grid, world_size, grid_cell_size)
            distance = math.sqrt((candidate_world[0] - goal_location[0])**2 +
                               (candidate_world[1] - goal_location[1])**2)

            if distance <= goal_radius:
                goal_cells.append(candidate_grid)

    return goal_cells

def astar_pathfind(start_pos: List[float], goal_location: List[float], goal_radius: float,
                  planner: PathPlanner) -> Optional[Dict[str, Any]]:
    """
    Find optimal path using A* algorithm

    Args:
        start_pos: [x, y] start position in world coordinates
        goal_location: [x, y] goal center in world coordinates
        goal_radius: Goal radius in meters
        planner: PathPlanner instance with grid and world config

    Returns:
        Dictionary with path info or None if no path found
    """
    world_size = planner.world_config["size"]
    grid_cell_size = planner.grid_cell_size

    # Convert positions to grid coordinates
    start_grid = world_to_grid_coords(start_pos, world_size, grid_cell_size)
    goal_cells = get_goal_cells(goal_location, goal_radius, world_size, grid_cell_size)

    if not goal_cells:
        print(f"[A*] ERROR: No valid goal cells found for goal at {goal_location}")
        return None

    # Check if start position is valid
    start_x, start_y = start_grid
    if (start_x < 0 or start_x >= planner.grid_width or
        start_y < 0 or start_y >= planner.grid_height or
        planner.grid[start_y, start_x]):
        print(f"[A*] ERROR: Invalid start position {start_pos} -> grid {start_grid}")
        return None

    # Initialize A* data structures
    open_set = []
    closed_set = set()
    nodes = {}  # position -> node mapping

    # Create start node
    start_node = AStarNode(start_grid, 0, 0)
    heapq.heappush(open_set, start_node)
    nodes[start_grid] = start_node

    goal_cells_set = set(goal_cells)

    while open_set:
        current = heapq.heappop(open_set)

        # Check if we reached any goal cell
        if current.position in goal_cells_set:
            # Reconstruct path
            path_grid = []
            node = current
            while node:
                path_grid.append(node.position)
                node = node.parent
            path_grid.reverse()

            # Convert path to world coordinates
            path_world = [grid_to_world_coords(pos, world_size, grid_cell_size) for pos in path_grid]

            # Calculate total distance
            total_distance = 0
            for i in range(1, len(path_world)):
                dx = path_world[i][0] - path_world[i-1][0]
                dy = path_world[i][1] - path_world[i-1][1]
                total_distance += math.sqrt(dx*dx + dy*dy)

            return {
                "path_world": path_world,
                "path_grid": path_grid,
                "distance": total_distance,
                "goal_cell": current.position,
                "success": True
            }

        closed_set.add(current.position)

        # Explore neighbors
        neighbors = get_neighbors(current.position, planner.grid_width, planner.grid_height)

        for neighbor_pos, move_cost in neighbors:
            if neighbor_pos in closed_set:
                continue

            # Check if neighbor is obstacle
            neighbor_x, neighbor_y = neighbor_pos
            if planner.grid[neighbor_y, neighbor_x]:
                continue

            tentative_g = current.g_cost + move_cost

            # Check if we've seen this position before
            if neighbor_pos in nodes:
                neighbor_node = nodes[neighbor_pos]
                if tentative_g >= neighbor_node.g_cost:
                    continue
                # Update existing node
                neighbor_node.g_cost = tentative_g
                neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                neighbor_node.parent = current
            else:
                # Create new node
                # Use closest goal cell for heuristic
                min_h = float('inf')
                for goal_cell in goal_cells:
                    h = euclidean_distance(neighbor_pos, goal_cell)
                    min_h = min(min_h, h)

                neighbor_node = AStarNode(neighbor_pos, tentative_g, min_h, current)
                nodes[neighbor_pos] = neighbor_node
                heapq.heappush(open_set, neighbor_node)

    # No path found
    print(f"[A*] No path found from {start_pos} to goal at {goal_location}")
    return None

def calculate_optimal_paths(world_name: str, spawn_goal_combinations: List[Dict],
                          wall_clearance: float) -> List[Dict]:
    """
    Calculate optimal paths for all spawn/goal combinations using A*

    Args:
        world_name: Name of world configuration
        spawn_goal_combinations: List of {start, goal, goal_radius} combinations
        wall_clearance: Wall clearance factor

    Returns:
        List of path results with distance and waypoints
    """
    planner = PathPlanner(world_name, wall_clearance)
    path_results = []

    print(f"[PATH_CALC] Calculating paths for {len(spawn_goal_combinations)} combinations")

    for i, combination in enumerate(spawn_goal_combinations):
        start_pos = combination["start"]
        goal_info = combination["goal"]
        goal_location = goal_info["location"]
        goal_radius = goal_info["radius"]

        print(f"[PATH_CALC] {i+1}/{len(spawn_goal_combinations)}: "
              f"{goal_info['name']} from {start_pos} to {goal_location}")

        path_result = astar_pathfind(start_pos, goal_location, goal_radius, planner)

        if path_result:
            path_results.append({
                "start": start_pos,
                "goal": goal_info,
                "path": path_result["path_world"],
                "distance": path_result["distance"],
                "success": True
            })
        else:
            path_results.append({
                "start": start_pos,
                "goal": goal_info,
                "path": [],
                "distance": float('inf'),
                "success": False
            })

    successful_paths = sum(1 for result in path_results if result["success"])
    print(f"[PATH_CALC] Successfully calculated {successful_paths}/{len(path_results)} paths")

    return path_results

def visualize_trial_setup(world_name: str, start_pos: List[float], goal_info: Dict,
                         path: List[List[float]], trial_info: Dict,
                         output_path: str, wall_clearance: float = 0.2):
    """
    Create visualization of trial setup for validation

    Args:
        world_name: Name of world configuration
        start_pos: Start position [x, y]
        goal_info: Goal information {name, location, radius}
        path: List of waypoints [[x, y], ...]
        trial_info: Trial information for labeling
        output_path: Path to save visualization
        wall_clearance: Wall clearance factor
    """
    # Create path planner to get grid and world info
    planner = PathPlanner(world_name, wall_clearance)
    world_size = planner.world_config["size"]

    # Set up the plot with fixed aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')

    # World bounds
    half_width = world_size[0] / 2
    half_height = world_size[1] / 2

    # Set axis limits with some padding
    padding = 1.0
    ax.set_xlim(-half_width - padding, half_width + padding)
    ax.set_ylim(-half_height - padding, half_height + padding)

    # Draw grid lines for verification (light gray)
    grid_spacing = 1.0  # 1 meter grid lines
    for x in np.arange(-half_width, half_width + grid_spacing, grid_spacing):
        ax.axvline(x, color='lightgray', alpha=0.3, linewidth=0.5)
    for y in np.arange(-half_height, half_height + grid_spacing, grid_spacing):
        ax.axhline(y, color='lightgray', alpha=0.3, linewidth=0.5)

    # Draw environment boundary
    boundary_rect = patches.Rectangle(
        (-half_width, -half_height), world_size[0], world_size[1],
        linewidth=2, edgecolor='black', facecolor='none', label='Environment Boundary'
    )
    ax.add_patch(boundary_rect)

    # Draw usable area boundary (with wall clearance)
    usable_bounds = planner.usable_bounds
    usable_width = usable_bounds[1][0] - usable_bounds[0][0]
    usable_height = usable_bounds[1][1] - usable_bounds[0][1]
    usable_rect = patches.Rectangle(
        (usable_bounds[0][0], usable_bounds[0][1]), usable_width, usable_height,
        linewidth=1, edgecolor='gray', facecolor='none', alpha=0.7, linestyle='--',
        label='Usable Area'
    )
    ax.add_patch(usable_rect)

    # Draw obstacles (red rectangles)
    for obstacle in planner.world_config["obstacles"]:
        if obstacle["type"] == "rectangle":
            bounds = obstacle["bounds"]
            min_x, min_y = bounds[0]
            max_x, max_y = bounds[1]

            # Original obstacle
            orig_width = max_x - min_x
            orig_height = max_y - min_y
            orig_rect = patches.Rectangle(
                (min_x, min_y), orig_width, orig_height,
                facecolor='red', alpha=0.7, edgecolor='darkred',
                label='Obstacle (Original)' if len([o for o in planner.world_config["obstacles"]
                                                   if o["type"] == "rectangle"]) == 1 else ""
            )
            ax.add_patch(orig_rect)

            # Obstacle with clearance (lighter red)
            expanded_min_x = min_x - wall_clearance
            expanded_min_y = min_y - wall_clearance
            expanded_max_x = max_x + wall_clearance
            expanded_max_y = max_y + wall_clearance
            expanded_width = expanded_max_x - expanded_min_x
            expanded_height = expanded_max_y - expanded_min_y

            expanded_rect = patches.Rectangle(
                (expanded_min_x, expanded_min_y), expanded_width, expanded_height,
                facecolor='red', alpha=0.3, edgecolor='red', linestyle='--',
                label='Obstacle + Clearance' if len([o for o in planner.world_config["obstacles"]
                                                    if o["type"] == "rectangle"]) == 1 else ""
            )
            ax.add_patch(expanded_rect)

    # Draw minimum spawn distance circle around goal (light red)
    min_spawn_distance = trial_info.get('min_spawn_distance', 6.0)
    spawn_exclusion_circle = patches.Circle(
        goal_info["location"], min_spawn_distance,
        facecolor='orange', alpha=0.2, edgecolor='orange', linestyle=':',
        label=f'Min Spawn Distance ({min_spawn_distance}m)'
    )
    ax.add_patch(spawn_exclusion_circle)

    # Draw goal area (blue circle)
    goal_circle = patches.Circle(
        goal_info["location"], goal_info["radius"],
        facecolor='blue', alpha=0.6, edgecolor='darkblue', linewidth=2,
        label=f'Goal: {goal_info["name"]} (r={goal_info["radius"]}m)'
    )
    ax.add_patch(goal_circle)

    # Draw start position (green circle)
    ax.plot(start_pos[0], start_pos[1], 'o', color='green', markersize=10,
            markeredgecolor='darkgreen', markeredgewidth=2, label='Start Position')

    # Draw optimal path (yellow line with markers)
    if path and len(path) > 1:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'o-', color='yellow', linewidth=3,
                markersize=4, markeredgecolor='orange', alpha=0.8,
                label=f'Optimal Path')

        # Mark start and end points of path
        ax.plot(path[0][0], path[0][1], 'o', color='lime', markersize=8,
                markeredgecolor='green', markeredgewidth=2)
        ax.plot(path[-1][0], path[-1][1], 'o', color='cyan', markersize=8,
                markeredgecolor='blue', markeredgewidth=2)

    # Set up labels and title
    ax.set_xlabel('X Position (meters)', fontsize=12)
    ax.set_ylabel('Y Position (meters)', fontsize=12)

    # Create detailed title with trial information
    path_distance = trial_info.get('path_distance', 'N/A')
    trial_number = trial_info.get('trial_number', 'N/A')

    title = (f"Trial {trial_number}: {goal_info['name'].title()} Goal\n"
             f"Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f}) → "
             f"Goal: ({goal_info['location'][0]:.1f}, {goal_info['location'][1]:.1f})\n"
             f"Path Distance: {path_distance:.2f}m | World: {world_name}")

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Ensure the plot directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save with high DPI for clear visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[VISUALIZATION] Saved: {output_path}")

def create_all_visualizations(world_name: str, goals: List[Dict], trials_per_goal: int,
                             min_spawn_distance: float, wall_clearance: float,
                             output_dir: str) -> int:
    """
    Generate all spawn locations, calculate paths, and create visualizations

    Args:
        world_name: Name of world configuration
        goals: List of goal definitions
        trials_per_goal: Number of trials per goal
        min_spawn_distance: Minimum spawn distance from goal
        wall_clearance: Wall clearance factor
        output_dir: Directory to save visualizations

    Returns:
        Number of visualizations created
    """
    print(f"[VIS_GEN] Creating visualizations for world: {world_name}")

    # Generate spawn locations
    spawn_locations = generate_spawn_locations(
        world_name, goals, trials_per_goal, min_spawn_distance, wall_clearance
    )

    # Create spawn/goal combinations
    combinations = []
    for goal in goals:
        goal_spawns = spawn_locations.get(goal["name"], [])
        for i, spawn_pos in enumerate(goal_spawns):
            combinations.append({
                "start": spawn_pos,
                "goal": goal,
                "trial_number": i + 1,
                "goal_name": goal["name"]
            })

    # Calculate optimal paths
    path_results = calculate_optimal_paths(world_name, combinations, wall_clearance)

    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    created_count = 0

    for i, (combination, path_result) in enumerate(zip(combinations, path_results)):
        if path_result["success"]:
            trial_info = {
                "trial_number": combination["trial_number"],
                "path_distance": path_result["distance"],
                "min_spawn_distance": min_spawn_distance
            }

            filename = f"{combination['goal_name']}_goal_trial_{combination['trial_number']:03d}.png"
            output_path = os.path.join(output_dir, filename)

            visualize_trial_setup(
                world_name=world_name,
                start_pos=combination["start"],
                goal_info=combination["goal"],
                path=path_result["path"],
                trial_info=trial_info,
                output_path=output_path,
                wall_clearance=wall_clearance
            )
            created_count += 1
        else:
            print(f"[VIS_GEN] Skipping visualization for failed path: "
                  f"{combination['goal_name']} trial {combination['trial_number']}")

    print(f"[VIS_GEN] Created {created_count} visualizations in: {output_dir}")
    return created_count

def save_path_visualizations(world_name: str, combinations: List[Dict[str, Any]],
                           path_results: List[Dict[str, Any]], output_dir: str,
                           wall_clearance: float = 0.5, min_spawn_distance: float = 6.0) -> int:
    """
    Save path visualizations for EXPLOIT_LOCATIONS_RANDOM trials

    Args:
        world_name: Name of the world
        combinations: List of spawn/goal combinations
        path_results: List of path calculation results
        output_dir: Directory to save visualizations
        wall_clearance: Wall clearance factor
        min_spawn_distance: Minimum spawn distance from goal

    Returns:
        Number of visualizations created
    """
    os.makedirs(output_dir, exist_ok=True)
    created_count = 0

    print(f"[PATH_VIS] Saving visualizations to: {output_dir}")

    for i, (combination, path_result) in enumerate(zip(combinations, path_results)):
        if path_result["success"]:
            trial_info = {
                "trial_number": combination["trial_number"],
                "path_distance": path_result["distance"],
                "min_spawn_distance": min_spawn_distance
            }

            filename = f"random_{combination['goal_name']}_trial_{combination['trial_number']:03d}.png"
            output_path = os.path.join(output_dir, filename)

            visualize_trial_setup(
                world_name=world_name,
                start_pos=combination["start"],
                goal_info=combination["goal"],
                path=path_result["path"],
                trial_info=trial_info,
                output_path=output_path,
                wall_clearance=wall_clearance
            )
            created_count += 1
        else:
            print(f"[PATH_VIS] Skipping visualization for failed path: "
                  f"{combination['goal_name']} trial {combination['trial_number']}")

    print(f"[PATH_VIS] Created {created_count} visualizations")
    return created_count

def test_path_planning_system():
    """
    Comprehensive test of the path planning system
    This mimics the parameters that would be passed from the controller
    """
    import time

    print("=" * 80)
    print("PATH PLANNING SYSTEM TEST")
    print("=" * 80)

    # Test parameters matching controller configuration
    test_worlds = ["20x20_multi_goal", "20x20_cross_multi_goal", "20x20_maze_multi_goal"]

    # Goals matching controller configuration
    goals = [
        {"name": "red", "location": [7, 7], "radius": 1.0},
        {"name": "green", "location": [-7, 7], "radius": 1.0},
        {"name": "blue", "location": [7, -7], "radius": 1.0},
        {"name": "yellow", "location": [-7, -7], "radius": 1.0}
    ]

    # Parameters matching expected controller values
    trials_per_goal = 20
    min_spawn_distance = 6.0  # 6 meter minimum distance from goal
    wall_clearance = 0.5      # 0.2 meter clearance around obstacles

    for world_name in test_worlds:
        print(f"\n{'-'*60}")
        print(f"TESTING WORLD: {world_name}")
        print(f"{'-'*60}")

        start_time = time.time()

        try:
            # Create output directory for this world
            output_dir = os.path.join("validation_images", world_name)

            # Run the full pipeline
            num_visualizations = create_all_visualizations(
                world_name=world_name,
                goals=goals,
                trials_per_goal=trials_per_goal,
                min_spawn_distance=min_spawn_distance,
                wall_clearance=wall_clearance,
                output_dir=output_dir
            )

            elapsed_time = time.time() - start_time

            print(f"\n[TEST_RESULTS] {world_name}:")
            print(f"  - Expected trials: {len(goals) * trials_per_goal}")
            print(f"  - Visualizations created: {num_visualizations}")
            print(f"  - Processing time: {elapsed_time:.2f} seconds")
            print(f"  - Output directory: {output_dir}")

            if num_visualizations == len(goals) * trials_per_goal:
                print(f"  - Status: SUCCESS ✓")
            else:
                print(f"  - Status: PARTIAL SUCCESS (some paths failed)")

        except Exception as e:
            print(f"[TEST_ERROR] Failed to process {world_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review generated visualizations in validation_images/ folder")
    print("2. Manually verify:")
    print("   - Environment accuracy (obstacles match Webots worlds)")
    print("   - Spawn location validity (proper clearances)")
    print("   - Path optimality and traversability")
    print("3. If validation passes, integrate with controller")

def main():
    """Main function for standalone testing"""
    print("Path Planning Module - Standalone Test")
    print("This will generate validation visualizations for manual review")

    # Set random seed for reproducible results during testing
    random.seed(42)
    np.random.seed(42)

    try:
        test_path_planning_system()
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
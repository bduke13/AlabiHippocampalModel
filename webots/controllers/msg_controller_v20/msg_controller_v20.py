import sys
import os
import re
import json
from pathlib import Path
from datetime import datetime

# Set project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Import modules
from msg_driver_v20 import Driver
from core.robot.robot_mode import RobotMode
from analysis.stats.stats_collector import stats_collector

##################################################
# MAIN CONFIGURATION - Change These Frequently
##################################################

# Primary Settings (modified most often)
SELECTED_MODE = "LEARN_LOCATIONS"                   # Current mode to run
SCALE_SELECTION = "multiscale"               # Which scale combination to use
START_LOCATION = [-7, 7]                     # Robot starting position [x, y]
TARGET_GOAL = "red"                         # For EXPLOIT_LOCATIONS mode

# Runtime Settings
RUN_TIME_HOURS = 2                          # Maximum runtime per session
MAX_DISTANCE = 25                           # Max sensor distance (adjust for world size)

# Development/Debug Settings  
PLOT_BVC = False                            # Enable BVC plotting (debug only)
RANDOMIZE_START = True                     # Use random start location
TD_LEARNING = False                         # Enable temporal difference learning (experimental)
USE_PROXIMITY_MOD = False                   # Enable proximity modulation (experimental)

##################################################
# Scale Combinations
##################################################

SCALE_COMBINATIONS = {
    "multiscale": ["small", "medium", "large"],    # Primary multi-scale setup
    "small": ["small"],                            # Fine detail only
    "medium": ["medium"],                          # Medium scale only  
    "large": ["large"],                            # Coarse scale only
    "xlarge": ["xlarge"],                          # Global scale only
    "fine_coarse": ["small", "large"],             # Skip medium scale
    "all_scales": ["small", "medium", "large", "xlarge"]  # All 4 scales
}

##################################################
# Goal Definitions
##################################################

# Single goal for most modes
SINGLE_GOAL = {
    "location": [-7, 7],
    "radius": 0.7,
    "name": "default"
}

# Multi-goal setups
MULTI_GOALS = {
    "explore": [  # Larger radius for learning phase
        {"name": "red", "location": [7, 7], "radius": 1.5},
        {"name": "green", "location": [-7, 7], "radius": 1.5},
        {"name": "blue", "location": [7, -7], "radius": 1.5},
        {"name": "yellow", "location": [-7, -7], "radius": 1.5}
    ],
    "exploit": [  # Smaller radius for exploitation phase
        {"name": "red", "location": [7, 7], "radius": 0.5},
        {"name": "green", "location": [-7, 7], "radius": 0.5},
        {"name": "blue", "location": [7, -7], "radius": 0.5},
        {"name": "yellow", "location": [-7, -7], "radius": 0.5}
    ]
}

# Start locations for multi-goal experiments
EXPERIMENT_START_LOCATIONS = [
    [7, -7], [-7, -7], [7, 7], [-7, 7], [0, 0]
]

##################################################
# Robot & Simulation Parameters
##################################################

ROBOT_PARAMS = {
    # Physical robot parameters
    "max_speed": 16,                        # Motor speed
    "wheel_radius": 0.031,                  # Physical wheel radius (meters)
    "axle_length": 0.271756,                # Distance between wheels (meters)
    
    # Sensor parameters
    "lidar_resolution": 720,                # LiDAR points per scan
    
    # Movement control
    "max_turn_degrees": 10.0,               # Max degrees per turn step
    "turn_error_tolerance": 5.0,            # Acceptable heading error (degrees)
    "max_turn_attempts": 30,                # Max turn attempts before giving up
}

SIMULATION_PARAMS = {
    # Timing parameters
    "timestep": 96,                         # Simulation timestep (milliseconds, 32*3)
    "tau_w": 10,                           # Movement steps per exploitation cycle
    "fallback_timeout": 900,                # Exploitation timeout (seconds, 15 minutes)
    
    # Environment parameters
    "max_influence_radius": 3.0,            # Wall influence radius for proximity calculation
}

##################################################
# Exploitation Algorithm Parameters
##################################################

EXPLOITATION_PARAMS = {
    # Core detection thresholds
    "reward_threshold": 0.1,                # Minimum reward to consider a scale valid
    "multistep_threshold": 0.2,             # Threshold for applying multi-step enhancement
    "obstacle_threshold": 0.5,              # Min distance to obstacle for safe movement (meters)
    "goal_activation_threshold": 0.1,       # Min activation for goal cell detection
    
    # Multi-step preplay parameters
    "multistep_steps": 3,                   # Default number of preplay steps
    "multistep_max_scales": 2,              # Max scales to enhance with multi-step
    "enhancement_bonus": 1.2,               # Boost factor for enhanced scales
    "multistep_decay_factor": 0.6,          # Exponential decay for step weighting
    
    # Momentum & direction persistence
    "momentum_type": "bonus",               # 'bonus', 'threshold', or 'none'
    "momentum_strength": 0.3,               # Strength of momentum effect (0.0-1.0)
    "momentum_steps": 15,                   # History length for momentum calculation
    "change_threshold": 0.5,                # Threshold for allowing direction changes
    
    # Loop detection & recovery
    "loop_threshold": 5,                    # Number of loops before forced exploration
    "max_steps_between_loops": 10,          # Steps allowed between loops
    "force_explore_duration": 5,            # Duration of forced exploration
    
    # Goal cell boosting (scale-dependent boost factors)
    "goal_boost_small": 5.0,                # Boost factor for small scale (most accurate)
    "goal_boost_medium": 2.0,               # Boost factor for medium scale
    "goal_boost_large": 1.0,                # Boost factor for large scale
    "goal_boost_xlarge": 1.0,               # Boost factor for xlarge scale (least accurate)
    
    # Debug & development
    "debug_exploit_v0": True,               # Debug output for exploit v0
    "debug_exploit_v1": True,              # Debug output for exploit v1
    "debug_exploit_v2": True,              # Debug output for exploit v2
    "step_decay_factor": 1.0,               # Experimental parameter for future use
}

##################################################
# Neural Network Parameter Definitions 
##################################################

# GLOBAL_DEFS contains default values for all parameters across scales.
# Parameters marked with (*) are typically overridden in SCALE_DEFS for scale-specific tuning.
# Unmarked parameters are usually kept the same across all scales.

# Global defaults with parameter descriptions
GLOBAL_DEFS = {
    # Head Direction 
    "num_hd": 8,                        # Number of head direction cells (more = finer angular resolution)
    
    # Boundary Vector Cells
    "sigma_r": 0.5,                     # * BVC distance tuning width in meters (smaller = sharper distance selectivity)
    "sigma_theta": 1.0,                 # * BVC angular tuning width in degrees (smaller = sharper angular selectivity)
    
    # Grid Cells 
    "num_grid_cells": 800,              # * Number of grid cells (more = richer spatial representation)
    "frequency_divisor": 0.25,          # * Grid spacing control (smaller = tighter grid fields)
    "rotation_range": (0, 360),         # Random rotation range in degrees (larger = more orientation diversity)
    "spread_range": (1.0, 1.0),         # Distance between grid activation peaks
    "translation_factor": 100.0,        # * Random translation range (larger = more spatial coverage)
    "grid_threshold": 0.7,              # Grid cell activation threshold
    "grid_sparsity": None,              # Grid cell sparsity (None=disabled(default), 0.3=keep top 30% defined in layer)
    
    # Place Cells 
    "num_pc": 2000,                     # * Number of place cells (more = finer spatial resolution)
    "gamma_pp": 0.9,                    # * Place cell recurrent inhibition (higher = more selective/sparse)
    "gamma_pb": 0.2,                    # BVC afferent inhibition (higher = less BVC influence)
    "gamma_pg": 0.3,                    # Grid cell afferent inhibition (higher = less grid influence)
    "grid_influence": 0.3,              # * Grid vs BVC ratio, 0=BVC only, 1=grid only (higher = more grid reliance)
    
    # Learning Parameters
    "stdp_lr": 0.05,                    # STDP learning rate when adaptive disabled (higher = faster learning)
    "adaptive_initial_lr": 0.10,        # Adaptive STDP starting rate (higher = faster initial learning)
    "adaptive_final_lr": 0.03,          # Adaptive STDP ending rate (higher = continued plasticity)
    "adaptive_decay_rate": 60,          # * Adaptive decay speed (higher = faster transition to final rate)
    "tau_hd": 0.5,                      # HD eligibility trace time constant (higher = longer trace)
    "correlation_window": 12,           # * Correlation tracking window in timesteps (longer = more stable correlations)
    "correlation_update_freq": 2,      # Correlation matrix update frequency (higher = more frequent updates)
    "correlation_scaling": 2.0,         # Correlation sigmoid scaling factor (higher = more selective)
    "min_correlation_weight": 0.1,      # Minimum connection weight for uncorrelated cells
    "correlation_threshold": 0.01,      # Minimum activation to include in correlation tracking
    "connection_decay_rate": 0.00005,   # Synaptic weight decay rate (higher = faster decay)
    
    # Proximity Suppression
    "proximity_threshold_factor": 1.0,  # Distance threshold as multiple of sigma_r (higher = larger threshold)
    "proximity_suppression_steepness": 10.0, # Sigmoid steepness for suppression curve
    "proximity_suppression_midpoint": 0.5,   # Sigmoid midpoint for suppression curve
    
    # Reward Cells
    "rcn_learning_rate": 0.1,           # RCN learning rate (higher = faster reward learning)
    "replay_timesteps": 20,             # Number of replay steps (more = longer memory consolidation)
    "replay_decay_constant": 6,         # Replay decay rate (higher = slower decay, longer influence)
    "multistep_steps": 2,               # Number of preplay steps in exploit
    
    # Feature Toggles
    "enable_adaptive_stdp": True,       # Use adaptive learning rates vs fixed
    "enable_correlation_weighting": True, # Weight connections by activation correlation
    "enable_proximity_suppression": True, # Suppress grid influence near walls
    "enable_connection_decay": True,    # Apply synaptic weight decay for homeostasis
    "enable_debug_prints": False,       # Print detailed debug information
    "enable_adaptive_logging": False,   # Log adaptive learning data to files
}

# Scale-specific overrides with rationale
SCALE_DEFS = {
    "small": {
        "scale_index": 0,
        "sigma_r": 0.5,                 # Fine spatial resolution
        "num_pc": 2000,                 # High resolution for detailed mapping
        "gamma_pp": 0.9,                # Strong inhibition for sparse, selective firing
        "num_grid_cells": 800,          # Dense grid representation
        "frequency_divisor": 0.25,      # Tight grid spacing for fine detail
        "grid_influence": 0.25,          # Moderate grid influence, boundary-driven accuracy
        "adaptive_decay_rate": 300,     # Fast adaptation for quick fine detail learning
        "correlation_window": 12,       # Longer window for stable fine-scale correlations
        "correlation_scaling": 3.5,
        "min_correlation_weight": 0.03, 
        "correlation_threshold": 0.015,
        "proximity_threshold_factor": 1.0,
        "multistep_steps": 3,  # More steps for fine detail planning
        "replay_decay_constant": 8,
    },
    
    "medium": {
        "scale_index": 1,
        "sigma_r": 1.0,                 # Intermediate spatial scale
        "num_pc": 1000,                 # Balanced resolution
        "gamma_pp": 1.0,                # Moderate inhibition
        "gamma_pg": 0.32,
        "num_grid_cells": 600,          # Medium grid density
        "frequency_divisor": 0.5,       # Medium grid spacing
        "grid_influence": 0.35,         # Balanced grid/boundary influence
        "translation_factor": 200.0, 
        "adaptive_decay_rate": 200,     # Moderate adaptation speed
        "correlation_window": 18,       # Medium correlation window
        "correlation_scaling": 3,
        "min_correlation_weight": 0.06, 
        "correlation_threshold": 0.03,
        "proximity_threshold_factor": 0.5,
        "multistep_steps": 2,           # Balanced planning depth
    },
    
    "large": {
        "scale_index": 2,
        "sigma_r": 1.5,                 # Coarse spatial representation
        "num_pc": 500,                  # Lower resolution for global structure
        "gamma_pp": 1.5,                # Weaker inhibition for broader fields
        "gamma_pb": 0.3, 
        "gamma_pg": 0.32,
        "num_grid_cells": 400,          # Sparser grid for large-scale patterns
        "frequency_divisor": 0.7,       # Large grid spacing
        "grid_influence": 0.4,          # More grid influence for global navigation
        "translation_factor": 400.0,
        "adaptive_decay_rate": 60,      # Slow adaptation for stable patterns
        "correlation_window": 24,       # Shorter window for coarse scale
        "correlation_scaling": 2.5,
        "min_correlation_weight": 0.12, 
        "correlation_threshold": 0.03,
        "proximity_threshold_factor": 0.25,
        "multistep_steps": 1,           # Fewer steps for coarse planning
    },
    
    "xlarge": {
        "scale_index": 3,
        "sigma_r": 2.0,                 # Very coarse global representation
        "num_pc": 250,                  # Minimal cells for global structure
        "gamma_pp": 2.0,                # Very weak inhibition for global coverage
        "num_grid_cells": 300,          # Sparse grid for global patterns
        "frequency_divisor": 1.0,       # Largest grid spacing
        "grid_influence": 0.45,         # High grid influence for global navigation
        "adaptive_decay_rate": 30,      # Slowest adaptation for stable global patterns
        "correlation_window": 30,       # Shortest window for very coarse scale
        "multistep_steps": 1,           # Minimal steps for global planning
    },
}

def merge_scale_parameters(global_defs, scale_defs, scale_names, mode_params):
    """
    Merge global defaults with scale-specific overrides and mode parameters.
    
    Args:
        global_defs: Global parameter defaults
        scale_defs: Scale-specific parameter overrides  
        scale_names: List of scale names to compile
        mode_params: Mode-specific parameters (enable_ojas, enable_stdp, etc.)
    
    Returns:
        List of fully merged scale configurations
    """
    compiled_scales = []
    
    for scale_name in scale_names:
        if scale_name not in scale_defs:
            raise ValueError(f"Unknown scale: {scale_name}")
        
        # Start with complete global defaults
        merged_config = global_defs.copy()
        
        # Apply scale-specific overrides
        merged_config.update(scale_defs[scale_name])
        
        # Apply mode-specific parameters (enable_ojas, enable_stdp, etc.)
        # These should override both global and scale-specific settings
        mode_overrides = {
            'enable_ojas': mode_params.get('enable_ojas'),
            'enable_stdp': mode_params.get('enable_stdp'),
            'clear_files': mode_params.get('clear_files'),
            'run_time_hours': mode_params.get('run_time_hours'),
            'max_dist': mode_params.get('max_dist'),
            'td_learning': mode_params.get('td_learning'),
            'use_prox_mod': mode_params.get('use_prox_mod'),
            'action_mode': mode_params.get('action_mode'),
        }
        
        # Only add non-None mode parameters
        for key, value in mode_overrides.items():
            if value is not None:
                merged_config[key] = value
        
        # Add the scale name for reference
        merged_config["name"] = scale_name
        
        compiled_scales.append(merged_config)
    
    return compiled_scales

def compile_scales(scale_names):
    """Updated compile_scales function using proper parameter merging."""
    return merge_scale_parameters(GLOBAL_DEFS, SCALE_DEFS, scale_names, CURRENT_MODE_PARAMS)

##################################################
# Mode Definitions
##################################################

# Map mode strings to RobotMode enums
MODES_MAP = {
    "OJAS": RobotMode.LEARNING,
    "HEBB": RobotMode.LEARNING, 
    "LEARNING": RobotMode.LEARNING,
    "DMTP": RobotMode.DMTP,
    "DMTP_EXPLOIT": RobotMode.DMTP,
    "EXPLOIT": RobotMode.EXPLOIT,
    "EXPLOIT_SAVE": RobotMode.EXPLOIT,
    "LEARNING_SAVE": RobotMode.EXPLOIT,
    "PLOTTING": RobotMode.PLOTTING,
    "LEARN_LOCATIONS": RobotMode.LEARN_LOCATIONS,
    "EXPLOIT_LOCATIONS": RobotMode.EXPLOIT_LOCATIONS
}

# Base configuration for all modes (reduces repetition)
BASE_CONFIG = {
    "scale_names": SCALE_COMBINATIONS[SCALE_SELECTION],
    "run_time_hours": RUN_TIME_HOURS,
    "max_dist": MAX_DISTANCE,
    "plot_bvc": PLOT_BVC,
    "td_learning": TD_LEARNING,
    "use_prox_mod": USE_PROXIMITY_MOD,
}

# Mode-specific configurations
MODE_CONFIGS = {
    "OJAS": {
        **BASE_CONFIG,
        "enable_ojas": True,
        "enable_stdp": False,
        "clear_files": True,
        "action_mode": "explore",
        "goal_config": {"type": "single", **SINGLE_GOAL},
        "trial_config": {"type": "simple", "count": 1, "start_locations": [START_LOCATION]}
    },
    
    "HEBB": {
        **BASE_CONFIG,
        "enable_ojas": False,
        "enable_stdp": True,
        "clear_files": False,
        "action_mode": "explore",
        "goal_config": {"type": "single", **SINGLE_GOAL},
        "trial_config": {"type": "simple", "count": 1, "start_locations": [START_LOCATION]}
    },
    
    "LEARNING": {
        **BASE_CONFIG,
        "enable_ojas": True,
        "enable_stdp": True,
        "clear_files": True,
        "action_mode": "explore",
        "goal_config": {"type": "single", **SINGLE_GOAL},
        "trial_config": {"type": "simple", "count": 1, "start_locations": [START_LOCATION]}
    },
    
    "PLOTTING": {
        **BASE_CONFIG,
        "enable_ojas": False,
        "enable_stdp": False,
        "clear_files": False,
        "action_mode": "explore",
        "goal_config": {"type": "single", **SINGLE_GOAL},
        "trial_config": {"type": "simple", "count": 1, "start_locations": [START_LOCATION]}
    },
    
    "EXPLOIT": {
        **BASE_CONFIG,
        "enable_ojas": True,
        "enable_stdp": True,
        "clear_files": False,
        "action_mode": "exploit_v1",
        "goal_config": {"type": "single", **SINGLE_GOAL},
        "trial_config": {"type": "simple", "count": 1, "start_locations": [START_LOCATION]}
    },
    
    "LEARN_LOCATIONS": {
        **BASE_CONFIG,
        "enable_ojas": True,
        "enable_stdp": True,
        "clear_files": False,
        "action_mode": "explore",
        "goal_config": {"type": "multi", "goals": MULTI_GOALS["explore"]},
        "trial_config": {"type": "simple", "count": 1, "start_locations": [START_LOCATION]}
    },
    
    "EXPLOIT_LOCATIONS": {
        **BASE_CONFIG,
        "enable_ojas": False,
        "enable_stdp": False,
        "clear_files": False,
        "action_mode": "exploit_locations",
        "save_data": True,
        "goal_config": {
            "type": "multi", 
            "goals": MULTI_GOALS["exploit"], 
            "target_goal": TARGET_GOAL
        },
        "trial_config": {
            "type": "combinations",
            "count": 5,
            "start_locations": EXPERIMENT_START_LOCATIONS,
            "target_goals": ["red", "green", "blue", "yellow"]
        }
    },
    "DMTP": {
        **BASE_CONFIG,
        "enable_ojas": True,
        "enable_stdp": True,
        "clear_files": False,
        "action_mode": "explore",
        "goal_config": {"type": "single", **SINGLE_GOAL},
        "trial_config": {"type": "simple", "count": 1, "start_locations": [START_LOCATION]}
    },
    
    "DMTP_EXPLOIT": {
        **BASE_CONFIG,
        "enable_ojas": True,
        "enable_stdp": True,
        "td_learning": True,
        "clear_files": False,
        "action_mode": "exploit_timed_fallback",
        "goal_config": {"type": "single", **SINGLE_GOAL},
        "trial_config": {"type": "simple", "count": 1, "start_locations": [START_LOCATION]}
    },
    
    "EXPLOIT_SAVE": {
        **BASE_CONFIG,
        "enable_ojas": False,
        "enable_stdp": False,
        "clear_files": False,
        "action_mode": "exploit_v0",
        "save_data": True,
        "goal_config": {"type": "single", **SINGLE_GOAL},
        "trial_config": {"type": "simple", "count": 20, "start_locations": [START_LOCATION]}
    },
    
    "LEARNING_SAVE": {
        **BASE_CONFIG,
        "enable_ojas": True,
        "enable_stdp": True,
        "td_learning": True,
        "clear_files": False,
        "action_mode": "exploit_v2",
        "save_data": True,
        "goal_config": {"type": "single", **SINGLE_GOAL},
        "trial_config": {"type": "simple", "count": 51, "start_locations": [START_LOCATION]}
    }
}

#################################
# Utility Functions
#################################

def save_run_parameters(world_name, selected_mode, mode_params, scale_names, trial_id=None):
    """
    Save all run parameters to a JSON file in the controller's world directory.
    
    Args:
        world_name: Name of the current world
        selected_mode: The selected mode string
        mode_params: Dictionary of parameters for the selected mode
        scale_names: List of scale names being used
        trial_id: Optional trial ID for this run
    """
    # Get the controller directory (where this file is located)
    controller_dir = Path(__file__).resolve().parent
    
    # Create the directory structure in controller/pkl/<world_name>/
    world_dir = os.path.join(controller_dir, "pkl", world_name)
    os.makedirs(world_dir, exist_ok=True)
    
    # Compile the final scale definitions (global + scale-specific + mode overrides)
    compiled_scales = merge_scale_parameters(GLOBAL_DEFS, SCALE_DEFS, scale_names, mode_params)

    # Collect all parameters
    run_parameters = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "selected_mode": selected_mode,
            "trial_id": trial_id,
            "world_name": world_name
        },
        "mode_parameters": mode_params,
        "active_scales": {
            name: SCALE_DEFS[name] for name in scale_names
        },
        "final_compiled_scales": {
            scale_config["name"]: scale_config for scale_config in compiled_scales
        },
        "all_available_scales": SCALE_DEFS,
        "scale_configuration": {
            "selected_scale_names": scale_names,
            "scale_combination_string": "_".join(scale_names)
        },
        "robot_params": ROBOT_PARAMS,
        "simulation_params": SIMULATION_PARAMS,
        "exploitation_params": EXPLOITATION_PARAMS,
        "global_neural_params": GLOBAL_DEFS
    }
    
    # Save to JSON file (overwrites existing file)
    json_path = os.path.join(world_dir, "run_parameters.json")
    with open(json_path, 'w') as f:
        json.dump(run_parameters, f, indent=2, default=str)
    
    print(f"[INFO] Saved run parameters to: {json_path}")
    return json_path

def get_highest_trial_id(stats_folder, start_loc, target_goal=None):
    """
    Reads the stats directory and determines the highest trial ID.
    """
    trial_ids = []
    if os.path.exists(stats_folder):
        # Escape dots in float coordinates for regex
        start_x = str(start_loc[0]).replace('.', r'\.')
        start_y = str(start_loc[1]).replace('.', r'\.')
        
        for file_name in os.listdir(stats_folder):
            if target_goal:
                # Pattern: trial_X_start_Y_Z_goal_GOALNAME
                pattern = rf"trial_(\d+)_start_{start_x}_{start_y}_goal_{target_goal}"
            else:
                # Pattern: trial_X_start_Y_Z (no goal suffix)
                pattern = rf"trial_(\d+)_start_{start_x}_{start_y}(?!_goal)"
            
            match = re.match(pattern, file_name)
            if match:
                trial_ids.append(int(match.group(1)))
    return max(trial_ids) if trial_ids else 0

def get_world_name(bot):
    """
    Determines the current world name dynamically from the .wbt file.
    """
    world_path = bot.getWorldPath()
    return os.path.basename(world_path).replace('.wbt', '')

def _run_simple_trials(mode, trial_config, **kwargs):
    """Handle simple trial execution"""
    start_locations = trial_config["start_locations"]
    trials_per_start = trial_config["count"]
    save_data = kwargs.get("save_data", False)
    
    bot = Driver()
    world_name = get_world_name(bot)
    
    # Setup stats if needed
    if save_data and mode == RobotMode.EXPLOIT:
        scale_names = kwargs.get("scale_names", [])
        scale_name_str = "_".join(scale_names)
        stats_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats", world_name, scale_name_str, "JSON"
        )
        os.makedirs(stats_folder, exist_ok=True)
        stats_collector_instance = stats_collector(output_dir=stats_folder)
    else:
        stats_collector_instance = None
    
    for start_loc in start_locations:
        for trial_num in range(1, trials_per_start + 1):
            # Skip if goal and start are the same (only for exploitation modes)
            goal_config = kwargs.get("goal_config", {})
            action_mode = kwargs.get("action_mode", "")

            # Only skip for exploitation modes, not exploration/learning modes
            skip_same_location = (
                goal_config.get("type") == "single" and
                goal_config["location"] == start_loc and
                "exploit" in action_mode.lower()
            )

            if skip_same_location:
                print(f"[INFO] Skipping trial - start location {start_loc} equals goal location (exploitation mode)")
                continue
            
            # Get existing trial count
            if save_data:
                current_trial_id = get_highest_trial_id(stats_folder, start_loc)
                trial_id = f"trial_{current_trial_id + 1}_start_{start_loc[0]}_{start_loc[1]}"
            else:
                trial_id = f"trial_{trial_num}_start_{start_loc[0]}_{start_loc[1]}"
            
            print(f"[INFO] Running trial: {trial_id}")
            
            # Run single trial
            _run_single_trial(bot, mode, trial_id, start_loc, None, stats_collector_instance, **kwargs)

    # Pause simulation after all trials complete
    print("[INFO] All trials completed - pausing simulation")
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

def _run_combination_trials(mode, trial_config, **kwargs):
    """Handle combination trial execution (start locations x target goals)"""
    start_locations = trial_config["start_locations"]
    target_goals = trial_config.get("target_goals", [None])
    trials_per_combo = trial_config["count"]
    save_data = kwargs.get("save_data", False)
    
    bot = Driver()
    world_name = get_world_name(bot)
    
    # Setup stats if needed
    print(f"[DEBUG] _run_combination_trials: save_data={save_data}, mode={mode}")
    if save_data:
        scale_names = kwargs.get("scale_names", [])
        scale_name_str = "_".join(scale_names)
        stats_folder = os.path.join(
            PROJECT_ROOT, "analysis", "stats", world_name, scale_name_str, "JSON"
        )
        os.makedirs(stats_folder, exist_ok=True)
        stats_collector_instance = stats_collector(output_dir=stats_folder)
        print(f"[DEBUG] Created stats_collector for folder: {stats_folder}")
    else:
        stats_collector_instance = None
        print(f"[DEBUG] No stats_collector created (save_data=False)")
    
    # Track overall trial progress
    trial_count = 0
    total_trials = 0

    # Calculate total expected trials for progress tracking
    for start_loc in start_locations:
        for target_goal in target_goals:
            if target_goal:
                goal_config = kwargs.get("goal_config", {})
                goals = goal_config.get("goals", [])
                goal_location = next((g["location"] for g in goals if g["name"] == target_goal), None)
                if not (goal_location and goal_location == start_loc):
                    total_trials += trials_per_combo
            else:
                total_trials += trials_per_combo

    print(f"[INFO] Expected total trials: {total_trials}")

    # Run the trial combinations with soft reset approach
    first_trial = True
    for start_loc in start_locations:
        for target_goal in target_goals:
            # Skip if goal and start are the same (for goal-specific trials)
            if target_goal:
                goal_config = kwargs.get("goal_config", {})
                goals = goal_config.get("goals", [])
                goal_location = next((g["location"] for g in goals if g["name"] == target_goal), None)
                if goal_location and goal_location == start_loc:
                    print(f"[INFO] Skipping trial - start location {start_loc} equals goal {target_goal} location")
                    continue

            for trial_num in range(1, trials_per_combo + 1):
                trial_count += 1

                # Get existing trial count
                if save_data:
                    current_trial_id = get_highest_trial_id(stats_folder, start_loc, target_goal)
                    if target_goal:
                        trial_id = f"trial_{current_trial_id + 1}_start_{start_loc[0]}_{start_loc[1]}_goal_{target_goal}"
                    else:
                        trial_id = f"trial_{current_trial_id + 1}_start_{start_loc[0]}_{start_loc[1]}"
                else:
                    if target_goal:
                        trial_id = f"trial_{trial_num}_start_{start_loc[0]}_{start_loc[1]}_goal_{target_goal}"
                    else:
                        trial_id = f"trial_{trial_num}_start_{start_loc[0]}_{start_loc[1]}"

                print(f"[INFO] Running trial {trial_count}/{total_trials}: {trial_id}")
                print(f"[DEBUG] About to start trial {trial_num}/{trials_per_combo} for start_loc={start_loc}, target_goal={target_goal}")

                if first_trial:
                    # Run first trial normally with full initialization
                    _run_single_trial(bot, mode, trial_id, start_loc, target_goal, stats_collector_instance, **kwargs)
                    first_trial = False
                else:
                    # Use soft reset for subsequent trials
                    print(f"[DEBUG] Using soft reset for trial {trial_count}")
                    bot.reset_for_next_trial(start_loc, trial_id, target_goal, stats_collector_instance)

                    # Run the trial (bot.run() will use the reset state)
                    print(f"[DEBUG] Starting bot.run() for trial {trial_id}")
                    bot.run()
                    print(f"[DEBUG] bot.run() completed for trial {trial_id}")

                print(f"[DEBUG] Completed trial {trial_count}/{total_trials} for start_loc={start_loc}, target_goal={target_goal}")

    # Pause simulation after all trials complete
    print("[INFO] All trials completed - pausing simulation")
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

def _run_single_trial(bot, mode, trial_id, start_loc, target_goal, stats_collector_instance, **kwargs):
    """Run a single trial"""
    
    # Update goal config for this trial if target_goal specified
    trial_kwargs = kwargs.copy()
    if target_goal and "goal_config" in trial_kwargs:
        trial_kwargs["goal_config"]["target_goal"] = target_goal
    
    # Get scale info
    scale_names = kwargs.get("scale_names", [])
    scales_list = compile_scales(scale_names)
    
    world_name = get_world_name(bot)
    
    # Initialize bot for this trial
    bot.initialization(
        mode=mode,
        scales=scales_list,
        robot_params=ROBOT_PARAMS,
        simulation_params=SIMULATION_PARAMS,
        exploitation_params=EXPLOITATION_PARAMS,
        stats_collector=stats_collector_instance,
        trial_id=trial_id,
        world_name=world_name,
        start_loc=start_loc,
        **trial_kwargs  # Pass all remaining parameters including goal_config, trial_config, clear_files, run_time_hours
    )
    
    bot.trial_id = trial_id
    
    # Save parameters for this trial
    save_run_parameters(
        world_name=world_name,
        selected_mode=CURRENT_SELECTED_MODE,
        mode_params=CURRENT_MODE_PARAMS,
        scale_names=scale_names,
        trial_id=trial_id
    )
    
    # Run the trial
    print(f"[DEBUG] Starting bot.run() for trial {trial_id}")
    bot.run()
    print(f"[DEBUG] bot.run() completed for trial {trial_id}")

    # NOTE: Removed worldReload() - using soft reset approach instead to avoid Webots controller restart issues

def run_bot(mode, **kwargs):
    """
    Runs the bot in the specified mode with the given parameters.
    Supports both simple and combination trial configurations.
    """
    # Get trial configuration and remove it from kwargs to avoid conflict
    trial_config = kwargs.pop("trial_config", {
        "type": "simple", 
        "count": 1, 
        "start_locations": [[0, 0]]
    })
    
    print(f"[INFO] Starting run with mode: {mode}")
    print(f"[INFO] Trial type: {trial_config['type']}")
    
    if trial_config["type"] == "simple":
        _run_simple_trials(mode, trial_config, **kwargs)
    elif trial_config["type"] == "combinations":
        _run_combination_trials(mode, trial_config, **kwargs)
    else:
        raise ValueError(f"Unknown trial type: {trial_config['type']}")

    # Note: Simulation pausing is handled within the trial functions

##################################################
# Execution
##################################################

# Global variables to hold current run info (for parameter saving)
CURRENT_SELECTED_MODE = None
CURRENT_MODE_PARAMS = None

if __name__ == "__main__":
    # Validate configuration
    if SELECTED_MODE not in MODE_CONFIGS or SELECTED_MODE not in MODES_MAP:
        print(f"Error: Invalid mode '{SELECTED_MODE}' selected.")
        print(f"Available modes: {list(MODE_CONFIGS.keys())}")
        sys.exit(1)
    
    if SCALE_SELECTION not in SCALE_COMBINATIONS:
        print(f"Error: Invalid scale selection '{SCALE_SELECTION}'.")
        print(f"Available combinations: {list(SCALE_COMBINATIONS.keys())}")
        sys.exit(1)
    
    # Set global variables for parameter saving
    CURRENT_SELECTED_MODE = SELECTED_MODE
    CURRENT_MODE_PARAMS = MODE_CONFIGS[SELECTED_MODE]

    # Lookup the RobotMode enum and the parameter set
    mode_enum = MODES_MAP[SELECTED_MODE]
    params = MODE_CONFIGS[SELECTED_MODE]
    
    print(f"[INFO] Starting run with mode: {SELECTED_MODE}")
    print(f"[INFO] Scale names: {params['scale_names']}")
    print(f"[INFO] Start location: {START_LOCATION}")
    
    # Now call run_bot with all parameters from the dictionary
    run_bot(mode_enum, **params)

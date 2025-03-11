"""trial_generator.py - Flexible parameter sweep framework for grid cell experiments."""
import numpy as np
import itertools
from typing import Dict, List, Any, Callable, Union, Tuple, Optional
from core.robot.robot_mode import RobotMode

def linspace_param(start: float, stop: float, num: int) -> List[float]:
    """Create linearly spaced parameter values, rounded to 3 decimal places."""
    return [round(float(x), 3) for x in np.linspace(start, stop, num)]

def logspace_param(start: float, stop: float, num: int, base: float = 10) -> List[float]:
    """Create logarithmically spaced parameter values, rounded to 3 decimal places."""
    return [round(float(x), 3) for x in np.logspace(np.log10(start), np.log10(stop), num, base=base)]

def default_naming(variation: Dict[str, Any]) -> str:
    """Generate trial name from varied parameters."""
    parts = []
    for key, value in sorted(variation.items()):
        # Format parameter value nicely
        if isinstance(value, float):
            formatted_val = f"{value:.2f}".rstrip('0').rstrip('.')
        elif isinstance(value, tuple):
            formatted_val = f"{value[0]}-{value[1]}"
        else:
            formatted_val = str(value)
        parts.append(f"{key}_{formatted_val}")
    return "_".join(parts)

def generate_trials(
    base_params: Dict[str, Any], 
    param_variations: Dict[str, List[Any]], 
    strategy: str = "grid", 
    naming_function: Optional[Callable] = None
) -> List[Dict[str, Any]]:
    """
    Generate trial configurations by combining base parameters with variations.
    
    Args:
        base_params: Dictionary of default parameters used for all trials
        param_variations: Dictionary where keys are parameter names and values are 
                         lists of values to try for that parameter
        strategy: Strategy for exploring parameter space:
                 - "grid": Full grid search (all combinations)
                 - "one-at-a-time": Vary one parameter at a time from base params
        naming_function: Function to generate trial names based on varied parameters
    
    Returns:
        List of trial dictionaries
    """
    if not naming_function:
        naming_function = default_naming
    
    trials = []
    
    if strategy == "grid":
        # Full grid search (all combinations)
        param_names = list(param_variations.keys())
        value_combinations = itertools.product(*(param_variations[param] for param in param_names))
        
        for values in value_combinations:
            variation = dict(zip(param_names, values))
            trial = base_params.copy()
            trial.update(variation)
            
            # Generate trial name
            trial["trial_name"] = naming_function(variation)
            trials.append(trial)
            
    elif strategy == "one-at-a-time":
        # Vary one parameter at a time from base configuration
        for param_name, values in param_variations.items():
            for value in values:
                variation = {param_name: value}
                trial = base_params.copy()
                trial.update(variation)
                
                # Generate trial name
                trial["trial_name"] = naming_function(variation)
                trials.append(trial)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return trials
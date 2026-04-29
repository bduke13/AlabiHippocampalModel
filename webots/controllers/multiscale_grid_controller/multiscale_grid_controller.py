import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTROLLER_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, CONTROLLER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.robot.robot_mode import RobotMode
from multiscale_grid_driver import DEFAULT_SCALES, Driver


def run_bot(mode: RobotMode, **kwargs) -> Driver:
    bot = Driver()
    bot.initialization(mode=mode, **kwargs)
    bot.run()
    return bot


if __name__ == "__main__":
    modes = {
        "LEARN_OJAS": RobotMode.LEARN_OJAS,
        "LEARN_OJAS_AUTO": RobotMode.LEARN_OJAS_AUTO,
        "LEARN_HEBB": RobotMode.LEARN_HEBB,
        "LEARNING": RobotMode.LEARNING,
        "DMTP": RobotMode.DMTP,
        "LEARN_LOCATIONS": RobotMode.LEARN_LOCATIONS,
        "LEARN_LOCATIONS_COVERAGE": RobotMode.LEARN_LOCATIONS_COVERAGE,
        "LEARN_LOCATIONS_COVERAGE_AUTO": RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
        "LEARN_LOCATIONS_TWO_PHASE": RobotMode.LEARN_LOCATIONS_TWO_PHASE,
        "LEARN_LOCATIONS_ADAPTIVE": RobotMode.LEARN_LOCATIONS_ADAPTIVE,
        "EXPLOIT": RobotMode.EXPLOIT,
        "EXPLOIT_LOCATIONS": RobotMode.EXPLOIT_LOCATIONS,
        "EXPLOIT_LOCATIONS_RANDOM": RobotMode.EXPLOIT_LOCATIONS_RANDOM,
        "EXPLOIT_LOCATIONS_RANDOM_AUTO": RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
        "PLOTTING": RobotMode.PLOTTING,
        "PLOTTING_AUTO": RobotMode.PLOTTING_AUTO,
        "PLOTTING_COVERAGE": RobotMode.PLOTTING_COVERAGE_AUTO,
        "PLOTTING_COVERAGE_AUTO": RobotMode.PLOTTING_COVERAGE_AUTO,
        "MANUAL": RobotMode.MANUAL_CONTROL,
    }
    selected = "EXPLOIT_LOCATIONS_RANDOM"
    run_bot(
        modes[selected],
        run_time_hours=15.0,
        start_loc=[1.0, 1.0],
        randomize_start_loc=False,
        scale_configs=DEFAULT_SCALES,
        reset_networks=False,
        use_cuda=False,
        max_dist=25.0,
        pcn_recurrent_inhibition_mode="all_scales",
        pcn_learning_stdp_start_steps=0,
        pcn_cross_scale_inhibition_base_enabled=True,
        pcn_cross_scale_inhibition_gaussian_enabled=True,
        enable_live_diagnostics=True,
        use_bvc_context_modulation=True,
        bvc_context_modulation_mode="pcn_excitation",
        bvc_context_gain_floor=0.10,
        bvc_context_gain_strength=1.0,
        environment_size=[20.0, 20.0],
        grid_size=0.5,
        coverage_percentage=0.95,
        minimum_learning_duration_hours=4.0,
        min_goal_visits=5,
        two_phase_learning=True,
        two_phase_keep_ojas_in_phase2=True,
        phase1_min_steps=2500,
        phase1_max_steps=20000,
        phase1_bin_size=0.5,
        phase1_min_revisit_bins=15,
        phase1_revisit_cosine_threshold=0.90,
        phase1_revisit_window=200,
        trials_per_goal=20,
        min_spawn_distance=6.0,
        wall_clearance=0.5,
        path_failure_ratio=10.0,
        auto_trial_name="GC_FIXED",
        num_auto_trials=5,
        current_auto_trial=1,
        tau_w=10,
        unified_recurrent_preplay_horizon=2,
        unified_exploit_step_distance=0.25,
        unified_preplay_commit_threshold=0.30,
        unified_exploit_action_mode="stochastic_preplay_weighted",
    )

"""Robot modes for multiscale_grid_controller_v2.

This module defines a v2-facing behavior enum and a compatibility mapping to the
legacy RobotMode enum used in the runtime driver implementation.

Batch/auto execution is configured separately via execution_config in v2 and is
not represented as part of the public mode enum.
"""

from enum import Enum, auto

from core.robot.robot_mode import RobotMode as LegacyRobotMode


class RobotModeV2(Enum):
    LEARN_OJAS = auto()
    LEARN_HEBB = auto()
    LEARN_LOCATIONS = auto()
    PLOTTING = auto()
    BENCHMARK = auto()


V2_TO_LEGACY_MODE = {
    RobotModeV2.LEARN_OJAS: LegacyRobotMode.LEARN_OJAS,
    RobotModeV2.LEARN_HEBB: LegacyRobotMode.LEARN_HEBB,
    # Use coverage-capable learning runtime path for v2 LEARN_LOCATIONS.
    RobotModeV2.LEARN_LOCATIONS: LegacyRobotMode.LEARN_LOCATIONS_COVERAGE,
    RobotModeV2.PLOTTING: LegacyRobotMode.PLOTTING,
    # Benchmark runs with both Oja/STDP enabled and no goals; legacy runtime mode can be LEARN_HEBB.
    RobotModeV2.BENCHMARK: LegacyRobotMode.LEARN_HEBB,
}


def to_legacy_mode(mode: RobotModeV2) -> LegacyRobotMode:
    if mode not in V2_TO_LEGACY_MODE:
        raise ValueError(f"Unsupported RobotModeV2: {mode}")
    return V2_TO_LEGACY_MODE[mode]

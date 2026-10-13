"""Keep PHP control timing independent of IsaacSim rendering."""

from __future__ import annotations

import dataclasses
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holosoma.config_types.experiment import ExperimentConfig


CORE_ISAACSIM = "holosoma.simulator.isaacsim.isaacsim.IsaacSim"
PHP_ISAACSIM = "wbt_training.simulators.isaacsim.IsaacSim"


def use_php_isaacsim(config: ExperimentConfig) -> ExperimentConfig:
    """Select PHP's adapter after CLI parsing, including for old checkpoints.

    Keep this module import-light: the native simulator is imported only when
    Holosoma constructs the environment after starting the IsaacSim app.
    """
    if config.simulator._target_ != CORE_ISAACSIM:
        return config
    return dataclasses.replace(
        config,
        simulator=dataclasses.replace(config.simulator, _target_=PHP_ISAACSIM),
    )


def install_physics_clock_fix(context) -> None:
    """Split stepping and rendering on this PHP simulation context only.

    IsaacSim 5.1's step(render=True) calls app.update(), which can advance an
    entire rendering interval of physics. IsaacLab's render() pauses physics
    during the graphics update, so one call below always means one physics tick.
    """
    original_step = context.step
    if getattr(original_step, "_php_physics_clock_fixed", False):
        return

    @wraps(original_step)
    def step(render=True, *args, **kwargs):
        result = original_step(False, *args, **kwargs)
        if render:
            context.render()
        return result

    step._php_physics_clock_fixed = True
    context.step = step

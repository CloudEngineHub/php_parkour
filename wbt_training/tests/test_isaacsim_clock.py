"""PHP's rendering adapter must not change the physics clock or other backends."""

import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from wbt_training.utils.isaacsim_clock import (
    CORE_ISAACSIM,
    PHP_ISAACSIM,
    install_physics_clock_fix,
    use_php_isaacsim,
)


@dataclass(frozen=True)
class Simulator:
    _target_: str = CORE_ISAACSIM


@dataclass(frozen=True)
class Experiment:
    simulator: Simulator = Simulator()
    checkpoint: str = "model_19999.pt"


class Context:
    """Model the observed 5 ms physics / 20 ms rendering interval."""

    def __init__(self):
        self.ticks = 0
        self.renders = 0
        self.fail = False

    def step(self, render=True):
        if self.fail:
            raise RuntimeError("physics failed")
        self.ticks += 4 if render else 1
        return "stepped"

    def render(self):
        self.renders += 1


class IsaacSimClockTests(unittest.TestCase):
    def test_one_control_update_stays_four_ticks_with_recording(self):
        context = Context()
        install_physics_clock_fix(context)
        for render in (False, False, False, True):
            self.assertEqual(context.step(render=render), "stepped")
        self.assertEqual((context.ticks, context.renders), (4, 1))

    def test_headless_clock_and_idempotent_installation(self):
        context = Context()
        install_physics_clock_fix(context)
        wrapped = context.step
        install_physics_clock_fix(context)
        self.assertIs(context.step, wrapped)
        for _ in range(4):
            context.step(False)
        self.assertEqual((context.ticks, context.renders), (4, 0))

    def test_already_split_backend_does_not_render_twice(self):
        context = Context()
        install_physics_clock_fix(context)
        context.step(render=False)
        context.render()
        self.assertEqual((context.ticks, context.renders), (1, 1))

    def test_correction_is_instance_local(self):
        php = Context()
        other = Context()
        original_class_method = Context.step
        install_physics_clock_fix(php)
        php.step()
        other.step()
        self.assertEqual((php.ticks, other.ticks), (1, 4))
        self.assertIs(Context.step, original_class_method)

    def test_physics_exception_does_not_render(self):
        context = Context()
        context.fail = True
        install_physics_clock_fix(context)
        with self.assertRaisesRegex(RuntimeError, "physics failed"):
            context.step()
        self.assertEqual(context.renders, 0)

    def test_routing_preserves_saved_config_and_other_backends(self):
        saved = Experiment()
        routed = use_php_isaacsim(saved)
        self.assertEqual(routed.simulator._target_, PHP_ISAACSIM)
        self.assertEqual(routed.checkpoint, saved.checkpoint)
        self.assertEqual(saved.simulator._target_, CORE_ISAACSIM)
        self.assertIs(use_php_isaacsim(routed), routed)
        mujoco = Experiment(Simulator("holosoma.simulator.mujoco.mujoco.MuJoCo"))
        self.assertIs(use_php_isaacsim(mujoco), mujoco)

    def test_adapter_installs_after_core_constructor(self):
        module = ModuleType("holosoma.simulator.isaacsim.isaacsim")

        class Core:
            def __init__(self, config):
                self.config = config
                self.sim = Context()

        module.IsaacSim = Core
        path = Path(__file__).parents[1] / "simulators/isaacsim.py"
        spec = importlib.util.spec_from_file_location("php_clock_test_adapter", path)
        adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {module.__name__: module}):
            spec.loader.exec_module(adapter)
        simulator = adapter.IsaacSim("saved config")
        simulator.sim.step(render=True)
        self.assertEqual(simulator.config, "saved config")
        self.assertEqual((simulator.sim.ticks, simulator.sim.renders), (1, 1))
        self.assertEqual(adapter.IsaacSim.__name__.lower(), "isaacsim")


if __name__ == "__main__":
    unittest.main()

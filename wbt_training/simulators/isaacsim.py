"""PHP's IsaacSim adapter; import only after the simulation app starts."""

from holosoma.simulator.isaacsim.isaacsim import IsaacSim as HolosomaIsaacSim

from wbt_training.utils.isaacsim_clock import install_physics_clock_fix


class IsaacSim(HolosomaIsaacSim):
    """Use Holosoma unchanged while preserving the control clock when rendering.

    The class name intentionally matches the simulator config's isaacsim
    name, as required by Holosoma's simulator-type validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_physics_clock_fix(self.sim)

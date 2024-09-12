from adarl.envs.ControlledEnv import ControlledEnv
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from gymnasium import Space

class LocomotionEnv(ControlledEnv[BaseJointImpedanceAdapter]):
    def  __init__(self, maxStepsPerEpisode,
                        stepLength_sec,
                        environmentController: BaseJointImpedanceAdapter):
        super().__init__(maxStepsPerEpisode,
                         stepLength_sec,
                         environmentController,
                         action_space,
                         observation_space,
                         state_space,
                         startSimulation,
                         is_timelimited,
                         allow_multiple_steps,
                         step_precision_tolerance)
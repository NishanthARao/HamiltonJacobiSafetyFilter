__credits__ = ["Kallinteris-Andreas"]
import math
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from typing import Optional, Tuple, Dict

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class SafeInvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is the Cartpole environment, based on the work of Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    just like in the classic environments, but now powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity).
    This environment consists of a cart that can be moved linearly, with a pole attached to one end and having another end free.
    The cart can be pushed left or right, and the goal is to balance the pole on top of the cart by applying forces to the cart.


    ## Action Space
    The agent take a 1-element vector for actions.

    The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
    the numerical force applied to the cart (with magnitude representing the amount of
    force and sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint |Type (Unit)|
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |


    ## Observation Space
    The observation space consists of the following parts (in order):
    - *qpos (2 element):* Position values of the robot's cart and pole.
    - *qvel (2 elements):* The velocities of cart and pole (their derivatives).

    The observation space is a `Box(-Inf, Inf, (4,), float64)` where the elements are as follows:

    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
    | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
    | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
    | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s)  |


    ## Rewards
    The goal is to keep the inverted pendulum stand upright (within a certain angle limit) for as long as possible - as such, a reward of +1 is given for each timestep that the pole is upright.

    The pole is considered upright if:
    $|angle| < 0.2$.

    and `info` also contains the reward.


    ## Starting State
    The initial position state is $\\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.
    The initial velocity state is $\\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.

    where $\\mathcal{U}$ is the multivariate uniform continuous distribution.


    ## Episode End
    ### Termination
    The environment terminates when the Inverted Pendulum is unhealthy.
    The Inverted Pendulum is unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite.
    2. The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radians.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    InvertedPendulum provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
    ```

    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `xml_file`              | **str**    |`"inverted_pendulum.xml"`| Path to a MuJoCo model                                                                        |
    | `reset_noise_scale`     | **float**  | `0.01`                  | Scale of random perturbations of initial position and velocity (see `Starting State` section) |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Fixed bug: `healthy_reward` was given on every step (even if the Pendulum is unhealthy), now it is only given if the Pendulum is healthy (not terminated) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/500)).
        - Added `xml_file` argument.
        - Added `reset_noise_scale` argument to set the range of initial states.
        - Added `info["reward_survive"]` which contains the reward.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
    * v2: All continuous control environments now use mujoco-py >= 1.5. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum).
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "inverted_pendulum.xml",
        frame_skip: int = 2,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        max_episode_steps: int = 1000,
        safety_filter_args: Optional[dict] = None,
        eval_mode: bool = False,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }
        
        self.max_episode_steps = max_episode_steps
        self.eval_mode = eval_mode
        ####### Safety Filter Variables #######
        
        ## Add a variable to indicate if safety filter has to be used
        ## This is primarily used to compare the training of the fallback policy
        ## with the vanilla policy (i.e., when training without safety requirements
        ## since the rewards and the termination conditions are different)
        self.prob_near_boundary = safety_filter_args.get("PROB_NEAR_BOUNDARY", 0.1) if safety_filter_args else 0.1
        self.use_safety_filter = safety_filter_args["USE_SAFETY_FILTER"] if safety_filter_args else False
        self.safe_margin_values = safety_filter_args["SAFE_MARGIN_VALUES"] if safety_filter_args and "SAFE_MARGIN_VALUES" in safety_filter_args else None
        if self.safe_margin_values is None:
            # Default safe margin values
            self.safe_margin_values = {
                "pos": 2.4,    # Safe margin for cart position
                "theta": 0.2,  # Safe margin for pole angle
            }
        self.safety_filter_in_use = False
        
        self.terminated_because = {
            "POS": 0,
            "THETA": 0,
        }
        #########################################

    def _calculate_l_value(self, 
                           state: np.ndarray,
                           ) -> Tuple[float, int]:
        
        POS_GEQ_THETA = 1  # 1: POS, 2: THETA
        
        x, theta, _, _ = state
        
        l_pos = (self.safe_margin_values["pos"] - np.abs(x)) / self.safe_margin_values["pos"]
        l_theta = (self.safe_margin_values["theta"] - np.abs(theta)) / self.safe_margin_values["theta"]

        if l_theta > l_pos: POS_GEQ_THETA = 2
        
        return min(l_pos, l_theta), POS_GEQ_THETA

    def _generate_random_state(self,) -> np.ndarray:
        
        # Generate random state near the boundaries of the safe set
        X_1 = self.safe_margin_values["pos"] - 1.5
        X_2 = self.safe_margin_values["pos"] - 2.3
        THETA_1 = self.safe_margin_values["theta"] - 0.03
        THETA_2 = self.safe_margin_values["theta"] - 0.15
        
        assert X_2 > 0 and THETA_2 > 0, \
            "absolute value of lower bounds are negative, try to keep them closer to the margin values"
        assert X_1 > X_2 and THETA_1 > THETA_2, \
            "lower bounds are greater than upper bounds!"

        # For (prob_near_boundary * 100)% of the time, generate a state near the boundaries of the safe set
        is_near_boundaries = self.np_random.uniform(0, 1) < self.prob_near_boundary
        # 50% of the time, generate a positive or negative value near the boundaries
        coin_flip = self.np_random.uniform(0, 1) < 0.5
        
        random_x = (self.np_random.uniform(low=-X_1, high=-X_2) if coin_flip else self.np_random.uniform(low=X_2, high=X_1)) if is_near_boundaries else self.np_random.uniform(low=-X_2, high=X_2)
        random_theta = (self.np_random.uniform(low=-THETA_1, high=-THETA_2) if coin_flip else self.np_random.uniform(low=THETA_2, high=THETA_1)) if is_near_boundaries else self.np_random.uniform(low=-THETA_2, high=THETA_2)
        random_vel = self.np_random.uniform(low=-0.5, high=0.5) if abs(random_x) < 0.5 else self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale)
        random_om = self.np_random.uniform(-self._reset_noise_scale, self._reset_noise_scale)
        
        # Stack the values to create the state
        random_state = np.array([random_x, random_theta, random_vel, random_om], dtype=np.float32)
        return random_state

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        info = {}

        if self.use_safety_filter:
            # Calculate the l-value for the current state
            l_value, POS_GEQ_THETA = self._calculate_l_value(observation)
            # If the l-value is less than 0, the state is unsafe
            terminated = bool(l_value < 0)
            
            info["l_value"] = l_value
            
            if POS_GEQ_THETA == 1 and terminated:
                self.terminated_because["POS"] += 1
            elif POS_GEQ_THETA == 2 and terminated:
                self.terminated_because["THETA"] += 1

        else:

            terminated = bool(
                not np.isfinite(observation).all() or (np.abs(observation[1]) > 0.2)
            )

        reward = int(not terminated)

        info["reward_survive"] = reward
        info["terminated_because"] = self.terminated_because if self.use_safety_filter else {}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):
        if self.use_safety_filter:
            # Generate a random state near the boundaries of the safe set
            state = self._generate_random_state()
            self.set_state(state[:2], state[2:])
        else:
            noise_low = -self._reset_noise_scale
            noise_high = self._reset_noise_scale

            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=noise_low, high=noise_high
            )
            qvel = self.init_qvel + self.np_random.uniform(
                size=self.model.nv, low=noise_low, high=noise_high
            )
            self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

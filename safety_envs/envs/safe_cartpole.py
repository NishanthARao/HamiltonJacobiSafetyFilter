"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv


class SafeCartPoleEnv(gym.Env):
    
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ## Starting State
    @TODO: Change the range
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments
    @TODO: Change the example
    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
        safety_filter_args: Optional[dict] = None,
        eval_mode : bool = False,
    ):
        super().__init__()
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.max_episode_steps = max_episode_steps
        self.eval_mode = eval_mode

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.low = -0.05
        self.high = 0.05

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None
        
        ####### Safety Filter Variables #######
        
        ## Add a variable to indicate if safety filter has to be used
        ## This is primarily used to compare the training of the fallback policy
        ## with the vanilla policy (i.e., when training without safety requirements
        ## since the rewards and the termination conditions are different)
        self.use_safety_filter = safety_filter_args["USE_SAFETY_FILTER"] if safety_filter_args else False
        self.safe_margin_values = safety_filter_args["SAFE_MARGIN_VALUES"] if safety_filter_args and "SAFE_MARGIN_VALUES" in safety_filter_args else None
        if self.safe_margin_values is None:
            # Default safe margin values
            self.safe_margin_values = {
                "pos": 2.0,                     # Safe margin for cart position
                "theta": 10 * (math.pi / 180),  # Safe margin for pole angle
                "vel": 1.0,                     # Safe margin for cart velocity
                "om": 0.1,                      # Safe margin for pole angular velocity
            }
        self.safety_filter_in_use = False
        #########################################
        
        
    def _calculate_l_value(self, state: np.ndarray) -> np.ndarray:
        """
        Calculates the safety function l(x) as the "signed distance" to the failure set.
        l(x) >= 0 indicates a safe state.
        l(x) < 0 indicates an unsafe state.
        
        Input: Takes in a state vector of shape (num_envs, 4)
        output: Returns a vector of shape (num_envs,) with the safety function values.
        """
        
        x, vel, theta, om = state
        
        # Calculate the safety function values
        l_pos = (self.safe_margin_values["pos"] - np.abs(x)) / self.safe_margin_values["pos"]
        l_theta = (self.safe_margin_values["theta"] - np.abs(theta)) / self.safe_margin_values["theta"]
        l_vel = (self.safe_margin_values["vel"] - np.abs(vel)) / self.safe_margin_values["vel"]
        l_om = (self.safe_margin_values["om"] - np.abs(om)) / self.safe_margin_values["om"]
        
        # Fallback policy should also bring the cart pole "towards" the safe set

        
        # l_pos = (self.safe_margin_values["pos"] - np.abs(np.tanh(2 * x))) / self.safe_margin_values["pos"]
        # l_theta = self.safe_margin_values["theta"] - np.abs(np.tanh(2 * theta)) / self.safe_margin_values["theta"]
        # l_vel = self.safe_margin_values["vel"] - np.abs(np.tanh(2 * vel)) / self.safe_margin_values["vel"]
        # l_om = self.safe_margin_values["om"] - np.abs(np.tanh(2 * om)) / self.safe_margin_values["om"]
        
        # Combine the safety function values
        # Calculate minimum value
        l_value = min(l_pos, l_theta, l_vel, l_om)
        
        return l_value
    
    def _generate_random_state(self, prob_near_boundary = 0.3) -> np.ndarray:
        """
        Generates a random state for the environment. We need the safety filter to be robust
        and thus, we encourage the reset to happen (~60% of the time) near the boundaries of the safe set, as 
        defined by the safe_margin_values. 
        
        Returns:
            A numpy array of shape (num_envs, 4) with the random state.
        """

        # Generate random state near the boundaries of the safe set
        X_1 = self.safe_margin_values["pos"] - 0.4
        X_2 = self.safe_margin_values["pos"] - 1.0
        THETA_1 = self.safe_margin_values["theta"] - 0.02
        THETA_2 = self.safe_margin_values["theta"] - 0.05
        VEL_1 = self.safe_margin_values["vel"] - 0.4
        VEL_2 = self.safe_margin_values["vel"] - 0.8
        OM_1 = self.safe_margin_values["om"] - 0.4
        OM_2 = self.safe_margin_values["om"] - 0.8
        
        assert X_2 > 0 and THETA_2 > 0 and VEL_2 > 0 and OM_2 > 0, \
            "absolute value of lower bounds are negative, try to keep them closer to the margin values"
        assert X_1 > X_2 and THETA_1 > THETA_2 and VEL_1 > VEL_2 and OM_1 > OM_2, \
            "lower bounds are greater than upper bounds!"

        # For (prob_near_boundary * 100)% of the time, generate a state near the boundaries of the safe set
        is_near_boundaries = self.np_random.uniform(0, 1) < prob_near_boundary
        # 50% of the time, generate a positive or negative value near the boundaries
        coin_flip = self.np_random.uniform(0, 1) < 0.5
        
        random_x = (self.np_random.uniform(low=-X_1, high=-X_2) if coin_flip else self.np_random.uniform(low=X_2, high=X_1)) if is_near_boundaries else self.np_random.uniform(low=-X_2, high=X_2)
        random_vel = (self.np_random.uniform(low=-VEL_1, high=-VEL_2) if coin_flip else self.np_random.uniform(low=VEL_2, high=VEL_1)) if is_near_boundaries else self.np_random.uniform(low=-VEL_2, high=VEL_2)
        random_theta = (self.np_random.uniform(low=-THETA_1, high=-THETA_2) if coin_flip else self.np_random.uniform(low=THETA_2, high=THETA_1)) if is_near_boundaries else self.np_random.uniform(low=-THETA_2, high=THETA_2)
        random_om = (self.np_random.uniform(low=-OM_1, high=-OM_2) if coin_flip else self.np_random.uniform(low=OM_2, high= OM_1)) if is_near_boundaries else self.np_random.uniform(low=-OM_2, high=OM_2)
        
        # Stack the values to create the state
        random_state = np.array([random_x, random_vel, random_theta, random_om], dtype=np.float32)
        return random_state
    
    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float32)

        # To keep the step function signature consistent with gym environments,
        # we return a dictionary with the l_values.
        # Note that this is the reward that the safety filter would use, 
        # and the "safe_dones" are calculated based on the l_values.
        info = {}

        ##################################
        if self.use_safety_filter:
        
            # Calculate the safety function values
            l_value = self._calculate_l_value(self.state)
        
            terminated = bool(self._calculate_l_value(self.state) < 0.0)  # Unsafe state if l(x) < 0
            #terminated = False
            
            info["l_value"] = l_value
            
        else:
            terminated = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )
        ##################################

        # ##################################
        # if any(done):
        #     if self.use_safety_filter:
        #         self.state[:, done] = self._generate_random_state()[:, done]
        #         self.steps[done] = 0
        #     else:
        #         # This code was generated by copilot, need to check if it works
        #         self.state[:, done] = self.np_random.uniform(
        #             low=self.low, high=self.high, size=(4, done.sum())
        #         ).astype(np.float32)
        #         self.steps[done] = 0
        # ##################################

        # reward = np.ones_like(terminated, dtype=np.float32)
        
        
        if not terminated: reward =  1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = 0.0
        

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        ##################
        if self.use_safety_filter and not self.eval_mode:
            self.state = self._generate_random_state(prob_near_boundary=0.05)
        else:
            self.low, self.high = utils.maybe_parse_reset_bounds(
                options, -0.05, 0.05  # default low
            )  # default high
            self.state = self.np_random.uniform(
                low=self.low, high=self.high, size=(4,)
            ).astype(np.float32)
        ###################    
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        if self.use_safety_filter:
            safety_box_color = (0, 255, 0)
            if self.safety_filter_in_use: safety_box_color = (255, 0, 0)
            
            cx_box, cy_box = 475, 250
            safety_box_coords = [(cx_box - 30, cy_box - 15), (cx_box - 30, cy_box + 15), 
                        (cx_box + 30, cy_box + 15), (cx_box + 30, cy_box - 15)]
            gfxdraw.aapolygon(self.surf, safety_box_coords, safety_box_color)
            gfxdraw.filled_polygon(self.surf, safety_box_coords, safety_box_color)

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == "__main__":
    env = SafeCartPoleEnv(render_mode="rgb_array", 
                          safety_filter_args={
                              "USE_SAFETY_FILTER": True,
                              "SAFE_MARGIN_VALUES": {
                                  "pos": 2.0,
                                  "theta": 10 * (math.pi / 180),
                                  "vel": 1.0,
                                  "om": 1.0,
                              }
                          })
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # Sample random action
        env.step(action)
        env.reset()
    env.close()
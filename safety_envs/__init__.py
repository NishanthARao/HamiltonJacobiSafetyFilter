from gymnasium.envs.registration import register

register(
    id="SafeCartPole-v1",
    entry_point="safety_envs.envs:SafeCartPoleEnv",
    max_episode_steps=500,
)

register(
    id="SafeInvertedPendulum-v1",
    entry_point="safety_envs.envs:SafeInvertedPendulumEnv",
    max_episode_steps=1000,
)
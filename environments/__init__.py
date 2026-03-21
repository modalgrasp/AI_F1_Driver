"""Environment package for F1 autonomous racing."""

from gymnasium.envs.registration import register

register(
    id="F1Racing-v0",
    entry_point="environments.f1_racing_env:F1RacingEnv",
    kwargs={"action_mode": "continuous"},
)

register(
    id="F1RacingDiscrete-v0",
    entry_point="environments.f1_racing_env:F1RacingEnv",
    kwargs={"action_mode": "discrete"},
)

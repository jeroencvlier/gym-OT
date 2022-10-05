from gym.envs.registration import register

register(
    id="OT-v0",
    entry_point="gym_OT.envs:OTGym_v0",
    max_episode_steps=1000
    )

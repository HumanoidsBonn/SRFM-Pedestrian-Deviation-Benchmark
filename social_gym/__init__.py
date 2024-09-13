from gymnasium.envs.registration import register

register(
    id='SocialForceEnv-v0',
    entry_point='social_gym.envs.sfm_env:SocialForceEnv',
    kwargs={}
)

register(
    id='PybulletSocialForceEnv-v0',
    entry_point='social_gym.envs.pybullet_srfm_env:SocialForceEnv',
    kwargs={'render': True,}
)

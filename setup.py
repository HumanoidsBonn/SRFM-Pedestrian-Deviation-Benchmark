from setuptools import setup

setup(name='social_gym',
      version='0.0.1',
      install_requires=['gymnasium',
                        "stable-baselines3[extra]==2.*",
                        "sb3-contrib==2.*",
                        "tqdm",
                        "tensorboard",
                        "numpy",
                        "wandb"
                        ]
)

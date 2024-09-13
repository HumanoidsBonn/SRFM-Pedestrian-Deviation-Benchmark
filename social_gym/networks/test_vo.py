import sys, os
#import for RL
from stable_baselines3 import TD3, PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan
from tqdm import tqdm
import torch
import numpy as np
import gymnasium as gym
import social_gym
from decentralized.velocity_obstacle import velocity_obstacle as vo
import math

#VO code from - https://github.com/atb033/multi_agent_path_planning

class SavingCallback(BaseCallback):
    def __init__(self, log_dir, save_freq=100000, verbose=0):
        super(SavingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print("Save intermediate model and replay buffer")
            self.model.save(os.path.join(self.log_dir, 'intermediate_saved_model'))
            self.model.save_replay_buffer(os.path.join(self.log_dir, "replay_buffer"))
        return True


def load_model(algo, env):
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                     sigma=0.3 * np.ones(env.action_space.shape[-1]))
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
    print("choosing algorithm: ", algo)
    if algo == "TD3":
        policy = "MlpPolicy"
        training_starts_at = 10000
        model = TD3(policy, env, action_noise=action_noise, learning_rate=1e-4, buffer_size=int(1e6), learning_starts=training_starts_at, batch_size=256, 
	            tau=0.005, gamma=0.99, train_freq=(5, 'step'), gradient_steps=-1, replay_buffer_class=None, replay_buffer_kwargs=None, optimize_memory_usage=False, 
	            policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5, stats_window_size=100, tensorboard_log="./tensorboard_logs/", policy_kwargs=policy_kwargs, verbose=1, seed=None, device='cuda', _init_setup_model=True)
    
    if algo == "SAC":
        policy = "MlpPolicy"
        training_starts_at = 10000
        model = SAC(policy, env, action_noise=action_noise, learning_rate=1e-4, buffer_size=int(1e6), learning_starts=training_starts_at, batch_size=256, 
                tau=0.005, gamma=0.99, train_freq=(5, 'step'), gradient_steps=-1, replay_buffer_class=None, replay_buffer_kwargs=None, optimize_memory_usage=False, 
                ent_coef='auto', tensorboard_log="./tensorboard_logs/", target_update_interval=1, target_entropy='auto', use_sde=False, sde_sample_freq=-1, use_sde_at_warmup=False, stats_window_size=100,
                policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)

    #change later
    if algo == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=5, batch_size=128, n_epochs=1, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                    use_sde=False, sde_sample_freq=-1, tensorboard_log="../tensorboard_logs/",
                    create_eval_env=False, policy_kwargs=None, verbose=1, device='cuda')
        
    return model

def load_train(log_dir_name, tensorboard_log_name, env, render=False):
    # Set training mode to true and render to false
    env.set_training_mode(True, render, True)

    if os.path.exists(log_dir_name + "intermediate_saved_model.zip"):
        print("LOAD saved model")
        model = TD3.load(log_dir_name + "intermediate_saved_model.zip")
        model.set_env(env)
        if os.path.exists(log_dir_name + "replay_buffer.pkl"):
            model.load_replay_buffer(log_dir_name + "replay_buffer")
    else:  model = load_model("TD3", env)
    max_iterations = 2000000
    wandb_run = setup_wandb(max_iterations)
    saving_callback = SavingCallback(log_dir_name, save_freq=50000)
    wandb_callback = WandbCallback(model_save_path=f"models/{wandb_run.id}",verbose=2,)
    model.learn(total_timesteps=max_iterations, tb_log_name=tensorboard_log_name,reset_num_timesteps=False, progress_bar=True, callback=[saving_callback, wandb_callback])
    wandb_run.finish()
    model.save(log_dir_name + "final_model")
    print("saved final model")

def setup_wandb(max_iterations):
    config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": max_iterations,
    "env_id": "SocialForceEnv-v1"}
    
    run = wandb.init(
        project="social_robot_force",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    return run

def load_test(log_dir_name, env, scenario, render=False, useRobot=True):
    # Set training mode to false
    env.set_training_mode(False,render,useRobot)

    # saved_model_path = log_dir_name + "intermediate_saved_model"
    saved_model_path = log_dir_name + "final_model"
    #model = load_model("TQC", env, "")
    # model = TD3.load(saved_model_path, env=env, device='cuda', force_reset = True)
    # model.set_env(env)

    #Number of times to run the evaluation scenario for
    eval_num = 100

    if scenario == 0:
        #Run all scenarios - 100 runs of each
        for scene in range(1,6):
            env.set_scenario(scene)
            avg_reward = 0
            for i in tqdm(range(eval_num)):
                for rob in [True,False]:
                    env.set_episode(i, rob)
                    obs, _ = env.reset()
                    Done = False
                    while not Done:
                        # action, _states = model.predict(obs)
                        pos, goal, peds = env.get_vo_arguments()

                        # print(np.shape(peds))

                        start = pos

                        vo.ROBOT_RADIUS = 2/2.2
                        vo.VMAX = 1

                        v_desired = vo.compute_desired_velocity(start,goal,vo.ROBOT_RADIUS, vo.VMAX)
                        control_vel = vo.compute_velocity(start, peds, v_desired)
                        action = control_vel
                        obs, reward, Done, _, info = env.step(action)
                        avg_reward = + reward
                    avg_reward = avg_reward / eval_num
                    useRobots, trajectories = env.get_pedestrian_trajectory()
                    save_trajectories(i, trajectories, useRobots, scene)
                    #print('The agent collected %f reward on average during %d iterations' % (avg_reward, i))
    else:
        #Run the scenario number based on provided value - 100 runs
        env.set_scenario(scenario)
        avg_reward = 0
        for i in tqdm(range(eval_num)):
            for rob in [True,False]:
                env.set_episode(i, rob)
                obs, _ = env.reset()
                Done = False
                while not Done:
                    # action, _states = model.predict(obs)
                    pos, goal, peds = env.get_vo_arguments()

                    # print(np.shape(peds))

                    start = pos

                    vo.ROBOT_RADIUS = 1.4/2.2
                    vo.VMAX = 1

                    v_desired = vo.compute_desired_velocity(start,goal,vo.ROBOT_RADIUS, vo.VMAX)
                    control_vel = vo.compute_velocity(start, peds, v_desired)
                    action = control_vel
                    obs, reward, Done, _, info = env.step(action)
                    avg_reward = + reward
                avg_reward = avg_reward / eval_num
                useRobots, trajectories = env.get_pedestrian_trajectory()
                save_trajectories(i, trajectories, useRobots, scenario)
                #print('The agent collected %f reward on average during %d iterations' % (avg_reward, i))

def save_trajectories(episode, trajectories, useRobots, scenario):
    # Set the filename
    filename = f"/home/sagrawal/Desktop/SocialRobotForce/social_gym/networks/Traj_Logs/vo/vorandomized_{scenario}_ep{episode}_{int(useRobots)}.txt"

    # Open the file, write the list, and close the file
    with open(filename, 'w') as file:
        for item in trajectories:
            file.write(f"{item[0]},{item[1]},{item[2]}\n")

def polar_to_cartesian_velocity(v_r, v_theta, theta):
    """
    Convert polar velocities to Cartesian velocities.
    
    Parameters:
    v_r (float): Radial velocity.
    v_theta (float): Angular velocity.
    theta (float): Heading angle in radians.

    Returns:
    tuple: Cartesian velocities (v_x, v_y).
    """
    v_x = v_r * np.cos(theta) - v_theta * np.sin(theta)
    v_y = v_r * np.sin(theta) + v_theta * np.cos(theta)
    
    return np.array([v_x, v_y])

def cartesian_to_polar_velocity(v_x, v_y, theta):
    """
    Convert Cartesian velocities to polar velocities.
    
    Parameters:
    v_x (float): Velocity in the x-direction.
    v_y (float): Velocity in the y-direction.
    theta (float): Heading angle in radians.
    
    Returns:
    tuple: Polar velocities (v_r, v_theta).
    """
    v_r = v_x * np.cos(theta) + v_y * np.sin(theta)
    v_theta = -v_x * np.sin(theta) + v_y * np.cos(theta)
    
    return (v_r, v_theta)

def rl(env_name):
    model_name = "model_clean_attempt3_mainmodel"
    log_dir_name = f"./Training_Logs/TD3_Logs/{model_name}/"
    tensorboard_log_name = f"TD3_{model_name}"

    print("cuda is available: ", torch.cuda.is_available)
    torch.device("cuda")
    torch.set_num_threads(4)

    os.makedirs(log_dir_name, exist_ok=True)

    # Initialize environments
    env = gym.make(env_name)
    
    for val in [True]:
        # load_train(log_dir_name, tensorboard_log_name, env, render=False)
        load_test(log_dir_name, env, scenario=5, render=False, useRobot=val)

    # Best models:
    # Baseline - model_clean_attempt3_baseline2
    # Ours - model_clean_attempt3_mainmodel

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    rl('SocialForceEnv-v0')
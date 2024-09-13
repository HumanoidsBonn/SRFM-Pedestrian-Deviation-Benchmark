import numpy as np
from numpy.random import RandomState
import pygame
import random
import yaml
import time
import math

import gymnasium as gym
from gymnasium import spaces
import social_gym

import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env

# Load YAML configuration for testing
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    


config = load_config('config_crosswalk.yaml')  # Adjust path as needed - config_footpath or config_crosswalk
configNoise = load_config('config_noise.yaml')

MAX_SPEED = 1.25
MAX_SPEED_ROBOT = 0.5

class Pedestrian:
    def __init__(self, A, B, lambda_, r, Tau, pos, goal, A_r, B_r, init_vel=None):
        self.pos = np.array(pos, dtype=np.float32)
        self.unrobot_pos = np.array(pos, dtype=np.float32)
        self.ghost_pos = np.array(pos, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.trajectory = []  # Initialize with the starting position
        self.collided = False
        self.collidedPedestrians = []
        
        # Initialize velocity
        if init_vel is None:
            self.vel = np.zeros(2)
        else:
            self.vel = np.array(init_vel, dtype=np.float32)  # Ensure the provided velocity is a numpy array
        
        self.unrobot_vel = self.vel # Initialize self velocity without robot affect same as starting velocity

        # Initializing the parameters
        self.radius = r
        self.A = A
        self.B = B
        self.Tau = Tau
        self.lambda_ = lambda_

        # Parameters for Robot Force
        self.A_r = A_r
        self.B_r = B_r

        self.goal_radius = 2
        self.trajectory_radius = 1
        self.dt = 0.067
    
    def compute_drive_force(self):
        desired_vel = (self.goal - self.pos)
        if np.linalg.norm(desired_vel) > MAX_SPEED:
            desired_vel = (desired_vel / np.linalg.norm(desired_vel)) * MAX_SPEED
        return (desired_vel - self.vel) / self.Tau
    
    def compute_repulsive_force(self, others):
        force = np.zeros(2)
        for other in others:
            if other is self:
                continue

            d = self.radius + other.radius # Sum of the radii of the two interacting entities
            direction = self.pos - other.pos
            distance = np.linalg.norm(direction)
            if distance == 0:  # Avoid division by zero
                continue
            n_gj = direction / distance
            e_g = self.vel / np.linalg.norm(self.vel) if np.linalg.norm(self.vel) > 0 else np.zeros(2)
            
            cos_phi_gj = -np.dot(n_gj, e_g)
            w_phi_gj = self.lambda_ + ((1 - self.lambda_) * (1 + cos_phi_gj) / 2)
            
            force_magnitude = self.A * np.exp((d - distance) / self.B) * w_phi_gj
            force += force_magnitude * n_gj  # Apply force in direction of n_gj
        return force
    
    def compute_obstacle_repulsive_force(self, obstacles):
        force = np.zeros(2)
        for obstacle in obstacles:
            direction = self.pos - obstacle['pos']
            distance = np.linalg.norm(direction)
            d = self.radius + obstacle['radius']
            if distance == 0:
                continue
            n_gj = direction / distance
            
            force_magnitude = self.A * np.exp((d - distance) / self.B)
            force += force_magnitude * n_gj
        return force
    
    def compute_robot_repulsive_force(self, robots):
        force = np.zeros(2)
        for robot in robots:
            direction = self.pos - robot['pos']
            distance = np.linalg.norm(direction)
            d = self.radius + robot['radius']
            if distance == 0:
                continue
            n_gj = direction / distance

            # Robot-specific parameters could be used here if different from pedestrian parameters
            force_magnitude = self.A_r * np.exp((d - distance) / self.B_r)
            force += force_magnitude * n_gj
        return force

    def update(self, others, obstacles=None, robots=None, useRobotRepulsion=False):
        drive_force = self.compute_drive_force()
        repulsive_force = self.compute_repulsive_force(others)
        obstacle_repulsive_force = self.compute_obstacle_repulsive_force(obstacles) if obstacles else np.zeros(2)
        robot_repulsive_force = self.compute_robot_repulsive_force(robots) if useRobotRepulsion and robots else np.zeros(2)

        total_nonrobot_force = drive_force + repulsive_force + obstacle_repulsive_force
        self.unrobot_vel = total_nonrobot_force * self.dt
        if np.linalg.norm(self.unrobot_vel) > MAX_SPEED:
            self.unrobot_vel = self.unrobot_vel / np.linalg.norm(self.unrobot_vel) * MAX_SPEED

        total_force = drive_force + repulsive_force + obstacle_repulsive_force + robot_repulsive_force
        self.vel += total_force * self.dt
        if np.linalg.norm(self.vel) > MAX_SPEED:
            self.vel = self.vel / np.linalg.norm(self.vel) * MAX_SPEED
        self.unrobot_pos = self.pos + self.unrobot_vel * self.dt
        self.pos += self.vel * self.dt
        self.trajectory.append(self.pos.copy())        

class SocialForceEnv(gym.Env):
    def __init__(self):
        super(SocialForceEnv, self).__init__()
        self.pedestrians = []
        self.pedestrianTrajectories = []
        self.nearby_pedestrians = []
        self.robots = []
        self.vel = []
        self.heading = 0
        self.prev_pos = np.array([0,0])

        self.goalDistanceCheck = False
        self.outOfBoundsCheck = False
        self.stepCounterCheck = False
        self.collision = False
        self.social_zone = 2
        self.useRobots = True
        self.episode = 0
        self.scenario = 0

        self.isTrainingMode = True
        self.trainAll = False
        self.isRender = False

        self.leftbound = 0
        self.rightbound  = 15
        self.bottombound = 0
        self.topbound  = 15


        if self.isTrainingMode:
            # RANDOM ROBOT TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Defining start and goal points for the robot randomly
            self.robot_start = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
            self.robot_goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
            while np.linalg.norm(self.robot_start - self.robot_goal) < 5:
                self.robot_goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
            # RANDOM ROBOT TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else:
            # EVALUATION ROBOT BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            self.robot_start = np.array(config['Robot']['Start'][0])
            self.robot_goal = np.array(config['Robot']['Goal'][0])
            self.vel = np.array([0,0])
            # EVALUATION ROBOT BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + 2*10 + 2 + 4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + 4*10 + 2 + 4,), dtype=np.float32)
        self.dt = 0.067
        self.step_counter = 0

        self.fig, self.ax = plt.subplots()
        self.fig.set_figheight(self.topbound-self.bottombound+2)
        self.fig.set_figwidth(self.rightbound-self.leftbound+2)
        plt.ion()

    def step(self, action):

        if self.isTrainingMode:
            # TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Update robot position
            self.robots[0]['pos'] = self.robots[0]['pos'].astype(np.float32)  # Ensure pos is a float array
            self.real_action = action * MAX_SPEED_ROBOT 
            self.robots[0]['pos'] += self.real_action * self.dt
            # TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else:
            # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if self.useRobots == True:
                self.prev_pos = self.robots[0]['pos'].astype(np.float32)
                # Update robot position
                self.robots[0]['pos'] = self.robots[0]['pos'].astype(np.float32)  # Ensure pos is a float array
                self.real_action = action * MAX_SPEED_ROBOT 
                self.robots[0]['pos'] += self.real_action * self.dt
                self.vel = (self.robots[0]['pos'] - self.prev_pos)/self.dt
                self.heading = self.calculate_heading(self.prev_pos, self.robots[0]['pos'])
            # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        # Update pedestrians
        for id,ped in enumerate(self.pedestrians):
            if (self.isTrainingMode) and (self.scenario != 5):
                # RANDOM PEDESTRIAN TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                ped.update(self.pedestrians, robots=self.robots, useRobotRepulsion=self.useRobots)
                self.pedestrianTrajectories.append([id,ped.pos.copy()])
                if np.linalg.norm(ped.pos-ped.goal) < 0.1:
                    goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                    while (np.linalg.norm(self.robot_goal - goal) < self.social_zone) or (np.linalg.norm(ped.pos - goal) < 7):
                        goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                    ped.goal = goal
                # RANDOM PEDESTRIAN TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            else:
                # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                ped.update(self.pedestrians, robots=self.robots, useRobotRepulsion=self.useRobots)
                distance_to_robot = np.linalg.norm(ped.pos - self.robots[0]['pos'])
                if self.scenario == 5:
                    self.pedestrianTrajectories.append([id,ped.pos.copy(),distance_to_robot])
                elif np.linalg.norm(ped.pos-ped.goal) > 0.1:
                    self.pedestrianTrajectories.append([id,ped.pos.copy(),distance_to_robot])
                # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



        # This adds all nearby pedestrians to a separate list - condition distance less than 2m
        self.nearby_pedestrians = [ped for ped in self.pedestrians if (np.linalg.norm(ped.pos - self.robots[0]['pos']) < 2)]
                
        obs = self._get_obs()
        self.success, self.oob, self.oot, self.collision = self._check_done()
        if self.success:
            print(np.linalg.norm(self.robot_start - self.robot_goal))
        done = self.success or self.oob or self.oot or self.collision
        reward = self._compute_reward()
        
        self.step_counter += 1
        info = {"is_success": self.success,
                "is_collision": self.collision,}
        
        # print("ROBOT VELOCITY: ", np.sqrt([self.vel[0]**2 + self.vel[1]**2]))

        if self.isRender:
            self.render()
        return obs, reward, done, False, info

    def reset(self, seed = None, options = None):
        # Reset the environment
        self.success, self.oot, self.oob, self.collision = False, False, False, False
        self.step_counter = 0
        self.real_action = np.array([0, 0])
        self.pedestrianTrajectories = []
        self.vel = np.array([0,0])

        if self.episode > 19:
            self.episode = self.episode%20

        if self.isTrainingMode == True and self.trainAll == True:
            self.scenario = np.random.randint(6)

        # Setting the config of test scenario - Adjust path as needed - config_footpath or config_crosswalk
        if self.scenario == 0:
            print('Random scenario selected')
        elif self.scenario == 1:
            config = load_config('config_footpath.yaml')
        elif self.scenario == 2:
            config = load_config('config_crosswalk.yaml')
        elif self.scenario == 3:
            config = load_config('config_crossfootpath.yaml')
        elif self.scenario == 4:
            config = load_config('config_assault.yaml')
        elif self.scenario == 5:
            config = load_config('config_concert.yaml')
        else:
            print('Not a valid scenario! Selecting footpath as default!!!')
            config = load_config('config_footpath.yaml')  
        configNoise = load_config('config_noise.yaml')

        if self.isTrainingMode:
            if self.trainAll == True:
                if self.scenario == 0:
                    # TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Defining start and goal points for the robot randomly
                    self.robot_start = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                    self.robot_goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                    while np.linalg.norm(self.robot_start - self.robot_goal) < 5:
                        self.robot_goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                    # TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                else:
                    self.robot_start = np.array(config['Robot']['Start'][0])
                    self.robot_goal = np.array(config['Robot']['Goal'][0])
            else:
                # Defining start and goal points for the robot randomly
                self.robot_start = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                self.robot_goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                while np.linalg.norm(self.robot_start - self.robot_goal) < 5:
                    self.robot_goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
        else:
            # # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if self.useRobots == True:
                self.robot_start = np.array(config['Robot']['Start'][0])
                self.robot_goal = np.array(config['Robot']['Goal'][0])
            else:
                self.robot_start = np.array([100,100])
                self.robot_goal = np.array([200,200])
            # # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.pedestrians = []

        if self.isTrainingMode:
            if self.trainAll == True:
                if self.scenario == 0:
                    # RANDOM PEDESTRIAN TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    for i in range(10):
                        start = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                        while (np.linalg.norm(self.robot_start - start) < self.social_zone):
                            start = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])

                        goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                        while (np.linalg.norm(self.robot_goal - goal) < self.social_zone) or (np.linalg.norm(start - goal) < 7):
                            goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])

                        self.pedestrians.append(Pedestrian(2, 0.89, 0.4, 0.3, 0.6, start, goal, 7.93, 0.99))
                    # RANDOM PEDESTRIAN TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                else:
                    for start, end in zip(config['Pedestrians']['Positions'], config['Goals']['Positions']):
                        st = np.array(start) + np.array(configNoise['Noise'][self.episode])
                        ed = np.array(end) - np.array(configNoise['Noise'][self.episode])
                        self.pedestrians.append(Pedestrian(2, 0.89, 0.4, 0.3, 0.6, st, ed, 7.93, 0.99))
            else:
                # RANDOM PEDESTRIAN TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                for i in range(10):
                    start = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                    while (np.linalg.norm(self.robot_start - start) < self.social_zone):
                        start = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])

                    goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])
                    while (np.linalg.norm(self.robot_goal - goal) < self.social_zone) or (np.linalg.norm(start - goal) < 7):
                        goal = np.array([random.uniform(self.leftbound, self.rightbound), random.uniform(self.bottombound, self.topbound)])

                    self.pedestrians.append(Pedestrian(2, 0.89, 0.4, 0.3, 0.6, start, goal, 7.93, 0.99))
                # RANDOM PEDESTRIAN TRAINING BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else:
            # # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            randseed = RandomState(self.episode)
            for start, end in zip(config['Pedestrians']['Positions'], config['Goals']['Positions']):
                noise = randseed.uniform(-0.5, 0.5, np.array(start).shape)
                noisy_st = np.array(start) + noise
                noisy_ed = np.array(end) + noise
                # st = np.array(start) + np.array(configNoise['Noise'][self.episode])
                # ed = np.array(end) - np.array(configNoise['Noise'][self.episode])
                # print(self.episode, self.useRobots, noise)
                st = noisy_st
                ed = noisy_ed
                self.pedestrians.append(Pedestrian(2, 0.89, 0.4, 0.3, 0.6, st, ed, 7.93, 0.99))
                # self.pedestrians.append(Pedestrian(1.5, 0.5, 0.6, 0.2, 4.71, 1.18, 0.15, start, end, 0))
            # # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.robots = [{'pos': self.robot_start, 'radius': 0.2}]
        return self._get_obs(), {}
    
    def get_relative_distance_angle(self, a, b):
        rel_angle = np.arctan2(b[1] - a[1], b[0] - a[0])
        rel_distance = np.linalg.norm(a - b)

        return np.array([rel_distance, rel_angle])

    def _get_obs(self):
        # Observation space is the positions of the robot and pedestrians
        robot_pos = self.robots[0]['pos']

        # ped info without ped velocity
        # ped_info = np.concatenate([self.get_relative_distance_angle(robot_pos, ped.pos) for ped in self.nearby_pedestrians] + (10 - len(self.nearby_pedestrians)) * [np.zeros(2)])
        
        # ped info with cartesian absolute ped velocity
        # ped_info = np.concatenate([np.concatenate([self.get_relative_distance_angle(robot_pos, ped.pos),ped.vel]) for ped in self.nearby_pedestrians] + (10 - len(self.nearby_pedestrians)) * [np.zeros(4)])
        
        # ped info with polar relative ped velocity
        ped_info = np.concatenate([np.concatenate([self.get_relative_distance_angle(robot_pos, ped.pos),self.get_relative_velocity(self.vel, ped.vel)]) for ped in self.nearby_pedestrians] + (10 - len(self.nearby_pedestrians)) * [np.zeros(4)])
        
        goal_info = self.get_relative_distance_angle(robot_pos, self.robot_goal)

        return np.concatenate([goal_info, ped_info, self.real_action, np.array([int(self.success), int(self.oot), int(self.oob), int(self.collision)])])
        # return np.concatenate([obs_x, obs_y, self.real_action])
        #return np.concatenate([robot_pos, self.robot_goal, pedestrian_positions])

    def _compute_reward(self):
        # Rewards

        # 1 - Distance reward
        robot_pos = self.robots[0]['pos']
        goal = self.robot_goal
        max_distance = np.linalg.norm(self.robot_start - goal)
        current_distance = np.linalg.norm(robot_pos - goal)
        norm_distance = self.dist_normalized(0, max_distance, current_distance)

        # Movement penalty
        movement_threshold = MAX_SPEED_ROBOT*0.067/2
        if np.linalg.norm(self.robots[0]['pos'] - self.prev_pos) < movement_threshold:
            alpha = 2
        else:
            alpha = 1

        # This one line removes movement penalty
        alpha = 1

        distance_reward = alpha * (-norm_distance)
        # distance_reward = alpha * (-current_distance)

        # 2 - Pedestrian Penalty
        ped_penalty = 0 * sum([np.linalg.norm(ped.pos - ped.unrobot_pos) for ped in self.nearby_pedestrians])

        # 3 - Time Penalty - a small one
        time_penalty = -0.1

        # Return Criteria
        if self.success:
            return 100
        elif self.oob:
            return -100
        elif self.oot:
            return -30
        elif self.collision:
            return -10
        else:
            return distance_reward + ped_penalty + time_penalty
    
    def dist_normalized(self, c_min, c_max, dist):
        normed_min, normed_max = 0, 1
        x_normed = (dist - c_min) / (c_max - c_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        return round(x_normed, 4)
    
    def _check_done(self):
        # Episode is done if the robot reaches the goal
        robot_pos = self.robots[0]['pos']
        goal = self.robot_goal
        
        self.goalDistanceCheck = np.linalg.norm(robot_pos - goal) < 0.5
        self.outOfBoundsCheck = (robot_pos[0] < self.leftbound-1) or (robot_pos[0] > self.rightbound+1) or (robot_pos[1] < self.bottombound-1) or (robot_pos[1] > self.topbound+1)
        self.stepCounterCheck = self.step_counter > 750
        self.collision_ped = [True if (np.linalg.norm(ped.pos - self.robots[0]['pos']) < 0.3) else False for ped in self.nearby_pedestrians]

        if not self.isTrainingMode:
            # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if self.useRobots == False:
                self.outOfBoundsCheck = False
                self.stepCounterCheck = self.step_counter > 750
            else:
                self.stepCounterCheck = False
            # EVALUATION BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # self.collision_ped = False # Just for plotting
        if (self.goalDistanceCheck or self.outOfBoundsCheck or self.stepCounterCheck or np.any(self.collision_ped)):
            print('Reached Goal:',self.goalDistanceCheck, ' | ', 'Out of Bounds:', self.outOfBoundsCheck, ' | ', 'Steps Over:', self.stepCounterCheck, ' | ','Collided:', np.any(self.collision_ped))
            print('Robot position: ', robot_pos)

        return self.goalDistanceCheck, self.outOfBoundsCheck, self.stepCounterCheck, np.any(self.collision_ped)
    
    def get_pedestrian_trajectory(self):
        return self.useRobots, self.pedestrianTrajectories

    # Setting the episode number of the run
    def set_episode(self, ep, rob):
        self.episode = ep
        self.useRobots = rob

    # Setting the scenario to use for testing
    def set_scenario(self, scene):
        self.scenario = scene
    
    # Setting the mode of the code to train or test
    def set_training_mode(self, areYouTraining=False, doYouRender=False, wantToUseRobots=True, allscenarios=False):
        self.isTrainingMode = areYouTraining
        if self.isTrainingMode == False:
            self.useRobots = wantToUseRobots
        self.trainAll = allscenarios
        self.isRender = doYouRender


    def calculate_heading(self, prev_position, new_position):
        """
        Calculate the heading (theta) of the robot given the previous and new positions.

        :param prev_position: Tuple (x_prev, y_prev) of the previous position.
        :param new_position: Tuple (x_new, y_new) of the new position.
        :return: Heading theta in radians.
        """
        x_prev, y_prev = prev_position
        x_new, y_new = new_position
        
        # Calculate differences in positions
        delta_x = x_new - x_prev
        delta_y = y_new - y_prev
        
        # Calculate the heading
        theta = math.atan2(delta_y, delta_x)
        
        return theta

    def get_dwa_arguments(self):
        # pointclouds = np.array([ped.pos for ped in self.pedestrians])
        pointclouds = np.array([ped.pos for ped in self.nearby_pedestrians] + (10 - len(self.nearby_pedestrians)) * [np.zeros(2, dtype=np.float32)])

        # if len(self.nearby_pedestrians) < 1:
        #     pointclouds = np.array([np.array([0,0])])
        # else:
        #     pointclouds = np.array([ped.pos for ped in self.nearby_pedestrians])

        pos = np.append(self.robots[0]['pos'],[self.heading])
        return tuple(pos), tuple(self.vel), tuple(self.robot_goal), pointclouds

    def get_relative_velocity(self, va, vb):
        v_ba = vb - va
        v_r, v_theta = self.cartesian_to_polar_velocity(v_ba[0], v_ba[1])
        return np.array([v_r, v_theta])
        
    def cartesian_to_polar_velocity(self, vx, vy):
        # Calculate the magnitude of the velocity vector
        vr = math.sqrt(vx**2 + vy**2)
        # Calculate the direction of the velocity vector
        vtheta = math.atan2(vy, vx)
        return vr, vtheta
    
    def get_vo_arguments(self):
        peds = []
        # for ped in self.pedestrians:
        for ped in self.nearby_pedestrians:
            info = np.array([ped.pos[0], ped.pos[1], ped.vel[0], ped.vel[1]])
            peds.append(info)

        if len(peds) < len(self.pedestrians):
            blank_info = np.array([0,0,0,0])
            peds.append(blank_info)

        # Printing pedestrian information going to VO
        # print(peds)

        peds = np.array(peds).T
        pos = np.append(self.robots[0]['pos'],self.vel)
        return pos, np.append(self.robot_goal,np.array([0,0])), peds

    def render(self, mode='human'):
        # Clear the current plot
        self.ax.clear()
        
        # Plot the robot - While travelling
        robot_pos = self.robots[0]['pos']

        # Plot the robot - while standing still
        # robot_pos = np.array(config['Robot']['Start'][0])
        robot_endpos = self.robot_goal
        circle = plt.Circle(robot_pos, self.social_zone/2, color='r', fill=False)
        # self.ax.axis("equal")
        self.ax.set_aspect('equal')
        self.ax.plot(robot_pos[0], robot_pos[1], 'ro', markersize=20, label='Robot')
        self.ax.add_patch(circle)

        theta = self.heading
        arrow_length = 1
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)

        self.ax.arrow(robot_pos[0], robot_pos[1], dx, dy, length_includes_head=True, head_width=0.2)

        self.ax.plot(robot_endpos[0], robot_endpos[1], 'g*', markersize=25, label='Robot Goal')
        # print('This many pixels in one y and x unit: ',self.ax.transData.transform([(0,1),(1,0)])-self.ax.transData.transform((0,0)))
        
        # Plot the pedestrians
        for i,ped in enumerate(self.pedestrians):
            ped_circle = circle = plt.Circle(ped.pos, 0.2, color='b', fill=False)

            if i==0:
                self.ax.plot(ped.pos[0], ped.pos[1], 'bo', label='Pedestrian', markersize=20, alpha=0.3)
            self.ax.plot(ped.pos[0], ped.pos[1], 'bo', markersize=20, alpha=0.3)
            # self.ax.add_patch(ped_circle)

            # These are for trajectory - Comment them for no trajectories
            x = [point[0] for point in ped.trajectory]
            y = [point[1] for point in ped.trajectory]
            self.ax.scatter(x,y,s=2)

        # Set plot limits
        self.ax.set_xlim(self.leftbound-1, self.rightbound+1)
        self.ax.set_ylim(self.bottombound-1, self.topbound+1)
        
        # Add a legend
        self.ax.legend(fontsize=30)
        
        # Display the plot
        plt.draw()
        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.close()


if __name__ == '__main__':

    env = gym.make('SocialForceEnv-v0')
    #env = make_vec_env('SocialForceEnv-v0', n_envs=1)
    #env = SocialForceEnv()
    obs = env.reset(seed=42)
    done = False
    total_reward = 0
    
    while not done:
        action = env.action_space.sample()  # Sample random action
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        env.render()
        #env.envs[0].env.env.render()
    
    print("Total Reward:", total_reward)
    env.close()

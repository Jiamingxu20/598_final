from __future__ import print_function
import gym
from gym import spaces
import math

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions
from diffusion_policy.devices.spacemouse import SpaceMouse


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom


class PlanarSweepingEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            reset_to_state=None,
            num_agents=1,
            sincos_vs_2points=False,
            goal_selection="L"

        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100 # was 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        self.balls_quantity = 18
        self.balls_radius = 12
        self.reward = 0

        self.sincos_vs_2points = sincos_vs_2points # True = sin/cos representations; False = 2points representations
        self.goal_selection = goal_selection 

        Dagentpos = 2
        Dagentsincos = 2
        Dagent2points = 4
        Dagent = Dagentpos + Dagentsincos if self.sincos_vs_2points else Dagentpos + Dagent2points

        Dobs = self.balls_quantity * 2 + Dagent

        # agent_pos, block_pos, block_angle, U_object_pos, U_object_angle
        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        high[:] = 1.

        self.observation_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float64)

        # positional goal for agent
        if self.sincos_vs_2points:
            self.action_space = spaces.Box(
                low=np.array([0,0,-1,-1], dtype=np.float64),
                high=np.array([ws,ws,1,1], dtype=np.float64),
                shape=(Dagent,),
                dtype=np.float64
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([0,0,0,0,0,0], dtype=np.float64),
                high=np.array([ws,ws,ws,ws,ws,ws], dtype=np.float64),
                shape=(Dagent,),
                dtype=np.float64
            )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action1 = None
        self.latest_action2 = None
        self.reset_to_state = reset_to_state

        self.agentmode = 0
        self.draw_keypoint = False

        self.done = False

        self.print_counter = 0
        self.pusher_length = 70

    def select_agent_mode(self, agentmode):
        self.agentmode = agentmode

    def select_draw_keypoint(self, draw_keypoint):
        self.draw_keypoint = draw_keypoint

    # True = sin/cos representations; False = 2points representations
    def select_sincos_vs_2points(self, sincos_vs_2points):
        self.sincos_vs_2points = sincos_vs_2points

    # I: I-shape goal
    # L: L-shape goal
    def select_goal(self, goal_selection):
        self.goal_selection = goal_selection

    def reset(self):
        seed = self._seed
        self.print_counter = 0
        self._setup()
        if self.block_cog is not None:
            self.L0_object.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        rs = np.random.RandomState(seed=seed)
        self.pos_Balls_init = np.array([])
        if self.agentmode == 0:
            if state is None:
                # Balls starting position
                for i in range(self.balls_quantity):
                    x = rs.randint(30, 470)
                    y = rs.randint(30, 470)
                    self.pos_Balls_init = np.append(self.pos_Balls_init, x)
                    self.pos_Balls_init = np.append(self.pos_Balls_init, y)
                state = self.pos_Balls_init
                state = np.insert(state, 0, np.pi*rs.randint(0, 360)/360)
                state = np.insert(state, 0, rs.randint(50, 450))
                state = np.insert(state, 0, rs.randint(50, 450))
        else:
            if state is None:
                for i in range(self.balls_quantity):
                    x = rs.randint(30, 470)
                    y = rs.randint(30, 470)
                    self.pos_Balls_init = np.append(self.pos_Balls_init, x)
                    self.pos_Balls_init = np.append(self.pos_Balls_init, y)
                state = self.pos_Balls_init
                state = np.insert(state, 0, np.pi*rs.randint(0, 360)/360)
                state = np.insert(state, 0, rs.randint(50, 450))
                state = np.insert(state, 0, rs.randint(50, 450))
                state = np.insert(state, 0, np.pi*rs.randint(0, 360)/360)
                state = np.insert(state, 0, rs.randint(20, 300))
                state = np.insert(state, 0, rs.randint(20, 300))

        # Calculate point1 and point2 and sincos for agents to replace angle
        self.sincos_agent1 = [np.sin(self.agent1.angle), np.cos(self.agent1.angle)]
        self.point1_agent1 = [self.agent1.position[0] + np.cos(self.agent1.angle)*self.pusher_length, self.agent1.position[1] + np.sin(self.agent1.angle)*self.pusher_length]
        self.point2_agent1 = [self.agent1.position[0] - np.cos(self.agent1.angle)*self.pusher_length, self.agent1.position[1] - np.sin(self.agent1.angle)*self.pusher_length]
        if (self.agentmode != 0):
            self.sincos_agent2 = [np.sin(self.agent2.angle), np.cos(self.agent2.angle)]
            self.point1_agent2 = [self.agent2.position[0] + np.cos(self.agent2.angle)*self.pusher_length, self.agent2.position[1] + np.sin(self.agent2.angle)*self.pusher_length]
            self.point2_agent2 = [self.agent2.position[0] - np.cos(self.agent2.angle)*self.pusher_length, self.agent2.position[1] - np.sin(self.agent2.angle)*self.pusher_length]


        self._set_state(state)
        observation = self._get_obs()
        return observation


    def step(self, action):
        self.print_counter += 1
        action1 = None
        action2 = None
        if not action is None:
            if self.agentmode == 0:           # Mouse
                action1 = action
                self.handle_agent_mode_0(action1)
            elif self.agentmode == 1:         # Mouse & Space Mouse
                if (self.select_sincos_vs_2points):
                    action1, action2 = action[:4], action[4:]
                else: 
                    action1, action2 = action[:6], action[6:]
                self.handle_agent_mode_1(action1, action2)
            elif self.agentmode == 2:         # Space Mouse & Space Mouse
                if (self.select_sincos_vs_2points):
                    action1, action2 = action[:4], action[4:]
                else: 
                    action1, action2 = action[:6], action[6:]
                self.handle_agent_mode_2(action1, action2)

        # Compute Done and Reward
        intersection_counter = 0

        scale = 4
        for i in range(self.balls_quantity):
            curr_x = self.Balls[i].position[0] - 250
            curr_y = self.Balls[i].position[1] - 250
            if (self.goal_selection == "I"):
                # I-shape
                if curr_y >= scale*24 and curr_y <= scale*30 and curr_x >= scale*-20 and curr_x <= scale*20: # top
                    intersection_counter += 1
                elif curr_y >= scale*-24 and curr_y <= scale*24 and curr_x >= scale*-3 and curr_x <= scale*3: # middle
                    intersection_counter += 1
                elif curr_y >= scale*-30 and curr_y <= scale*-24 and curr_x >= scale*-20 and curr_x <= scale*20: # bottom
                    intersection_counter += 1
            elif (self.goal_selection == "L"):
                # L-shape
                if curr_y >= scale*30 and curr_y <= scale*36 and curr_x >= scale*-24 and curr_x <= scale*24:   # _
                    intersection_counter += 1
                if curr_y >= scale*-30 and curr_y <= scale*36 and curr_x >= scale*-24 and curr_x <= scale*-18: #  |
                    intersection_counter += 1

        # Update the pos_Balls
        for i in range(self.balls_quantity):
            self.pos_Balls[i*2] = self.Balls[i].position[0]
            self.pos_Balls[i*2+1] = self.Balls[i].position[1]
        
        # Compute Done and Reward
        succ_balls_interc = int(self.balls_quantity * self.success_threshold)
        self.done = intersection_counter >= succ_balls_interc
        reward = np.clip(intersection_counter/succ_balls_interc, 0, 1)
        self.reward = reward
        
        # if (self.print_counter % 100 == 0):
        #     print("reward: ", reward)
        #     print("intersection_counter / succ_balls_interc: ", intersection_counter, " / ", succ_balls_interc)

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, self.done, info
    
    def handle_agent_mode_0(self, action1):
        self.simulate_agent(self.agent1, self.sim_hz, self.control_hz, self.k_p, self.k_v, action1)

    def handle_agent_mode_1(self, action1, action2):
        self.handle_agent_mode_0(action1)
        self.simulate_agent(self.agent2, self.sim_hz, self.control_hz, self.k_p, self.k_v, action2)

    def handle_agent_mode_2(self, action1, action2):
        self.simulate_agent(self.agent1, self.sim_hz, self.control_hz, self.k_p, self.k_v, action1)
        self.simulate_agent(self.agent2, self.sim_hz, self.control_hz, self.k_p, self.k_v, action2)

    def simulate_agent(self, agent, sim_hz, control_hz, k_p, k_v, action):
        dt = 1.0 / sim_hz
        self.n_contact_points = 0
        n_steps = sim_hz // control_hz
        if action is not None:
            for _ in range(n_steps):
                position = action[:2]
                if (self.sincos_vs_2points):
                    sincos = action[2:4]
                    agent.angle = math.atan2(sincos[0], sincos[1])
                else:
                    point1 = action[2:4]
                    point2 = action[4:6] # Not in use 
                    cossin = ((point1[0] - position[0])/self.pusher_length, (point1[1] - position[1])/self.pusher_length)
                    if (cossin[0] == 0):
                        cossin[0] += 1e-8
                    agent.angle = math.atan2(cossin[1], (cossin[0]))

                acceleration = k_p * (position - agent.position) + k_v * (Vec2d(0, 0) - agent.velocity)
                agent.velocity += acceleration * dt
                self.space.step(dt)

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent1.position).length < 30:
                self.teleop = True
                if (self.sincos_vs_2points):
                    act = (mouse_position[0], mouse_position[1], self.sincos_agent1[0], self.sincos_agent1[1])
                else:
                    act = (mouse_position[0], mouse_position[1], self.point1_agent1[0], self.point1_agent1[1],
                           self.point2_agent1[0], self.point2_agent1[1])
            return act
        return TeleopAgent(act)
    
    def teleop_agent_spacemouse(self, select = 0):
        # global mouse_positionx, mouse_positiony
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        # if select == 0:
        #     device0 = SpaceMouse(pos_sensitivity=100.0, rot_sensitivity=100.0, SpaceMouseNumber=0)
        # else:
        #     device0 = SpaceMouse(pos_sensitivity=100.0, rot_sensitivity=100.0, SpaceMouseNumber=1)
        # device0.start_control()
        if select == 0:
            device0 = SpaceMouse(pos_sensitivity=2000.0, rot_sensitivity=2000.0, SpaceMouseNumber=0)
        else:
            device0 = SpaceMouse(pos_sensitivity=2000.0, rot_sensitivity=2000.0, SpaceMouseNumber=1)
        device0.start_control()
        def act(obs):
            state0 = device0.get_controller_state()
            dpos0 = (state0["dpos"])
            raw_drotation = (state0["raw_drotation"])
            angle = raw_drotation[2]
            
            if (self.agentmode == 0) or (self.agentmode == 2 and select == 0):
                newpos = Vec2d(self.agent1.position[0] + dpos0[1], self.agent1.position[1] + dpos0[0])
                
                if int(newpos[0]) > 5 and int(newpos[0]) < 500 and int(newpos[1]) > 5 and int(newpos[1]) < 500:
                    self.agent1.position = newpos
                    self.agent1.angle = angle*0.005 + self.agent1.angle
                
                    # Keep the angle between 0 and 2*pi
                    if (self.agent1.angle >= 2*np.pi):
                        self.agent1.angle -= 2*np.pi 
                    elif (self.agent1.angle < 0):
                        self.agent1.angle += 2*np.pi 

                    # Calculate point1 and point2 and sincos for Angle representations
                    self.sincos_agent1 = [np.sin(self.agent1.angle), np.cos(self.agent1.angle)]
                    self.point1_agent1 = [self.agent1.position[0] + np.cos(self.agent1.angle)*self.pusher_length, self.agent1.position[1] + np.sin(self.agent1.angle)*self.pusher_length]
                    self.point2_agent1 = [self.agent1.position[0] - np.cos(self.agent1.angle)*self.pusher_length, self.agent1.position[1] - np.sin(self.agent1.angle)*self.pusher_length]

                # Action 
                if (self.sincos_vs_2points):
                    act = (self.agent1.position[0], self.agent1.position[1], self.sincos_agent1[0], self.sincos_agent1[1])
                else:
                    act = (self.agent1.position[0], self.agent1.position[1], self.point1_agent1[0], self.point1_agent1[1],
                           self.point2_agent1[0], self.point2_agent1[1])
                
                return act
            
            elif (self.agentmode == 1) or (self.agentmode == 2 and select == 1):
                newpos = Vec2d(self.agent2.position[0] + dpos0[1], self.agent2.position[1] + dpos0[0])

                if int(newpos[0]) > 5 and int(newpos[0]) < 500 and int(newpos[1]) > 5 and int(newpos[1]) < 500:
                    self.agent2.position = newpos
                    self.agent2.angle = angle*0.005 + self.agent2.angle
                
                    # Keep the angle between 0 and 2*pi
                    if (self.agent2.angle >= 2*np.pi):
                        self.agent2.angle -= 2*np.pi 
                    elif (self.agent2.angle < 0):
                        self.agent2.angle += 2*np.pi 

                    # Calculate point1 and point2 and sincos for Angle representations
                    self.sincos_agent2 = [np.sin(self.agent2.angle), np.cos(self.agent2.angle)]
                    self.point1_agent2 = [self.agent2.position[0] + np.cos(self.agent2.angle)*self.pusher_length, self.agent2.position[1] + np.sin(self.agent2.angle)*self.pusher_length]
                    self.point2_agent2 = [self.agent2.position[0] - np.cos(self.agent2.angle)*self.pusher_length, self.agent2.position[1] - np.sin(self.agent2.angle)*self.pusher_length]

                # Action 
                if (self.sincos_vs_2points):
                    act = (self.agent2.position[0], self.agent2.position[1], self.sincos_agent2[0], self.sincos_agent2[1])
                else:
                    act = (self.agent2.position[0], self.agent2.position[1], self.point1_agent2[0], self.point1_agent2[1],
                           self.point2_agent2[0], self.point2_agent2[1])
                
                return act

        return TeleopAgent(act)


    def _get_obs(self):

        if self.agentmode == 0:
            obs = self.pos_Balls
            obs = np.concatenate([obs, np.array([self.agent1.position[0], self.agent1.position[1]]) ])
            if (self.sincos_vs_2points):
                obs = np.concatenate((obs, tuple(self.sincos_agent1)), axis=-1)
            else:
                obs = np.concatenate((obs, tuple(self.point1_agent1)), axis=-1)
                obs = np.concatenate((obs, tuple(self.point2_agent1)), axis=-1)
        else:
            obs = self.pos_Balls
            obs = np.concatenate((obs, tuple(self.agent1.position)), axis=-1)
            if (self.sincos_vs_2points):
                obs = np.concatenate((obs, tuple(self.sincos_agent1)), axis=-1)
            else:
                obs = np.concatenate((obs, tuple(self.point1_agent1)), axis=-1)
                obs = np.concatenate((obs, tuple(self.point2_agent1)), axis=-1)

            obs = np.concatenate((obs, tuple(self.agent2.position)), axis=-1)
            if (self.sincos_vs_2points):
                obs = np.concatenate((obs, tuple(self.sincos_agent2)), axis=-1)
            else:
                obs = np.concatenate((obs, tuple(self.point1_agent2)), axis=-1)
                obs = np.concatenate((obs, tuple(self.point2_agent2)), axis=-1)

        return obs
    
    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        if self.agentmode == 0:
            info = {
                'pos_agent1': np.array(self.agent1.position),
                'ang_agent1': np.array(self.agent1.angle),
                'vel_agent1': np.array(self.agent1.velocity),
                "sincos_agent1": np.array(self.sincos_agent1),
                "point1_agent1": np.array(self.point1_agent1),
                "point2_agent1": np.array(self.point2_agent1),
                'pos_Balls': np.array(self.pos_Balls),
                'goal': self.goal_pose,
                'n_contacts': n_contact_points_per_step,
                'success': self.done,
                'reward': self.reward}
        else: 
            info = {
                'pos_agent1': np.array(self.agent1.position),
                'ang_agent1': np.array(self.agent1.angle),
                'vel_agent1': np.array(self.agent1.velocity),
                "sincos_agent1": np.array(self.sincos_agent1),
                "point1_agent1": np.array(self.point1_agent1),
                "point2_agent1": np.array(self.point2_agent1),

                'pos_agent2': np.array(self.agent2.position),
                'ang_agent2': np.array(self.agent2.angle),
                'vel_agent2': np.array(self.agent2.velocity),
                "sincos_agent2": np.array(self.sincos_agent2),
                "point1_agent2": np.array(self.point1_agent2),
                "point2_agent2": np.array(self.point2_agent2),

                'pos_Balls': np.array(self.pos_Balls),
                'goal': self.goal_pose,
                'n_contacts': n_contact_points_per_step,
                
                'success': self.done,
                'reward': self.reward}
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas
        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        for shape in self.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(self.goal_bodys.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, pygame.Color(18, 4, 76), goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        # Draw keypoints
        if (self.draw_keypoint):
            pygame.draw.circle(canvas, pygame.Color(0, 0, 0), ([self.agent1.position[0] + np.cos(self.agent1.angle)*self.pusher_length, self.agent1.position[1] + np.sin(self.agent1.angle)*self.pusher_length]), 3)
            pygame.draw.circle(canvas, pygame.Color(0, 0, 0), ([self.agent1.position[0] - np.cos(self.agent1.angle)*self.pusher_length, self.agent1.position[1] - np.sin(self.agent1.angle)*self.pusher_length]), 3)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # the clock is already ticked during in step for "human"

        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action1 is not None):
                action1 = np.array(self.latest_action1)
                coord1 = (action1 / 512 * 96).astype(np.int32)
                marker_size1 = int(8/96*self.render_size)
                thickness1 = int(1/96*self.render_size)
                cv2.drawMarker(img, coord1,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size1, thickness=thickness1)
                
            if self.render_action and (self.latest_action2 is not None):
                action2 = np.array(self.latest_action2)
                coord2 = (action2 / 512 * 96).astype(np.int32)
                marker_size2 = int(8/96*self.render_size)
                thickness2 = int(1/96*self.render_size)
                cv2.drawMarker(img, coord2,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size2, thickness=thickness2)
        
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if self.agentmode == 0:
            if isinstance(state, np.ndarray):
                state = state.tolist()
            pos_agent1 = state[:2]
            ang_agent1 = state[2]
            self.pos_Balls = np.array([])
            for i in range(self.balls_quantity):
                self.pos_Balls = np.append(self.pos_Balls, state[3+i*2:5+i*2])
            self.agent1.position = pos_agent1
            self.agent1.angle = ang_agent1
        else:
            if isinstance(state, np.ndarray):
                state = state.tolist()
            pos_agent1 = state[:2]
            ang_agent1 = state[2]
            pos_agent2 = state[3:5]
            ang_agent2 = state[5]
            self.pos_Balls = np.array([])
            for i in range(self.balls_quantity):
                self.pos_Balls = np.append(self.pos_Balls, state[6+i*2:8+i*2])
            self.agent1.position = pos_agent1
            self.agent2.position = pos_agent2
            self.agent1.angle = ang_agent1
            self.agent2.angle = ang_agent2

        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.

        # for compatibility with legacy data
        for i in range(self.balls_quantity):
            self.Balls[i].position = [self.pos_Balls[i*2], self.pos_Balls[i*2+1]]
        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    


    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Add walls
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent
        if self.agentmode == 0:
            self.agent1 = self.add_rectangle((270, 250), 0)
        else:
            self.agent1 = self.add_rectangle((270, 250), 0)
            self.agent2 = self.add_rectangle((250, 270), 0)

        # Add Balls
        self.Balls = list()
        for i in range(self.balls_quantity):
            rs = np.random#.RandomState(seed=0)
            x = rs.randint(30, 470)
            y = rs.randint(30, 470)
            self.Balls.append(self.add_ball((x, y), self.balls_radius))
        
        
        # Add Goals
        if (self.goal_selection == "I"):
            self.goal_bodys = self.add_I_goals((250, 250), 0, 4)
            # print("I goal")
        elif (self.goal_selection == "L"):
            self.goal_bodys = self.add_L_goals((250, 250), 0, 4)
            # print("L goal")

        self.goal_pose = np.array([250, 250, 0])
        
        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.9


    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_ball(self, position, radius):
        body = pymunk.Body()
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        shape.density = 10000
        shape.elasticity = 1
        self.space.add(body, shape)
        return body
    
    def add_rectangle(self, position, angle):
        vertices = [(self.pusher_length, 10),
                    (-self.pusher_length, 10),
                    (-self.pusher_length, -10),
                    (self.pusher_length, -10)]
        body = pymunk.Body(100000)
        body.position = position
        body.angle = angle
        shape = pymunk.Poly(body, vertices)
        shape.color = pygame.Color('RoyalBlue')
        shape.density = 10000
        shape.elasticity = 1
        self.space.add(body, shape)
        return body


    def add_I_goals(self, position, angle, scale):
        mass = 1
        mask=pymunk.ShapeFilter.ALL_MASKS()

        vertices1 = [(scale*-20, scale*30),
                     (scale*20,  scale*30),
                     (scale*20,  scale*24),
                     (scale*-20, scale*24)]
        vertices2 = [(scale*-3, scale*24),
                     (scale*3,  scale*24),
                     (scale*3,  scale*-24),
                     (scale*-3, scale*-24)]
        vertices3 = [(scale*-20, scale*-30),
                     (scale*20,  scale*-30),
                     (scale*20,  scale*-24),
                     (scale*-20, scale*-24)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        inertia3 = pymunk.moment_for_poly(mass, vertices=vertices3)
        
        body = pymunk.Body(mass, inertia1+inertia2+inertia3)
        
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.filter = pymunk.ShapeFilter(mask=mask)

        body.center_of_gravity = (shape1.center_of_gravity+shape2.center_of_gravity+shape3.center_of_gravity)/3
        body.position = position
        body.angle = angle
        body.friction = 1

        self.shapes = set()
        self.shapes.add(shape1)
        self.shapes.add(shape2)
        self.shapes.add(shape3)

        return body




    def add_L_goals(self, position, angle, scale):
        mass = 1

        vertices1 = [(scale*-24, scale*36),
                     (scale*24, scale*36),
                     (scale*24, scale*30),
                     (scale*-24, scale*30)]
        vertices2 = [(scale*-18, scale*30),
                     (scale*-24, scale*30),
                     (scale*-24, scale*-30),
                     (scale*-18, scale*-30)]
        
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
    
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1

        self.shapes = set()
        self.shapes.add(shape1)
        self.shapes.add(shape2)
        
        return body

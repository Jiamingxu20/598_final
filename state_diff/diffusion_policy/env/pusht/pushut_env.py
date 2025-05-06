from __future__ import print_function
import numpy as np
from gym import spaces
from pymunk.vec2d import Vec2d
from diffusion_policy.env.pusht.pusht_env import PushTEnv, pymunk_to_shapely
import pygame
import gym
from gym import spaces
import collections
import numpy as np
import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions
# from diffusion_policy.devices.spacemouse import SpaceMouse
import math
import copy



class PushUTEnv(PushTEnv):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle, U_object_pos, U_object_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2,ws,ws,np.pi*2], dtype=np.float64),
            shape=(8,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
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

        # Mode 0: Single Mouse
        # Mode 1: Space Mouse & Mouse
        # Mode 2: Space Mouse & Space Mouse

        self.done = False

        self.U_object_keys = np.array([
                                  (15, 0), (45, 0),
                                  (45, 30), (45, 60),
                                  (45, 90), (45, 120),
                                  (15, 120), (15, 90),
                                  (15, 60), (15, 30),
                                #   (0, 0), (0, 0),
                                  (-15, 30), (-15, 60),
                                  (-15, 90), (-15, 120),
                                  (-45, 120), (-45, 90),
                                  (-45, 60), (-45, 30),
                                  (-45, 0), (-15, 0),
                                  ])
        self.T_object_keys = np.array([
                                  (15, 0), (45, 0),
                                  (45, 30), (15, 30),
                                  (15, 60), (15, 90),
                                  (15, 120), 
                                #   (0, 0), 
                                  (-15, 120), (-15, 90), 
                                  (-15, 60), (-15, 30), 
                                  (-45, 30), (-45, 0),
                                  (-15, 0)
                                  ])


    def select_agent_mode(self, agentmode):
        self.agentmode = agentmode
    
    def select_draw_keypoint(self, draw_keypoint):
        self.draw_keypoint = draw_keypoint


    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if self.agentmode == 0:
            if state is None:
                rs = np.random.RandomState(seed=seed)
                state = np.array([
                    rs.randint(50, 450), rs.randint(50, 450),
                    rs.randint(120, 200), rs.randint(120, 200),
                    rs.randn() * 2 * np.pi - np.pi,
                    rs.randint(300, 380), rs.randint(300, 380),
                    rs.randn() * 2 * np.pi - np.pi,
                    ])
        else:
            if state is None:
                rs = np.random.RandomState(seed=seed)
                state = np.array([
                    rs.randint(20, 300), rs.randint(20, 300),
                    rs.randint(50, 450), rs.randint(50, 450),
                    rs.randint(100, 400), rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                    rs.randint(100, 400), rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                    ])
        self._set_state(state)

        observation = self._get_obs()
        return observation


    def _get_structured_keypoints(self, object, keys):
        object_x, object_y = object.position
        keypoints = np.zeros((len(keys), 2))  # Directly allocate with the correct shape
        rotate_theta = -object.angle % (2 * np.pi)

        for x in range(len(keys)):
            object_theta = math.atan2(keys[x][1], keys[x][0]) + rotate_theta
            object_length = np.linalg.norm(keys[x])
            keypoints[x] = [object_x - np.cos(object_theta)*object_length,
                            object_y + np.sin(object_theta)*object_length]

        return keypoints

    def U_object_get_keypoints(self):
        return self._get_structured_keypoints(self.U_object, self.U_object_keys)


    def T_object_get_keypoints(self):
        return self._get_structured_keypoints(self.block, self.T_object_keys)


    def _get_obs(self):
        if self.agentmode == 0:
            obs = np.array(
                tuple(self.agent1.position) \
                + tuple(self.block.position) \
                + tuple(self.U_object.position) \
                + (self.block.angle % (2 * np.pi),)
                + (self.U_object.angle % (2 * np.pi),))
        else:
            obs = np.array(
                tuple(self.agent1.position) \
                + tuple(self.agent2.position) \
                + tuple(self.block.position) \
                + tuple(self.U_object.position) \
                + (self.block.angle % (2 * np.pi),)
                + (self.U_object.angle % (2 * np.pi),))
        print("obs inside _get_obs_: ", obs)
        return obs
        

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        if self.agentmode == 0:
            info = {
                'pos_agent1': np.array(self.agent1.position),
                'vel_agent1': np.array(self.agent1.velocity),
                'T_object_pose': np.array(list(self.block.position) + [self.block.angle]),
                'U_object_pose': np.array(list(self.U_object.position) + [self.U_object.angle]),
                'T_goal_pose': self.T_goal_pose,
                'U_goal_pose': self.U_goal_pose,
                'n_contacts': n_contact_points_per_step,
                'success': self.done}
        else:
            info = {
                'pos_agent1': np.array(self.agent1.position),
                'vel_agent1': np.array(self.agent1.velocity),
                'pos_agent2': np.array(self.agent2.position),
                'vel_agent2': np.array(self.agent2.velocity),
                'T_object_pose': np.array(list(self.block.position) + [self.block.angle]),
                'U_object_pose': np.array(list(self.U_object.position) + [self.U_object.angle]),
                'T_goal_pose': self.T_goal_pose,
                'U_goal_pose': self.U_goal_pose,
                'n_contacts': n_contact_points_per_step,
                'success': self.done}
        return info

    def step(self, action1, action2=None):

        if self.agentmode == 0:           # Mouse
            self.handle_agent_mode_0(action1)
        elif self.agentmode == 1:         # Mouse & Space Mouse
            self.handle_agent_mode_1(action1, action2)
        elif self.agentmode == 2:         # Space Mouse & Space Mouse
            self.handle_agent_mode_2(action1, action2)

        # compute reward
        goal_body = self._get_goal_pose_body(self.T_goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)

        U_goal_body = self._get_goal_pose_body(self.U_goal_pose)
        U_goal_geom = pymunk_to_shapely(U_goal_body, self.U_object.shapes)

        block_geom = pymunk_to_shapely(self.block, self.block.shapes)
        U_object_geom = pymunk_to_shapely(self.U_object, self.U_object.shapes)

        block_intersection_area = goal_geom.intersection(block_geom).area
        U_object_intersection_area = U_goal_geom.intersection(U_object_geom).area

        block_goal_area = goal_geom.area
        U_object_goal_area = U_goal_geom.area

        block_coverage = block_intersection_area / block_goal_area
        U_object_coverage = U_object_intersection_area / U_object_goal_area

        reward = np.clip((block_coverage + U_object_coverage) / (2 * self.success_threshold), 0, 1)
        self.done = (block_coverage > self.success_threshold) and (U_object_coverage > self.success_threshold)

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
                acceleration = k_p * (action - agent.position) + k_v * (Vec2d(0, 0) - agent.velocity)
                agent.velocity += acceleration * dt
                self.space.step(dt)

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
        goal_body = self._get_goal_pose_body(self.T_goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        goal_body = self._get_goal_pose_body(self.U_goal_pose)
        for shape in self.U_object.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        if self.draw_keypoint:
            # Draw structured_keypoint
            U_st_keypoints = self.U_object_get_keypoints()
            T_st_keypoints = self.T_object_get_keypoints()
            for i in range (20):
                curr_pos = np.array([U_st_keypoints[i][0], U_st_keypoints[i][1]])
                # curr_coord = (curr_pos / 512 * 96).astype(np.int32)
                pygame.draw.circle(canvas, self.structured_keypoint_color, curr_pos, 15)
            for i in range (14):
                curr_pos2 = np.array([T_st_keypoints[i][0], T_st_keypoints[i][1]])
                # curr_coord2 = (curr_pos2 / 512 * 96).astype(np.int32)
                pygame.draw.circle(canvas, self.structured_keypoint_color, curr_pos2, 15)
            # Draw keypoint
            obs_key = self._get_obs()
            UT_keypoints = obs_key.reshape(2,-1)[0].reshape(-1,2)[:9+13]
            for i in range (19):
                curr_pos3 = np.array([UT_keypoints[i][0], UT_keypoints[i][1]])
                # curr_coord3 = (curr_pos3 / 512 * 96).astype(np.int32)
                pygame.draw.circle(canvas, self.keypoint_color, curr_pos3, 13)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

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


    def _set_state(self, state):
        if self.agentmode == 0:
            if isinstance(state, np.ndarray):
                state = state.tolist()
            pos_agent1 = state[:2]
            pos_block = state[2:4]
            rot_block = state[4]
            pos_U = state[5:7]
            rot_U = state[7]
            self.agent1.position = pos_agent1
        else:
            if isinstance(state, np.ndarray):
                state = state.tolist()
            pos_agent1 = state[:2]
            pos_agent2 = state[2:4]
            pos_block = state[4:6]
            rot_block = state[6]
            pos_U = state[7:9]
            rot_U = state[9]
            self.agent1.position = pos_agent1
            self.agent2.position = pos_agent2

        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
            self.U_object.position = pos_U
            self.U_object.angle = rot_U
        else:
            self.block.angle = rot_block
            self.block.position = pos_block
            self.U_object.angle = rot_U
            self.U_object.position = pos_U

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent1.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def teleop_agent_spacemouse(self, select = 0):
        # global mouse_positionx, mouse_positiony
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        if select == 0:
            device0 = SpaceMouse(pos_sensitivity=400.0, rot_sensitivity=400.0, SpaceMouseNumber=0)
        else:
            device0 = SpaceMouse(pos_sensitivity=400.0, rot_sensitivity=400.0, SpaceMouseNumber=1)
        device0.start_control()
        
        def act(obs):
            if self.agentmode == 1:
                state0 = device0.get_controller_state()
                dpos0 = (state0["dpos"])
                newpos = Vec2d(self.agent2.position[0] + dpos0[1], self.agent2.position[1] + dpos0[0])
                if int(newpos[0]) > 5 and int(newpos[0]) < 500 and int(newpos[1]) > 5 and int(newpos[1]) < 500:
                    self.agent2.position = newpos
                return (self.agent2.position[0], self.agent2.position[1])
            elif self.agentmode == 2:
                if select == 0:
                    state0 = device0.get_controller_state()
                    dpos0 = (state0["dpos"])
                    newpos = Vec2d(self.agent1.position[0] + dpos0[1], self.agent1.position[1] + dpos0[0])
                    if int(newpos[0]) > 5 and int(newpos[0]) < 500 and int(newpos[1]) > 5 and int(newpos[1]) < 500:
                        self.agent1.position = newpos
                    return (self.agent1.position[0], self.agent1.position[1])
                elif select == 1:
                    state0 = device0.get_controller_state()
                    dpos0 = (state0["dpos"])
                    newpos = Vec2d(self.agent2.position[0] + dpos0[1], self.agent2.position[1] + dpos0[0])
                    if int(newpos[0]) > 5 and int(newpos[0]) < 500 and int(newpos[1]) > 5 and int(newpos[1]) < 500:
                        self.agent2.position = newpos
                    return (self.agent2.position[0], self.agent2.position[1])
                
        return TeleopAgent(act)


    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        if self.agentmode == 0:
            self.agent1 = self.add_circle((250, 270), 15)
        else:
            self.agent1 = self.add_circle((270, 250), 15)
            self.agent2 = self.add_circle((250, 270), 15)
        
        self.block = self.add_T((400, 400), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.keypoint_color = pygame.Color('black')
        self.structured_keypoint_color = pygame.Color('red')
        self.T_goal_pose = np.array([336,156,np.pi/4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        # self.success_threshold = 0.95    # 95% coverage.
        self.success_threshold = 0.90

        self.U_object = self.add_U((200, 200), 0)
        U_goal_pos = (self.T_goal_pose[:2].copy())
        goal_distance = 107
        U_goal_pos[0] -= goal_distance
        U_goal_pos[1] += goal_distance
        self.U_goal_pose = np.array([*U_goal_pos,np.pi*5/4])  # x, y, theta (in radians)



    def add_U(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        clearance = 3
        vertices1 = [(-scale*1.5, scale), (scale*1.5, scale), (scale*1.5, 0), (-scale*1.5, 0)]
        vertices2 = [(-scale*1.5, scale), (-scale/2 - clearance, scale), (-scale/2 - clearance, length*scale), (-scale*1.5, length*scale)]
        vertices3 = [(scale/2 + clearance, scale), (scale*1.5, scale), (scale*1.5, length*scale), (scale/2 + clearance, length*scale)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        inertia3 = pymunk.moment_for_poly(mass, vertices=vertices3)
        body = pymunk.Body(mass, inertia1 + inertia2 + inertia3)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape3.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity + shape3.center_of_gravity) / 3
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2, shape3)
        return body




    def add_T(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-scale*1.5, scale),
                     ( scale*1.5, scale),
                     ( scale*1.5, 0),
                     (-scale*1.5, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                     (-scale/2, length*scale),
                     ( scale/2, length*scale),
                     ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
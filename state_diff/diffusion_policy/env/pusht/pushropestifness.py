from __future__ import print_function
import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.constraints
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


class PushRopeEnv(gym.Env):
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
        self.sim_hz = 100 # was 100
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

        self.rope_length = 18
        self.rope_ballsize = 10
        self.rope_ballspacing = 27

        self.done = False

    def select_agent_mode(self, agentmode):
        self.agentmode = agentmode

    def select_draw_keypoint(self, draw_keypoint):
        self.draw_keypoint = draw_keypoint

    def set_rope_parameters(self, rope_length, rope_ballsize, rope_ballspacing):
        self.rope_length = rope_length
        self.rope_ballsize = rope_ballsize
        self.rope_ballspacing = rope_ballspacing

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.L0_object.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if self.agentmode == 0:
            if state is None:
                rs = np.random.RandomState(seed=seed)
                # Balls starting position
                self.pos_Balls_init = np.array([])
                rand_x = 0
                for i in range(self.rope_length):
                    if i == 0:
                        rand_x = rs.randint(25, 475)
                        self.pos_Balls_init = np.append(self.pos_Balls_init, rand_x)
                        self.pos_Balls_init = np.append(self.pos_Balls_init, 25)
                    else:
                        self.pos_Balls_init = np.append(self.pos_Balls_init, rand_x)
                        self.pos_Balls_init = np.append(self.pos_Balls_init, 25 + i*self.rope_ballspacing)
                state = self.pos_Balls_init
                state = np.insert(state, 0, rs.randint(50, 450))
                state = np.insert(state, 0, rs.randint(50, 450))
        else:
            if state is None:
                rs = np.random.RandomState(seed=seed)
                # Balls starting position
                self.pos_Balls_init = np.array([])
                rand_x = 0
                for i in range(self.rope_length):
                    if i == 0:
                        rand_x = rs.randint(25, 475)
                        self.pos_Balls_init = np.append(self.pos_Balls_init, rand_x)
                        self.pos_Balls_init = np.append(self.pos_Balls_init, 25)
                    else:
                        self.pos_Balls_init = np.append(self.pos_Balls_init, rand_x)
                        self.pos_Balls_init = np.append(self.pos_Balls_init, 25 + i*self.rope_ballspacing)
                state = self.pos_Balls_init
                state = np.insert(state, 0, rs.randint(50, 450))
                state = np.insert(state, 0, rs.randint(50, 450))
                state = np.insert(state, 0, rs.randint(20, 300))
                state = np.insert(state, 0, rs.randint(20, 300))

        self._set_state(state)
        observation = self._get_obs()
        return observation


    def step(self, action1, action2=None):

        if self.agentmode == 0:           # Mouse
            self.handle_agent_mode_0(action1)
        elif self.agentmode == 1:         # Mouse & Space Mouse
            self.handle_agent_mode_1(action1, action2)
        elif self.agentmode == 2:         # Space Mouse & Space Mouse
            self.handle_agent_mode_2(action1, action2)

        # Compute Done and Reward
        part1_intersection_counter = 0
        part2_intersection_counter = 0
        part3_intersection_counter = 0
        for i in range(self.rope_length):
            curr_x = self.Balls[i].position[0]-200
            curr_y = self.Balls[i].position[1]-200
            if curr_y >= 60 and curr_y <= 90 and curr_x >= -90 and curr_x <= 90: # top
                part1_intersection_counter += 1
            elif curr_y >= -90 and curr_y <= -60 and curr_x >= -90 and curr_x <= 90: # middle
                part2_intersection_counter += 1
            elif curr_y <= -0.857*curr_x + 17.15 and curr_y >= -0.857*curr_x - 17.15 and curr_y <= 60 and curr_y >= -60: # bottom
                part3_intersection_counter += 1

        # Update the pos_Balls
        for i in range(self.rope_length ):
            self.pos_Balls[i] = self.Balls[i].position
        
        total_intersection = part1_intersection_counter + part2_intersection_counter + part3_intersection_counter
        self.done = part1_intersection_counter*4/self.rope_length >= self.success_threshold and part2_intersection_counter*3/self.rope_length >= self.success_threshold and part3_intersection_counter*4/self.rope_length >= self.success_threshold 
    
        reward = np.clip(total_intersection/self.rope_length / (2 * self.success_threshold), 0, 1)


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

    def render(self, mode):
        return self._render_frame(mode)

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
            device0 = SpaceMouse(pos_sensitivity=300.0, rot_sensitivity=300.0, SpaceMouseNumber=0)
        else:
            device0 = SpaceMouse(pos_sensitivity=300.0, rot_sensitivity=300.0, SpaceMouseNumber=1)
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


    def _get_obs(self):
        if self.agentmode == 0:
            obs = self.pos_Balls
            obs = np.concatenate((obs, tuple(self.agent1.position)), axis=-1)

        else:
            obs = self.pos_Balls
            obs = np.concatenate((obs, tuple(self.agent1.position)), axis=-1)
            obs = np.concatenate((obs, tuple(self.agent2.position)), axis=-1)
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
                'vel_agent1': np.array(self.agent1.velocity),
                'pos_Balls': np.array(self.pos_Balls),
                'Z_goal': self.Z_goal_pose,
                'n_contacts': n_contact_points_per_step}
        else: 
            info = {
                'pos_agent1': np.array(self.agent1.position),
                'vel_agent1': np.array(self.agent1.velocity),
                'pos_agent2': np.array(self.agent1.position),
                'vel_agent2': np.array(self.agent1.velocity),
                'pos_Balls': np.array(self.pos_Balls),
                'Z_goal': self.Z_goal_pose,
                'n_contacts': n_contact_points_per_step}
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
            goal_points = [pymunk.pygame_util.to_pygame(self.Z_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

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
            self.pos_Balls = list()
            for i in range(self.rope_length):
                self.pos_Balls.append(state[2+i*2:4+i*2])
            self.agent1.position = pos_agent1
        else:
            if isinstance(state, np.ndarray):
                state = state.tolist()
            pos_agent1 = state[:2]
            pos_agent2 = state[2:4]
            self.pos_Balls = list()
            for i in range(self.rope_length):
                self.pos_Balls.append(state[4+i*2:6+i*2])
            self.agent1.position = pos_agent1
            self.agent2.position = pos_agent2

        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.

        # for compatibility with legacy data
        for i in range(self.rope_length):
            self.Balls[i].position = self.pos_Balls[i]
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
            self.agent1 = self.add_ball((250, 270), 13)
        else:
            self.agent1 = self.add_ball((270, 250), 13)
            self.agent2 = self.add_ball((250, 270), 13)

        # Add Balls
        self.Balls = list()
        for i in range(self.rope_length):
            self.Balls.append(self.add_ball((50+i*self.rope_ballspacing, 50), self.rope_ballsize))
 

        # # Add Strings
        # self.Strings = list()
        # for i in range(self.rope_length-1):
        #     self.Strings.append(self.add_string(self.Balls[i], self.Balls[i+1]))

        for i in range(self.rope_length-1):
            self.connect_balls(self.Balls[i], self.Balls[i+1])
            print("constraintsadded")

        # # Add Goals
        self.goal_color = pygame.Color('LightGreen')
        self.Z_body = self.add_Z_goals((200, 200), 0)
        self.Z_goal_pose = np.array([200, 200, np.pi/4])
        
        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.9    # 80% coverage.


    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_ball(self, position, radius):
        body = pymunk.Body()
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        shape.density = 1
        shape.elasticity = 1
        self.space.add(body, shape)
        return body

    def add_box(self, position, height=3, width=3):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_string(self, body1, body2):
        body_1 = body1
        body_2 = body2
        joint = pymunk.PinJoint(body_1, body_2)
        self.space.add(joint)
        return joint

    def add_Z_goals(self, position, angle):
        color = 'LightSlateGray'
        mass = 1
        mask=pymunk.ShapeFilter.ALL_MASKS()

        vertices1 = [(-90, -90),
                     (90,  -90),
                     (90,  -60),
                     (-90, -60)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(50, -60),
                     (90,  -60),
                     (-50, 60),
                     (-90, 60)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        vertices3 = [(-90, 60),
                     (90,  60),
                     (90,  90),
                     (-90, 90)]
        inertia3 = pymunk.moment_for_poly(mass, vertices=vertices3)

        body = pymunk.Body(mass, inertia1+inertia2+inertia3)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        
        shape1.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.color = pygame.Color(color)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.color = pygame.Color(color)
        shape3.filter = pymunk.ShapeFilter(mask=mask)

        body.center_of_gravity = (shape1.center_of_gravity+shape2.center_of_gravity+shape3.center_of_gravity)/3
        body.position = position
        body.angle = angle
        body.friction = 1
        # self.space.add(body, shape1, shape2, shape3)
        self.shapes = set()
        self.shapes.add(shape1)
        self.shapes.add(shape2)
        self.shapes.add(shape3)
        
        return body

    def connect_balls(self, ball1, ball2):
        stiffness = 100000000
        damping = 1
        spring = pymunk.DampedRotarySpring(ball1, ball2, np.pi/2, stiffness, damping)
        def post_solve_func(constraint, space):
            print("Hello from pre-solve")
        spring.post_solve = post_solve_func(spring, self.space)
        spring.post_solve = None
        self.space.add(spring)



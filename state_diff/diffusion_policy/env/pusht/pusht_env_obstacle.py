import gym
from gym import spaces

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

from diffusion_policy.env.pusht.pusht_env import PushTEnv

class PushTEnvWithStaticCircle(PushTEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add a new attribute for the static circle
        self.static_circle = None

    def reset(self):
        # Remove the static circle from the last episode
        if self.static_circle is not None:
            self.space.remove(self.static_circle)
            self.static_circle = None

        # Reset the environment
        observation = super().reset()

        # Add a static circle to a random position in the environment
        circle_radius = int(list(self.agent.shapes)[0].radius * 2)  # Slightly larger than the agent
        circle_position = (np.random.randint(50, 450), np.random.randint(50, 450))
        self.static_circle = self.add_static_circle(circle_position, circle_radius)

        return observation

    def add_static_circle(self, position, radius):
        # Create a static body
        body = pymunk.Body(body_type=pymunk.Body.STATIC)

        # Set the position of the static body
        body.position = position

        # Create a circle shape and add it to the static body
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('DarkOrange')

        # Add the static body and its shape to the space
        self.space.add(body, shape)

        return body

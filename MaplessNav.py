import os;
import time

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy, pygame, gymnasium, math


class MaplessNav(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 16}
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "obstacle": (0, 0, 0),
        "agent": (35, 31, 32),
        "target": (255, 255, 0)
    }

    def __init__(self, render_mode=None):

        self.map_size = 7  # m

        self._linear_velocity = 0.1  # m/sec
        self._angular_velocity = 10  # deg/sec
        self._agent_size = 0.34 * 0.5  # m (radius)
        self._lidar_length = 1.5  # m
        self._target_size = 0.7 * 0.5  # m (radius)

        # self.lidar_angles = 15
        # self.lidar_range = [-120, 120]
        self.lidar_angles = 7
        self.lidar_range = [-90, 90]

        self._step_limit = 250

        self.single_observation_space = self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(self.lidar_angles + 2,),
            dtype=numpy.float32
        )

        self.single_action_space = self.action_space = gymnasium.spaces.Box(
            low=numpy.array([-1, -1]),
            high=numpy.array([1, 1])
        )

        # Rendering paramters
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = 500
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Object rect for collision check
        self.agent_rect = None
        self.target_rect = None
        self.lidar_scan_reading = []
        self.walls_rects = []
        self.obstacles_rects = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.random(2) * (self.map_size - 0.6) + 0.3
        self._target_location = self.np_random.random(2) * (self.map_size - 0.6) + 0.3
        self._agent_orientation = self.np_random.random() * 360
        self._obstacles_definition = [
            [3.5 - 1.5, 3.3 - 0.2, 3, 0.4],
            # [1.9 - 0.2, 4.1 - 1.5, 0.4, 3],
            # [5.1 - 0.2, 4.1 - 1.5, 0.4, 3]
        ]
        # self._obstacles_definition = []     

        # Temporary increase the size of the agent to avoid spawining
        # too close to an obstacle
        self._agent_size *= 3
        # If needed, render the screen
        self.agent_rect = self.target_rect = None
        while self._collision_test(self.agent_rect) or self._collision_test(self.target_rect):
            if self.render_mode == "human": self.render_mode = "temp"
            self._agent_location = self.np_random.random(2) * (self.map_size - 0.6) + 0.3
            self._target_location = self.np_random.random(2) * (self.map_size - 0.6) + 0.3
            self._agent_orientation = self.np_random.random() * 360
            self._render_frame()
        if self.render_mode == "temp": self.render_mode = "human"
        self._agent_size /= 3

        # self._agent_location = [3, 3]
        # self._agent_orientation = 0
        # self._target_location = [6, 3]
        # self._obstacles_definition = []
        self._render_frame()

        self._step_counter = 0

        observation = self._get_obs()
        self.old_distance = observation[-1]

        return observation, self._get_info()

    def step(self, action):

        self._step_counter += 1

        # Extract the actions and normalize the linear velocity
        # to be always positive
        linear_action, angular_action = action[0], -action[1]
        linear_action = (linear_action + 1) / 2


        # Perform the actions
        self._agent_orientation += angular_action * self._angular_velocity
        self._agent_location += numpy.array([math.cos(math.radians(self._agent_orientation)), -math.sin(
            math.radians(self._agent_orientation))]) * self._linear_velocity * linear_action

        # If needed, render the screen
        self._render_frame()

        # Initialize the elements to return
        info, reward, terminated, truncated = self._get_info(), 0, False, False
        truncated = self._step_counter > self._step_limit

        observation = self._get_obs()

        # Computation of the reward for collisions
        if info["collision"]:
            #print("COLLISIONE")
            #print(info["collision"])
            terminated = True
            reward = 0

        # Computation of the reward for task completed
        if info["goal-reached"]:
            terminated = True
            reward = 1

        # Computing the basic step reward with the new
        # formula: a bonus to move towards the target adding a nomralization
        # multiplier and a small penalty for each step
        if not terminated:
            reward_multiplier, step_penalty = 3, 0.0001 #0.0001
            new_distance = observation[-1]
            reward = (self.old_distance - new_distance) * reward_multiplier - step_penalty
            self.old_distance = new_distance

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        if self.render_mode == "rgb_array": return self._render_frame()

    def _get_obs(self):

        a = numpy.array(self._agent_location)
        b = numpy.array(self._target_location)
        distance = numpy.linalg.norm(a - b)

        heading = math.degrees(numpy.arctan2(a[1] - b[1], a[0] - b[0])) + 360
        heading = (heading + self._agent_orientation) % 360

        # Normalization step
        heading /= 360
        distance /= (self.map_size * 1.414213)

        # Get the lidar lectures
        if self.lidar_scan_reading == []:
            self.lidar_scan_reading = [0 for _ in range(self.lidar_angles)]

        return self.lidar_scan_reading + [heading, distance]

    def _lidar_scan(self, canvas):

        if self.agent_rect is None: return

        normalized_lidar_length = int(self._lidar_length * self.window_size / self.map_size)
        scans_ticks = []

        # Calculate the center of the origin rectangle
        origin_center = self.agent_rect.center

        # step = (120 - (-121)) / (self.lidar_angles)
        # for angle in numpy.arange(-120, 121, step):
        # step = abs(-120-120) / (self.lidar_angles-1)
        step = abs(self.lidar_range[0] - self.lidar_range[1]) / (self.lidar_angles - 1)

        for i in range(self.lidar_angles):
            angle = self.lidar_range[0] + (i * step)
            angle_rad = math.radians(angle - self._agent_orientation)

            scans_ticks.append([])

            for n in range(normalized_lidar_length):

                lidar_tick_center = [
                    origin_center[0] + 1 * n * math.cos(angle_rad),
                    origin_center[1] + 1 * n * math.sin(angle_rad)
                ]

                lidar_tic_rect = pygame.Rect(lidar_tick_center[0], lidar_tick_center[1], 1, 1)

                if self._collision_test(lidar_tic_rect): break
                scans_ticks[-1].append(lidar_tic_rect)

        for scan_ticks in scans_ticks:
            scan_color = self.colors["green"] if len(scan_ticks) == normalized_lidar_length else self.colors["red"]
            for tick in scan_ticks: pygame.draw.rect(canvas, scan_color, tick)

        self.lidar_scan_reading = [(len(scan_ticks) / normalized_lidar_length) for scan_ticks in scans_ticks]

    def _get_info(self):

        # Check for target reached
        goal_reached = self.agent_rect.colliderect(self.target_rect)

        # a = numpy.array(self._agent_location)
        # b = numpy.array(self._target_location)
        # distance = numpy.linalg.norm(a - b)
        #goal_reached = distance < 0.3

        obstacles = self._collision_test(self.agent_rect)

        infos = {
            "cost": [0.0],
            "goal-reached": goal_reached,
            "collision": obstacles
        }

        return infos

    def _collision_test(self, rect):
        if rect is None: return True

        collision_test_w = [rect.colliderect(wall) for wall in self.walls_rects]

        collision_test_o = [rect.colliderect(wall) for wall in self.obstacles_rects]
        # print(collision_test_o)
        #print(any(collision_test_w + collision_test_o))
        return any(collision_test_w + collision_test_o)

    def _render_frame(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((239, 230, 221))
        # pix_square_size = ( self.window_size / self.size ) 

        self.walls_rects = self._spawn_walls(canvas)
        self.obstacles_rects = self._spawn_obstacls(canvas)

        normalized_agent_size = self._agent_size * self.window_size / self.map_size
        normalized_target_size = self._target_size * self.window_size / self.map_size

        # Now we draw the agent
        self.agent_rect = pygame.draw.circle(
            canvas,
            self.colors["agent"],
            numpy.array(self._agent_location) * self.window_size / self.map_size,
            normalized_agent_size
        )

        # First we draw the target
        self.target_rect = pygame.draw.circle(
            canvas,
            self.colors["target"],
            numpy.array(self._target_location) * self.window_size / self.map_size,
            normalized_target_size
        )

        #  angles = range(0, 360, 360 // 8)
        self._lidar_scan(canvas)

        # Draw a mask for the agent (cover the lidar)
        self.agent_rect = pygame.draw.circle(
            canvas,
            self.colors["agent"],
            numpy.array(self._agent_location) * self.window_size / self.map_size,
            normalized_agent_size
        )

        # Draw a pointer for the orientation of the agent
        pygame.draw.circle(
            canvas,
            self.colors["agent"],
            (numpy.array(self._agent_location) * self.window_size / self.map_size) + \
            [normalized_agent_size * math.cos(math.radians(self._agent_orientation)),
             -normalized_agent_size * math.sin(math.radians(self._agent_orientation))],
            normalized_agent_size / 3,
        )

        if self.render_mode == "human":

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        # rgb_array
        else:
            return numpy.transpose(numpy.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _spawn_obstacls(self, canvas):

        rect_array = []

        for obstacle in self._obstacles_definition:
            # First we draw the target
            rect_array.append(pygame.draw.rect(
                canvas,
                self.colors["obstacle"],
                pygame.Rect(
                    numpy.array([obstacle[0], obstacle[1]]) * self.window_size / self.map_size,

                    numpy.array([obstacle[2], obstacle[3]]) * self.window_size / self.map_size
                ),
            ))

        return rect_array

    def _spawn_walls(self, canvas):

        rect_array = []

        rect_array.append(pygame.draw.rect(
            canvas,
            self.colors["obstacle"],
            pygame.Rect(
                numpy.array([0, 0]),
                (10, self.window_size),
            ),
        ))
        rect_array.append(pygame.draw.rect(
            canvas,
            self.colors["obstacle"],
            pygame.Rect(
                numpy.array([self.window_size - 10, 0]),
                (10, self.window_size),
            ),
        ))
        rect_array.append(pygame.draw.rect(
            canvas,
            self.colors["obstacle"],
            pygame.Rect(
                numpy.array([0, 0]),
                (self.window_size, 10),
            ),
        ))
        rect_array.append(pygame.draw.rect(
            canvas,
            self.colors["obstacle"],
            pygame.Rect(
                numpy.array([0, self.window_size - 10]),
                (self.window_size, 10),
            ),
        ))

        return rect_array

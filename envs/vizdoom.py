import gym
import numpy as np
import vizdoom as vzd
import pathlib

class ViZDoom(gym.Env):
    # Define scenario configurations
    SCENARIOS = {
        'basic': {
            'config_file': 'basic.cfg',
            'actions': [
                [1, 0, 0],  # Move Forward
                [0, 1, 0],  # Turn Left
                [0, 0, 1],  # Turn Right
            ],
            'buttons': [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT]
        },
        'deadly_corridor': {
            'config_file': 'deadly_corridor.cfg',
            'actions': [
                [1, 0, 0, 0, 0, 0],  # Move Forward
                [0, 1, 0, 0, 0, 0],  # Turn Left
                [0, 0, 1, 0, 0, 0],  # Turn Right
                [0, 0, 0, 1, 0, 0],  # Move Left
                [0, 0, 0, 0, 1, 0],  # Move Right
                [0, 0, 0, 0, 0, 1],  # Attack
            ],
            'buttons': [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
                       vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
        },
        'defend_the_center': {
            'config_file': 'defend_the_center.cfg',
            'actions': [
                [1, 0, 0],  # Turn Left
                [0, 1, 0],  # Turn Right
                [0, 0, 1],  # Attack
            ],
            'buttons': [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK]
        },
        'health_gathering': {
            'config_file': 'health_gathering.cfg',
            'actions': [
                [1, 0, 0, 0, 0],  # Move Forward
                [0, 1, 0, 0, 0],  # Turn Left
                [0, 0, 1, 0, 0],  # Turn Right
                [0, 0, 0, 1, 0],  # Move Left
                [0, 0, 0, 0, 1],  # Move Right
            ],
            'buttons': [vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
                       vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT]
        }
    }

    def __init__(self, task, size=(64, 64), action_repeat=1, seed=0, **kwargs):
        super().__init__()

        self._task = task
        self._size = tuple(size) if isinstance(size, list) else size
        self._action_repeat = action_repeat
        self._random = np.random.RandomState(seed)

        # Validate task
        if task not in self.SCENARIOS:
            raise ValueError(f"Unknown VizDoom task: {task}. Available tasks: {list(self.SCENARIOS.keys())}")

        # Create a VizDoom instance
        self.game = vzd.DoomGame()

        # Load scenario configuration
        scenario_config = self.SCENARIOS[task]
        scenario_path = pathlib.Path(__file__).parent / 'vizdoom_scenarios' / scenario_config['config_file']

        if scenario_path.exists():
            # Load config file and modify scenario path to use built-in scenarios
            self.game.load_config(str(scenario_path))
            # Override the scenario path to use VizDoom's built-in scenarios
            wad_name = scenario_config['config_file'].replace('.cfg', '.wad')
            self.game.set_doom_scenario_path(vzd.scenarios_path + "/" + wad_name)
        else:
            # Fallback to manual configuration if config file not found
            self._setup_manual_config(task)

        # Override some settings for consistency
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_window_visible(False)

        # Set up actions
        self.actions = scenario_config['actions']
        self.game.set_available_buttons(scenario_config['buttons'])

        # Initialize the game
        self.game.init()

        # Define observation and action spaces for the Gym interface
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        })
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.reward_range = [-np.inf, np.inf]

    def _setup_manual_config(self, task):
        """Fallback manual configuration if config files are not found."""
        if task == 'basic':
            self.game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")
            self.game.set_doom_map("map01")
            self.game.set_living_reward(-1)
            self.game.set_episode_timeout(300)
        elif task == 'deadly_corridor':
            self.game.set_doom_scenario_path(vzd.scenarios_path + "/deadly_corridor.wad")
            self.game.set_doom_map("map01")
            self.game.set_living_reward(-1)
            self.game.set_episode_timeout(2100)
        elif task == 'defend_the_center':
            self.game.set_doom_scenario_path(vzd.scenarios_path + "/defend_the_center.wad")
            self.game.set_doom_map("map01")
            self.game.set_living_reward(-1)
            self.game.set_episode_timeout(2100)
        elif task == 'health_gathering':
            self.game.set_doom_scenario_path(vzd.scenarios_path + "/health_gathering.wad")
            self.game.set_doom_map("map01")
            self.game.set_living_reward(-1)
            self.game.set_episode_timeout(2100)

    def _get_image(self):
        """Get and process the current screen image."""
        if self.game.is_episode_finished():
            return np.zeros(self._size + (3,), dtype=np.uint8)

        state = self.game.get_state()
        # VizDoom screen_buffer is already in HWC format (height, width, channels)
        image = state.screen_buffer  # No transpose needed!

        # Resize image if necessary
        if image.shape[:2] != self._size:
            try:
                import cv2
                image = cv2.resize(image, self._size, interpolation=cv2.INTER_AREA)
            except ImportError:
                # Fallback to PIL resize if cv2 not available
                from PIL import Image
                pil_image = Image.fromarray(image)
                pil_image = pil_image.resize(self._size, Image.NEAREST)
                image = np.array(pil_image)

        return image

    def step(self, action):
        # The agent outputs an integer, we map it to a VizDoom action
        doom_action = self.actions[action]

        # Perform action with action repeat
        total_reward = 0.0
        for _ in range(self._action_repeat):
            reward = self.game.make_action(doom_action)
            total_reward += reward
            if self.game.is_episode_finished():
                break

        done = self.game.is_episode_finished()
        image = self._get_image()

        obs = {
            'image': image,
            'is_first': False,
            'is_last': done,
            'is_terminal': done,
        }

        # Enhanced info dict for debugging and logging
        info = {
            'is_terminal': done,
        }

        # Add game variables if available
        if not done and self.game.get_state() is not None:
            game_vars = self.game.get_state().game_variables
            if len(game_vars) > 0:
                info['game_variables'] = game_vars
                # Common variables for different scenarios
                if self._task == 'basic' and len(game_vars) >= 1:
                    info['ammo'] = game_vars[0]
                elif self._task == 'health_gathering' and len(game_vars) >= 1:
                    info['health'] = game_vars[0]

        return obs, total_reward, done, info

    def reset(self):
        self.game.new_episode()
        image = self._get_image()

        obs = {
            'image': image,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }
        return obs

    def close(self):
        if hasattr(self, 'game'):
            self.game.close()


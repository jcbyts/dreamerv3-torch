#!/usr/bin/env python3
"""
Enhanced VizDoom environment that exposes ground truth factors
including semantic segmentation labels and depth information
"""

import gym
import numpy as np
import vizdoom as vzd
from vizdoom import GameVariable
import pathlib
from envs.vizdoom import ViZDoom


class EnhancedViZDoom(ViZDoom):
    """VizDoom environment with ground truth factors exposed."""
    
    def __init__(self, task, size=(64, 64), action_repeat=1, seed=0,
                 enable_labels=True, enable_depth=True, **kwargs):
        self._enable_labels = enable_labels
        self._enable_depth = enable_depth

        # Initialize parent class
        super().__init__(task, size, action_repeat, seed, **kwargs)

        # Check if buffers are actually enabled after initialization
        try:
            self._labels_available = self.game.is_labels_buffer_enabled()
            self._depth_available = self.game.is_depth_buffer_enabled()
            print(f"Labels buffer available: {self._labels_available}")
            print(f"Depth buffer available: {self._depth_available}")
        except Exception as e:
            print(f"Error checking buffer availability: {e}")
            self._labels_available = False
            self._depth_available = False

        # Enable game variables for ground truth factors
        self._setup_game_variables()

        # Track previous state for delta calculations
        self._prev_game_vars = None

    def _setup_game_variables(self):
        """Setup game variables for ground truth factor extraction."""
        try:
            # Essential variables for self-motion and world state
            game_vars = [
                GameVariable.POSITION_X,
                GameVariable.POSITION_Y,
                GameVariable.POSITION_Z,
                GameVariable.ANGLE,
                GameVariable.VELOCITY_X,
                GameVariable.VELOCITY_Y,
                GameVariable.VELOCITY_Z,
                GameVariable.HEALTH,  # Default in Deadly Corridor
            ]

            # Try to add optional variables (may not be available with viz_nocheat)
            optional_vars = [
                (GameVariable.PITCH, "PITCH"),
                (GameVariable.ARMOR, "ARMOR"),
                (GameVariable.KILLCOUNT, "KILLCOUNT"),
                (GameVariable.SELECTED_WEAPON, "SELECTED_WEAPON"),
                (GameVariable.SELECTED_WEAPON_AMMO, "SELECTED_WEAPON_AMMO"),
            ]

            for var, name in optional_vars:
                try:
                    game_vars.append(var)
                    print(f"Added optional variable: {name}")
                except:
                    print(f"Optional game variable {name} not available (possibly viz_nocheat)")

            self.game.set_available_game_variables(game_vars)
            self._game_var_names = [var.name for var in game_vars]
            print(f"Enabled game variables: {self._game_var_names}")

        except Exception as e:
            print(f"Error setting up game variables: {e}")
            self._game_var_names = []
        
        # Update observation space to include ground truth
        obs_spaces = {
            'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        }

        if self._labels_available:
            obs_spaces['labels'] = gym.spaces.Box(0, 255, self._size, dtype=np.uint8)
        if self._depth_available:
            obs_spaces['depth'] = gym.spaces.Box(0, 1, self._size, dtype=np.float32)

        self.observation_space = gym.spaces.Dict(obs_spaces)
        
        # Track label mappings
        self._label_mappings = {}
        self._discovered_labels = set()
    
    def _get_enhanced_observation(self):
        """Get observation with ground truth factors."""
        if self.game.is_episode_finished():
            obs = {
                'image': np.zeros(self._size + (3,), dtype=np.uint8),
                'is_first': False,
                'is_last': True,
                'is_terminal': True,
            }
            if self._labels_available:
                obs['labels'] = np.zeros(self._size, dtype=np.uint8)
            if self._depth_available:
                obs['depth'] = np.zeros(self._size, dtype=np.float32)
            return obs
        
        state = self.game.get_state()
        
        # Get RGB image
        image = state.screen_buffer
        if image.shape[:2] != self._size:
            try:
                import cv2
                image = cv2.resize(image, self._size, interpolation=cv2.INTER_AREA)
            except ImportError:
                from PIL import Image
                pil_image = Image.fromarray(image)
                pil_image = pil_image.resize(self._size, Image.NEAREST)
                image = np.array(pil_image)
        
        obs = {
            'image': image,
            'is_first': False,
            'is_last': False,
            'is_terminal': False,
        }
        
        # Get labels (semantic segmentation)
        if self._labels_available and state.labels_buffer is not None:
            labels = state.labels_buffer
            if labels.shape != self._size:
                try:
                    import cv2
                    labels = cv2.resize(labels, self._size, interpolation=cv2.INTER_NEAREST)
                except ImportError:
                    from PIL import Image
                    pil_labels = Image.fromarray(labels)
                    pil_labels = pil_labels.resize(self._size, Image.NEAREST)
                    labels = np.array(pil_labels)
            
            obs['labels'] = labels.astype(np.uint8)
            
            # Track discovered labels
            unique_labels = set(labels.flatten())
            self._discovered_labels.update(unique_labels)
        
        # Get depth buffer
        if self._depth_available and state.depth_buffer is not None:
            depth = state.depth_buffer
            if depth.shape != self._size:
                try:
                    import cv2
                    depth = cv2.resize(depth, self._size, interpolation=cv2.INTER_LINEAR)
                except ImportError:
                    from PIL import Image
                    pil_depth = Image.fromarray(depth)
                    pil_depth = pil_depth.resize(self._size, Image.BILINEAR)
                    depth = np.array(pil_depth)
            
            # Normalize depth to [0, 1]
            if depth.max() > depth.min():
                depth = (depth - depth.min()) / (depth.max() - depth.min())
            obs['depth'] = depth.astype(np.float32)

        # Add game variables (the real ground truth!)
        if hasattr(state, 'game_variables') and len(state.game_variables) > 0:
            obs['game_variables'] = state.game_variables.copy()

        return obs
    
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
        obs = self._get_enhanced_observation()
        obs['is_last'] = done
        obs['is_terminal'] = done

        # Update previous state tracking
        current_game_vars = obs.get('game_variables', None)
        if current_game_vars is not None:
            # Store for next step
            prev_vars = self._prev_game_vars
            self._prev_game_vars = current_game_vars.copy()
        else:
            prev_vars = None
        
        # Enhanced info dict with ground truth factors
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
        
        # Add ground truth factor summary
        if self._labels_available and 'labels' in obs:
            unique_labels = np.unique(obs['labels'])
            info['unique_labels'] = unique_labels.tolist()
            info['num_objects'] = len(unique_labels)
        
        return obs, total_reward, done, info
    
    def reset(self):
        self.game.new_episode()
        obs = self._get_enhanced_observation()
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False

        # Reset previous state tracking
        self._prev_game_vars = obs.get('game_variables', None)

        return obs
    
    def get_label_info(self):
        """Get information about discovered labels."""
        return {
            'discovered_labels': sorted(list(self._discovered_labels)),
            'num_unique_labels': len(self._discovered_labels),
            'label_mappings': self._label_mappings
        }


def extract_minimal_vizdoom_factors(obs_with_state, prev_state=None):
    """Extract minimal ground truth factors for VizDoom Deadly Corridor disentanglement.

    Following the specification for ~51-66 dimensional factor space:
    - Self/Ego: 11 dims (pose, orientation, velocity, Δpose)
    - Vest (goal): 4 dims (distance & vector to vest)
    - Enemies: 6 enemies × 6 dims = 36 dims
    - Projectiles: 3 × 5 dims = 15 dims (optional)

    Args:
        obs_with_state: Dict containing 'game_variables', 'labels', etc.
        prev_state: Previous game variables for delta calculations

    Returns:
        Dict with properly separated ego vs world factors
    """
    factors = {}

    # ===== A. SELF/EGO FACTORS (11 dims) =====
    if 'game_variables' in obs_with_state:
        game_vars = obs_with_state['game_variables']

        if len(game_vars) >= 8:  # pos_x, pos_y, pos_z, angle, vel_x, vel_y, vel_z, health
            pos_x, pos_y, pos_z, angle, vel_x, vel_y, vel_z, health = game_vars[:8]

            # 1. Pose (3 dims)
            factors['ego_pos_x'] = pos_x
            factors['ego_pos_y'] = pos_y
            factors['ego_pos_z'] = pos_z

            # 2. Orientation (2 dims - sin/cos encoding to handle wrapping)
            angle_rad = np.deg2rad(angle % 360)
            factors['ego_angle_sin'] = np.sin(angle_rad)
            factors['ego_angle_cos'] = np.cos(angle_rad)

            # 3. Velocities (3 dims)
            factors['ego_vel_x'] = vel_x
            factors['ego_vel_y'] = vel_y
            factors['ego_vel_z'] = vel_z

            # 4. Health (1 dim)
            factors['ego_health'] = health

            # 5. Derived ego-motion: Δpose between frames (4 dims)
            if prev_state is not None and len(prev_state) >= 4:
                prev_x, prev_y, prev_z, prev_angle = prev_state[:4]

                # Position deltas
                factors['ego_delta_x'] = pos_x - prev_x
                factors['ego_delta_y'] = pos_y - prev_y
                factors['ego_delta_z'] = pos_z - prev_z

                # Angle delta (handle wrapping)
                angle_diff = (angle - prev_angle + 180) % 360 - 180
                factors['ego_delta_angle'] = angle_diff
            else:
                # First frame - no delta
                factors['ego_delta_x'] = 0.0
                factors['ego_delta_y'] = 0.0
                factors['ego_delta_z'] = 0.0
                factors['ego_delta_angle'] = 0.0

    # ===== B. WORLD/EXOGENOUS FACTORS =====

    # Get object information from labels buffer
    enemy_objects = []
    vest_object = None
    projectile_objects = []

    if 'labels' in obs_with_state:
        labels = obs_with_state['labels']

        # Parse labels to identify objects by type
        unique_labels = np.unique(labels)

        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue

            mask = labels == label_id
            count = np.sum(mask)
            if count < 5:  # Skip tiny objects
                continue

            # Get object center in screen coordinates
            y_coords, x_coords = np.where(mask)
            center_x = np.mean(x_coords) / labels.shape[1]  # Normalize to [0,1]
            center_y = np.mean(y_coords) / labels.shape[0]
            size = count / labels.size

            # Classify object type (this is scenario-specific)
            # In Deadly Corridor: enemies are typically certain label IDs
            if label_id in [1, 2, 3, 4, 5, 6]:  # Assume these are enemies
                enemy_objects.append({
                    'id': label_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'size': size,
                    'alive': 1.0  # Present = alive
                })
            elif label_id == 255:  # Assume vest/goal object
                vest_object = {
                    'center_x': center_x,
                    'center_y': center_y,
                    'size': size,
                    'present': 1.0
                }
            # Add projectile detection if needed

    # B1. Vest/Goal object (4 dims)
    if vest_object:
        factors['vest_present'] = vest_object['present']
        factors['vest_screen_x'] = vest_object['center_x']
        factors['vest_screen_y'] = vest_object['center_y']
        # Distance approximation (closer = larger on screen)
        factors['vest_distance_proxy'] = 1.0 / (vest_object['size'] + 1e-6)
    else:
        factors['vest_present'] = 0.0
        factors['vest_screen_x'] = 0.5  # Center default
        factors['vest_screen_y'] = 0.5
        factors['vest_distance_proxy'] = 10.0  # Far away

    # B2. Enemies (6 enemies × 6 dims = 36 dims)
    max_enemies = 6
    enemy_objects.sort(key=lambda x: x['size'], reverse=True)  # Largest first

    for i in range(max_enemies):
        if i < len(enemy_objects):
            enemy = enemy_objects[i]
            factors[f'enemy{i}_alive'] = enemy['alive']
            factors[f'enemy{i}_id'] = float(enemy['id'])
            factors[f'enemy{i}_screen_x'] = enemy['center_x']
            factors[f'enemy{i}_screen_y'] = enemy['center_y']
            factors[f'enemy{i}_size'] = enemy['size']
            factors[f'enemy{i}_distance_proxy'] = 1.0 / (enemy['size'] + 1e-6)
        else:
            # No enemy in this slot
            factors[f'enemy{i}_alive'] = 0.0
            factors[f'enemy{i}_id'] = 0.0
            factors[f'enemy{i}_screen_x'] = 0.5
            factors[f'enemy{i}_screen_y'] = 0.5
            factors[f'enemy{i}_size'] = 0.0
            factors[f'enemy{i}_distance_proxy'] = 10.0

    # B3. Projectiles (3 × 5 dims = 15 dims) - Optional
    # Note: Projectiles are harder to detect from labels alone
    # Would need additional logic or different object detection
    max_projectiles = 3
    for i in range(max_projectiles):
        factors[f'projectile{i}_present'] = 0.0  # Placeholder
        factors[f'projectile{i}_screen_x'] = 0.5
        factors[f'projectile{i}_screen_y'] = 0.5
        factors[f'projectile{i}_vel_x'] = 0.0
        factors[f'projectile{i}_vel_y'] = 0.0

    return factors


# Backward compatibility alias
def extract_ground_truth_factors(obs_with_state, prev_state=None):
    """Backward compatibility wrapper."""
    return extract_minimal_vizdoom_factors(obs_with_state, prev_state)


if __name__ == "__main__":
    # Test the enhanced environment
    env = EnhancedViZDoom('basic', enable_labels=True, enable_depth=True)
    
    print("Testing enhanced VizDoom environment...")
    obs = env.reset()
    print(f"Observation keys: {obs.keys()}")
    
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"{key}: {value}")
    
    # Take a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        if 'labels' in obs:
            factors = extract_ground_truth_factors(obs)
            print(f"Step {i}: {len(factors)} ground truth factors extracted")
            
        if done:
            break
    
    label_info = env.get_label_info()
    print(f"Discovered labels: {label_info}")
    
    env.close()

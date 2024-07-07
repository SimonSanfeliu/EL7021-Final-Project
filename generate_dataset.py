import json
import random
from language_table.language_table.environments import blocks
from language_table.language_table.environments.rewards import block2block

def generate_instruction(block_color, target_position):
    return f"Mueve el bloque {block_color} al punto {target_position} sin tocar los obstáculos."

def calculate_action(initial_position, target_position):
    # Calculate the difference between the initial and target positions
    return [target_position[0] - initial_position[0], target_position[1] - initial_position[1]]

def generate_observation(initial_state, rgb_image):
    # Combine block positions, block colors, and obstacle positions into a single observation
    observation = {
        "block_positions": [block['position'] for block in initial_state['blocks']],
        "block_colors": [block['color'] for block in initial_state['blocks']],
        "obstacle_positions": [obstacle['position'] for obstacle in initial_state['obstacles']],
        "rgb": rgb_image.tolist()  # Include RGB image as part of observation
    }
    return observation

def render_environment_as_image(env, state):
    # Render the environment's state as an RGB image
    rgb_array = env.render(mode='rgb_array')
    return rgb_array

def generate_dataset(num_examples):
    colors = ["rojo", "azul", "verde", "amarillo"]
    env = CustomLanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,  # Specify the block mode
        reward_factory=block2block.BlockToBlockReward,
        control_frequency=10.0
    )
    dataset = []

    for _ in range(num_examples):
        state = env.reset()
        initial_blocks = state.get('blocks', [])
        obstacles = state.get('obstacles', [])
        if not initial_blocks:
            continue  # Skip if there are no blocks in the initial state
        target_block = random.choice(initial_blocks)
        target_position = [random.randint(0, 4), random.randint(0, 4)]

        instruction = generate_instruction(target_block['color'], target_position)
        action = calculate_action(target_block['position'], target_position)
        rgb_image = render_environment_as_image(env, state)
        observation = generate_observation(state, rgb_image)

        example = {
            "instruction": instruction,
            "initial_state": {
                "blocks": initial_blocks,
                "obstacles": obstacles
            },
            "target_state": {
                "blocks": [
                    {"color": target_block['color'], "position": target_position}
                ]
            },
            "action": action,
            "observation": observation
        }
        dataset.append(example)

    return dataset

# Generate and save the dataset
num_examples = 1000
dataset = generate_dataset(num_examples)

with open('spanish_obstacle_dataset.json', 'w') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

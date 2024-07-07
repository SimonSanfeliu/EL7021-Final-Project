import random
from language_table.language_table.environments import blocks, language_table

class CustomLanguageTable(language_table.LanguageTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles = []

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(3):  # For example, create 3 obstacles
            obstacle = {
                'position': [random.randint(0, 4), random.randint(0, 4)]
            }
            obstacles.append(obstacle)
        return obstacles

    def _generate_blocks(self):
        blocks = []
        for _ in range(5):  # For example, create 5 blocks
            block = {
                'color': random.choice(["rojo", "azul", "verde", "amarillo"]),
                'position': [random.randint(0, 4), random.randint(0, 4)]
            }
            blocks.append(block)
        return blocks

    def reset(self):
        state = super().reset()
        print(f"Initial state from super().reset(): {state}")  # Debug statement

        self.obstacles = self._generate_obstacles()  # Initialize obstacles
        state['obstacles'] = self.obstacles  # Add obstacles to the state

        self.blocks = self._generate_blocks()  # Initialize blocks
        state['blocks'] = self.blocks  # Add blocks to the state

        print(f"State after adding obstacles and blocks: {state}")  # Debug statement
        return state

    def step(self, action):
        state, reward, done, info = super().step(action)
        state['obstacles'] = self.obstacles  # Ensure obstacles are included in each step
        state['blocks'] = self.blocks  # Ensure blocks are included in each step

        return state, reward, done, info

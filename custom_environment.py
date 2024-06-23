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

    def reset(self):
        state = super().reset()
        self.obstacles = self._generate_obstacles()  # Initialize obstacles
        state['obstacles'] = self.obstacles  # Add obstacles to the state
        if 'blocks' not in state:
            state['blocks'] = []
        return state

    def step(self, action):
        state, reward, done, info = super().step(action)
        state['obstacles'] = self.obstacles  # Ensure obstacles are included in each step
        if 'blocks' not in state:
            state['blocks'] = []
        return state, reward, done, info
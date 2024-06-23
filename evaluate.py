from language_table import evaluation
from custom_environment import CustomLanguageTable
from language_table.environments import blocks
from language_table.environments.rewards import block2block

evaluator = evaluation.Evaluator(CustomLanguageTable(
    block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
    reward_factory=block2block.BlockToBlockReward,
    control_frequency=10.0
))
results = evaluator.evaluate(model, num_episodes=100)
print(f"Success Rate: {results['success_rate']}")

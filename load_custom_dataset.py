import json
import tensorflow as tf

def load_custom_dataset(dataset_path, batch_size, shuffle_buffer_size=10000):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    def gen():
        for item in data:
            yield {
                'instruction': item['instruction'],
                'initial_block_colors': [block['color'] for block in item['initial_state']['blocks']],
                'initial_block_positions': [block['position'] for block in item['initial_state']['blocks']],
                'initial_obstacle_positions': [obstacle['position'] for obstacle in item['initial_state']['obstacles']],
                'target_block_colors': [block['color'] for block in item['target_state']['blocks']],
                'target_block_positions': [block['position'] for block in item['target_state']['blocks']],
                'action': item['action'],
                'observation': item['observation'],
                'rgb_image': tf.convert_to_tensor(item['rgb_image'], dtype=tf.uint8)  # Convert back to tensor
            }

    output_signature = {
        'instruction': tf.TensorSpec(shape=(), dtype=tf.string),
        'initial_block_colors': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'initial_block_positions': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
        'initial_obstacle_positions': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
        'target_block_colors': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'target_block_positions': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
        'action': tf.TensorSpec(shape=(2,), dtype=tf.int32),
        'observation': {
            'block_positions': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
            'block_colors': tf.TensorSpec(shape=(None,), dtype=tf.string),
            'obstacle_positions': tf.TensorSpec(shape=(None, 2), dtype=tf.int32)
        },
        'rgb': tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)  # Adjust shape as necessary
    }

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )

    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

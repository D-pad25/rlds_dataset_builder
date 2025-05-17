import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load dataset (defaults to split='train')
ds = tfds.load('agrivle_dataset_v1', split='train', data_dir="~/tensorflow_datasets")

# Take the first episode
for episode in ds.take(1):
    steps = episode['steps']
    metadata = episode['episode_metadata']
    print(f"\n📁 File path: {metadata['file_path'].numpy().decode()}")
    
    # This dataset has one step per episode
    for step in steps:
        obs = step['observation']
        action = step['action'].numpy()
        state = obs['state'].numpy()
        instruction = step['language_instruction'].numpy().decode()

        base_img = obs['image'].numpy()
        wrist_img = obs['wrist_image'].numpy()

        # Display both images
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(base_img)
        axs[0].set_title("Base RGB")
        axs[0].axis("off")

        axs[1].imshow(wrist_img)
        axs[1].set_title("Wrist RGB")
        axs[1].axis("off")

        plt.suptitle("AgrivleDatasetV1 Visual Sample")
        plt.tight_layout()
        plt.show()

        print(f"🦾 Joint State: {np.round(state, 3)}")
        print(f"🎮 Action: {np.round(action, 3)}")
        print(f"🗣️ Instruction: \"{instruction}\"\n")

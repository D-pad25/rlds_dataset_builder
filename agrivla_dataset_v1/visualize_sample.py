import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

# 👇 Import your dataset builder explicitly
from agrivla_dataset_v1_dataset_builder import AgrivlaDatasetV1

# 👇 Instantiate and load it locally
builder = AgrivlaDatasetV1(data_dir="~/tensorflow_datasets")
builder.download_and_prepare()
ds = builder.as_dataset(split='train')

# ✅ Visualize one example
for episode in tfds.as_numpy(ds.take(1)):
    steps = episode['steps']
    meta = episode['episode_metadata']

    for step in steps:
        obs = step['observation']
        base_img = obs['image']
        wrist_img = obs['wrist_image']

        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.imshow(base_img)
        # plt.title("Base RGB")
        # plt.axis("off")

        # plt.subplot(1, 2, 2)
        # plt.imshow(wrist_img)
        # plt.title("Wrist RGB")
        # plt.axis("off")
        # plt.show()

        print(f"🦾 State: {step['observation']['state']}")
        print(f"🎮 Action: {step['action']}")
        print(f"🗣️ Instruction: {step['language_instruction']}")

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset(dataset_name: str, data_dir: str = "~/tensorflow_datasets", limit=5):
    print(f"🔍 Loading dataset: {dataset_name} from {data_dir}")
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = builder.as_dataset(split='train')
    ds_numpy = tfds.as_numpy(ds)

    print(f"📦 Dataset loaded. Showing up to {limit} episodes...\n")
    count = 0

    for episode in ds_numpy:
        steps = episode['steps']
        meta = episode['episode_metadata']['file_path'].decode('utf-8')
        print(f"🗂️ Episode: {meta} — {len(steps)} step(s)")

        if len(steps) == 0:
            print("⚠️ Empty episode. Skipping.\n")
            continue

        step = steps[0]  # Each episode has 1 step
        obs = step['observation']
        base_img = obs['image']
        wrist_img = obs['wrist_image']
        state = obs['state']
        action = step['action']
        instruction = step['language_instruction'].decode('utf-8')

        # Display side-by-side
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Instruction: {instruction}", fontsize=10)

        plt.subplot(1, 2, 1)
        plt.imshow(base_img)
        plt.title("Base RGB")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(wrist_img)
        plt.title("Wrist RGB")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        print(f"🦾 State:  {np.round(state, 2)}")
        print(f"🎮 Action: {np.round(action, 2)}\n")

        count += 1
        if count >= limit:
            break

if __name__ == "__main__":
    visualize_dataset("agrivla_dataset_v1", data_dir="/home/d_pad25/tensorflow_datasets", limit=5)

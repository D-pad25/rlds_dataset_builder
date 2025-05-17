import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the dataset
dataset = tfds.load(
    'agrivle_dataset_v1',
    split='train',
    data_dir='/home/d_pad25/tensorflow_datasets',
    shuffle_files=False
)

# Take one example
example = next(iter(dataset))

# Extract step data
step = example['steps']
obs = step['observation']
image = obs['image'].numpy()
wrist_image = obs['wrist_image'].numpy()
state = obs['state'].numpy()
action = step['action'].numpy()
instruction = step['language_instruction'].numpy().decode('utf-8')

# Visualize
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image)
axs[0].set_title("Base RGB Image")
axs[0].axis('off')

axs[1].imshow(wrist_image)
axs[1].set_title("Wrist RGB Image")
axs[1].axis('off')

plt.suptitle(f"Instruction: {instruction}\nState: {state.round(2)}\nAction: {action.round(2)}", fontsize=10)
plt.tight_layout()
plt.show()

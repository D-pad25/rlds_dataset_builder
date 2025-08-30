from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import pickle
import joblib

class AgrivlaDatasetV1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for AgrivlaDatasetV1."""

    VERSION = tfds.core.Version('1.0.1')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Added language instruction and prompt features.'
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [6x robot joint angles, '
                                '1x gripper position].',
                        ),
                        'prompt': tfds.features.Text(
                            doc='Prompt for the robot, if available. ',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [6x joint absoloute joing angle, '
                            '1x absoloute gripper position, sode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(
            # path='/mnt/e/VLA_data/CleanData/*'),
            # Test on E:\VLA_data\CleanData224\v5
            # E:\VLA_data\JoblibData224_Steps
            path='/mnt/e/Round1_addedPrompt/joblib/*'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of episodes from single-step .joblib files."""

        # Step 1: Find all episode directories
        episode_dirs = sorted(glob.glob(path))  # e.g. /data/train/episode_*
        print(f"Found {len(episode_dirs)} episodes")  # Add this

        def _parse_example(episode_dir):
            # Step 2: Load and sort all step .pkl files within the episode
            step_files = sorted(glob.glob(os.path.join(episode_dir, '*.joblib')))

            # Language instruction (can be customized later)
            instruction = "Pick a ripe, red tomato and drop it in the blue bucket."
            # print("Generating language embedding...")
            language_embedding = self._embed([instruction])[0].numpy()
            # print("Language embedding done.")

            episode = []
            for i, step_path in enumerate(step_files):
                # with open(step_path, 'rb') as f:
                #     step = pickle.load(f)
                step = joblib.load(step_path)  # Load the step data

                episode.append({
                    'observation': {
                        'image': step['base_rgb'],
                        'wrist_image': step['wrist_rgb'],
                        'state': step['joint_positions'].astype(np.float32),
                        'prompt': step['prompt'],
                    },
                    'action': step['control'].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (len(step_files) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(step_files) - 1),
                    'is_terminal': i == (len(step_files) - 1),
                    'language_instruction': instruction,
                    'language_embedding': language_embedding,
                })

            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_dir
                },
            }

            return episode_dir, sample

        # Use single-thread parsing for now
        for episode_dir in episode_dirs:
            print(f"Parsing: {episode_dir}")  # Add this
            yield _parse_example(episode_dir)

        # For large datasets, consider switching to Apache Beam:
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #     beam.Create(episode_dirs)
        #     | beam.Map(_parse_example)
        # )


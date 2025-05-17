from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
import pickle

class AgrivleDatasetV1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for AgrivleDatasetV1."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
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
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [6x robot joint angles, '
                                '1x gripper position].',
                        )
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
            path='/mnt/c/Users/Danie/Documents/QUT - Local/ThesisLocal/*'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of episodes by grouping step .pkl files."""

        # Step 1: Find all episode directories
        episode_dirs = sorted(glob.glob(path))  # e.g. /data/train/episode_*

        def _parse_example(episode_dir):
            # Step 2: Load and sort all step .pkl files within the episode
            step_files = sorted(glob.glob(os.path.join(episode_dir, '*.pkl')))

            episode = []
            for i, step_path in enumerate(step_files):
                with open(step_path, 'rb') as f:
                    step = pickle.load(f)

                # Language instruction (can be customized later)
                instruction = "Pick a ripe tomato and drop it in the grey bucket."
                language_embedding = self._embed([instruction])[0].numpy()

                episode.append({
                    'observation': {
                        'image': step['base_rgb'],
                        'wrist_image': step['wrist_rgb'],
                        'state': step['joint_positions'].astype(np.float32),
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
            yield _parse_example(episode_dir)

        # For large datasets, consider switching to Apache Beam:
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #     beam.Create(episode_dirs)
        #     | beam.Map(_parse_example)
        # )

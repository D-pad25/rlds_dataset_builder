from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
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
            path='/mnt/c/Users/Danie/Documents/QUT - Local/ThesisLocal/0513_183547_unique_wrist/*.pkl'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of single-step episodes from .pkl files."""

        episode_paths = glob.glob(path)

        for episode_path in episode_paths:
            with open(episode_path, 'rb') as f:
                step = pickle.load(f)

            # Provide a default or dynamic instruction
            instruction = "Perform manipulation task"
            language_embedding = self._embed([instruction])[0].numpy()

            sample = {
                'steps': [{
                    'observation': {
                        'image': step['base_rgb'],
                        'wrist_image': step['wrist_rgb'],
                        'state': step['joint_positions'].astype(np.float32),
                    },
                    'action': step['control'].astype(np.float32),
                    'discount': 1.0,
                    'reward': 1.0,
                    'is_first': True,
                    'is_last': True,
                    'is_terminal': True,
                    'language_instruction': instruction,
                    'language_embedding': language_embedding,
                }],
                'episode_metadata': {
                    'file_path': episode_path,
                },
            }

            yield episode_path, sample

def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            with open(episode_path, 'rb') as f:
                data = pickle.load(f)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(len(data['timestamp'])):
                # compute Kona language embedding
                language_embedding = self._embed([data['task']])[0].numpy()

                episode.append({
                    'observation': {
                        'image': data['external_image'][i][...,::-1],
                        'wrist_image': data['hand_image'][i][...,::-1],
                        'state': np.array(data['joint_positions'][i] + data['joint_velocities'][i][:-2]).astype(np.float32),
                    },
                    'action': np.concatenate([data['gripper_position'][i], data['gripper_quaternion'][i], data['joint_positions'][i][-2:]]).astype(np.float32),
                    'discount': 1.0,
                    'reward': float(data['joint_positions'][i][-1]<0.005),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': data['task'],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
           yield _parse_example(sample)
            

        #for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        #beam = tfds.core.lazy_imports.apache_beam
        #return (
        #        beam.Create(episode_paths)
        #        | beam.Map(_parse_example)
        #)
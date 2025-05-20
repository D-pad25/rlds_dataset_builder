## AgrivleDatasetV1

**AgrivleDatasetV1** is a real-world, single-step RLDS-compatible dataset collected for fine-tuning generalist Vision-Language-Action (VLA) models in agricultural manipulation tasks. It was captured using an xArm6 robot equipped with both base and wrist-mounted Intel RealSense RGB cameras.

### 📦 Contents

Each `.pkl` file represents a single timestep and contains:

- `base_rgb`: *(224×224×3)* RGB image from the base camera  
- `wrist_rgb`: *(224×224×3)* RGB image from the wrist camera  
- `joint_positions`: *7-DOF* arm joint angles  
- `joint_velocities`: *7-DOF* joint velocities (not used in RLDS format, but retained in source)  
- `ee_pos_quat`: *7D* end-effector position and orientation  
- `gripper_position`: *Scalar* gripper state  
- `control`: *7D* action (absolute joint target positions)  
- `language_instruction`: Natural language command (default placeholder)  
- `language_embedding`: *512-dim* Universal Sentence Encoder embedding  

### 🧩 Format

This dataset is formatted using the [RLDS](https://github.com/google-research/rlds) standard and [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets).  
It is compatible with pipelines such as [OpenPI](https://github.com/Physical-Intelligence/openpi) and [LeRobot](https://github.com/Physical-Intelligence/lerobot).

### 🛠️ Preprocessing Notes

- Each timestep is saved as an individual `.pkl` file.
- Dataset is **single-step** (1 step per episode).
- All RGB images are preserved at full resolution *(480×640×3)*.
- Language instructions are currently uniform: `"Perform manipulation task"`.
- Sentence embeddings are generated using [Universal Sentence Encoder Large v5](https://tfhub.dev/google/universal-sentence-encoder-large/5).
- No corrupted or outlier files were removed for v1.0.0.
- Dataset size: **~633 MiB**  
  Number of episodes: **811**

---

*Version 1.0.0 - May 2025*

import cv2
import numpy as np
import tensorflow_datasets as tfds
from agrivla_dataset_v2_dataset_builder import AgrivlaDatasetV2
from pathlib import Path
def render_frame(base_img, wrist_img, state, action, prompt):
    # Combine images side by side
    base = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
    combined = np.hstack([base, wrist])

    # Resize for video
    combined = cv2.resize(combined, (224*2, 224))  # (width, height)

    # Draw overlay text
    overlay = combined.copy()
    cv2.rectangle(overlay, (0, 0), (combined.shape[1], 80), (0, 0, 0), -1)
    combined = cv2.addWeighted(overlay, 0.6, combined, 0.4, 0)

    # Text
    y = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    green = (0, 255, 0)

    def put(text):
        nonlocal y
        cv2.putText(combined, text, (10, y), font, scale, green, 1, cv2.LINE_AA)
        y += 20

    put("Prompt: " + (prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)))
    put("State: " + np.array2string(state, precision=2, separator=","))
    put("Action: " + np.array2string(action, precision=2, separator=","))

    return combined


def create_episode_video(output_path="~/Thesis/Data/sem2/rlds_converted/v2Test.mp4", fps=30):
    builder = AgrivlaDatasetV2(data_dir="~/tensorflow_datasets")
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train")

    # --- Check if dataset has any episodes ---
    try:
        episode = next(iter(tfds.as_numpy(ds.take(1))))
    except StopIteration:
        print("⚠️ No episodes found in dataset.")
        return

    steps = episode.get("steps", None)
    if steps is None:
        print("⚠️ Episode has no 'steps' field.")
        return

    steps_iter = iter(steps)
    first_step = next(steps_iter, None)
    if first_step is None:
        print("⚠️ Episode contains no steps.")
        return

    # Restart steps (since we consumed one for the check)
    steps = [first_step] + list(steps_iter)

    frames = []
    for step in steps:
        obs = step["observation"]
        frame = render_frame(
            base_img=obs["image"],
            wrist_img=obs["wrist_image"],
            state=obs["state"],
            prompt=obs["prompt"],
            action=step["action"],
        )
        frames.append(frame)

    if not frames:
        print("⚠️ No frames were generated.")
        return

    # Write video
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(str(output_path).replace("~", str(Path.home())), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"✅ Video saved to: {output_path}")


if __name__ == "__main__":
    create_episode_video(fps=30)

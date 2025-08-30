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
    combined = cv2.resize(combined, (224 * 2, 224))  # (width, height)

    # Draw overlay text
    overlay = combined.copy()
    cv2.rectangle(overlay, (0, 0), (combined.shape[1], 80), (0, 0, 0), -1)
    combined = cv2.addWeighted(overlay, 0.6, combined, 0.4, 0)

    # Text
    y = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 255, 0)

    def put(text):
        nonlocal y
        cv2.putText(combined, text, (10, y), font, scale, color, 1, cv2.LINE_AA)
        y += 20

    if isinstance(prompt, bytes):
        try:
            prompt = prompt.decode("utf-8", errors="ignore")
        except Exception:
            prompt = str(prompt)

    put("Prompt: " + (str(prompt) if prompt is not None else ""))
    put("State: " + np.array2string(np.asarray(state), precision=2, separator=","))
    put("Action: " + np.array2string(np.asarray(action), precision=2, separator=","))

    return combined


def create_videos(
    output_dir="~/Thesis/Data/sem_2_rlds_converted_fromSSD",
    fps=30,
    split="train",
    max_episodes=None,   # set to an int to limit how many episodes you export
):
    # Prepare dataset
    data_dir = str(Path("~/tensorflow_datasets").expanduser())
    builder = AgrivlaDatasetV2(data_dir=data_dir)
    builder.download_and_prepare()
    ds = builder.as_dataset(split=split)

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_iter = tfds.as_numpy(ds)
    exported = 0
    any_episode = False

    for ep_idx, episode in enumerate(episodes_iter):
        any_episode = True
        steps_ds = episode.get("steps")
        if steps_ds is None:
            print(f"⚠️ Episode {ep_idx}: missing 'steps' field, skipping.")
            continue

        # Iterate steps as numpy
        step_iter = steps_ds

        writer = None
        frame_count = 0
        out_path = out_dir / f"v2_ep{ep_idx:04d}.mp4"

        for step in step_iter:
            # Defensive access: RLDS often nests observations/actions like this
            obs = step.get("observation", {})
            base_img = obs.get("image")
            wrist_img = obs.get("wrist_image", base_img)
            state = obs.get("state")
            action = step.get("action")

            # Prompt might live in different places depending on your builder
            prompt = (
                step.get("prompt")
                or step.get("language_instruction")
                or episode.get("episode_metadata", {}).get("prompt")
                or ""
            )

            # Skip if essential images are missing
            if base_img is None or wrist_img is None:
                continue

            frame = render_frame(base_img, wrist_img, state, action, prompt)

            # Lazily open the writer once we know frame size
            if writer is None:
                h, w, _ = frame.shape
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )

            writer.write(frame)
            frame_count += 1

        if writer is not None:
            writer.release()

        if frame_count == 0:
            # If no frames, remove empty file if any
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            print(f"⚠️ Episode {ep_idx}: contained no valid frames, skipped.")
            continue

        print(f"✅ Saved: {out_path}  ({frame_count} frames)")
        exported += 1

        if max_episodes is not None and exported >= max_episodes:
            break

    if not any_episode:
        print("⚠️ No episodes found in dataset.")
    elif exported == 0:
        print("⚠️ Dataset had episodes, but none produced frames.")

if __name__ == "__main__":
    # Example: export first 5 episodes from the train split
    create_videos(fps=30, max_episodes=5)

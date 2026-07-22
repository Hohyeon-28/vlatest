import csv
import json
import os
import pprint
import statistics
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import tqdm
import tyro


_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_libero_init_state_compat(*args, **kwargs):
    """Allow trusted LIBERO init-state pickle files on PyTorch 2.6+."""
    if "weights_only" not in kwargs and args:
        try:
            load_path = os.fspath(args[0])
        except TypeError:
            load_path = ""
        normalized_path = load_path.replace("\\", "/")
        if "/LIBERO/" in normalized_path or "/init_files/" in normalized_path:
            kwargs["weights_only"] = False
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_libero_init_state_compat

from libero.libero import benchmark

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)

log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)  # ensures directory exists


LATENCY_METRIC_KEYS = [
    "obs_preprocess_ms",
    "client_get_action_ms",
    "client_roundtrip_ms",
    "server_handler_ms",
    "policy_total_ms",
    "policy_prepare_input_ms",
    "policy_apply_transforms_ms",
    "policy_model_get_action_ms",
    "policy_unapply_transforms_ms",
    "action_convert_ms",
    "env_step_ms",
    "step_total_ms",
]


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile / 100.0)))
    return ordered[idx]


def summarize_latency(records):
    summary = {"num_action_steps": len(records)}
    for key in LATENCY_METRIC_KEYS:
        values = [float(item[key]) for item in records if item.get(key) is not None]
        if not values:
            continue
        summary[key] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": _percentile(values, 90),
            "p99": _percentile(values, 99),
            "min": min(values),
            "max": max(values),
        }
    return summary


def write_latency_csv(path, records):
    if not records:
        return
    fieldnames = sorted({key for item in records for key in item.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def log_latency_summary(prefix, records, log_file):
    summary = summarize_latency(records)
    if not records:
        print(f"{prefix} latency: no action steps recorded")
        log_file.write(f"{prefix} latency: no action steps recorded\n")
        return summary
    step = summary.get("step_total_ms", {})
    request = summary.get("client_roundtrip_ms", {})
    model = summary.get("policy_model_get_action_ms", {})
    line = (
        f"{prefix} latency: "
        f"step_mean={step.get('mean', 0.0):.2f}ms, step_p90={step.get('p90', 0.0):.2f}ms, "
        f"request_mean={request.get('mean', 0.0):.2f}ms, "
        f"model_mean={model.get('mean', 0.0):.2f}ms"
    )
    print(line)
    log_file.write(line + "\n")
    return summary


def summarize_obs(obs_dict):
    summary = {}
    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": tuple(v.shape), "dtype": v.dtype, "device": v.device}
        elif isinstance(v, np.ndarray):
            summary[k] = {"shape": v.shape, "dtype": v.dtype}
        else:
            summary[k] = type(v).__name__
    pprint.pprint(summary)


def show_obs_images_cv2(new_obs):
    # remove batch dim
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]

    # convert RGB -> BGR for OpenCV
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

    # show in separate windows
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


@dataclass
class GenerateConfig:
    # fmt: off
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                     # Number of rollouts per task
    #################################################################################################################
    # fmt: on
    """Port to connect to."""
    port: int = 5555
    """Headless mode (no GUI)."""
    headless: bool = False
    """Run only the specified task indices (overrides order if provided)."""
    task_ids: list[int] | None = None
    """Run tasks in this explicit order."""
    task_order: list[int] | None = None
    """Record per-action latency JSONL/CSV and summary files."""
    record_timing: bool = True
    """Print latency progress every N action steps. 0 disables progress prints."""
    timing_print_every: int = 100
    """Optional suffix for log/latency files, e.g. fake or real."""
    result_tag: str = ""
    """Directory for eval logs and latency files."""
    result_dir: str = log_dir


class GR00TPolicy:
    """GR00T Policy wrapper for Libero environments."""

    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(self, host="localhost", port=5555, headless=False):
        from gr00t.eval.service import ExternalRobotInferenceClient

        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless
        self.last_timing = {}

    def get_action(self, observation_dict, lang: str):
        """Get action from GR00T policy given observation and language instruction."""
        total_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        obs_dict = self._process_observation(observation_dict, lang)
        obs_preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        # summarize_obs(obs_dict)
        client_start = time.perf_counter()
        action_chunk = self.policy.get_action(obs_dict)
        client_get_action_ms = (time.perf_counter() - client_start) * 1000.0
        convert_start = time.perf_counter()
        action = self._convert_to_libero_action(action_chunk, 0)
        action_convert_ms = (time.perf_counter() - convert_start) * 1000.0

        server_timing = action_chunk.get("__server_timing__", {})
        client_timing = action_chunk.get("__client_timing__", {})
        policy_timing = action_chunk.get("__policy_timing__", {})
        self.last_timing = {
            "obs_preprocess_ms": obs_preprocess_ms,
            "client_get_action_ms": client_get_action_ms,
            "client_roundtrip_ms": client_timing.get("roundtrip_ms"),
            "server_handler_ms": server_timing.get("handler_ms"),
            "action_convert_ms": action_convert_ms,
            "action_call_total_ms": (time.perf_counter() - total_start) * 1000.0,
        }
        self.last_timing.update(policy_timing)
        return action

    def _process_observation(self, obs, lang: str):
        """Convert Libero observation to GR00T format."""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        if not self.headless:
            show_obs_images_cv2(new_obs)
        return new_obs

    def _convert_to_libero_action(
        self, action_chunk: dict[str, np.array], idx: int = 0
    ) -> np.ndarray:
        """Convert GR00T action chunk to Libero format.

        Args:
            action_chunk: Dictionary of action components from GR00T policy
            idx: Index of action to extract from chunk (default: 0 for first action)

        Returns:
            7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in self.action_keys
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array


def eval_libero(cfg: GenerateConfig) -> None:
    result_dir = cfg.result_dir
    os.makedirs(result_dir, exist_ok=True)
    result_tag = cfg.result_tag.strip()
    result_suffix = f"_{result_tag}" if result_tag else ""
    result_prefix = f"libero_eval_{cfg.task_suite_name}{result_suffix}"
    rollout_root = f"{result_dir}/rollouts/{result_prefix}"

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    if result_tag:
        print(f"Result tag: {result_tag}")
    log_file = open(f"{result_dir}/{result_prefix}.log", "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    if result_tag:
        log_file.write(f"Result tag: {result_tag}\n")
    latency_records = []
    task_latency_summaries = {}
    latency_jsonl_file = None
    latency_jsonl_path = f"{result_dir}/{result_prefix}_latency_steps.jsonl"
    latency_csv_path = f"{result_dir}/{result_prefix}_latency_steps.csv"
    latency_summary_path = f"{result_dir}/{result_prefix}_latency_summary.json"
    if cfg.record_timing:
        latency_jsonl_file = open(latency_jsonl_path, "w")
        log_file.write(f"Latency JSONL: {latency_jsonl_path}\n")
        log_file.write(f"Latency CSV: {latency_csv_path}\n")
        log_file.write(f"Latency summary: {latency_summary_path}\n")

    # Decide which task indices to run
    if cfg.task_ids:
        task_indices = cfg.task_ids
    elif cfg.task_order:
        task_indices = cfg.task_order
    else:
        task_indices = list(range(num_tasks_in_suite))

    # Clamp indices to valid range and warn if needed
    task_indices = [idx for idx in task_indices if 0 <= idx < num_tasks_in_suite]

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_indices):
        task_latency_records = []
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)

        gr00t_policy = GR00TPolicy(host="localhost", port=cfg.port, headless=cfg.headless)

        # Start episodes
        task_episodes, task_successes = 0, 0
        max_trials = min(cfg.num_trials_per_task, len(initial_states))
        for episode_idx in tqdm.tqdm(range(max_trials)):
            episode_latency_records = []
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            top_view = []
            wrist_view = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 600  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 1000  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            done = False
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # # Get preprocessed image
                    img, wrist_img = get_libero_image(obs)

                    # # Save preprocessed image for replay video
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    action_step_index = t - cfg.num_steps_wait
                    step_start = time.perf_counter()

                    # Query model to get action
                    action = gr00t_policy.get_action(
                        obs,
                        task.language,
                    )

                    # Execute action in environment
                    env_step_start = time.perf_counter()
                    obs, reward, done, info = env.step(action.tolist())
                    env_step_ms = (time.perf_counter() - env_step_start) * 1000.0
                    step_total_ms = (time.perf_counter() - step_start) * 1000.0

                    if cfg.record_timing:
                        timing_record = {
                            "task_suite": cfg.task_suite_name,
                            "task_id": task_id,
                            "task_description": task_description,
                            "episode_idx": episode_idx,
                            "global_episode": total_episodes + 1,
                            "sim_t": t,
                            "action_step_index": action_step_index,
                            "done": bool(done),
                            "reward": float(reward),
                            "env_step_ms": env_step_ms,
                            "step_total_ms": step_total_ms,
                        }
                        timing_record.update(gr00t_policy.last_timing)
                        latency_records.append(timing_record)
                        task_latency_records.append(timing_record)
                        episode_latency_records.append(timing_record)
                        if latency_jsonl_file is not None:
                            latency_jsonl_file.write(json.dumps(timing_record) + "\n")
                            if cfg.timing_print_every and len(latency_records) % cfg.timing_print_every == 0:
                                latency_jsonl_file.flush()
                        if cfg.timing_print_every and len(latency_records) % cfg.timing_print_every == 0:
                            request_ms = timing_record.get("client_roundtrip_ms") or 0.0
                            model_ms = timing_record.get("policy_model_get_action_ms") or 0.0
                            print(
                                f"[latency] steps={len(latency_records)} "
                                f"step_total={step_total_ms:.2f}ms "
                                f"request={request_ms:.2f}ms "
                                f"model={model_ms:.2f}ms"
                            )

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                top_view,
                wrist_view,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                result_tag=result_tag,
                rollout_root=rollout_root,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            if cfg.record_timing:
                log_latency_summary("Episode", episode_latency_records, log_file)
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        if cfg.record_timing:
            task_latency_summaries[str(task_id)] = {
                "task_description": task_description,
                "summary": log_latency_summary("Task", task_latency_records, log_file),
            }
        log_file.flush()

    if cfg.record_timing:
        if latency_jsonl_file is not None:
            latency_jsonl_file.close()
        write_latency_csv(latency_csv_path, latency_records)
        latency_summary = {
            "task_suite": cfg.task_suite_name,
            "result_tag": result_tag,
            "result_prefix": result_prefix,
            "num_tasks": len(task_indices),
            "num_episodes": total_episodes,
            "num_successes": total_successes,
            "success_rate": float(total_successes) / float(total_episodes) if total_episodes else 0.0,
            "overall": summarize_latency(latency_records),
            "tasks": task_latency_summaries,
            "files": {
                "jsonl": latency_jsonl_path,
                "csv": latency_csv_path,
                "summary": latency_summary_path,
                "rollouts": rollout_root,
            },
        }
        with open(latency_summary_path, "w") as f:
            json.dump(latency_summary, f, indent=2)
        print(f"Saved latency JSONL at {latency_jsonl_path}")
        print(f"Saved latency CSV at {latency_csv_path}")
        print(f"Saved latency summary at {latency_summary_path}")
        log_file.write(f"Saved latency JSONL at {latency_jsonl_path}\n")
        log_file.write(f"Saved latency CSV at {latency_csv_path}\n")
        log_file.write(f"Saved latency summary at {latency_summary_path}\n")

    # Save local log file
    log_file.close()


if __name__ == "__main__":
    cfg = tyro.cli(GenerateConfig)
    eval_libero(cfg)

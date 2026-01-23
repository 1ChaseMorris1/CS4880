import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import A2C, DQN, PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecVideoRecorder

from envs.snake_env import SnakeEnv


def make_env(
    grid_size: int,
    reward_type: str,
    record_video: bool,
):
    def _init():
        env = SnakeEnv(
            grid_size=grid_size,
            reward_type=reward_type,
            render_mode="rgb_array" if record_video else None,
        )
        env = gym.wrappers.FlattenObservation(env)
        return env

    return _init


def build_model(algo: str, env):
    algo = algo.lower()
    if algo == "ppo":
        return PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs")
    if algo == "a2c":
        return A2C("MlpPolicy", env, verbose=1, tensorboard_log="runs")
    if algo == "dqn":
        return DQN("MlpPolicy", env, verbose=1, tensorboard_log="runs")
    raise ValueError("Unsupported algorithm. Use: ppo, a2c, dqn")


class WandbMetricsAndVideoCallback(BaseCallback):
    def __init__(self, record_video: bool, video_dir: str):
        super().__init__()
        self.record_video = record_video
        self.video_dir = video_dir
        self.logged_videos: set[str] = set()

    def _on_step(self) -> bool:
        step = self.num_timesteps
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                wandb.log(
                    {
                        "global_step": step,
                        "episode_reward": ep.get("r"),
                        "episode_length": ep.get("l"),
                    }
                )

        if self.record_video and os.path.isdir(self.video_dir):
            for name in sorted(os.listdir(self.video_dir)):
                if not name.endswith(".mp4"):
                    continue
                path = os.path.join(self.video_dir, name)
                if path in self.logged_videos:
                    continue
                wandb.log(
                    {"global_step": step, "video": wandb.Video(path, format="mp4")}
                )
                self.logged_videos.add(path)

        return True


def _compute_video_length() -> int:
    video_length = int(os.environ.get("VIDEO_LENGTH", 0))
    if video_length > 0:
        return video_length
    video_seconds = int(os.environ.get("VIDEO_SECONDS", 60))
    fps = int(SnakeEnv.metadata.get("render_fps", 12))
    return max(1, video_seconds * fps)


def main():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    grid_size = int(os.environ.get("GRID_SIZE", 10))
    reward_type = os.environ.get("REWARD_TYPE", "dense")
    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", 200_000))
    algorithms = os.environ.get("ALGORITHMS", "ppo").split(",")
    record_video = os.environ.get("RECORD_VIDEO", "1") == "1"
    video_dir = os.environ.get("VIDEO_DIR", "videos")
    log_video_every_steps = int(os.environ.get("LOG_VIDEO_EVERY_STEPS", 50000))
    video_length = _compute_video_length()

    for algo in [a.strip().lower() for a in algorithms if a.strip()]:
        run = wandb.init(
            project="snake-rl",
            group="snake-rl-algorithms",
            name=f"{algo}-grid{grid_size}-{reward_type}",
            config={
                "algorithm": algo.upper(),
                "grid_size": grid_size,
                "reward_type": reward_type,
                "total_timesteps": total_timesteps,
                "record_video": record_video,
            },
        )
        run.define_metric("global_step")
        run.define_metric("episode_reward", step_metric="global_step")
        run.define_metric("episode_length", step_metric="global_step")
        run.define_metric("video", step_metric="global_step")

        algo_video_dir = os.path.join(video_dir, algo)
        env = DummyVecEnv([make_env(grid_size, reward_type, record_video)])
        env = VecMonitor(env)
        if record_video:
            env = VecVideoRecorder(
                env,
                algo_video_dir,
                record_video_trigger=lambda step: step % log_video_every_steps == 0,
                video_length=video_length,
                name_prefix=f"snake-{algo}",
            )

        model = build_model(algo, env)
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(
                [
                    WandbCallback(verbose=1),
                    WandbMetricsAndVideoCallback(record_video=record_video, video_dir=algo_video_dir),
                ]
            ),
            log_interval=10,
        )

        model.save(f"snake_{algo}")
        env.close()
        run.finish()


if __name__ == "__main__":
    main()

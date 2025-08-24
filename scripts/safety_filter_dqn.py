import os
import time
import yaml
import wandb
import torch
import random
import argparse
import numpy as np
import safety_envs.envs
import gymnasium as gym
from pathlib import Path
from safety_filters.safe_dqn import SafeDQN

scripts_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_dir = scripts_dir.parent
config_dir = project_dir / "config"

with open(config_dir / "config_safedqn_cartpole.yaml", "r") as file:
    config = yaml.safe_load(file)

logs_dir = project_dir / "logs"
run_foldername = config["exp_name"] + time.strftime("_%Y-%m-%d_%H-%M-%S")
run_dir = logs_dir / run_foldername
os.makedirs(run_dir, exist_ok=True)

run = wandb.init(
    project=config["wandb_project_name"],
    config=config,
    monitor_gym=True,
    save_code=True,
)

#Update the config with the new directories
config["run_dir"] = str(run_dir)
config["project_dir"] = str(project_dir)
config["scripts_dir"] = str(scripts_dir)
config["logs_dir"] = str(logs_dir)
config["run_id"] = run.id

# Save the config to a YAML file in the run directory
with open(run_dir / "config.yaml", "w") as file:
    yaml.dump(config, file)

# Save the source code of the project to a zip file
os.system(f'zip -r -q "{run_dir / "source_code.zip"}" "{project_dir}" -x "*__pycache__*" -x "*.pytest_cache*" -x "*logs*" -x "*.venv*"')

# Set random seeds for reproducibility
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.backends.cudnn.deterministic = config["torch_deterministic"]
device = torch.device("cuda" if torch.cuda.is_available() and config["cuda"] else "cpu")

env = gym.make(config["env_id"], safety_filter_args=config.get("safety_filter_args", None))
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(config["seed"])

model = SafeDQN(
    env=env,
    learning_rate=config["learning_rate"],
    buffer_size=config["buffer_size"],
    learning_starts=config["learning_starts"],
    batch_size=config["batch_size"],
    tau=config["tau"],
    gamma=config["gamma"],
    train_frequency=config["train_frequency"],
    target_network_frequency=config["target_network_frequency"],
    exploration_fraction=config["exploration_fraction"],
    exploration_start_eps=config["eps_start"],
    exploration_end_eps=config["eps_end"],
    wandb_log=f"runs/{run.id}",
    device=device,
    safety_filter_args=config.get("safety_filter_args", None),
)

if config["train_model"]: 
    model.learn(total_timesteps=config["total_timesteps"])
else:
    model_path = Path(config["eval_model_path"])
    if model_path.exists():
        model.q_network.load_state_dict(torch.load(model_path, map_location=model.device))
        model.target_network.load_state_dict(model.q_network.state_dict())
        model.target_network.eval()  # Set the target network to evaluation mode
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print("No pre-trained model found. Starting with a fresh model.")
        model.learn(total_timesteps=config["total_timesteps"])

model_dir = run_dir / "model"
model_path = model_dir / "q_network.pth"
os.makedirs(model_dir, exist_ok=True)
torch.save(model.q_network.state_dict(), model_path)
print(f"Model saved to {model_path}")

if not config["manual_mode"]:
    
    video_path = run_dir / "videos"
    os.makedirs(video_path, exist_ok=True)
    eval_env = gym.make(config["env_id"], safety_filter_args=config.get("safety_filter_args", None), render_mode="rgb_array")
    eval_env = gym.wrappers.RecordVideo(eval_env, video_path, episode_trigger=lambda x: True)

    total_reward = 0
    num_episodes = 10
    print(f"Evaluating the model for {num_episodes} episodes...")
    model.target_network.eval()  # Set the target network to evaluation mode
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        epsisode_reward = 0
        eps_step = 0
        safety_filter_interventions = 0
        MAX_EPISODE_STEPS = 2000
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
            #rand_action = eval_env.action_space.sample() if random.random() < 0.75 else None
            
            rand_action = eval_env.action_space.sample()
            #rand_action = 0
            
            # abcd = model._consult_safety_filter(obs_tensor, task_action=rand_action, use_qcbf=True)
            # if model.target_network(obs_tensor).max(1)[0].item() > config["safety_filter_args"]["SAFETY_FILTER_EPSILON"] and rand_action is not None:
            #     # Task policy is random inputs to the model
            #     action = eval_env.action_space.sample()
            #     eval_env.unwrapped.safety_filter_in_use = False
            # else:
            #     # Safety filter policy
            #     action = model._predict_action(epsilon=0.0, observation=obs)
            #     eval_env.unwrapped.safety_filter_in_use = True
            action, filter_in_use = model._consult_safety_filter(obs_tensor, task_action=rand_action, use_qcbf=True)
            safety_filter_interventions += filter_in_use
            eval_env.unwrapped.safety_filter_in_use = filter_in_use
            obs, reward, terminated, truncated, info = eval_env.step(action)
            #done = terminated or truncated
            eps_step += 1
            done = terminated or eps_step >= MAX_EPISODE_STEPS
            epsisode_reward += reward
        total_reward += epsisode_reward
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {epsisode_reward}, Safety Filter Interventions: {safety_filter_interventions}/{MAX_EPISODE_STEPS} ({(safety_filter_interventions * 100)/MAX_EPISODE_STEPS})%")
        wandb.log({
            "eval/episode_reward": epsisode_reward,
            "eval/safety_filter_interventions": safety_filter_interventions,
            },)
    avg_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    wandb.log({"avg_statistics/average_reward": avg_reward},)


    for i, video_name in enumerate(video_path.glob("*.mp4")):
        wandb.log({"Viz/video": wandb.Video(str(video_name), format="mp4")})

    run.finish()
    env.close()
    
else:
    
    eval_env = gym.make(config["env_id"], safety_filter_args=config.get("safety_filter_args", None), render_mode="human")
    
    print("\033[33mManual mode is enabled!\033[0m")
    print("You can now interact with the environment manually.")
    
    import sys
    import select
    import tty
    import termios
    
    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        for i in range(100):
            obs_array, _ = eval_env.reset()
            while True:
            ## Keyboard inputs to control the agent
                if isData():
                    key = sys.stdin.read(1)
                    print(key, end="\r")
                    if key == 'a':
                        manual_action = np.array([0,])
                    elif key == 'd':
                        manual_action = np.array([1,])
                    elif key == 'q':
                        print("Exiting test mode.")
                        break
                    elif key == 'r':
                        print("Resetting environment.")
                        obs_array, _ = eval_env.reset()
                    else:
                        print(f"Unknown command: {key}")

                else:
                    manual_action = None

                action, filter_in_use = model._consult_safety_filter(obs_array, task_action=manual_action, use_qcbf=True)
                eval_env.unwrapped.safety_filter_in_use = filter_in_use
                obs, reward, terminated, truncated, info = eval_env.step(action)
                #print(terminated, info)
                if terminated: 
                    break
            
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
run.finish()
#eval_env.close()
env.close()
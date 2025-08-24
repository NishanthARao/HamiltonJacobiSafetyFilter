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
from safety_filters.safe_sac_continuous import SafeSACContinuous

scripts_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_dir = scripts_dir.parent
config_dir = project_dir / "config"

with open(config_dir / "config_safe_sac_inverted_pendulum.yaml", "r") as file:
    config = yaml.safe_load(file)

logs_dir = project_dir / "logs"
run_foldername = config["exp_name"] + time.strftime("_%Y-%m-%d_%H-%M-%S")
run_dir = logs_dir / run_foldername
os.makedirs(run_dir, exist_ok=True)
video_path = run_dir / "videos"
os.makedirs(video_path, exist_ok=True)

if not config.get("dont_log", True): 
    run = wandb.init(
        project=config["wandb_project_name"],
        config=config,
        monitor_gym=True,
        save_code=True,)
    config["run_id"] = run.id

#Update the config with the new directories
config["run_dir"] = str(run_dir)
config["project_dir"] = str(project_dir)
config["scripts_dir"] = str(scripts_dir)
config["logs_dir"] = str(logs_dir)
config["video_path"] = str(video_path)

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

#env = gym.make(config["env_id"])
env = gym.make(config["env_id"], safety_filter_args=config.get("safety_filter_args", None), reset_noise_scale=config.get("reset_noise_scale", 0.1),) 
env = gym.wrappers.RecordEpisodeStatistics(env)
env.action_space.seed(config["seed"])

model = SafeSACContinuous(
    env=env,
    q_learning_rate=float(config["q_learning_rate"]),
    policy_learning_rate=float(config["policy_learning_rate"]),
    alpha_learning_rate=float(config.get("alpha_learning_rate", None)),
    buffer_size=config["buffer_size"],
    learning_starts=config["learning_starts"],
    batch_size=config["batch_size"],
    tau=config["tau"],
    gamma=config["gamma"],
    train_frequency=config["train_frequency"],
    policy_update_frequency=config["policy_update_frequency"],
    target_network_frequency=config["target_network_frequency"],
    alpha=config["alpha"],
    autotune_alpha=config["autotune_alpha"],
    wandb_log=f"runs/{run.id}" if not config.get("dont_log", True) else None,
    device=device,
    safety_filter_args=config.get("safety_filter_args", None),
    dont_log=config.get("dont_log", False),
)

if config["train_model"]: 
    model.learn(total_timesteps=config["total_timesteps"])
else:
    model_path = Path(config["eval_model_path"])
    model_path_q1 = model_path / "q1_network.pth"
    model_path_q2 = model_path / "q2_network.pth"
    model_path_target_q1 = model_path / "target_q1_network.pth"
    model_path_target_q2 = model_path / "target_q2_network.pth"
    model_path_actor = model_path / "actor_network.pth"
    if model_path.exists():
        model.q_network_1.load_state_dict(torch.load(model_path_q1, map_location=model.device))
        model.q_network_2.load_state_dict(torch.load(model_path_q2, map_location=model.device))
        model.target_q_network_1.load_state_dict(torch.load(model_path_target_q1, map_location=model.device))
        model.target_q_network_2.load_state_dict(torch.load(model_path_target_q2, map_location=model.device))
        model.actor.load_state_dict(torch.load(model_path_actor, map_location=model.device))
        
        model.target_q_network_1.eval()
        model.target_q_network_2.eval()  # Set the target network to evaluation mode
        model.actor.eval()
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print("No pre-trained model found. Starting with a fresh model.")
        model.learn(total_timesteps=config["total_timesteps"])

model_dir = run_dir / "model"
model_path_q1 = model_dir / "q1_network.pth"
model_path_q2 = model_dir / "q2_network.pth"
model_path_target_q1 = model_dir / "target_q1_network.pth"
model_path_target_q2 = model_dir / "target_q2_network.pth"
model_path_actor = model_dir / "actor_network.pth"
os.makedirs(model_dir, exist_ok=True)
torch.save(model.q_network_1.state_dict(), model_path_q1)
torch.save(model.q_network_2.state_dict(), model_path_q2)
torch.save(model.target_q_network_1.state_dict(), model_path_target_q1)
torch.save(model.target_q_network_2.state_dict(), model_path_target_q2)
torch.save(model.actor.state_dict(), model_path_actor)
print(f"Model saved to {model_dir}")

if config["eval_model"] and not config["manual_mode"]:
    
    eval_env = gym.make(config["env_id"], safety_filter_args=config.get("safety_filter_args", None), eval_mode = True, render_mode="rgb_array", reset_noise_scale=config.get("reset_noise_scale", 0.1),)
    #eval_env = gym.make(config["env_id"], render_mode="rgb_array")
    eval_env = gym.wrappers.RecordVideo(eval_env, video_path, episode_trigger=lambda x: True)

    total_reward = 0
    num_episodes = 10
    print(f"Evaluating the model for {num_episodes} episodes...")
    model.target_q_network_1.eval()  # Set the target network to evaluation mode
    model.target_q_network_2.eval()
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
            
            #rand_action = eval_env.action_space.sample()
            rand_action = np.random.uniform(-1, 1, size=(1,))
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
            action, filter_in_use = model.consult_safety_filter(obs_tensor, task_action=rand_action, use_qcbf=True)
            safety_filter_interventions += filter_in_use
            eval_env.unwrapped.safety_filter_in_use = filter_in_use
            #action, _, _ = model.actor.get_actions(obs_tensor)
            obs, reward, terminated, truncated, info = eval_env.step(action.detach().view(-1))
            #done = terminated or truncated
            eps_step += 1
            done = terminated or eps_step >= MAX_EPISODE_STEPS
            epsisode_reward += reward
        total_reward += epsisode_reward
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {epsisode_reward}, Safety Filter Interventions: {safety_filter_interventions}/{MAX_EPISODE_STEPS} ({(safety_filter_interventions * 100)/MAX_EPISODE_STEPS})%")
        if not config.get("dont_log", True):
            wandb.log({
                        "eval/episode_reward": epsisode_reward,
                        "eval/safety_filter_interventions": safety_filter_interventions,
                    },)
    avg_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    
    if not config.get("dont_log", True):
        for i, video_name in enumerate(video_path.glob("*.mp4")):
            wandb.log({"Viz/video": wandb.Video(str(video_name), format="mp4")})
        wandb.log({"avg_statistics/average_reward": avg_reward},)
        
    #eval_env.close()
    
elif config["eval_model"] and config["manual_mode"]:
    
    eval_env = gym.make(config["env_id"], safety_filter_args=config.get("safety_filter_args", None), render_mode="human", eval_mode=True)
    
    print("\033[33mManual mode is enabled!\033[0m")
    print("You can now interact with the environment manually.")
    
    import sys
    import select
    import tty
    import termios
    
    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    old_settings = termios.tcgetattr(sys.stdin)
    exit_flag = False
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        for i in range(100):
            obs, _ = eval_env.reset()
            while True:
            ## Keyboard inputs to control the agent
                if isData():
                    key = sys.stdin.read(1)
                    print(key, end="\r")
                    if key == 'a':
                        manual_action = np.array([-1.5,])
                    elif key == 'd':
                        manual_action = np.array([1.5,])
                    elif key == 'q':
                        print("Exiting test mode.")
                        # Double break
                        exit()
                    elif key == 'r':
                        print("Resetting environment.")
                        obs, _ = eval_env.reset()
                    else:
                        print(f"Unknown command: {key}")

                else:
                    manual_action = None

                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)

                action, filter_in_use = model.consult_safety_filter(obs_tensor, task_action=manual_action, use_qcbf=True)
                eval_env.unwrapped.safety_filter_in_use = filter_in_use
                obs, reward, terminated, truncated, info = eval_env.step(action.detach().view(-1).numpy())
                #print(terminated, info)
                if terminated: 
                    break
            
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        #eval_env.close()
        
if not config.get("dont_log", True): run.finish()
env.close()

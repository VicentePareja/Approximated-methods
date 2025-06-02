import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

NUM_RUNS = 30
NUM_EPISODES = 1500
TOTAL_TIMESTEPS = 300000
ENV_ID = "MountainCar-v0"

DQN_PARAMS = {
    "learning_rate": 1e-3,
    "buffer_size": 50000,
    "learning_starts": 1000,
    "batch_size": 32,
    "tau": 1.0,
    "target_update_interval": 500,
    "train_freq": 1,
    "gradient_steps": 1,
    "gamma": 0.99,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.02,
    "policy_kwargs": dict(net_arch=[64, 64]),
    "verbose": 0,
}

DATA_DIR = "data"
MONITOR_DIR = os.path.join(DATA_DIR, "monitor_dqn")
os.makedirs(MONITOR_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def run_dqn_and_collect_lengths(run_idx):
    base_path = os.path.join(MONITOR_DIR, f"run_{run_idx:02d}")
    env = gym.make(ENV_ID)
    env = Monitor(env, filename=base_path)
    model = DQN("MlpPolicy", env, **DQN_PARAMS)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10)
    env.close()

    monitor_csv = base_path + ".monitor.csv"
    # Leer usando header=0 para omitir la fila "r,l,t" y comment="#" para ignorar líneas iniciadas con "#"
    df = pd.read_csv(monitor_csv, comment="#", header=0)
    lengths = df["l"].to_numpy().astype(int)

    if lengths.shape[0] < NUM_EPISODES:
        raise ValueError(f"Run {run_idx}: sólo se registraron {lengths.shape[0]} episodios, se requieren {NUM_EPISODES}.")
    return lengths[:NUM_EPISODES]

all_lengths_dqn = np.zeros((NUM_RUNS, NUM_EPISODES), dtype=np.int32)

for run in range(NUM_RUNS):
    print(f"=== DQN: Iniciando corrida {run+1}/{NUM_RUNS} ===")
    np.random.seed(run + 1000)
    all_lengths_dqn[run, :] = run_dqn_and_collect_lengths(run)

def average_every_k(episode_lengths, k=10):
    mean_over_runs = np.mean(episode_lengths, axis=0)
    num_blocks = mean_over_runs.shape[0] // k
    result = np.zeros(num_blocks)
    for i in range(num_blocks):
        block = mean_over_runs[i*k : (i+1)*k]
        result[i] = np.mean(block)
    return result

avg_dqn = average_every_k(all_lengths_dqn, k=10)

np.save(os.path.join(DATA_DIR, "lengths_dqn.npy"), all_lengths_dqn)
print(f"-> Guardado: {os.path.join(DATA_DIR, 'lengths_dqn.npy')}")

header = "block_index,avg_length"
dqn_rows = np.vstack((np.arange(len(avg_dqn))+1, avg_dqn)).T
np.savetxt(
    os.path.join(DATA_DIR, "avg_dqn.csv"),
    dqn_rows,
    delimiter=",",
    header=header,
    comments=""
)
print(f"-> Guardado: {os.path.join(DATA_DIR, 'avg_dqn.csv')}")

print("\nResumen DQN (promedio cada 10 episodios):")
print(" Bloques 1–5:", avg_dqn[:5])
print("\n¡Todos los datos de DQN quedan en la carpeta 'data/' para análisis posterior!")
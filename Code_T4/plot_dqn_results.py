# plot_dqn_results.py

import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Archivos CSV con promedios cada 10 episodios
AVG_DQN_CSV    = os.path.join(DATA_DIR, "avg_dqn.csv")
AVG_SARSA_CSV  = os.path.join(DATA_DIR, "avg_sarsa.csv")
AVG_Q_CSV      = os.path.join(DATA_DIR, "avg_q.csv")

# Cargar datos
dqn_data   = np.loadtxt(AVG_DQN_CSV,   delimiter=",", skiprows=1)
sarsa_data = np.loadtxt(AVG_SARSA_CSV, delimiter=",", skiprows=1)
q_data     = np.loadtxt(AVG_Q_CSV,     delimiter=",", skiprows=1)

blocks_dqn  = dqn_data[:, 0]
avg_dqn     = dqn_data[:, 1]

blocks_sarsa = sarsa_data[:, 0]
avg_sarsa    = sarsa_data[:, 1]

blocks_q  = q_data[:, 0]
avg_q     = q_data[:, 1]

# Gráfico 1: DQN solo
plt.figure(figsize=(8, 5))
plt.plot(blocks_dqn, avg_dqn, label="DQN", linewidth=2, color='C2')
plt.title("DQN en MountainCar-v0\n(promedio de longitud cada 10 episodios)")
plt.xlabel("Bloque de 10 episodios")
plt.ylabel("Longitud promedio del episodio")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "dqn_only.png"), dpi=300)
plt.close()

# Gráfico 2: Comparación SARSA vs Q-Learning vs DQN
plt.figure(figsize=(8, 5))
plt.plot(blocks_sarsa, avg_sarsa, label="SARSA",      linewidth=2, color='C0')
plt.plot(blocks_q,     avg_q,     label="Q-Learning", linewidth=2, linestyle="--", color='C1')
plt.plot(blocks_dqn,    avg_dqn,   label="DQN",       linewidth=2, linestyle=":",  color='C2')
plt.title("Comparación: SARSA vs Q-Learning vs DQN\n(promedio de longitud cada 10 episodios)")
plt.xlabel("Bloque de 10 episodios")
plt.ylabel("Longitud promedio del episodio")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "comparison_all.png"), dpi=300)
plt.close()

print(f"Gráficos guardados en '{RESULTS_DIR}/':")
print(" - DQN solo:       results/dqn_only.png")
print(" - Comparación:    results/comparison_all.png")
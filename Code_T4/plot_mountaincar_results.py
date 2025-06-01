# plot_mountaincar_results.py

import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1) Ubicaciones de archivos de datos
# --------------------------------------------------
DATA_DIR = "data"
AVG_SARSA_CSV = os.path.join(DATA_DIR, "avg_sarsa.csv")
AVG_Q_CSV = os.path.join(DATA_DIR, "avg_q.csv")

# --------------------------------------------------
# 2) Directorio para guardar gráficos
# --------------------------------------------------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------
# 3) Cargar los datos promediados cada 10 episodios
# --------------------------------------------------
# Cada CSV tiene dos columnas: block_index, avg_length
# (block_index va de 1 a 100, avg_length es el valor promedio).
sarsa_data = np.loadtxt(AVG_SARSA_CSV, delimiter=",", skiprows=1)
q_data     = np.loadtxt(AVG_Q_CSV,     delimiter=",", skiprows=1)

# Separar columnas
blocks_sarsa = sarsa_data[:, 0]    # 1, 2, ..., 100
avg_sarsa    = sarsa_data[:, 1]
blocks_q     = q_data[:, 0]
avg_q        = q_data[:, 1]

# --------------------------------------------------
# 4) Crear el gráfico comparativo
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(blocks_sarsa, avg_sarsa, label="SARSA", linewidth=2)
plt.plot(blocks_q,     avg_q,     label="Q-Learning", linewidth=2, linestyle="--")

plt.title("Comparación de SARSA vs Q-Learning en MountainCar-v0\n(promedio de longitud cada 10 episodios)")
plt.xlabel("Bloque de 10 episodios")
plt.ylabel("Longitud promedio del episodio")
plt.legend()
plt.grid(alpha=0.3)

# Guardar figura
output_path = os.path.join(RESULTS_DIR, "sarsa_vs_q_comparison.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Gráfico guardado en: {output_path}")
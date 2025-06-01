# mountaincar_linear_rl.py

import os
import numpy as np
import random
import gymnasium as gym
from FeatureExtractor import FeatureExtractor

# --------------------------------------------------
# 1) Parámetros generales
# --------------------------------------------------
NUM_RUNS = 30         # Número de corridas independientes
NUM_EPISODES = 1000   # Episodios por corrida
GAMMA = 1.0           # Factor de descuento
EPSILON = 0.0         # Epsilon-greedy (en este caso siempre greedy)
ALPHA = 0.5 / 8.0     # Paso de aprendizaje (0.5 dividido por num_tilings = 8)

# --------------------------------------------------
# 2) Directorio para guardar datos
# --------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------
# 3) Función auxiliar: elección epsilon-greedy
# --------------------------------------------------
def epsilon_greedy_action(weights, feature_extractor, observation):
    """
    Con ε = 0, siempre devolvemos la acción greedy:
    a = argmax_a Q(s,a) = w^T phi(s,a)
    En caso de empate, elegimos al azar entre los máximos.
    """
    q_values = []
    for a in [0, 1, 2]:
        f_sa = feature_extractor.get_features(observation, a)
        q_values.append(np.dot(weights, f_sa))
    max_q = max(q_values)
    best_actions = [i for i, q in enumerate(q_values) if q == max_q]
    return random.choice(best_actions)

# --------------------------------------------------
# 4) Sarsa con aproximación lineal y tile coding
# --------------------------------------------------
def run_sarsa(env_name="MountainCar-v0"):
    """
    Corre NUM_RUNS corridas, cada una de NUM_EPISODES episodios.
    Retorna: array de forma (NUM_RUNS, NUM_EPISODES) con la longitud de cada episodio.
    """
    env = gym.make(env_name)
    feature_extractor = FeatureExtractor()
    num_features = feature_extractor.num_of_features

    all_lengths = np.zeros((NUM_RUNS, NUM_EPISODES), dtype=np.int32)

    for run in range(NUM_RUNS):
        random.seed(run + 123)
        np.random.seed(run + 123)

        w = np.zeros(num_features, dtype=np.float64)

        for ep in range(NUM_EPISODES):
            obs, info = env.reset()
            a = epsilon_greedy_action(w, feature_extractor, obs)
            f_sa = feature_extractor.get_features(obs, a)

            done = False
            t = 0

            while not done:
                next_obs, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                if not done:
                    a_next = epsilon_greedy_action(w, feature_extractor, next_obs)
                    f_saprime = feature_extractor.get_features(next_obs, a_next)
                    q_saprime = np.dot(w, f_saprime)
                else:
                    q_saprime = 0.0
                    f_saprime = np.zeros_like(f_sa)

                q_sa = np.dot(w, f_sa)
                delta = reward + GAMMA * q_saprime - q_sa
                w += ALPHA * delta * f_sa

                obs = next_obs
                a = a_next if not done else None
                f_sa = f_saprime.copy()
                t += 1

            all_lengths[run, ep] = t

    env.close()
    return all_lengths

# --------------------------------------------------
# 5) Q-Learning con aproximación lineal y tile coding
# --------------------------------------------------
def run_q_learning(env_name="MountainCar-v0"):
    """
    Corre NUM_RUNS corridas, cada una de NUM_EPISODES episodios.
    Retorna: array de forma (NUM_RUNS, NUM_EPISODES) con la longitud de cada episodio.
    """
    env = gym.make(env_name)
    feature_extractor = FeatureExtractor()
    num_features = feature_extractor.num_of_features

    all_lengths = np.zeros((NUM_RUNS, NUM_EPISODES), dtype=np.int32)

    for run in range(NUM_RUNS):
        random.seed(run + 456)
        np.random.seed(run + 456)

        w = np.zeros(num_features, dtype=np.float64)

        for ep in range(NUM_EPISODES):
            obs, info = env.reset()
            done = False
            t = 0

            while not done:
                a = epsilon_greedy_action(w, feature_extractor, obs)
                f_sa = feature_extractor.get_features(obs, a)
                next_obs, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                if not done:
                    q_values_next = []
                    for a2 in [0, 1, 2]:
                        f_saprime_candidate = feature_extractor.get_features(next_obs, a2)
                        q_values_next.append(np.dot(w, f_saprime_candidate))
                    q_saprime_max = max(q_values_next)
                else:
                    q_saprime_max = 0.0

                q_sa = np.dot(w, f_sa)
                delta = reward + GAMMA * q_saprime_max - q_sa
                w += ALPHA * delta * f_sa

                obs = next_obs
                t += 1

            all_lengths[run, ep] = t

    env.close()
    return all_lengths

# --------------------------------------------------
# 6) Función para promediar cada 10 episodios
# --------------------------------------------------
def average_every_k(episode_lengths, k=10):
    """
    episode_lengths: array de forma (NUM_RUNS, NUM_EPISODES)
    Devuelve arreglo de tamaño (NUM_EPISODES // k) con el promedio (sobre runs y sobre cada bloque de k) de longitudes.
    """
    mean_over_runs = np.mean(episode_lengths, axis=0)  # shape = (NUM_EPISODES,)
    num_blocks = mean_over_runs.shape[0] // k
    result = np.zeros(num_blocks)
    for i in range(num_blocks):
        block = mean_over_runs[i*k : (i+1)*k]
        result[i] = np.mean(block)
    return result

# --------------------------------------------------
# 7) Main: correr ambos métodos, guardar resultados
# --------------------------------------------------
if __name__ == "__main__":
    print("Ejecutando SARSA sobre MountainCar-v0 ...")
    lengths_sarsa = run_sarsa()
    avg_sarsa = average_every_k(lengths_sarsa, k=10)

    print("Ejecutando Q-Learning sobre MountainCar-v0 ...")
    lengths_q = run_q_learning()
    avg_q = average_every_k(lengths_q, k=10)

    # -------------------------
    # Guardar datos en data/
    # -------------------------
    # 1) Guardar longitudes crudas (NumPy arrays)
    np.save(os.path.join(DATA_DIR, "lengths_sarsa.npy"), lengths_sarsa)
    np.save(os.path.join(DATA_DIR, "lengths_q.npy"), lengths_q)
    print(f"-> Guardado: {os.path.join(DATA_DIR, 'lengths_sarsa.npy')}")
    print(f"-> Guardado: {os.path.join(DATA_DIR, 'lengths_q.npy')}")

    # 2) Guardar promedios cada 10 episodios en CSV
    #    (cada archivo tendrá 100 filas, una por bloque de 10 episodios)
    header = "block_index,avg_length"
    # Para SARSA:
    sarsa_rows = np.vstack((np.arange(len(avg_sarsa))+1, avg_sarsa)).T
    np.savetxt(
        os.path.join(DATA_DIR, "avg_sarsa.csv"),
        sarsa_rows,
        delimiter=",",
        header=header,
        comments=""
    )
    # Para Q-Learning:
    q_rows = np.vstack((np.arange(len(avg_q))+1, avg_q)).T
    np.savetxt(
        os.path.join(DATA_DIR, "avg_q.csv"),
        q_rows,
        delimiter=",",
        header=header,
        comments=""
    )
    print(f"-> Guardado: {os.path.join(DATA_DIR, 'avg_sarsa.csv')}")
    print(f"-> Guardado: {os.path.join(DATA_DIR, 'avg_q.csv')}")

    # 3) También imprimimos un pequeño resumen para confirmar
    print("\nResumen (promedio cada 10 episodios):")
    print(" SARSA, bloques 1–5:", avg_sarsa[:5])
    print(" Q-Learning, bloques 1–5:", avg_q[:5])
    print("\nTodos los datos quedan en la carpeta 'data/' para análisis y graficación posterior.")
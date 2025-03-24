import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]  # Берем только первые два признака для 2D-визуализации


def kmeans_custom(X, k, max_iters=100):
    # Инициализация центроидов случайным образом
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    frames = []  # Список для сохранения кадров

    for iteration in range(max_iters):
        # Шаг 1: Назначение точек ближайшему центроиду
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
        labels = np.argmin(distances, axis=0)

        # Сохранение кадра для визуализации
        plt.figure(figsize=(6, 6))
        for i in range(k):
            cluster_points = X[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Кластер {i}')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Центроиды')
        plt.xlabel('Длина чашелистика (cm)')
        plt.ylabel('Ширина чашелистика (cm)')
        plt.title(f'Итерация {iteration + 1}')
        plt.legend()
        plt.savefig(f'frame_{iteration}.png')
        frames.append(imageio.imread(f'frame_{iteration}.png'))
        plt.close()

        # Шаг 2: Пересчет центроидов
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i]
                                  for i in range(k)])

        # Проверка сходимости
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids


    imageio.mimsave('kmeans_iris_animation.gif', frames, fps=1)

    # Очистка временных файлов
    for i in range(len(frames)):
        os.remove(f'frame_{i}.png')

    return labels, centroids

# Запуск алгоритма с k=3 (на основе метода локтя)
k = 3
labels, centroids = kmeans_custom(X, k)

# # Итоговая визуализация
# plt.figure(figsize=(6, 6))
# for i in range(k):
#     cluster_points = X[labels == i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Кластер {i}')
# plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Центроиды')
# plt.xlabel('Длина чашелистика (cm)')
# plt.ylabel('Ширина чашелистика (cm)')
# plt.title('Итоговый результат K-means (Iris)')
# plt.legend()
# plt.show()

# Итоговые проекции для всех пар признаков
feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
feature_names = ['Длина чашелистика', 'Ширина чашелистика', 'Длина лепестка', 'Ширина лепестка']

plt.figure(figsize=(15, 10))
for idx, (i, j) in enumerate(feature_pairs, 1):
    plt.subplot(2, 3, idx)
    X_pair = iris.data[:, [i, j]]
    labels, centroids = kmeans_custom(X_pair, k)  # Повторный запуск для каждой пары
    for cluster in range(k):
        cluster_points = X_pair[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Кластер {cluster}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200)
    plt.xlabel(f'{feature_names[i]} (cm)')
    plt.ylabel(f'{feature_names[j]} (cm)')
    plt.title(f'Проекция {feature_names[i]} - {feature_names[j]}')
    plt.legend()
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
import warnings

# Отключаем предупреждение от Lasso
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# === Загрузка данных ===
df = pd.read_csv("AmesHousing.csv")

# === Предобработка ===
df.drop(columns=["Order", "PID"], inplace=True, errors='ignore')

# Удалим признаки с >30% пропусков
missing = df.isnull().mean()
df = df.loc[:, missing < 0.3]

# Заполним оставшиеся пропуски
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# One-hot кодирование категориальных признаков
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# === Разделение X и y ===
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# === Нормализация ===
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# === Удалим сильно коррелирующие признаки (>0.95) ===
corr_matrix = X_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_scaled.drop(columns=to_drop, inplace=True)

# === PCA для 3D визуализации (2 компоненты + целевая переменная) ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap="viridis")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("SalePrice")
plt.colorbar(sc, label="SalePrice")
plt.title(" 3D-график: PCA компоненты + SalePrice")
plt.show()


# === Разделение на train/test ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# === Lasso + подбор alpha ===
alphas = np.logspace(-3, 1, 20)
rmse_scores = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)


# === График RMSE от alpha ===
plt.figure(figsize=(8, 5))
plt.plot(alphas, rmse_scores, marker="o")
plt.xscale("log")
plt.xlabel("Коэффициент регуляризации (alpha)")
plt.ylabel("RMSE")
plt.title(" Зависимость RMSE от регуляризации (Lasso)")
plt.grid(True)
plt.show()

# === Лучшая модель ===
best_alpha = alphas[np.argmin(rmse_scores)]
best_model = Lasso(alpha=best_alpha, max_iter=1000)
best_model.fit(X_train, y_train)

# === Влияющие признаки ===
coef_series = pd.Series(best_model.coef_, index=X_train.columns)


# Убедимся, что есть хотя бы один ненулевой коэффициент
if coef_series.abs().sum() == 0:
    print(" Все коэффициенты нулевые — модель не смогла найти важные признаки.")
else:
    top_feature = coef_series.abs().idxmax()

    # === Вывод результатов ===
    print(f" Лучший alpha: {best_alpha:.4f}")
    print(f" Минимальный RMSE: {min(rmse_scores):.2f}")
    print(f" Наиболее влияющий признак на SalePrice: {top_feature}")

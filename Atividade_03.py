import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================
# 1. Criando o Dataset
# =========================
np.random.seed(42)  # Mantém os mesmos valores toda vez que rodar

dados = {
    'tamanho_msg': [20, 35, 50, 65, 80, 100, 120, 150, 180, 200]
}

df = pd.DataFrame(dados)

# Regra:
# Cada caractere adiciona ~0.5 min + 5 min base + ruído aleatório
df['tempo_real_espera'] = (
    df['tamanho_msg'] * 0.5
) + 5 + np.random.normal(0, 2, len(df))

print("=== DATASET GERADO ===")
print(df)

# =========================
# 2. Separando Variáveis
# =========================
X = df[['tamanho_msg']]   # Feature
y = df['tempo_real_espera']  # Target

# =========================
# 3. Treinamento do Modelo
# =========================
reg = LinearRegression()
reg.fit(X, y)

# =========================
# 4. Resultados
# =========================
print("\n=== RESULTADOS DO MODELO ===")
print(f"Intercepto (Tempo Base): {reg.intercept_:.2f} min")
print(f"Coeficiente (Aumento por Caractere): {reg.coef_[0]:.2f} min")

# =========================
# 5. Teste de Previsão
# =========================
previsao = reg.predict([[200]])
print(f"\nPrevisão para 200 caracteres: {previsao[0]:.2f} minutos")
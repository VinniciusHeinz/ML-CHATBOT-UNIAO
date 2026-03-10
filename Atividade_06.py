import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================
# DATASET
# =========================
data = {
    'tempo_conversa_min': [5, 12, 2, 20, 15, 8, 30, 10, 5, 25],
    'mensagens_trocadas': [10, 25, 4, 45, 30, 12, 60, 18, 9, 50],
    'custo_ads_real': [15.5, 32.0, 8.0, 55.0, 40.5, 20.0, 80.0, 25.0, 14.0, 65.0]
}

df = pd.DataFrame(data)

# =========================
# 1. Definindo X e y
# =========================
X = df[['tempo_conversa_min', 'mensagens_trocadas']]  # DUAS variáveis
y = df['custo_ads_real']  # Target

# =========================
# 2. Criando o modelo
# =========================
modelo = LinearRegression()

# =========================
# 3. Treinando o modelo
# =========================
modelo.fit(X, y)

# =========================
# 4. Coeficientes
# =========================
print("=== COEFICIENTES DO MODELO ===")
print(f"Intercepto (custo base): {modelo.intercept_:.2f}")
print(f"Coeficiente Tempo (min): {modelo.coef_[0]:.2f}")
print(f"Coeficiente Mensagens: {modelo.coef_[1]:.2f}")

# =========================
# 5. Previsão solicitada
# =========================
previsao = modelo.predict([[15, 35]])

print("\n=== PREVISÃO ===")
print(f"Custo previsto para 15 min e 35 mensagens: R$ {previsao[0]:.2f}")
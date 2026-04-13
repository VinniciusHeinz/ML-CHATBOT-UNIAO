# =============================================================================
# BLOCO 1 — GERADOR DE DATASET SINTÉTICO
# Sistema de Predição de Risco Clínico
# =============================================================================
# Objetivo: Gerar 2000 registros de pacientes fictícios com variáveis biomédicas
# realistas e uma variável alvo (risco) definida por regras clínicas coerentes.
#
# Bibliotecas utilizadas:
#   - numpy  → geração de números aleatórios com distribuições controladas
#   - pandas → criação e exportação do DataFrame
# =============================================================================

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA SEMENTE ALEATÓRIA
# Garante que o dataset gerado seja sempre o mesmo (reprodutibilidade).
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# QUANTIDADE DE REGISTROS
# ─────────────────────────────────────────────────────────────────────────────
N = 2000

# ─────────────────────────────────────────────────────────────────────────────
# LISTA DE NOMES FICTÍCIOS
# Nomes simples, sem sobrenomes, variados para dar realismo ao dataset.
# ─────────────────────────────────────────────────────────────────────────────
nomes_masculinos = [
    "Carlos", "João", "Pedro", "Lucas", "Marcos", "Rafael", "Bruno",
    "Diego", "Felipe", "André", "Gustavo", "Ricardo", "Eduardo", "Thiago",
    "Rodrigo", "Leonardo", "Fábio", "Henrique", "Vinicius", "Mateus",
    "Gabriel", "Leandro", "Renato", "Danilo", "Fernando", "Caio", "Murilo",
    "Otávio", "Sérgio", "Tiago"
]

nomes_femininos = [
    "Ana", "Maria", "Carla", "Fernanda", "Juliana", "Patrícia", "Aline",
    "Camila", "Beatriz", "Amanda", "Larissa", "Vanessa", "Luciana", "Sandra",
    "Gabriela", "Letícia", "Mariana", "Priscila", "Renata", "Simone",
    "Tatiane", "Viviane", "Yasmin", "Bruna", "Cristina", "Débora", "Elaine",
    "Fabiana", "Giovana", "Helena"
]

todos_os_nomes = nomes_masculinos + nomes_femininos

# Sorteia aleatoriamente N nomes (com reposição)
nomes = np.random.choice(todos_os_nomes, size=N, replace=True)

# ─────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DAS VARIÁVEIS BIOMÉDICAS
# Cada variável segue uma distribuição normal (gaussiana) com média e desvio
# padrão baseados em referências clínicas reais, depois é limitada (clip)
# para remover valores biologicamente impossíveis.
# ─────────────────────────────────────────────────────────────────────────────

# IDADE — distribuição entre 18 e 99 anos
# Média 50 anos, desvio 18 → cobre bem adultos jovens, meia-idade e idosos
idade = np.random.normal(loc=50, scale=18, size=N).clip(18, 99).astype(int)

# GLICOSE (mg/dL)
# Normal: 70–99 | Pré-diabetes: 100–125 | Diabetes: ≥126
# Média 105 reflete população com tendência a alterações metabólicas
glicose = np.random.normal(loc=105, scale=30, size=N).clip(60, 300).round(1)

# PRESSÃO ARTERIAL SISTÓLICA (mmHg)
# Normal: <120 | Elevada: 120–129 | Hipertensão estágio 1: 130–139 | Estágio 2: ≥140
# Média 125 reflete distribuição realista da população adulta
pressao_arterial = np.random.normal(loc=125, scale=20, size=N).clip(70, 220).round(1)

# IMC — Índice de Massa Corporal (kg/m²)
# Abaixo do peso: <18.5 | Normal: 18.5–24.9 | Sobrepeso: 25–29.9 | Obesidade: ≥30
# Média 27 é compatível com tendências populacionais atuais
imc = np.random.normal(loc=27, scale=6, size=N).clip(14, 55).round(1)

# COLESTEROL TOTAL (mg/dL)
# Desejável: <200 | Limítrofe: 200–239 | Alto: ≥240
# Média 210 reflete perfil lipídico comum na população adulta
colesterol = np.random.normal(loc=210, scale=45, size=N).clip(100, 400).round(1)

# ─────────────────────────────────────────────────────────────────────────────
# DEFINIÇÃO DA VARIÁVEL ALVO: RISCO CLÍNICO
# Lógica baseada em pontuação de fatores de risco — quanto mais variáveis
# fora dos limites saudáveis, maior o escore e consequentemente o risco.
#
# Cada condição abaixo gera 1 ponto de risco.
# Escore 0–1 → risco 0 (BAIXO)
# Escore 2–3 → risco 1 (MÉDIO)
# Escore 4–5 → risco 2 (ALTO)
# ─────────────────────────────────────────────────────────────────────────────

escore = np.zeros(N, dtype=int)

# Condição 1: Glicose elevada (≥126 mg/dL → zona diabética)
escore += (glicose >= 126).astype(int)

# Condição 2: Pressão arterial alta (≥140 mmHg → hipertensão estágio 2)
escore += (pressao_arterial >= 140).astype(int)

# Condição 3: Obesidade (IMC ≥ 30)
escore += (imc >= 30).astype(int)

# Condição 4: Colesterol alto (≥240 mg/dL)
escore += (colesterol >= 240).astype(int)

# Condição 5: Idade avançada (≥60 anos → fator de risco independente)
escore += (idade >= 60).astype(int)

# Classificação do risco com base no escore acumulado
risco = np.where(escore <= 1, 0,         # 0 ou 1 fatores → BAIXO
         np.where(escore <= 3, 1, 2))    # 2–3 fatores → MÉDIO | 4–5 → ALTO

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUÇÃO DO DATAFRAME
# Organiza todas as variáveis em um DataFrame Pandas estruturado.
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "nome"            : nomes,
    "idade"           : idade,
    "glicose"         : glicose,
    "pressao_arterial": pressao_arterial,
    "imc"             : imc,
    "colesterol"      : colesterol,
    "risco"           : risco
})

# ─────────────────────────────────────────────────────────────────────────────
# EXPORTAÇÃO DO DATASET
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv("pacientes.csv", index=False, encoding="utf-8-sig")

# ─────────────────────────────────────────────────────────────────────────────
# RELATÓRIO DE VALIDAÇÃO
# Exibe uma visão geral do dataset gerado para conferência em aula.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  DATASET GERADO COM SUCESSO → pacientes.csv")
print("=" * 60)
print(f"\nTotal de registros : {len(df)}")
print(f"Total de colunas   : {len(df.columns)}")
print("\n--- Primeiros 5 registros ---")
print(df.head())
print("\n--- Estatísticas descritivas ---")
print(df.describe().round(2))
print("\n--- Distribuição da variável alvo (risco) ---")
contagem = df["risco"].value_counts().sort_index()
rotulos   = {0: "Baixo Risco", 1: "Risco Médio", 2: "Risco Alto"}
for nivel, qtd in contagem.items():
    pct = qtd / N * 100
    print(f"  {rotulos[nivel]} ({nivel}): {qtd} pacientes ({pct:.1f}%)")
print("\n--- Valores nulos por coluna ---")
print(df.isnull().sum())
print("=" * 60)

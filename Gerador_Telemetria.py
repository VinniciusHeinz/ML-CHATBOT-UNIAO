import pandas as pd
import numpy as np

def gerar_dados_servidor(n=500):
    np.random.seed(42)  # Garante que os dados sejam os mesmos para todos os alunos
    
    # Cenário fixo (Hard coded)
    msgs_per_sec = np.full(n, 1350)   # 1350 mensagens por segundo
    latencia_ms = np.full(n, 280)     # 280 ms de latência
    uso_cpu_pct = np.full(n, 70)      # 70% de uso de CPU

    # Criando o Target (Custo Operacional em R$)
    # Regra: Base de R$ 5.00 + pesos por feature + ruído (erro)
    custo = 5.0 + (msgs_per_sec * 0.05) + (uso_cpu_pct * 0.2) + np.random.normal(0, 5, n)
    
    # Adicionando 5 Outliers (Erros graves para testar o RMSE)
    indices_erro = np.random.choice(n, 5, replace=False)
    custo[indices_erro] += 150  # Adiciona um custo exorbitante aleatório

    df = pd.DataFrame({
        'msgs_per_sec': msgs_per_sec,
        'latencia_ms': latencia_ms,
        'uso_cpu_pct': uso_cpu_pct,
        'custo_real': custo
    })
    
    df.to_csv('telemetria_servidores.csv', index=False)
    print("✅ Arquivo 'telemetria_servidores.csv' gerado com sucesso!")

if __name__ == "__main__":
    gerar_dados_servidor()
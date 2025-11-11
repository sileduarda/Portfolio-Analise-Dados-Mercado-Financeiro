# --- Instalação das bibliotecas necessárias ---
# pip install pandas yfinance scikit-learn matplotlib seaborn

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("--- Iniciando Projeto 1: Detecção de Anomalias de Mercado ---")

# --- 1. Coleta de Dados ---
# Vamos usar a PETR4.SA como exemplo
ticker = "PETR4.SA"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

if data.empty:
    print(f"Não foi possível baixar os dados para {ticker}.")
else:
    print(f"Dados de {ticker} baixados com sucesso. Total de {len(data)} dias.")

    # --- 2. Engenharia de Features Estatísticas ---
    # Variação diária do preço
    data['Daily_Return'] = data['Close'].pct_change()
    # Log do Volume (para normalizar a distribuição)
    data['Log_Volume'] = np.log(data['Volume'])

    # Remover valores nulos (primeiro dia não tem 'Daily_Return')
    data = data.dropna()

    # Features que o modelo de ML usará
    features = ['Log_Volume', 'Daily_Return']
    data_features = data[features]

    # --- 3. Detecção de Anomalia (Machine Learning) ---
    # O Isolation Forest é ótimo para isso.
    # 'contamination' é a proporção esperada de anomalias (ex: 1%)
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    
    # Treinar o modelo
    model.fit(data_features)
    
    # Fazer a predição (1 = normal, -1 = anomalia)
    data['Anomaly_ML'] = model.predict(data_features)
    
    # --- 4. Detecção de Anomalia (Estatística Simples) ---
    # Regra: Z-Score do Volume > 3 (mais de 3 desvios padrão da média)
    data['Volume_ZScore'] = (data['Volume'] - data['Volume'].rolling(30).mean()) / data['Volume'].rolling(30).std()
    data['Anomaly_Stats'] = data['Volume_ZScore'].apply(lambda x: -1 if x > 3 else 1)

    data = data.dropna() # Z-score cria NaNs no início

    print(f"Anomalias (ML) detectadas: {np.sum(data['Anomaly_ML'] == -1)} dias")
    print(f"Anomalias (Stats) detectadas: {np.sum(data['Anomaly_Stats'] == -1)} dias")

    # --- 5. Visualização e Insights ---
    
    # ----- Gráfico 1: Anomalias por Machine Learning (Isolation Forest) -----
    plt.figure(figsize=(15, 7))
    sns.set_style("whitegrid")
    
    # Plotar todos os dias (normal)
    plt.plot(data.index, data['Close'], color='blue', label='Preço de Fechamento (Normal)', alpha=0.5)
    
    # Destacar os dias de anomalia
    anomalies_ml = data[data['Anomaly_ML'] == -1]
    plt.scatter(anomalies_ml.index, anomalies_ml['Close'], color='red', marker='o', s=100, label='Anomalia (Isolation Forest)')
    
    plt.title(f"Detecção de Anomalias (Isolation Forest) para {ticker}", fontsize=16)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Preço de Fechamento (R$)", fontsize=12)
    plt.legend()
    plt.savefig("anomalias_ml.png") # Salva o gráfico
    print("Gráfico 'anomalias_ml.png' salvo.")

    # ----- Gráfico 2: Anomalias por Estatística (Z-Score) -----
    plt.figure(figsize=(15, 7))
    
    # Plotar todos os dias (normal)
    plt.plot(data.index, data['Close'], color='blue', label='Preço de Fechamento (Normal)', alpha=0.5)
    
    # Destacar os dias de anomalia
    anomalies_stats = data[data['Anomaly_Stats'] == -1]
    plt.scatter(anomalies_stats.index, anomalies_stats['Close'], color='orange', marker='X', s=100, label='Anomalia (Z-Score > 3)')
    
    plt.title(f"Detecção de Anomalias (Z-Score de Volume > 3) para {ticker}", fontsize=16)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Preço de Fechamento (R$)", fontsize=12)
    plt.legend()
    plt.savefig("anomalias_stats.png") # Salva o gráfico
    print("Gráfico 'anomalias_stats.png' salvo.")

# --- 6. Insights do Projeto 1 ---
print("\n--- Insights (Projeto 1) ---")
print(f"""
1.  **Contexto:** O objetivo deste projeto foi identificar dias com padrões de negociação atípicos (volume e preço) para a {ticker}, que poderiam justificar uma investigação mais profunda por parte da área de supervisão de mercados.
2.  **Método ML (Isolation Forest):** O modelo sinalizou {np.sum(data['Anomaly_ML'] == -1)} dias. Essas anomalias tendem a ser dias com uma *combinação* atípica de alto volume e grande variação de preço (para cima ou para baixo).
3.  **Método Estatístico (Z-Score):** O método, focado *apenas* no volume, sinalizou {np.sum(data['Anomaly_Stats'] == -1)} dias onde o volume foi mais de 3 desvios padrão acima da média móvel de 30 dias.
4.  **Conclusão:** Os dias sinalizados (especialmente os que aparecem em ambos os métodos) seriam os primeiros a serem analisados. Um analista investigaria se houve notícias relevantes (ex: resultados, fatos relevantes) que justificassem o padrão. A ausência de notícias poderia ser um forte indicativo de *insider trading* ou manipulação.
""")

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, pearsonr
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# # CONFIGURAÇÃO##

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
OUTDIR = "outputs_hw2"
FIGDIR = "figures_hw2"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Tradução dos nomes das características
FEATURE_NAMES_PT = {
    'fixed acidity': 'Acidez Fixa',
    'volatile acidity': 'Acidez Volátil', 
    'citric acid': 'Ácido Cítrico',
    'residual sugar': 'Açúcar Residual',
    'chlorides': 'Cloretos',
    'free sulfur dioxide': 'Dióxido de Enxofre Livre',
    'total sulfur dioxide': 'Dióxido de Enxofre Total',
    'density': 'Densidade',
    'pH': 'pH',
    'sulphates': 'Sulfatos',
    'alcohol': 'Álcool',
    'quality': 'Qualidade'
}

FEATURE_NAMES_EN = list(FEATURE_NAMES_PT.keys())[:-1] 
FEATURE_NAMES_PT_LIST = [FEATURE_NAMES_PT[name] for name in FEATURE_NAMES_EN]

# # FUNÇÕES AUXILIARES##

def load_and_combine_data():
    """Carrega e combina os datasets de vinho tinto e branco"""
    print("Carregando dados...")
    
    def download_if_needed(url, filename):
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
        return pd.read_csv(filename, sep=';')
    
    df_red = download_if_needed(RED_URL, "winequality-red.csv")
    df_white = download_if_needed(WHITE_URL, "winequality-white.csv")
    
    df_red['tipo_vinho'] = 'Tinto'
    df_white['tipo_vinho'] = 'Branco'
    df = pd.concat([df_red, df_white], ignore_index=True)
    df = df.rename(columns=FEATURE_NAMES_PT)
    
    print(f"✓ Dados carregados: {df.shape[0]} observações, {df.shape[1] - 2} características")
    return df

def calculate_rmse(y_true, y_pred):
    """Calcula RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_r2(y_true, y_pred):
    """Calcula R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def manual_kfold_cv(X, y, model_func, k=10, random_state=42):
    """Implementação de validação cruzada k-fold"""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    fold_size = len(X) // k
    rmse_scores, r2_scores = [], []
    
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else len(X)
        test_idx = indices[start_idx:end_idx]
        train_idx = np.setdiff1d(indices, test_idx)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model_func(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse_scores.append(calculate_rmse(y_test, y_pred))
        r2_scores.append(calculate_r2(y_test, y_pred))
    
    return np.mean(rmse_scores), np.std(rmse_scores), np.mean(r2_scores), np.std(r2_scores)

# # IMPLEMENTAÇÕES DOS MODELOS##

class OLSManual:
    """Implementação de Regressão Linear Ordinária"""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        try:
            self.coef_full_ = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(X_with_intercept, full_matrices=False)
            s_inv = np.diag(1 / s)
            self.coef_full_ = Vt.T @ s_inv @ U.T @ y
        
        self.intercept_ = self.coef_full_[0]
        self.coef_ = self.coef_full_[1:]
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")
        return self.intercept_ + X @ self.coef_

class RidgeManual:
    """Implementação de Regressão Ridge (L2)"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        n_features = X_with_intercept.shape[1]
        penalty_matrix = self.alpha * np.eye(n_features)
        penalty_matrix[0, 0] = 0
        
        try:
            self.coef_full_ = np.linalg.inv(X_with_intercept.T @ X_with_intercept + penalty_matrix) @ X_with_intercept.T @ y
        except np.linalg.LinAlgError:
            self.coef_full_ = np.linalg.pinv(X_with_intercept.T @ X_with_intercept + penalty_matrix) @ X_with_intercept.T @ y
        
        self.intercept_ = self.coef_full_[0]
        self.coef_ = self.coef_full_[1:]
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")
        return self.intercept_ + X @ self.coef_

def pca_manual(X, n_components=None):
    """Implementação de PCA"""
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]
    
    X_pca = X_centered @ eigenvectors
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    return X_pca, eigenvectors, explained_variance_ratio

## ANÁLISE EXPLORATÓRIA (PASSO 0)##

def analise_exploratoria(df):
    """Realiza análise exploratória completa dos dados"""
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*60)
    
    X = df[FEATURE_NAMES_PT_LIST].values
    y = df['Qualidade'].values
    
    # 1. Estatísticas descritivas ANTES da transformação
    print("\n1. Estatísticas descritivas:")
    stats_before = []
    for i, col in enumerate(FEATURE_NAMES_PT_LIST):
        valores = X[:, i]
        stats_before.append([
            col, np.mean(valores), np.std(valores), skew(valores),
            np.min(valores), np.percentile(valores, 25), np.percentile(valores, 50),
            np.percentile(valores, 75), np.max(valores)
        ])
    
    stats_before_df = pd.DataFrame(stats_before, columns=[
        'Característica', 'Média', 'Desvio Padrão', 'Assimetria',
        'Mínimo', 'Q1', 'Mediana', 'Q3', 'Máximo'
    ])
    stats_before_df.to_csv(os.path.join(OUTDIR, 'estatisticas_antes_transformacao.csv'), index=False, encoding='utf-8')
    
    # 2. Análise de skewness
    skewness_values = skew(X, axis=0)
    skewness_df = pd.DataFrame({
        'Característica': FEATURE_NAMES_PT_LIST,
        'Skewness': skewness_values,
        '|Skewness| > 0.5': np.abs(skewness_values) > 0.5,
        '|Skewness| > 1.0': np.abs(skewness_values) > 1.0
    })
    skewness_df.to_csv(os.path.join(OUTDIR, 'analise_skewness.csv'), index=False, encoding='utf-8')
    
    print(f"   • {sum(skewness_df['|Skewness| > 0.5'])} características com |skewness| > 0.5")
    print(f"   • {sum(skewness_df['|Skewness| > 1.0'])} características com |skewness| > 1.0")
    
    # 3. Correlação com o target
    correlations_with_target = []
    for i, col in enumerate(FEATURE_NAMES_PT_LIST):
        corr, _ = pearsonr(X[:, i], y)
        correlations_with_target.append((col, corr))
    
    correlations_with_target.sort(key=lambda x: abs(x[1]), reverse=True)
    corr_target_df = pd.DataFrame(correlations_with_target, columns=['Característica', 'Correlação'])
    corr_target_df.to_csv(os.path.join(OUTDIR, 'correlacao_com_target_antes.csv'), index=False, encoding='utf-8')
    
    print("\n2. Top 5 características mais correlacionadas com qualidade:")
    for i, (feature, corr) in enumerate(correlations_with_target[:5]):
        print(f"   {i+1}. {feature}: {corr:.3f}")
    
    # 4. Gerar visualizações
    print("\n3. Gerando visualizações...")
    
    # Histogramas
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_NAMES_PT_LIST):
        ax = axes[i]
        ax.hist(X[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'Histograma de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequência')
        ax.grid(alpha=0.3)
    for i in range(len(FEATURE_NAMES_PT_LIST), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'histogramas_antes_transformacao.png'), dpi=150)
    plt.close()
    
    # Boxplots
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_NAMES_PT_LIST):
        ax = axes[i]
        ax.boxplot(X[:, i], vert=True)
        ax.set_title(f'Boxplot de {col}')
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
    for i in range(len(FEATURE_NAMES_PT_LIST), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'boxplots_antes_transformacao.png'), dpi=150)
    plt.close()
    
    # Matriz de correlação
    corr_matrix_before = np.corrcoef(X, rowvar=False)
    corr_df_before = pd.DataFrame(corr_matrix_before, index=FEATURE_NAMES_PT_LIST, columns=FEATURE_NAMES_PT_LIST)
    corr_df_before.to_csv(os.path.join(OUTDIR, 'matriz_correlacao_antes.csv'), encoding='utf-8')
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr_matrix_before, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(FEATURE_NAMES_PT_LIST)), FEATURE_NAMES_PT_LIST, rotation=45, ha='right')
    plt.yticks(range(len(FEATURE_NAMES_PT_LIST)), FEATURE_NAMES_PT_LIST)
    plt.title('Matriz de Correlação (Antes da Transformação)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'heatmap_correlacao_antes.png'), dpi=150)
    plt.close()
    
    print("✓ Análise exploratória concluída. Arquivos salvos em:")
    print(f"   - {OUTDIR}/ (dados)")
    print(f"   - {FIGDIR}/ (gráficos)")
    
    return X, y

## PRÉ-PROCESSAMENTO E TRANSFORMAÇÃO ##

def preprocessamento(X_train, X_test, y_train, y_test):
    """Aplica transformação de potência e padronização"""
    print("\n" + "="*60)
    print("PRÉ-PROCESSAMENTO DOS DADOS")
    print("="*60)
    
    preprocessor = Pipeline([
        ('power_transform', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Análise após transformação
    skewness_after = skew(X_train_transformed, axis=0)
    skewness_before = skew(X_train, axis=0)
    
    reducao_media = np.mean(100 * (1 - np.abs(skewness_after) / np.abs(np.where(skewness_before != 0, skewness_before, 0.01))))
    
    print(f"\n1. Redução média de skewness: {reducao_media:.1f}%")
    print(f"2. Dados transformados: {X_train_transformed.shape}")
    
    # Gerar histogramas após transformação
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_NAMES_PT_LIST):
        ax = axes[i]
        ax.hist(X_train_transformed[:, i], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_title(f'{col} (Transformado)')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequência')
        ax.grid(alpha=0.3)
    for i in range(len(FEATURE_NAMES_PT_LIST), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'histogramas_depois_transformacao.png'), dpi=150)
    plt.close()
    
    return X_train_transformed, X_test_transformed, preprocessor

## ANÁLISE PCA ##

def analise_pca_manual(X_train_transformed):
    """Realiza análise PCA"""
    print("\n" + "="*60)
    print("ANÁLISE PCA MANUAL")
    print("="*60)
    
    X_pca, eigenvectors, explained_variance_ratio = pca_manual(X_train_transformed)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Salvar resultados
    pca_results = pd.DataFrame({
        'Componente': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        'Autovalor': np.linalg.eigvalsh(np.cov(X_train_transformed.T))[::-1],
        'Variância_Explicada': explained_variance_ratio,
        'Variância_Acumulada': cumulative_variance
    })
    pca_results.to_csv(os.path.join(OUTDIR, 'pca_resultados_manual.csv'), index=False, encoding='utf-8')
    
    comp_80 = (cumulative_variance < 0.8).sum() + 1
    comp_90 = (cumulative_variance < 0.9).sum() + 1
    comp_95 = (cumulative_variance < 0.95).sum() + 1
    
    print(f"\n1. Variância explicada pelos componentes:")
    print(f"   • 80% da variância: {comp_80} componentes")
    print(f"   • 90% da variância: {comp_90} componentes")
    print(f"   • 95% da variância: {comp_95} componentes")
    
    # Gráfico de variância explicada
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, color='steelblue', label='Individual')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'r-', marker='o', label='Acumulada')
    plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='80%')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95%')
    plt.xlabel('Componente Principal')
    plt.ylabel('Variância Explicada')
    plt.title('Variância Explicada por Componente Principal (PCA Manual)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'pca_variancia_explicada.png'), dpi=150)
    plt.close()
    
    print("✓ Análise PCA concluída")
    return X_pca, pca_results

## MODELAGEM - PASSO 1: REGRESSÃO OLS##

def passo1_regressao_ols(X_train, X_test, y_train, y_test):
    """Passo 1: Regressão OLS manual e com sklearn"""
    print("\n" + "="*60)
    print("PASSO 1: REGRESSÃO OLS")
    print("="*60)
    
    resultados_ols = {}
    
    # OLS Manual
    ols_manual = OLSManual()
    ols_manual.fit(X_train, y_train)
    y_pred_manual = ols_manual.predict(X_test)
    rmse_manual = calculate_rmse(y_test, y_pred_manual)
    r2_manual = calculate_r2(y_test, y_pred_manual)
    
    # OLS Sklearn
    ols_sklearn = LinearRegression()
    ols_sklearn.fit(X_train, y_train)
    y_pred_sklearn = ols_sklearn.predict(X_test)
    rmse_sklearn = calculate_rmse(y_test, y_pred_sklearn)
    r2_sklearn = calculate_r2(y_test, y_pred_sklearn)
    
    # Validação Cruzada
    def ols_model_func(X_train_fold, y_train_fold):
        model = OLSManual()
        model.fit(X_train_fold, y_train_fold)
        return model
    
    rmse_cv_mean, rmse_cv_std, r2_cv_mean, r2_cv_std = manual_kfold_cv(
        X_train, y_train, ols_model_func, k=10, random_state=RANDOM_SEED
    )
    
    print(f"\n1. Resultados OLS:")
    print(f"   • Manual:      RMSE = {rmse_manual:.4f}, R² = {r2_manual:.4f}")
    print(f"   • Sklearn:     RMSE = {rmse_sklearn:.4f}, R² = {r2_sklearn:.4f}")
    print(f"   • Validação Cruzada (10-fold):")
    print(f"     RMSE = {rmse_cv_mean:.4f} ± {rmse_cv_std:.4f}")
    print(f"     R²   = {r2_cv_mean:.4f} ± {r2_cv_std:.4f}")
    
    # Salvar resultados
    resultados_ols = {
        'manual': {'rmse': rmse_manual, 'r2': r2_manual},
        'sklearn': {'rmse': rmse_sklearn, 'r2': r2_sklearn},
        'cv_manual': {'rmse_mean': rmse_cv_mean, 'rmse_std': rmse_cv_std, 
                     'r2_mean': r2_cv_mean, 'r2_std': r2_cv_std}
    }
    
    # Gráfico de resultados
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, y_pred_manual, alpha=0.5, s=20)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Qualidade Real')
    axes[0].set_ylabel('Qualidade Predita')
    axes[0].set_title('OLS Manual: Reais vs Preditos')
    axes[0].grid(alpha=0.3)
    
    residuals = y_test - y_pred_manual
    axes[1].scatter(y_pred_manual, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Valores Preditos')
    axes[1].set_ylabel('Resíduos')
    axes[1].set_title('OLS Manual: Análise de Resíduos')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'ols_resultados.png'), dpi=150)
    plt.close()
    
    print("✓ Gráficos OLS salvos")
    return resultados_ols, ols_manual, ols_sklearn

## MODELAGEM - PASSO 2: REGRESSÃO RIDGE (L2) ##

def passo2_regressao_ridge(X_train, X_test, y_train, y_test):
    """Passo 2: Regressão Ridge manual e com sklearn"""
    print("\n" + "="*60)
    print("PASSO 2: REGRESSÃO RIDGE (L2)")
    print("="*60)
    
    # Determinar alpha ótimo
    alphas = np.logspace(-3, 3, 50)
    ridge = Ridge(random_state=RANDOM_SEED)
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid={'alpha': alphas},
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_alpha = grid_search.best_params_['alpha']
    
    # Ridge Manual
    ridge_manual = RidgeManual(alpha=best_alpha)
    ridge_manual.fit(X_train, y_train)
    y_pred_ridge_manual = ridge_manual.predict(X_test)
    rmse_ridge_manual = calculate_rmse(y_test, y_pred_ridge_manual)
    r2_ridge_manual = calculate_r2(y_test, y_pred_ridge_manual)
    
    # Ridge Sklearn
    ridge_sklearn = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
    ridge_sklearn.fit(X_train, y_train)
    y_pred_ridge_sklearn = ridge_sklearn.predict(X_test)
    rmse_ridge_sklearn = calculate_rmse(y_test, y_pred_ridge_sklearn)
    r2_ridge_sklearn = calculate_r2(y_test, y_pred_ridge_sklearn)
    
    print(f"\n1. Parâmetro ótimo: α = {best_alpha:.4f}")
    print(f"\n2. Resultados Ridge:")
    print(f"   • Manual:      RMSE = {rmse_ridge_manual:.4f}, R² = {r2_ridge_manual:.4f}")
    print(f"   • Sklearn:     RMSE = {rmse_ridge_sklearn:.4f}, R² = {r2_ridge_sklearn:.4f}")
    
    # Gráfico do perfil CV
    cv_results = grid_search.cv_results_
    mean_scores = np.sqrt(-cv_results['mean_test_score'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, mean_scores, 'o-', alpha=0.5)
    plt.axvline(best_alpha, color='red', linestyle='--', label=f'Melhor α: {best_alpha:.4f}')
    plt.xscale('log')
    plt.xlabel('Alpha (λ) - Escala Logarítmica')
    plt.ylabel('RMSE (Validação Cruzada)')
    plt.title('Perfil de Validação Cruzada para Regressão Ridge')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'ridge_cv_profile_completo.png'), dpi=150)
    plt.close()
    
    # Gráfico de coeficientes
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(FEATURE_NAMES_PT_LIST))
    width = 0.35
    plt.bar(x_pos - width/2, ridge_manual.coef_, width, label='Ridge Manual', alpha=0.8)
    plt.bar(x_pos + width/2, ridge_sklearn.coef_, width, label='Ridge Sklearn', alpha=0.8)
    plt.xlabel('Características')
    plt.ylabel('Valor do Coeficiente')
    plt.title('Comparação de Coeficientes: Ridge Manual vs Sklearn')
    plt.xticks(x_pos, FEATURE_NAMES_PT_LIST, rotation=45, ha='right')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'ridge_coeficientes_comparacao.png'), dpi=150)
    plt.close()
    
    print("✓ Gráficos Ridge salvos")
    
    resultados_ridge = {
        'best_alpha': best_alpha,
        'manual': {'rmse': rmse_ridge_manual, 'r2': r2_ridge_manual},
        'sklearn': {'rmse': rmse_ridge_sklearn, 'r2': r2_ridge_sklearn}
    }
    
    return resultados_ridge, ridge_manual, ridge_sklearn

## MODELAGEM - PASSO 3: REGRESSÃO PLS##

def passo3_regressao_pls(X_train, X_test, y_train, y_test):
    """Passo 3: Regressão PLS com seleção de componentes"""
    print("\n" + "="*60)
    print("PASSO 3: REGRESSÃO PLS")
    print("="*60)
    
    max_components = min(10, X_train.shape[1])
    n_components_range = range(1, max_components + 1)
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    
    rmse_scores_cv = []
    
    print(f"\n1. Buscando melhor número de componentes (1-{max_components})...")
    for n_comp in n_components_range:
        pls = PLSRegression(n_components=n_comp)
        rmse_scores = np.sqrt(-cross_val_score(
            pls, X_train, y_train,
            scoring='neg_mean_squared_error', cv=kf
        ))
        rmse_scores_cv.append(rmse_scores.mean())
    
    best_n_components = np.argmin(rmse_scores_cv) + 1
    
    # Treinar modelo final
    pls_final = PLSRegression(n_components=best_n_components)
    pls_final.fit(X_train, y_train)
    y_pred_pls = pls_final.predict(X_test).ravel()
    
    rmse_pls = calculate_rmse(y_test, y_pred_pls)
    r2_pls = calculate_r2(y_test, y_pred_pls)
    
    print(f"\n2. Resultados PLS:")
    print(f"   • Componentes ótimos: {best_n_components}")
    print(f"   • RMSE: {rmse_pls:.4f}")
    print(f"   • R²: {r2_pls:.4f}")
    
    # Perfil de validação cruzada
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, rmse_scores_cv, 'o-', linewidth=2, markersize=8)
    plt.axvline(best_n_components, color='red', linestyle='--', 
                label=f'Ótimo: {best_n_components} componentes')
    plt.xlabel('Número de Componentes')
    plt.ylabel('RMSE (Validação Cruzada)')
    plt.title('Perfil de Validação Cruzada PLS')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'pls_cv_profile_completo.png'), dpi=150)
    plt.close()
    
    print("✓ Gráfico PLS salvo")
    
    resultados_pls = {
        'best_n_components': best_n_components,
        'test': {'rmse': rmse_pls, 'r2': r2_pls}
    }
    
    return resultados_pls, pls_final

## MODELAGEM - PASSO 4: REDE NEURAL##

def passo4_rede_neural(X_train, X_test, y_train, y_test):
    """Passo 4: Rede Neural para regressão"""
    print("\n" + "="*60)
    print("PASSO 4: REDE NEURAL (MLP REGRESSOR)")
    print("="*60)
    
    print("\n1. Configurando rede neural...")
    print("   Arquitetura: 100 → 50 → 20 neurônios")
    print("   Função de ativação: ReLU")
    print("   Otimizador: Adam")
    print("   Early stopping: Ativado")
    
    model_nn = MLPRegressor(
        hidden_layer_sizes=(100, 50, 20), 
        activation='relu',
        solver='adam',
        alpha=0.001, 
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=RANDOM_SEED,
        early_stopping=True,
        n_iter_no_change=20,
        verbose=False
    )
    
    model_nn.fit(X_train, y_train)
    y_pred_nn = model_nn.predict(X_test)
    
    rmse_nn = calculate_rmse(y_test, y_pred_nn)
    r2_nn = calculate_r2(y_test, y_pred_nn)
    
    print(f"\n2. Resultados Rede Neural:")
    print(f"   • RMSE: {rmse_nn:.4f}")
    print(f"   • R²: {r2_nn:.4f}")
    print(f"   • Épocas treinadas: {model_nn.n_iter_}")
    
    # Curva de aprendizado
    if hasattr(model_nn, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model_nn.loss_curve_, 'b-', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Loss (MSE)')
        plt.title('Curva de Aprendizado da Rede Neural')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, 'nn_curva_aprendizado.png'), dpi=150)
        plt.close()
        print("✓ Curva de aprendizado salva")
    
    resultados_nn = {
        'test': {'rmse': rmse_nn, 'r2': r2_nn},
        'architecture': '100x50x20',
        'n_iterations': model_nn.n_iter_
    }
    
    return resultados_nn, model_nn

## ANÁLISE COMPARATIVA FINAL##

def analise_comparativa_final(resultados):
    """Realiza análise comparativa final entre todos os modelos"""
    print("\n" + "="*60)
    print("ANÁLISE COMPARATIVA FINAL")
    print("="*60)
    
    tabela_comparativa = []
    
    for nome, res in resultados.items():
        if nome == 'ols':
            tabela_comparativa.append(['OLS (Sklearn)', 'N/A', 
                                      f"{res['sklearn']['rmse']:.4f}",
                                      f"{res['sklearn']['r2']:.4f}"])
        elif nome == 'ridge':
            tabela_comparativa.append(['Ridge', f"α={res['best_alpha']:.2f}", 
                                      f"{res['sklearn']['rmse']:.4f}",
                                      f"{res['sklearn']['r2']:.4f}"])
        elif nome == 'pls':
            tabela_comparativa.append(['PLS', f"Comp={res['best_n_components']}", 
                                      f"{res['test']['rmse']:.4f}",
                                      f"{res['test']['r2']:.4f}"])
        elif nome == 'nn':
            tabela_comparativa.append(['Rede Neural', '100x50x20', 
                                      f"{res['test']['rmse']:.4f}",
                                      f"{res['test']['r2']:.4f}"])
    
    print("\nDESEMPENHO DOS MODELOS:")
    print(tabulate(tabela_comparativa, 
                  headers=['Modelo', 'Parâmetros', 'RMSE', 'R²'],
                  tablefmt='simple', stralign='center'))
    
    # Determinar melhor modelo
    melhor_modelo = None
    melhor_rmse = float('inf')
    
    for nome, res in resultados.items():
        if nome == 'ols':
            rmse_atual = res['sklearn']['rmse']
            nome_completo = 'OLS'
        elif nome == 'ridge':
            rmse_atual = res['sklearn']['rmse']
            nome_completo = 'Ridge'
        elif nome == 'pls':
            rmse_atual = res['test']['rmse']
            nome_completo = 'PLS'
        elif nome == 'nn':
            rmse_atual = res['test']['rmse']
            nome_completo = 'Rede Neural'
        else:
            continue
            
        if rmse_atual < melhor_rmse:
            melhor_rmse = rmse_atual
            melhor_modelo = nome_completo
    
    print(f"\n✓ MELHOR MODELO: {melhor_modelo} (RMSE: {melhor_rmse:.4f})")
    
    # Gráfico comparativo
    modelos_plot = []
    rmse_vals = []
    r2_vals = []
    
    for linha in tabela_comparativa:
        modelos_plot.append(linha[0])
        rmse_vals.append(float(linha[2]))
        r2_vals.append(float(linha[3]))
    
    x = np.arange(len(modelos_plot))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    bars1 = ax1.bar(x, rmse_vals, width=0.4, label='RMSE', color='skyblue', alpha=0.8)
    ax1.set_xlabel('Modelo')
    ax1.set_ylabel('RMSE (menor é melhor)', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modelos_plot, rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    ax2.plot(x, r2_vals, 'ro-', linewidth=2, markersize=8, label='R²')
    ax2.set_ylabel('R² (maior é melhor)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Comparação de Desempenho: RMSE e R² por Modelo')
    fig.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'comparacao_final_modelos.png'), dpi=150)
    plt.close()
    
    print("✓ Gráfico comparativo salvo")
    
    return tabela_comparativa

## FUNÇÃO PRINCIPAL##

def main():
    """Função principal"""
    print("="*60)
    print("HW2 - MODELOS DE REGRESSÃO PARA PREVISÃO DA QUALIDADE DO VINHO")
    print("="*60)
    
    # 1. Carregar dados
    df = load_and_combine_data()
    
    # 2. Análise exploratória
    X, y = analise_exploratoria(df)
    
    # 3. Divisão treino-teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED
    )
    print(f"\nDivisão dos dados:")
    print(f"  • Treino: {X_train.shape[0]} observações")
    print(f"  • Teste:  {X_test.shape[0]} observações")
    
    # 4. Pré-processamento
    X_train_transformed, X_test_transformed, preprocessor = preprocessamento(
        X_train, X_test, y_train, y_test
    )
    
    # 5. Análise PCA
    X_pca, pca_results = analise_pca_manual(X_train_transformed)
    
    # 6. Modelagem
    print("\n" + "="*60)
    print("INICIANDO MODELAGEM")
    print("="*60)
    
    todos_resultados = {}
    
    # Passo 1: OLS
    resultados_ols, ols_manual, ols_sklearn = passo1_regressao_ols(
        X_train_transformed, X_test_transformed, y_train, y_test
    )
    todos_resultados['ols'] = resultados_ols
    
    # Passo 2: Ridge
    resultados_ridge, ridge_manual, ridge_sklearn = passo2_regressao_ridge(
        X_train_transformed, X_test_transformed, y_train, y_test
    )
    todos_resultados['ridge'] = resultados_ridge
    
    # Passo 3: PLS
    resultados_pls, pls_model = passo3_regressao_pls(
        X_train_transformed, X_test_transformed, y_train, y_test
    )
    todos_resultados['pls'] = resultados_pls
    
    # Passo 4: Rede Neural
    resultados_nn, nn_model = passo4_rede_neural(
        X_train_transformed, X_test_transformed, y_train, y_test
    )
    todos_resultados['nn'] = resultados_nn
    
    # 7. Análise comparativa final
    tabela_final = analise_comparativa_final(todos_resultados)
    
    # 8. Conclusões
    print("\n" + "="*60)
    print("CONCLUSÕES")
    print("="*60)
    
    # Determinar se há relações não-lineares
    rmse_nn = todos_resultados['nn']['test']['rmse']
    melhor_linear_rmse = min(
        todos_resultados['ols']['sklearn']['rmse'],
        todos_resultados['ridge']['sklearn']['rmse'],
        todos_resultados['pls']['test']['rmse']
    )
    
    if rmse_nn < melhor_linear_rmse:
        diferenca_percentual = 100 * (melhor_linear_rmse - rmse_nn) / melhor_linear_rmse
        print(f"✓ A Rede Neural é {diferenca_percentual:.1f}% melhor que o melhor modelo linear")
        print("  → Sugere existência de relações não-lineares significativas")
    else:
        print("✓ Os modelos lineares tiveram desempenho similar ou melhor")
        print("  → As relações podem ser predominantemente lineares")
    
    # Características mais importantes
    if 'ridge' in todos_resultados:
        coef_abs = np.abs(ridge_sklearn.coef_)
        indices_ordenados = np.argsort(coef_abs)[::-1]
        
        print(f"\nCaracterísticas mais importantes (Ridge):")
        for i, idx in enumerate(indices_ordenados[:3]):
            print(f"  {i+1}. {FEATURE_NAMES_PT_LIST[idx]}")
    
    print("\n" + "="*60)
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print(f"Arquivos salvos em: {OUTDIR}/ e {FIGDIR}/")

if __name__ == "__main__":
    main()
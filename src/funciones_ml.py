# =============================================================================
# funciones_ml.py
# Proyecto: Machine Learning 2026
#
# Objetivo:
# - Centralizar funciones auxiliares comunes de EDA, evaluación,
#   visualización y comparación de modelos.
# - Mantener nomenclatura, formato y salidas homogéneas en todos los scripts.
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from scipy.stats import chi2_contingency
from matplotlib.lines import Line2D

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, balanced_accuracy_score,
                             roc_curve, auc, confusion_matrix, 
                             classification_report)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance

RANDOM_STATE = 12345

COLORES_MODELOS = {
    'SVM lineal': 'deepskyblue',
    'SVM RBF': 'coral',
    'SVM Polinómico': 'limegreen',
    'lineal': 'deepskyblue',
    'RBF': 'coral',
    'POLY': 'limegreen',
    'Bagging SVM RBF': 'firebrick',
    'Árbol de decisiones': 'limegreen',
    'Bagging Árbol': 'seagreen',
    'R.Logística': 'aqua',
    'Bagging R.Logística': 'royalblue',
    'Random Forest': 'forestgreen',
    'XGBoost': 'gold',
    'Stacking (R.Log.)': 'mediumpurple'
    }

# FUNCIONES UTILIZADAS EN LOS SCRIPTS

def v_cramer(x, y):
    """
    Calcula la V de Cramer entre dos variables categóricas.
    """
    tabla = pd.crosstab(x, y)
    
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return np.nan
    
    chi2 = chi2_contingency(tabla)[0]
    n = tabla.to_numpy().sum()
    k = min(tabla.shape)
    
    if n == 0 or k <= 1:
        return np.nan
    
    return np.sqrt(chi2 / (n * (k - 1)))


def matriz_corr(matriz: pd.DataFrame, titulo):
    """
    Representa gráficamente una matriz.
    """
    mask = np.triu(np.ones_like(matriz, dtype=bool), k=1)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(matriz, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                mask=mask)
    
    plt.title(f'Matriz de correlación: {titulo}', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()


def histogramas(df, variables, columnas=2, bins=25, titulo='Histogramas'):
    """
    Representa histogramas simples para diagnóstico de distribución y normalidad.
    """
    n = len(variables)
    filas = (n + columnas - 1) // columnas
    plt.figure(figsize=(6 * columnas, 4 * filas))

    for i, var in enumerate(variables, 1):
        plt.subplot(filas, columnas, i)
        plt.hist(df[var].dropna(), bins=bins, edgecolor='black', 
                 color='steelblue', alpha=0.7)
        plt.title(f'Distribución: {var}', fontsize=12, weight='bold')
    
    plt.suptitle(titulo, fontsize=16, weight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    
def resumen_categoricas(df, variables, y='Y', columnas=2):
    """
    Representa la tasa de Y=1 para cada variable categórica.
    """
    n = len(variables)
    filas = (n + columnas - 1) // columnas
    plt.figure(figsize=(8 * columnas, 4 * filas))

    for i, var in enumerate(variables, 1):
        plt.subplot(filas, columnas, i)
        tasa = df.groupby(var)[y].mean()
        tasa.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Tasa de {y}=1 según {var}', fontsize=13, weight='bold')
        plt.ylabel(f'Proporción de {y}=1')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def analisis_categoricas(df_cat):
    """
    Representa en una sola figura el número de dummies y el 
    porcentaje de observaciones en categorías minoritarias.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: número de dummies
    sns.barplot(
        data=df_cat.sort_values('n_dummies', ascending=False), x='variable',
        y='n_dummies', palette='Set2', ax=ax1)
    ax1.set_title('Número de dummies por variable', fontsize=14, weight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Número de dummies')
    ax1.tick_params(axis='x', rotation=45)

    # Gráfico 2: porcentaje de observaciones en categorías raras
    df_plot_raras = df_cat.melt(
        id_vars='variable', value_vars=['pct_obs_cat_<1%', 'pct_obs_cat_<5%'],
        var_name='tipo', value_name='porcentaje')

    sns.barplot(data=df_plot_raras, x='variable', y='porcentaje', hue='tipo',
                palette='Set2', ax=ax2)
    ax2.set_title('Observaciones en categorías minoritarias', 
                  fontsize=14, weight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('Porcentaje de observaciones')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
    
def boxplot_numericas(df, variables, y='Y', columnas=3):
    """
    Representa boxplots de variables numéricas respecto a la variable objetivo.
    """
    n = len(variables)
    filas = (n + columnas - 1) // columnas
    plt.figure(figsize=(5 * columnas, 4 * filas))

    for i, var in enumerate(variables, 1):
        plt.subplot(filas, columnas, i)
        sns.boxplot(data=df, x=y, y=var, palette='Set2')
        plt.title(var, fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.show()


def histogramas_densidad_y(df, variables, y='Y', columnas=3, bins=25):
    """
    Representa histogramas por clase de Y con curvas de densidad.
    """
    n = len(variables)
    filas = (n + columnas - 1) // columnas

    plt.figure(figsize=(5 * columnas, 4 * filas))
    for i, var in enumerate(variables, 1):
        plt.subplot(filas, columnas, i)

        sns.histplot(data=df, x=var, hue=y, kde=True, bins=bins,
                     stat='count', common_norm=False,
                     alpha=0.35, edgecolor='white')
        plt.title(var, fontsize=13, weight='bold')
        plt.xlabel(var)
        plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()
    

# =============================================================================
# SCRIPT 2
# =============================================================================
def heatmap_ranking_vars(df_vars, top_n=12):
    """
    Representa:
    1. heatmap de puntuaciones
    2. ranking por métodos que seleccionan la variable
    """
    df_plot = df_vars.copy()
    
    for col in ['imp_rf', 'coef_l1', 'score_f']:
        maximo = df_plot[col].max()
        if maximo != 0:
            df_plot[col] = df_plot[col]/maximo
            
    df_plot = df_plot.sort_values(
        ['n_metodos', 'imp_rf', 'coef_l1', 'score_f'],
        ascending=[False, False, False, False]).head(top_n).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), width_ratios=(1.5, 1))
    hm_puntos = df_plot.set_index('var')[['imp_rf', 'coef_l1', 'score_f']]
    hm_puntos.columns = ['RF', 'L1', 'SKB']
    
    sns.heatmap(hm_puntos, annot=True, fmt='.2f', cmap='YlOrRd', cbar=False, 
                ax=axes[0])
    
    axes[0].set_title('Puntuaciones', fontsize=14, weight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    # Ranking
    df_rank = df_plot.sort_values(['n_metodos', 'imp_rf', 'coef_l1'],
                                  ascending=[False, False, False]).copy()
    y_pos = range(len(df_rank))
    
    axes[1].hlines(y=y_pos, xmin=0, xmax=df_rank['n_metodos'], linewidth=2,
                   color='orange')
    axes[1].plot(df_rank['n_metodos'], y_pos, 'o', color='darkorange')
    
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(df_rank['var'])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 3.5)
    axes[1].set_xticks([0, 1, 2, 3])
    
    axes[1].set_title('Variables ordenadas por número de métodos', fontsize=14,
                      weight='bold')
    axes[1].set_xlabel('Número de métodos')
    axes[1].set_ylabel('')
    
    plt.title('Clasificación de las variables para el modelo SVM', fontsize=16,
              weight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
    
    
def tabla_grid(df_grid, cols_param):
    """
    Construye una tabla resumen ordenada a partir de cv_results_.
    """
    if 'class_weight' in df_grid.columns:
        df_grid['class_weight'] = df_grid['class_weight'].fillna('None').astype(str)
    
    auc_cv = df_grid['mean_test_roc_auc']
    recall_cv = df_grid['mean_test_recall']
    accuracy_cv = df_grid['mean_test_accuracy']
    balance_auc_recall = 2*(auc_cv*recall_cv)/(auc_cv+recall_cv)
    equilibrio = 3/((1/auc_cv) + (1/recall_cv) + (1/accuracy_cv))
    tabla = pd.DataFrame({
        **{col: df_grid[f'param_{col}'] for col in cols_param},
        'auc_cv': auc_cv,
        'accuracy_cv': accuracy_cv ,
        'recall_cv': recall_cv,
        'balance_a_r': balance_auc_recall,
        'equilibrio_scores': equilibrio,
        })
    
    tabla = tabla.sort_values(['auc_cv', 'recall_cv', 'balance_a_r',
                               'accuracy_cv'], ascending=False).reset_index(
                                   drop=True).copy()
    
    return tabla


def grafico_parametros(df, parametros, metricas, nombre):
    """
    A partir del DF del grid crea un gráfico por cada parámetro, 
    mostrando la evolución de los scores.
    """
    paramet = [p for p in parametros if p in df.columns and df[p].nunique()>1
               and p!= 'class_weight']
    cw =  'class_weight' in df.columns
    n_params = len(paramet)
    n_met = len(metricas)
    sns.set_style("whitegrid")
    
    n_cols = min(3, n_params)
    n_filas = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_filas, n_cols, figsize=(18, 5*n_filas),
                             sharey=True, squeeze=False)
    axes = axes.flatten()
    
    for i, col in enumerate(paramet):
        ax = axes[i]
        id_vars = [col]
        if cw: id_vars.append('class_weight')
        
        df_plot = df.melt(id_vars=id_vars, value_vars=metricas, 
                          var_name='Métrica', value_name='Score')
        df_plot['Métrica'] = df_plot['Métrica'].str.replace('_cv','').str.upper()
        
        if df[col].dtype == object:
            df_plot['x'] = df_plot[col].astype(str)
            if cw:
                df_plot['x'] += '\n' + df_plot['class_weight'].astype(str)
            
            sns.barplot(data=df_plot, x='x', y='Score', hue='Métrica', ax=ax, 
                        palette='Set2', alpha=0.8, errorbar='sd', capsize=0.8,
                        err_kws={'alpha': 0.1}, legend=False)
            ax.set_xlabel('')
        else:
            estilo = 'class_weight' if cw else None
            sns.lineplot(data=df_plot, x=col, y='Score', hue='Métrica', ax=ax,
                         palette='Set2', linewidth=2.5, style=estilo,
                         errorbar='sd', err_style='band', err_kws={'alpha': 0.1},
                         legend=True)
                        
            if col.upper() == 'C': ax.set_xscale('log')
            if col.upper() == 'FINAL_ESTIMATOR__C': ax.set_xscale('log')
            ax.set_xlabel(col, fontsize=12)
        
        ax.set_title(col.upper(), fontsize=14, weight='bold')
        ax.set_ylabel('Score' if i % n_cols == 0 else '', fontsize=12)
        ax.set_ylim(df[metricas].values.min()*0.85, 1)
        ax.grid(True, which='both', ls='-', alpha=0.2, color='gray')
    
    for ax in axes[n_params:]:
        ax.axis('off')
    
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if len(h) > len(handles):
            handles, labels = h, l
        if ax.get_legend(): ax.get_legend().remove()
    
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.05, 0.98),
               ncol=n_met+ (4 if cw else 0), frameon=False)
    fig.suptitle(f'{nombre}: evolución de métricas por parámetro', fontsize=16,
                 fontweight='bold', ha='left', x=0.05, y=1.01)
    
    fig.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.tight_layout()
    plt.show()


def heatmap_params(df_grid, nombre, fila, columna, metricas, params):
    """
    Representa un heatmap de un modelo para más de dos hiperparámetros.
    """
    params_n = {k: v for k, v in params.items() if len(v) > 1}
    params_base = {k: v for k, v in params.items() if len(v) == 1}
    referencias = params_base.items()
        
    nombres_col = list(params_n.keys())
    valores_col = list(params_n.values())
    combinaciones = list(itertools.product(*valores_col)) if params_n else [()]
    
    n_met = len(metricas)
    n_cols = len(combinaciones)
    datos_fila = df_grid[fila].nunique()
    datos_col = df_grid[columna].nunique()
    
    alto = (datos_fila*n_met) + 2.5
    ancho = (datos_col*n_cols) + 3
    texto = datos_fila < 15 and datos_col < 15
    
    fig, axes = plt.subplots(
        nrows=n_met, ncols=n_cols+1, figsize=(ancho, alto), squeeze=False, 
        gridspec_kw={'width_ratios': [1]*n_cols+[0.04]})
    
    for f, met in enumerate(metricas):
        df_rango = df_grid.copy()
        for k, v in referencias: df_rango = df_rango[df_rango[k] == v[0]]
        vmin, vmax = df_rango[met].min(), df_rango[met].max()
        
        for c, comb in enumerate(combinaciones):
            ax = axes[f,c]
            cbx = axes[f, -1] if c == n_cols-1 else None
            df_plot = df_grid.copy()
            for k, v in referencias: df_plot = df_plot[df_plot[k] == v[0]]
            for i, n in enumerate(nombres_col):
                df_plot = df_plot[df_plot[n] == comb[i]]
                
            datos = df_plot.pivot(index=fila, columns=columna, values=met)
            sns.heatmap(
                datos, annot=texto, cmap='YlOrRd', fmt='.3f', vmin=vmin, 
                vmax=vmax, ax=ax, annot_kws={'size': 8}, cbar=(c==n_cols-1), 
                cbar_ax=cbx)
            ax.set_aspect('equal', adjustable='box')
            ax.set(xlabel='', ylabel='') 
            ax.tick_params(labelbottom=True, labelleft=True)
            
            if f == 0:
                ax.set_title(
                    '\n'.join([f'{k}:{v}' for k, v in zip(nombres_col, comb)]),
                    fontsize=10, weight='bold')
            if c == 0:
                ax.set_ylabel(f"{met.upper().replace('_CV','')}\n{fila}",
                              fontsize=10, weight='bold')
            ax.set_xlabel(f'{columna}' if f == n_met-1 else '', 
                          fontsize=10, weight='bold')
            
    subtitulo = ' | '.join([f'{k}: {v[0]}' for k, v in referencias])            
    plt.suptitle(f'Heatmap Parámetros {nombre}\n{subtitulo}',
                 fontsize=16, fontweight='bold', y=0.96, va='bottom')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.15, 
                        wspace=0.2, hspace=0.2)    
    plt.show()
    
    
def boxplot_candidatos(resultados, metricas, nombre):
    """
    Representa boxplots de una métrica de validación cruzada para varios 
    candidatos.
    """
    n_met = len(metricas)
    n_box = len(resultados) # Número de modelos a comparar

    fig, axes = plt.subplots(
        nrows=n_met, ncols=1, figsize=(12 + 0.5*n_box, 8*n_met), sharex=True,
        squeeze=False)
    sns.set_style('whitegrid')
    
    axes = axes.flatten()

    mapping = {
        'candidato': '', 'param_C': 'C', 'param_gamma': 'gamma', 
        'param_degree': 'grado','param_class_weight': 'cw', 
        'param_penalty': 'penalty', 'param_criterion': 'criterio',
        'param_max_depth': 'max_depth', 'param_min_samples_split': 'split', 
        'param_min_samples_leaf': 'min_leaf',         
        'param_n_estimators': 'n_estimators',
        'param_max_samples': 'max_samples', 'param_max_features': 'max_features',
        'param_bootstrap': 'bootstrap',
        'param_bootstrap_features': 'boot_features'
        }
    cols_x = [c for c in mapping if c in resultados.columns]
    
    etiquetas = [
        '\n'.join(
            f'{mapping[c]}={fila[c]}' if mapping[c] != '' else f'{fila[c]}' 
            for c in cols_x)
        for _, fila in resultados[cols_x].iterrows()]

    for i, met in enumerate(metricas):
        ax=axes[i]
        cols_test = resultados.filter(regex=f'^split.*_test_{met}').columns
        
        datos = resultados[cols_test].assign(etiqueta=etiquetas).melt(
            id_vars=['etiqueta'], var_name='split', value_name='score')
        
        sns.boxplot(data=datos, x='etiqueta', y='score', ax=ax, 
                    palette='Set3')
        ax.set_title(f'{met.upper()}', fontsize=14, pad=7)
        ax.set_ylabel('Score', fontsize=12)
        
        if i == n_met - 1:
            ax.set_xlabel('Parámetros', fontsize=12)
            ax.tick_params(axis='x', rotation=30, labelbottom=True)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
    
    plt.suptitle(
        f'BOXPLOT comparación de candidatos {nombre}', fontsize=16, 
        weight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
 

def evaluar_modelo(modelo, nombre, X_train, X_test, y_train, y_test,
                   **parametros):
    """
    Ajusta, predice y resume un modelo de clasificación.
    """
    # Modelado
    modelo.fit(X_train, y_train)
    # Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    # Score
    if hasattr(modelo, 'decision_function'):
        y_score_train = modelo.decision_function(X_train)
        y_score_test = modelo.decision_function(X_test)
    elif hasattr(modelo, 'predict_proba'):
        y_score_train = modelo.predict_proba(X_train)[:, 1]
        y_score_test = modelo.predict_proba(X_test)[:, 1]
    else:
        y_score_train = y_pred_train
        y_score_test = y_pred_test
    # Resumen del modelo    
    df_resumen = resumen_modelo(
        nombre, y_train, y_test, y_pred_train, y_pred_test, y_score_train, 
        y_score_test, **parametros)
    
    return {
        'modelo': modelo,
        'resumen': df_resumen,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'y_score_train': y_score_train,
        'y_score_test': y_score_test
        }


def resumen_modelo(nombre, y_train, y_test, y_pred_train, y_pred_test,
                   y_score_train, y_score_test, **parametros):
    """
    Crea un dataframe resumen con métricas de train y test,
    junto con la matriz de confusión y los parámetros del modelo.
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fila = {
        'modelo': nombre,
        'auc_train': roc_auc_score(y_train, y_score_train),
        'auc_test': roc_auc_score(y_test, y_score_test),
        'accuracy_train': accuracy_score(y_train, y_pred_train),
        'accuracy_test': accuracy_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'f1': f1_score(y_test, y_pred_test, zero_division=0),
        'f1_macro': f1_score(y_test, y_pred_test, average='macro',
                             zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
        'specificity': specificity,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    fila.update(parametros)
    df = pd.DataFrame([fila])
    
    return df


def curva_auc_roc(y_real, y_score, nombre, thr_est=0):
    """
    Representa la curva ROC del modelo.
    Muestra:
        El punto estándar
        El punto óptimo según el índice de Youden.
    """
    fpr, tpr, thresholds = roc_curve(y_real, y_score)
    auc_modelo = auc(fpr, tpr)
    # Punto Youden
    youden_vals = tpr - fpr
    idx_youden = np.argmax(tpr - fpr)
    pt_youden = {
        'threshold': thresholds[idx_youden],
        'fpr': fpr[idx_youden],
        'tpr': tpr[idx_youden],
        'youden': youden_vals[idx_youden],
        'idx': idx_youden
        }
    # Punto estandar
    idx_est = np.argmin(np.abs(thresholds - thr_est))
    pt_est = {
        'threshold': thresholds[idx_est],
        'fpr': fpr[idx_est],
        'tpr': tpr[idx_est]
        }
    
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc_modelo:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    
    plt.scatter(pt_youden['fpr'], pt_youden['tpr'], color='red', s=50,
                label=f"Youden máx (thr={pt_youden['threshold']:.4f})")
    plt.scatter(pt_est['fpr'], pt_est['tpr'], color='green', s=50,
                label='Threshold estándar (thr=0)')
    
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title(f'Curva ROC - {nombre}', fontsize=16, weight='bold', y=1.01)
    plt.legend(loc='lower right')    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'auc': auc_modelo,
        'youden': pt_youden,
        'estandar': pt_est
        }


def grafico_thresholds(y_real, y_score, nombre, thr_youden):
    """
    Representa la evolución de accuracy, recall, specificity y balanced
    accuracy según el valor del threshold.
    """  
    thresholds = np.linspace(y_score.min(), y_score.max(), 100)
    data = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
        accuracy = accuracy_score(y_real, y_pred)
        recall = recall_score(y_real, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        bal_acc = balanced_accuracy_score(y_real, y_pred)
        data.append([t, accuracy, recall, specificity, bal_acc])
    datos = pd.DataFrame(data, columns=['Threshold', 'Accuracy', 'Recall',
                 'Specificity', 'Balanced Accuracy'])
    datos = datos.melt('Threshold', var_name='Métrica', value_name='Score')

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=datos, x='Threshold', y='Score', hue='Métrica', 
                 palette = 'Set2')
    plt.axvline(0, color='gray', ls='--', label='thr = 0')
    plt.axvline(thr_youden,color='red',ls='--',label=f'Youden = {thr_youden:.4f}')
    plt.title(f'{nombre} - Comparación de thresholds', fontsize=16, 
              weight='bold', y=1.01)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    
def matrices_confusion(y_real, predicciones, nombres):
    """
    Representa la matriz de confusión de uno o varios modelos.
    """
    if not isinstance(predicciones, list):
        predicciones = [predicciones]
    if isinstance(nombres, str):
        nombres = [nombres]
        
    n_pred = len(predicciones)
    n_cols = (n_pred if n_pred <= 3 else int(np.ceil(np.sqrt(n_pred))))
    filas = int(np.ceil(n_pred/n_cols))
    
    fig, axes = plt.subplots(filas, n_cols, figsize=(6*n_cols, 5*filas), 
                             squeeze=False)
    
    axes = axes.flatten()

    for ax, y_pred, nombre in zip(axes, predicciones, nombres):
        mc = confusion_matrix(y_real, y_pred)
        sns.heatmap(
            mc, annot=True, fmt='d', cmap='Blues', xticklabels=['NO','SI'],
            yticklabels=['NO','SI'], cbar=False, ax=ax,
            annot_kws={'size': 11, 'weight': 'bold'})
        
        ax.set_title(nombre, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Real', fontsize=12)
    
    for ax in axes[n_pred:]:
        ax.axis('off')
        
    fig.suptitle('Matriz de Confusión', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def heatmap_classification_reports(y_real, predicciones, nombres):
    """
    Representa precision, recall y f1-score para NO y SI en varios modelos.
    """
    if not isinstance(predicciones, list):
        predicciones = [predicciones]
    if isinstance(nombres, str):
        nombres = [nombres]
        
    n_pred = len(predicciones)
    n_cols = n_pred if n_pred <= 3 else int(np.ceil(np.sqrt(n_pred)))
    filas = int(np.ceil(n_pred/n_cols))
    fig, axes = plt.subplots(filas, n_cols, figsize=(5*n_cols, 4*filas), 
                             squeeze=False)
    
    axes = axes.flatten()

    for ax, y_pred, nombre in zip(axes, predicciones, nombres):
        report = classification_report(
            y_real, y_pred, target_names=['NO','SI'], output_dict=True, 
            zero_division=0)
        tabla = pd.DataFrame(report).T.loc[
            ['NO','SI'], ['precision', 'recall', 'f1-score']]
        sns.heatmap(tabla,annot=True, fmt='.3f',cmap='YlOrRd',cbar=False,ax=ax)
        ax.set_title(nombre)
        
    for ax in axes[n_pred:]:
        ax.axis('off')
    plt.tight_layout()
    
    plt.show()


def grafico_comparacion_metricas(df_resumen, nombre):
    """
    Representa comparativamente las métricas principales de varios modelos.
    """
    metricas = ['auc_test', 'accuracy_test', 'recall', 'precision', 'f1', 
                'f1_macro', 'balanced_accuracy', 'specificity']
    datos = df_resumen[['modelo'] + metricas].melt(
        id_vars='modelo', var_name='Métrica', value_name='Valor')
    modelos = df_resumen['modelo'].unique().tolist()
    sns.set_style("whitegrid")
    
    paleta_color = {m: COLORES_MODELOS.get(m, sns.color_palette('Set3')[i % 12]) 
                    for i, m in enumerate(modelos)}

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=datos, x='Métrica', y='Valor', hue='modelo', marker='o', 
                 linewidth=2, palette=paleta_color)
    
    if isinstance(nombre, list):
        titulo = ' vs '.join(nombre) if len(nombre) > 1 else nombre[0]
    else:
        titulo = nombre
        
    plt.title(f'Comparación de métricas: {titulo}', fontsize=16, 
              fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.ylim(datos['Valor'].min()*0.9, 1)
    plt.legend(title='Modelos', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def curva_auc_roc_comparativa(y_real, scores, nombres):
    """
    Representa una comparación de curvas ROC para varios modelos.
    """
    paleta_color = {
        m: COLORES_MODELOS.get(m, sns.color_palette('Set3')[i % 12])
        for i, m in enumerate(nombres)
        }

    plt.figure(figsize=(8, 6))
    
    for n, y_score in zip(nombres, scores):
        fpr, tpr, _ = roc_curve(y_real, y_score)
        auc_modelo = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2,
                 label=f'{n} (AUC={auc_modelo:.4f})',
                 color=paleta_color[n])
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Comparación de curvas ROC', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()


def sobreajuste(df_resumen):
    """
    Calcula el gap train-test para AUC y accuracy y lo representa.
    """
    df_sobreajuste = df_resumen[[
        'modelo', 'auc_train', 'auc_test', 'accuracy_train',
        'accuracy_test']].copy().reset_index(drop=True)
    df_sobreajuste['gap_auc'] = (
        df_sobreajuste['auc_train'] - df_sobreajuste['auc_test'])
    df_sobreajuste['gap_accuracy'] = (
        df_sobreajuste['accuracy_train'] - df_sobreajuste['accuracy_test'])
    df_sobreajuste = df_sobreajuste.sort_values(
        ['auc_test','gap_auc'], ascending=[False,True]).copy()

    n_modelos = len(df_sobreajuste)
    fig, axes = plt.subplots(n_modelos, 1, figsize=(12, 3*n_modelos), 
                             squeeze=False)
    axes = axes.flatten()
    
    color_test = 'deepskyblue'
    color_train = 'darkorange'

    for i, fila in df_sobreajuste.iterrows():
        metricas = ['AUC ROC', 'Accuracy']
        y_pos = np.array([1, 0], dtype=float)
        alto = 0.36
        train_valor = [fila['auc_train'], fila['accuracy_train']]
        test_valor = [fila['auc_test'], fila['accuracy_test']]
        bar_train = axes[i].barh(y_pos+alto/2, train_valor, height=alto,
                                 color=color_train, label='Train')
        bar_test = axes[i].barh(y_pos-alto/2, test_valor, height=alto,
                                color=color_test, label='Test')
        for bar in bar_train:
            valor = bar.get_width()
            axes[i].text(valor-0.07, bar.get_y() + bar.get_height()/2,
                         f'{valor:.4f}', va='center', ha='right',
                         color='white', weight='bold')
        for bar in bar_test:
            valor = bar.get_width()
            axes[i].text(valor-0.07, bar.get_y() + bar.get_height()/2,
                         f'{valor:.4f}', va='center', ha='right',
                         color='white', weight='bold')
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(metricas)
        axes[i].set_xlim(0, 1.01)
        
        if i == n_modelos - 1:
            axes[i].set_xlabel('Score', fontsize=12)
            axes[i].tick_params(axis='x', labelbottom=True)
        else:
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', labelbottom=False)
            
        axes[i].set_title(f"Modelo: {fila['modelo']}", fontsize=14, pad=7)
        axes[i].grid(True, axis='x', linestyle='--', alpha=0.6)
        axes[i].grid(False, axis='y')
        axes[i].legend(loc='lower right')
        
    fig.suptitle('Análisis de Sobreajuste (Train vs Test Gap)', fontsize=16, 
                 weight='bold', y=1)
    plt.tight_layout()
    plt.show()
    
    return df_sobreajuste


# =============================================================================
# SCRIPT 3
# =============================================================================
def visualiza_arbol(modelo, columnas, nombre):
    """
    Dibuja la estructura de un árbol de decisión entrenado.
    """
    fig, axes = plt.subplots(1, 2, figsize=(30, 12),
                             gridspec_kw={'width_ratios': [3, 1]})
    plot_tree(modelo, feature_names=columnas, class_names=['No', 'Sí'], 
              filled=True, rounded=True, fontsize=10, precision=2, ax=axes[0])
    axes[0].set_title(f'Estructura de {nombre}', fontsize=16, weight='bold')
    
    imp_vars = pd.Series(modelo.feature_importances_,
                             index=columnas).sort_values(ascending=True)
    imp_vars.plot(kind='barh', color='teal', ax=axes[1])
    axes[1].set_title("Importancia de las variables", fontsize=14, weight='bold')
    axes[1].set_xlabel('')
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    
def curvas_aprendizaje(modelos, X, y, cv, scoring='roc_auc'):
    """
    Calcula y representa la curva de aprendizaje de los modelos en TRAIN.
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    resultados_curva = []
    train_sizes = np.linspace(0.1, 1.0, 5)

    colores_inicio = sns.color_palette(palette='Set3', n_colors=len(modelos))
    paleta_color = {}
    
    for i, m in enumerate(modelos):
        if m in COLORES_MODELOS:
            paleta_color[m] = COLORES_MODELOS[m]
        else:
            paleta_color[m] = colores_inicio[i]

    for nombre, modelo in modelos.items():
        size, train_scores, test_scores = learning_curve(
            modelo, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring)
        df_modelo = pd.DataFrame({
            'Modelo': nombre,
            'Tamaño': np.r_[size, size],
            'Score medio': np.r_[
                train_scores.mean(axis=1), test_scores.mean(axis=1)],
            'Score std': np.r_[train_scores.std(axis=1), test_scores.std(axis=1)],
            'Tipo': ['Train']*len(size) + ['Test']*len(size)
            })
        m_train = train_scores.mean(axis=1)
        m_test = test_scores.mean(axis=1)
        df_modelo['Gap'] = np.r_[m_train - m_test, m_train - m_test]
        resultados_curva.append(df_modelo)
    
    df_curva = pd.concat(resultados_curva, ignore_index=True)
    sns.lineplot(
        data=df_curva, x='Tamaño', y='Score medio', hue='Modelo', style='Tipo',
        palette=paleta_color, markers=True, linewidth=2)

    for nombre in modelos.keys():
        for tipo in ['Train', 'Test']:
            sub = df_curva[
                (df_curva['Modelo'] == nombre) & (df_curva['Tipo'] == tipo)]
            plt.fill_between(
                sub['Tamaño'], sub['Score medio'] - sub['Score std'],
                sub['Score medio'] + sub['Score std'], alpha=0.1,
                color=paleta_color[nombre])

    plt.title(f'Comparación Curvas de Aprendizaje ({scoring.upper()})',
              fontsize=16, weight='bold')
    plt.xlabel('Tamaño de muestra')
    plt.ylabel(scoring.upper())
    plt.ylim(df_curva['Score medio'].min()-0.05, 1.01)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    return df_curva


def tabla_oob_bagging(modelos_bag):
    """
    Construye una tabla resumen con el OOB score de los modelos bagging.
    """
    filas = []
    for nombre, modelo in modelos_bag.items():
        fila = {'modelo': nombre}
        if hasattr(modelo, 'oob_score_'):
            fila['oob_score'] = modelo.oob_score_
        else:
            fila['oob_score'] = np.nan
        filas.append(fila)
    df_oob = pd.DataFrame(
        filas).sort_values('oob_score', ascending=False).reset_index(drop=True)

    return df_oob


def importancia_variables(modelos, X, y, scoring='roc_auc', n_repeats=10,
                               top_n=10):
    """
    Calcula y representa la importancia por permutación de cada variable
    según la métrica de evaluación indicada.
    """
    resultados = []

    for nombre, modelo in modelos.items():        
        perm = permutation_importance(
            modelo, X, y, scoring=scoring, n_repeats=n_repeats,
            random_state=RANDOM_STATE
            )
        
        df_perm = pd.DataFrame({
            'modelo': nombre,
            'variable': X.columns,
            'importancia_media': perm.importances_mean,
            'importancia_std': perm.importances_std
            }).sort_values('importancia_media', ascending=False)
        
        resultados.append(df_perm)

    df_importancia = pd.concat(resultados, ignore_index=True)

    modelos_lista = list(modelos.keys())
    n_modelos = len(modelos_lista)
    n_cols = 2 if n_modelos > 1 else 1
    n_filas = int(np.ceil(n_modelos / n_cols))
    fig, axes = plt.subplots(n_filas, n_cols, figsize=(6*n_cols, 5*n_filas), 
                             squeeze=False)
    axes = axes.flatten()

    for ax, nombre in zip(axes, modelos_lista):
        sub = df_importancia[
            df_importancia['modelo'] == nombre].head(top_n).sort_values(
                'importancia_media', ascending=True)
        
        color_plot = COLORES_MODELOS.get(nombre, 'steelblue')
        
        ax.barh(sub['variable'], sub['importancia_media'], color=color_plot)
        ax.set_title(f'Variables clave en: {nombre}', fontsize=12, weight='bold')
        ax.set_xlabel(f'Pérdida de {scoring.upper()} al eliminar la variable')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.4)
    
    for ax in axes[n_modelos:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return df_importancia


def caracteristicas_stacking(modelo_stack, X, y, nombre):
    """
    Representa las características metaaprendidas generadas por los modelos
    base del stacking.
    """
    X_meta = modelo_stack.transform(X)
    columnas = [nombre_modelo for nombre_modelo, _ in modelo_stack.estimators]
    
    df_meta = pd.DataFrame(X_meta, columns=columnas)
    df_meta['target'] = y.values if hasattr(y, 'values') else y
    
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.color': '.93'})

    # Representación de gráficos
    g = sns.pairplot(
        df_meta, hue='target', palette='Set1', diag_kind='kde', height=3.5,
        aspect=1.2, diag_kws={'fill': True, 'alpha': 0.5}, corner=False,
        plot_kws={'s': 20, 'alpha': 0.8, 'linewidth': 0.5, 'edgecolor': 'w'})
    g._legend.remove()
    
    plt.suptitle(f'Características Meta-aprendidas - {nombre}', y=1.02, 
                 fontsize=16, weight='bold', ha='center')
    colors = sns.color_palette('Set1', n_colors=2)
    handle_no = Line2D(
        [0], [0], marker='o', color='w', markerfacecolor=colors[0], 
        markersize=8, label='NO Abandona')
    handle_si = Line2D(
        [0], [0], marker='o', color='w', markerfacecolor=colors[1], 
        markersize=8, label='SÍ Abandona')
    g.fig.legend(
        handles=[handle_no, handle_si], title='Estado', loc='lower center',
        bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True, fontsize=11,
        title_fontsize=12, shadow=False)
   
    plt.margins(x=0.02, y=0.02)
    
    plt.tight_layout()
    plt.show()
    
    # Correlación entre las salidas
    corr_meta = df_meta[columnas].corr()
    # Tabla de información
    
    # Resumen descriptivo
    filas_info = []
    y_real = df_meta['target'].values
    
    for col in columnas:
        auc_modelo = roc_auc_score(y_real, df_meta[col])
        media_no = df_meta.loc[df_meta['target'] == 0, col].mean()
        media_si = df_meta.loc[df_meta['target'] == 1, col].mean()
        mediana_no= df_meta.loc[df_meta['target'] == 0, col].median()
        mediana_si = df_meta.loc[df_meta['target'] == 1, col].median()
        std_no = df_meta.loc[df_meta['target'] == 0, col].std()
        std_si = df_meta.loc[df_meta['target'] == 1, col].std()
        
        corr_rest = corr_meta.loc[col, corr_meta.columns != col].abs()
        
        filas_info.append({
            'modelo_base': col,
            'auc_individual': auc_modelo,
            'media_clase_0 (NO)': media_no,
            'media_clase_1 (SI)': media_si,
            'diferencia_medias': media_si - media_no,
            'mediana_clase_0 (NO)': mediana_no,
            'mediana_clase_1 (SI)': mediana_si,
            'diferencia_medianas': mediana_si - mediana_no,
            'std_clase_0 (NO)': std_no,
            'std_clase_1 (SI)': std_si,
            'corr_media_abs': corr_rest.mean(),
            'corr_max_abs': corr_rest.max()
            })
    info_meta = pd.DataFrame(filas_info).sort_values(
        ['auc_individual', 'diferencia_medias'],
        ascending=False).reset_index(drop=True)

    return {
        'df_meta': df_meta,
        'info_meta': info_meta,
        'corr_meta': corr_meta
        }

# =============================================================================
# SCRIPT 4
# =============================================================================
def resumen_global(resultados):
    """
    Transforma la lista de diccionarios de modelos en un DataFrame consolidado
    con métricas de rendimiento y gaps de sobreajuste.
    """
    resumen_global = []
    
    columnas_finales = ['modelo', 'tipo_modelo', 'auc_train', 'auc_test',
                        'accuracy_train', 'accuracy_test', 'gap_auc', 
                        'gap_accuracy', 'recall', 'precision', 'f1', 
                        'f1_macro', 'balanced_accuracy', 'specificity',
                        'tn', 'fp', 'fn', 'tp']
    
    for df in resultados:
        df_res = df['resumen'].copy()
        df_res['tipo_modelo'] = type(df['modelo']).__name__
        df_res['gap_auc'] = df_res['auc_train']-df_res['auc_test']
        df_res['gap_accuracy'] = df_res['accuracy_train']-df_res['accuracy_test']
        
        resumen_global.append(df_res)
    
    df_resumen = pd.concat(resumen_global, ignore_index=True)
    
    df_final = df_resumen[columnas_finales]
    
    df_final = df_final.sort_values('auc_test', 
                                    ascending=False).reset_index(drop=True)
    
    return df_final


def curvas_validacion(modelo, rango, X, y, cv, metrica, n_cols=3):
    """
    Representa varias curvas de validación en una única figura con subplots.  
    """
    resultados = {}
    
    for nombre, mod in modelo.items():
        
        rang = rango[nombre]
        n_params = len(rang)
        
        col = min(n_params, n_cols)
        n_filas = int(np.ceil(n_params / col))
        
        fig, axes = plt.subplots(n_filas, col, figsize=(6*col, 5*n_filas),
                                 squeeze=False)
        axes = axes.flatten()
        
        for i, (p_n, p_r) in enumerate(rang.items()):
            ax = axes[i]
            
            train_scores, valid_scores = validation_curve(
                estimator=mod, X=X, y=y, param_name=p_n, param_range=p_r, 
                cv=cv, scoring=metrica, n_jobs=-1)
            
            media_train = train_scores.mean(axis=1)
            std_train = train_scores.std(axis=1)
            media_valid = valid_scores.mean(axis=1)
            std_valid = valid_scores.std(axis=1)
            
            resultados[f'{nombre}_{p_n}'] = pd.DataFrame({
                'parametro': p_r,
                'train_mean': media_train,
                'valid_mean': media_valid
                })
            
            ax.plot(p_r, media_train, marker='o', label='Train', 
                    color='darkorange')
            ax.plot(p_r, media_valid, marker='o', label='CV', 
                    color='steelblue')
            ax.fill_between(
                p_r, media_train - std_train, media_train + std_train,
                alpha=0.1, color='darkorange')
            ax.fill_between(
                p_r, media_valid - std_valid, media_valid + std_valid,
                alpha=0.1, color='steelblue')
            
            ax.set_title(f'Parámetro: {p_n}', fontsize=12)
            ax.set_xlabel('Valor del parámetro')
            ax.set_ylabel(metrica.upper())
            ax.legend()
            ax.grid(alpha=0.3)

        for ax in axes[n_params:]:
            ax.axis('off')
            
        fig.suptitle(f'Curvas de Validación: {nombre}', fontsize=16,
                     weight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    return resultados


def grafico_ranking(df_resumen, metricas, n_cols=3):
    """
    Genera un panel de rankings por métricas y errores usando formato
    lollipop horizontal.
    Devuelve un DataFrame con el ranking de cada métrica.
    """
    df_rankings = df_resumen[['modelo']].copy()
    for met in metricas:
        menor = met.lower() in ['fp', 'fn', 'gap_auc', 'gap_accuracy']
        df_rankings[met] = df_resumen[met].rank(ascending=menor).astype(int)
        
    n_met = len(metricas)
    n_filas = int(np.ceil(n_met / n_cols))

    fig, axes = plt.subplots(n_filas, n_cols, figsize=(6*n_cols, 3*n_filas), 
                             squeeze=False)
    axes = axes.flatten()

    for i, met in enumerate(metricas):
        ax = axes[i]
        
        menor = met.lower() in ['fp', 'fn', 'gap_auc', 'gap_accuracy']
    
        ranking = df_resumen[['modelo', met]].sort_values(
            met, ascending=not menor).reset_index(drop=True)
                
        y_pos = np.arange(len(ranking))
        colores = [COLORES_MODELOS.get(m, 'gray') for m in ranking['modelo']]
        x_min = ranking[met].min()
        x_max = ranking[met].max()
        rango = x_max - x_min if x_max != x_min else 1
        
        ax.hlines(
            y=y_pos, xmin=x_min-(rango*0.02), xmax=ranking[met]-(rango*0.025),
            color=colores, alpha=0.35, linewidth=10)
        
        ax.scatter(ranking[met], y_pos, s=250, c=colores, alpha=0.6, 
                   ec=colores, linewidth=1.5)
                
        for y, valor in zip(y_pos, ranking[met]):
            fmt = f'{valor:.4f}' if valor < 2 else f'{valor:.0f}'
            ax.text(valor + (rango*0.06), y, fmt, va='center', ha='left',
                    fontsize=8, fontweight='bold')
            
        ax.set_ylim(-0.3, len(ranking)-0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranking['modelo'], fontsize=8, rotation=35)
        ax.get_yticklabels()[-1].set_weight('bold')
        
        ax.set_xlim(x_min-(rango*0.05), x_max+(rango*0.25))
        ax.set_xticklabels([])

        ax.set_title(met.upper(), weight='bold', fontsize=12, pad=10)
        ax.grid(axis='both', linestyle='-', alpha=0.3)
        
    for ax in axes[n_met:]:
        ax.axis('off')
        
    fig.suptitle('Ranking Comparativo de Modelos', fontsize=16, weight='bold',
                 y=0.96, va='center')
        
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, wspace=0.25, 
                        hspace=0.3) 
    plt.show()
    
    return df_rankings


def heatmap_metricas(df_resumen):
    """
    Heatmap que ordena los modelos por su desempeño medio. Identifica al 
    ganador global visualmente.
    """
    metricas = ['auc_test', 'accuracy_test', 'recall', 'precision', 'f1',
                'f1_macro', 'balanced_accuracy', 'specificity', 'fp', 'fn', 
                'gap_auc', 'gap_accuracy']
    
    # 1. Normalización
    df_plot = df_resumen.set_index('modelo')[metricas].copy()
    
    rangos = df_plot.max() - df_plot.min()
    df_norm = (df_plot - df_plot.min()).div(rangos.replace(0, np.nan))
    df_norm.loc[:, rangos.eq(0)] = 0.5

    # 2. Métricas de orden inverso
    cols_inv = ['fp', 'fn', 'gap_auc', 'gap_accuracy']
    df_norm[cols_inv] = 1 - df_norm[cols_inv]
    
    # 3. Ordenar modelos
    df_norm['score_medio'] = df_norm.mean(axis=1)
    df_norm = df_norm.sort_values('score_medio',
                                  ascending=False).drop(columns='score_medio')

    plt.figure(figsize=(13, 5))
    ax = sns.heatmap(df_norm, annot=False, fmt='.2f', cmap='YlGnBu', 
                    linewidths=0.5, linecolor='white', 
                    cbar_kws={'aspect':20})
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticks([])
    cbar.set_label('Score Relativo', size=8, weight='bold', labelpad=5)       
        
    plt.title('Heatmap comparativo global de métricas (Normalizado)',
              fontsize=16, weight='bold', pad=10)
    plt.xlabel('MÉTRICAS EVALUADAS', fontsize=10, weight='bold', labelpad=10)
    plt.ylabel('MODELOS', fontsize=10, weight='bold', labelpad=10)
    plt.xticks(fontsize=10, rotation=30, ha='right')
    plt.yticks(fontsize=10, ha='right')
    plt.tight_layout()
    plt.show()
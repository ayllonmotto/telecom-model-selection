# -*- coding: utf-8 -*-
# =============================================================================
# 03_ensemble_ml.py
# Proyecto: Machine Learning 2026
#
# Objetivo:
# - Cargar la base preparada en los scripts anteriores.
# - Comprobar que se mantiene la misma partición train/test.
# - Aplicar bagging con distintos estimadores base.
# - Comparar cada modelo base con su versión bagging.
# - Comparar los modelos bagging entre sí.
# - Estimadores base para stacking.
# - Implementar RF y XGBoost para contrastes.
# - Ajuste del meta-modelo.
# - Selección de candidatos y modelo final.
# - Guardar la información necesaria para el bloque de comparación.
#
# Nota:
# - Se reutiliza la partición del Script 02 para mantener comparabilidad.
# - Se prueban tres estimadores base:
#   1. SVM RBF: mejor modelo individual del Script 02.
#   2. Regresión logística: contraste con un modelo más simple y estable.
#   3. Árbol de decisión: modelo inestable donde bagging suele aportar valor.
# - Se desarrollan dos modelos para contraste en Stacking
#   1. Random Forest
#   2. XGBoost: contraste boosting al bagging
# =============================================================================
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (BaggingClassifier, StackingClassifier, 
                              RandomForestClassifier)

from xgboost import XGBClassifier

from funciones_ml import (
    RANDOM_STATE, tabla_grid, grafico_parametros, heatmap_params,
    boxplot_candidatos, evaluar_modelo, matrices_confusion,
    heatmap_classification_reports, grafico_comparacion_metricas,
    curva_auc_roc_comparativa, sobreajuste, curvas_aprendizaje,
    visualiza_arbol, tabla_oob_bagging, importancia_variables,
    caracteristicas_stacking
    )

# -----------------------------------------------------------------------------
# AVISOS DE COMPATIBILIDAD
# -----------------------------------------------------------------------------
# El proyecto fue desarrollado utilizando la parametrización clásica de
# LogisticRegression mediante el argumento `penalty`. Versiones recientes de
# scikit-learn muestran avisos de deprecación para esta sintaxis.
#
# Se silencian únicamente estos avisos para mantener la implementación original
# del proyecto sin ocultar otros warnings potencialmente relevantes.

warnings.filterwarnings(
    'ignore', message=r".*'penalty' was deprecated.*", category=FutureWarning
    )

warnings.filterwarnings(
    'ignore', message=r'.*Inconsistent values: penalty=.*', 
    category=UserWarning
    )

# -----------------------------------------------------------------------------
# RUTAS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RUTA_SVM = BASE_DIR / 'outputs' / 'BBDD_svm.pickle'
RUTA_SALIDA = BASE_DIR / 'outputs' / 'BBDD_ensemble.pickle'

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

if not RUTA_SVM.exists():
    raise FileNotFoundError(
        'No se encontró BBDD_svm.pickle. '
        'Ejecuta primero 02_svm_ml.py.'
    )



# =============================================================================
# 1. CARGA DE DATOS Y CONFIGURACIÓN GENERAL
# =============================================================================

with open(RUTA_SVM, 'rb') as archivo:
    bbdd_svm = pickle.load(archivo)
    
# Se reutiliza exactamente la partición train/test fijada en el Script 02
# para mantener la comparabilidad directa entre los modelos individuales,
# los ensembles y la comparación global final.
X_train = bbdd_svm['X_train_svm'].copy()
X_test = bbdd_svm['X_test_svm'].copy()
y_train = bbdd_svm['y_train'].copy()
y_test = bbdd_svm['y_test'].copy()

# Parámetros del SVM RBF final seleccionados en el Script 02.
# Este modelo se toma aquí como mejor clasificador individual de referencia
# para el bloque bagging y como uno de los posibles modelos base del stacking.
params_rbf = bbdd_svm['parametros_rbf']

# CONFIGURACIÓN
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# MÉTRICAS DE EVALUACIÓN
# Se mantienen AUC y accuracy como métricas principales del proyecto:
# - AUC mide la capacidad global de discriminación entre clases.
# - Accuracy resume el porcentaje total de clasificaciones correctas.
#
# Además, se incorpora recall como métrica estratégica complementaria,
# ya que interesa reducir falsos negativos y mejorar la detección de clientes
# con riesgo real de abandono.
score = ['roc_auc', 'accuracy', 'recall']
score_grid = ['auc_cv', 'accuracy_cv', 'recall_cv']

# =============================================================================
# 2. ESTIMADORES BASE - JUSTIFICACIÓN METODOLÓGICA DEL BLOQUE BAGGING
# =============================================================================
# En esta primera parte del script se fijan tres modelos base que servirán
# para analizar el efecto de bagging sobre clasificadores de naturaleza distinta.
#
# La idea no es aplicar bagging de forma automática sobre cualquier modelo,
# sino comprobar empíricamente en qué tipo de estimadores aporta una mejora
# real y en cuáles su utilidad es más limitada.
#
# Se consideran tres perfiles complementarios:
# 1. SVM RBF:
#    mejor modelo individual seleccionado en el Script 02 y referencia principal
#    del proyecto a nivel de clasificador individual.
#
# 2. Regresión logística:
#    modelo más simple, lineal y estable, incorporado como contraste para
#    comprobar si bagging aporta valor también en estimadores de menor varianza.
#
# 3. Árbol de decisión:
#    modelo más inestable y sensible a perturbaciones de la muestra, donde
#    bagging suele resultar especialmente útil al reducir varianza y ganar
#    robustez predictiva.
#
# Se sigue, por tanto, una secuencia metodológica en tres pasos:
# 1) fijar un modelo base representativo en cada familia;
# 2) construir su versión bagging;
# 3) comparar modelo individual frente a ensemble antes de pasar a la
#    comparación global entre baggings.

# -----------------------------------------------------------------------------
# 2.1. ESTIMADOR BASE 1: SVM RBF
# -----------------------------------------------------------------------------
print('\nESTIMADOR BASE 1: SVM RBF')
nombre_svm = 'SVM RBF'
modelo_svm = SVC(kernel='rbf', C=params_rbf['C'], gamma=params_rbf['gamma'],
                 class_weight=params_rbf['class_weight'], probability=True)
resultados_svm = evaluar_modelo(
    modelo_svm, nombre_svm, X_train, X_test, y_train, y_test, C=params_rbf['C'],
    gamma=params_rbf['gamma'], class_weight=params_rbf['class_weight'])
print(nombre_svm, params_rbf)

# -----------------------------------------------------------------------------
# 2.2. ESTIMADOR BASE 2: REGRESIÓN LOGÍSTICA
# -----------------------------------------------------------------------------
# La regresión logística se incorpora como modelo base de contraste dentro del
# bloque bagging. Frente a otros clasificadores más flexibles o más inestables,
# representa una alternativa lineal, simple y relativamente estable, útil para
# comprobar si la agregación bootstrap aporta una mejora real también en una
# familia de modelos con menor varianza.
#
# Además, su interpretación metodológica resulta clara: permite evaluar hasta
# qué punto un modelo clásico y parsimonioso sigue siendo competitivo frente a
# enfoques más complejos cuando el objetivo principal es predecir.

print('\nESTIMADOR BASE 2: REGRESIÓN LOGÍSTICA')
nombre_log = 'R.Logística'
parametros_log = ['penalty', 'C', 'class_weight']

param_grid_log = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 0.5, 1, 5, 10],
    'solver':['liblinear'],
    'class_weight': [None, 'balanced']
    }
rlog = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)

grid_log = GridSearchCV(
    rlog, param_grid_log, cv=cv, scoring=score, refit='roc_auc', verbose=3,
    n_jobs=-1)
grid_log.fit(X_train, y_train)
print('\nMejores parámetros:')
print(grid_log.best_params_)
print('Mejor AUC CV:', grid_log.best_score_)

df_cv_log = pd.DataFrame(grid_log.cv_results_)
if 'param_class_weight' in df_cv_log.columns:
    df_cv_log['param_class_weight'] = df_cv_log[
        'param_class_weight'].fillna('None').astype(str)
df_grid_log = tabla_grid(df_cv_log, parametros_log)

grafico_parametros(df_grid_log, parametros_log, score_grid, nombre_log)

# Aunque algunas configuraciones sin balanceo mantienen valores aceptables de
# AUC o accuracy, muestran un recall claramente insuficiente sobre la clase
# positiva. Por ello, la comparación final de candidatos se centra en la rama
# con class_weight='balanced', más coherente con el objetivo operativo del
# problema.

df_grid_log = df_grid_log[df_grid_log['class_weight']=='balanced']

# ---------------------------------------------
# CANDIDATOS REGRESIÓN LOGÍSTICA
# ---------------------------------------------
candidatos_log = pd.DataFrame({
    'candidato': ['rlog_1', 'rlog_2', 'rlog_3', 'rlog_4', 'rlog_5', 'rlog_6'],
    'penalty': ['l2', 'l2', 'l1', 'l1', 'l2', 'l1'],
    'C': [1, 5, 5, 1, 0.5, 0.5],
    'class_weight': 'balanced',
    'solver': 'liblinear'
    })
seleccion_log = candidatos_log.merge(
    df_cv_log, left_on= parametros_log,
    right_on=['param_penalty', 'param_C', 'param_class_weight'], how='left')

boxplot_candidatos(seleccion_log, score, nombre_log)

# ---------------------------------------------
# MODELO REGRESIÓN LOGÍSTICA
# ---------------------------------------------
# El mejor valor automático del grid en AUC CV corresponde a una configuración
# con C=5 y penalty='l2'. No obstante, la comparación manual de candidatos
# muestra que las diferencias entre las mejores alternativas son pequeñas y que
# varios modelos presentan boxplots muy parecidos en AUC, accuracy y recall.
#
# En este contexto, se selecciona rlog_4 (C=1, penalty='l1',
# class_weight='balanced') como modelo representativo del bloque. La decisión
# se apoya en tres ideas:
# 1) mantiene un comportamiento muy próximo al de los mejores candidatos del
#    grid en las métricas principales;
# 2) ofrece una solución más parsimoniosa, con mayor regularización efectiva;
# 3) resulta metodológicamente adecuada como modelo base sencillo y estable
#    para contrastar posteriormente el efecto de bagging.
#
# Por tanto, no se elige por maximizar de forma aislada una única métrica en
# validación cruzada, sino por ofrecer un perfil global competitivo y una
# complejidad algo más contenida dentro de un bloque donde las diferencias son
# muy reducidas.

param_rlog ={
    'C': 1,
    'class_weight': 'balanced',
    'penalty': 'l1',
    }
modelo_log = LogisticRegression(**param_rlog, solver='liblinear', max_iter=5000,
                                random_state=RANDOM_STATE)

resultados_log = evaluar_modelo(
    modelo_log, nombre_log, X_train, X_test, y_train, y_test, **param_rlog)
print('\nResumen Regresión logística:')
print(resultados_log['resumen'].round(4).to_string(index=False))

matrices_confusion(y_test, resultados_log['y_pred_test'], nombre_log)
heatmap_classification_reports(y_test, resultados_log['y_pred_test'], nombre_log)
sobreajuste(resultados_log['resumen'])

# En test, la regresión logística mantiene un comportamiento coherente con lo
# observado en validación cruzada: AUC=0.8481, accuracy=0.7684 y recall=0.7730.
# Además, el hecho de que auc_test sea ligeramente superior a auc_train no se
# interpreta como un problema, sino como una ausencia de sobreajuste relevante
# dentro de la variabilidad esperable de una partición concreta train/test.

# -----------------------------------------------------------------------------
# 2.3. ESTIMADOR BASE 3: ÁRBOL DE DECISIONES
# -----------------------------------------------------------------------------
# El árbol de decisión se incorpora como tercer estimador base porque aporta
# un perfil muy distinto al de SVM y regresión logística. Frente a ambos,
# permite capturar relaciones no lineales, interacciones y reglas de decisión
# de forma explícita, además de ofrecer una interpretabilidad estructural muy
# útil en esta fase del proyecto.
#
# Su inclusión también resulta especialmente adecuada dentro del bloque
# ensemble, ya que los árboles son modelos con mayor varianza y, por tanto,
# candidatos naturales para comprobar si técnicas como bagging pueden mejorar
# su estabilidad y capacidad de generalización.

print('\nESTIMADOR BASE 3: ÁRBOL DE DECISIONES')
nombre_arbol = 'Árbol de decisiones'
parametros_arbol = ['criterion', 'max_depth', 'min_samples_split', 
                    'min_samples_leaf', 'class_weight']

param_grid_arbol = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [10, 20, 30],
    'class_weight': [None, 'balanced']
    }
arbol_d = DecisionTreeClassifier(random_state=RANDOM_STATE)

grid_arbol = GridSearchCV(
    arbol_d, param_grid_arbol, cv=cv, scoring=score, refit='roc_auc', 
    return_train_score=True, verbose=3, n_jobs=-1)
grid_arbol.fit(X_train, y_train)
print('\nMejores parámetros:')
print(grid_arbol.best_params_)
print('Mejor AUC CV:', grid_arbol.best_score_)
df_cv_arbol = pd.DataFrame(grid_arbol.cv_results_)

if 'param_class_weight' in df_cv_arbol.columns:
    df_cv_arbol['param_class_weight'] = df_cv_arbol[
        'param_class_weight'].fillna('None').astype(str)
df_grid_arbol = tabla_grid(df_cv_arbol, parametros_arbol)

grafico_parametros(df_grid_arbol, parametros_arbol, score_grid, nombre_arbol)

# HEATMAPS
criterion = ['gini', 'entropy']
for c in criterion:
    param_arbol_heat = {
        'criterion': [c],
        'min_samples_leaf': [10, 20, 30],
        'class_weight': ['balanced']
        }
    heatmap_params(df_grid_arbol, nombre_arbol, 'max_depth', 'min_samples_split',
                   score_grid, param_arbol_heat)

# Las configuraciones sin ponderación se descartan de la comparación final
# porque, en general, muestran un recall inferior sobre la clase positiva,
# menos coherente con el objetivo operativo del problema.

df_grid_arbol = df_grid_arbol[df_grid_arbol['class_weight']=='balanced']

# ---------------------------------------------
# CANDIDATOS ARBOL DE DECISIÓN
# ---------------------------------------------
df_grid_arbol = df_grid_arbol[df_grid_arbol['class_weight']=='balanced']
candidatos_arbol = pd.DataFrame({
    'candidato': ['arb_1', 'arb_2', 'arb_3', 
                  'arb_4', 'arb_5', 'arb_6',
                  'arb_7', 'arb_8', 'arb_9'],
    'criterion': ['entropy', 'entropy', 'entropy', 
                  'entropy', 'entropy', 'entropy',
                  'gini', 'gini', 'gini'],
    'max_depth': [5, 5, 5, 
                  7, 7, 7,
                  7, 5, 5],
    'min_samples_split': [20]*9,
    'min_samples_leaf': [30, 20, 10, 
                         30, 20, 10,
                         20, 20, 30],
    'class_weight': ['balanced']*9
    })
seleccion_arbol = candidatos_arbol.merge(
    df_cv_arbol, left_on= parametros_arbol, right_on=[
        'param_criterion', 'param_max_depth', 'param_min_samples_split',
        'param_min_samples_leaf', 'param_class_weight'],how='left')

boxplot_candidatos(seleccion_arbol, score, nombre_arbol)

# ---------------------------------------------
# MODELO ÁRBOL DE DECISIÓN
# ---------------------------------------------
# El mejor valor automático del grid en AUC CV corresponde a una configuración
# con criterion='entropy', max_depth=5, min_samples_leaf=30 y
# min_samples_split=10. No obstante, la selección del árbol base no se hace
# únicamente por el primer puesto del ranking, sino por observación conjunta de
# la tabla resumen, los heatmaps, los boxplots y el comportamiento esperado del
# modelo como estimador individual previo al bagging.
#
# En este contexto, se selecciona como árbol base el candidato arb_5
# (criterion='entropy', max_depth=5, min_samples_split=20,
# min_samples_leaf=20, class_weight='balanced'), ya que ofrece el mejor
# equilibrio global entre AUC, accuracy y recall dentro de una configuración
# suficientemente regularizada.
#
# Frente a alternativas cercanas, mantiene una capacidad discriminante muy alta
# y una accuracy elevada, sin forzar en exceso la complejidad del árbol. Esta
# elección resulta especialmente adecuada porque el objetivo aquí no es agotar
# el rendimiento del árbol individual, sino disponer de un modelo base sólido,
# interpretable y razonablemente estable antes de aplicar bagging.
#
# Por tanto, no se elige por maximizar de forma aislada una sola métrica en
# validación cruzada, sino por representar una solución robusta y coherente con
# la lógica posterior del bloque ensemble.

param_arbol = {
    'criterion': 'entropy',
    'max_depth': 5,
    'min_samples_split': 20,
    'min_samples_leaf': 20,
    'class_weight': 'balanced'
    }
modelo_arbol = DecisionTreeClassifier(**param_arbol, random_state=RANDOM_STATE)

resultados_arbol = evaluar_modelo(
    modelo_arbol, nombre_arbol, X_train, X_test, y_train, y_test, **param_arbol)
print(resultados_arbol['resumen'].round(4).to_string(index=False))

matrices_confusion(y_test, resultados_arbol['y_pred_test'], nombre_arbol)
heatmap_classification_reports(y_test, resultados_arbol['y_pred_test'], 
                               nombre_arbol)
sobreajuste(resultados_arbol['resumen'])
visualiza_arbol(resultados_arbol['modelo'], X_train.columns, nombre_arbol)

# En test, el árbol seleccionado confirma un comportamiento muy competitivo:
# AUC=0.9187, accuracy=0.9068 y recall=0.8298, con una matriz de confusión que
# refleja un buen compromiso entre detección de la clase positiva y control de
# falsos positivos. Además, el gap train-test es reducido, lo que sugiere una
# generalización razonable para un árbol individual de esta complejidad.
#
# En conjunto, este resultado refuerza su papel como árbol base adecuado para
# el bloque bagging posterior.

# =============================================================================
# 3. BAGGING CON DISTINTOS ESTIMADORES BASE
# =============================================================================
# -----------------------------------------------------------------------------
# 3.1. PARÁMETROS DE BÚSQUEDA PARA BAGGING
# -----------------------------------------------------------------------------
# Se utiliza un grid moderado para mantener el coste computacional
param_grid_bag_svm = {
    'n_estimators': [75, 100],
    'max_samples': [0.8, 0.9],
    'max_features': [0.7, 0.85, 1.0],
    'bootstrap': [True],
    'bootstrap_features': [False]
    }

param_grid_bag = {
    'n_estimators': [50, 100, 150],
    'max_samples': [0.6, 0.7, 0.85, 1.0],
    'max_features': [0.6, 0.7, 0.85, 1.0],
    'bootstrap': [True],
    'bootstrap_features': [False]
    }

parametros_bag = ['n_estimators', 'max_samples', 'max_features', 'bootstrap', 
                  'bootstrap_features']

# -----------------------------------------------------------------------------
# 3.2. BAGGING SVM RBF
# -----------------------------------------------------------------------------
# Se aplica bagging sobre el SVM RBF porque el enunciado exige evaluar esta
# técnica sobre el mejor modelo individual del bloque SVM. El objetivo es
# comprobar empíricamente si la agregación de múltiples réplicas bootstrap
# permite mejorar la capacidad de generalización del modelo, aunque el SVM no
# sea, en principio, el candidato más natural para bagging frente a otros
# clasificadores de mayor varianza, como los árboles.

print('\nBAGGING SVM RBF (MODELO 1)')
nombre_b_svm = 'Bagging SVM RBF'
bag_svm_grid = BaggingClassifier(
    estimator=modelo_svm, oob_score=True, random_state=RANDOM_STATE, n_jobs=-1)

grid_bag_svm = GridSearchCV(
    bag_svm_grid, param_grid_bag_svm, cv=cv, scoring=score, refit='roc_auc',
    return_train_score=True, verbose=3, n_jobs=-1)
grid_bag_svm.fit(X_train, y_train)

df_cv_bag_svm = pd.DataFrame(grid_bag_svm.cv_results_)
df_grid_bag_svm = tabla_grid(df_cv_bag_svm, parametros_bag)

grafico_parametros(df_grid_bag_svm, parametros_bag, score_grid, nombre_b_svm)

# ---------------------------------------------
# CANDIDATOS BAGGING SVM RBF
# ---------------------------------------------
# Se seleccionan varias configuraciones competitivas para comparar perfiles
# distintos dentro del bloque: combinaciones más globales, otras algo más
# conservadoras y otras orientadas a comprobar si pequeños cambios en
# max_samples o max_features alteran de forma apreciable el equilibrio entre
# AUC, accuracy y recall.

candidatos_bag_svm = pd.DataFrame({
    'candidato': ['bag_svm_1', 'bag_svm_2', 'bag_svm_3', 
                  'bag_svm_4', 'bag_svm_5', 'bag_svm_6',
                  'bag_svm_7', 'bag_svm_8', 'bag_svm_9'],
    'n_estimators': [75, 100, 75,
                     75, 100, 75,
                     100, 100, 75],
    'max_samples': [0.9, 0.9, 0.9,
                    0.8, 0.8, 0.8,
                    0.8, 0.9, 0.9],
    'max_features': [1.0, 1.0, 0.85,
                     1.0, 1.0, 0.85,
                     0.85, 0.85, 0.70],
    'bootstrap': [True]*9,
    'bootstrap_features': [False]*9
    })
seleccion_bag_svm = candidatos_bag_svm.merge(
    df_cv_bag_svm, left_on=parametros_bag, right_on=[
        'param_n_estimators', 'param_max_samples', 'param_max_features', 
        'param_bootstrap', 'param_bootstrap_features'], how='left')

boxplot_candidatos(seleccion_bag_svm, score, nombre_b_svm)

# ---------------------------------------------
# SELECCIÓN BAGGING SVM RBF
# ---------------------------------------------
# La configuración finalmente seleccionada para representar este bloque es la
# correspondiente a n_estimators=75, max_samples=0.8, max_features=1.0,
# bootstrap=True y bootstrap_features=False.
#
# En la tabla de candidatos, esta combinación corresponde a bag_svm_4. La
# elección no se apoya solo en el mejor valor puntual de una métrica, sino en
# una lectura conjunta de la tabla resumen, los gráficos paramétricos y los
# boxplots, valorando qué configuración ofrece un compromiso razonable entre
# AUC, accuracy y recall dentro del espacio de búsqueda definido.
#
# Se trata, por tanto, de un candidato representativo del bloque Bagging SVM
# RBF, elegido para contrastar si la agregación mejora realmente al SVM RBF
# individual en términos prácticos.

param_bag_svm = {
    'n_estimators': 75,
    'max_samples': 0.8,
    'max_features':1.00,
    'bootstrap': True,
    'bootstrap_features': False
    }

bagging_svm = BaggingClassifier(
    estimator= modelo_svm, **param_bag_svm, oob_score= True, 
    random_state= RANDOM_STATE, n_jobs=-1)
resultados_bag_svm = evaluar_modelo(
    bagging_svm, nombre_b_svm, X_train, X_test, y_train, y_test,
    **param_bag_svm)

matrices_confusion(y_test, resultados_bag_svm['y_pred_test'], nombre_b_svm)
heatmap_classification_reports(y_test, resultados_bag_svm['y_pred_test'],
                               nombre_b_svm)
sobreajuste(resultados_bag_svm['resumen'])

# -----------------------------------------------------------------------------
# 3.3. COMPARACIÓN SVM-RBF VS BAGGING SVM-RBF
# -----------------------------------------------------------------------------
print('\nCOMPARACIÓN - SVM RBF y BAGGING SVM RBF')
df_svm_vs_bag = pd.concat(
    [resultados_svm['resumen'], resultados_bag_svm['resumen']], ignore_index=True)
print(df_svm_vs_bag.round(4).to_string(index=False))

grafico_comparacion_metricas(df_svm_vs_bag, [nombre_svm, nombre_b_svm])
matrices_confusion(
    y_test, [resultados_svm['y_pred_test'], resultados_bag_svm['y_pred_test']],
    [nombre_svm, nombre_b_svm])
heatmap_classification_reports(
    y_test, [resultados_svm['y_pred_test'], resultados_bag_svm['y_pred_test']],
    [nombre_svm, nombre_b_svm])

curva_auc_roc_comparativa(
    y_test, [resultados_svm['y_score_test'], resultados_bag_svm['y_score_test']],
    [nombre_svm, nombre_b_svm])

df_sobre_svm_bag = sobreajuste(df_svm_vs_bag)
print(df_sobre_svm_bag.round(4).to_string(index=False))

# La comparación entre el SVM RBF individual y su versión Bagging muestra un
# resultado mixto. El ensemble mejora de forma moderada la AUC test
# (0.9189 -> 0.9210), la accuracy test (0.8828 -> 0.8941), la precision de la
# clase positiva (0.6480 -> 0.7895) y la specificity (0.8783 -> 0.9577),
# reduciendo además los falsos positivos (69 -> 24).
#
# Sin embargo, estas mejoras se consiguen a costa de un deterioro claro en la
# capacidad de detección de la clase positiva: el recall cae de 0.9007 a
# 0.6383, los falsos negativos aumentan de 14 a 51 y el f1-score de la clase
# positiva desciende de 0.7537 a 0.7059. También empeora el balanced accuracy,
# lo que indica una pérdida de equilibrio entre ambas clases.
#
# En consecuencia, el Bagging SVM RBF desplaza el modelo hacia un perfil mucho
# más conservador: comete menos falsos positivos, pero deja escapar un número
# claramente mayor de casos positivos reales. Desde el punto de vista operativo
# del problema, este intercambio no resulta especialmente favorable.
#
# Además, el análisis de generalización no permite afirmar una mejora clara del
# sobreajuste: aunque el gap en AUC es algo menor, el gap en accuracy aumenta
# de forma apreciable. Por tanto, el aporte del Bagging SVM RBF debe
# interpretarse como positivo pero no sobresaliente, y no parece suficiente
# como para justificar de forma concluyente su mayor coste computacional frente
# al SVM RBF individual.
#
# Precisamente por ello, tiene sentido extender el análisis a otros modelos
# base. El bagging suele aportar más valor en clasificadores más inestables o
# con mayor varianza, como los árboles de decisión, mientras que en modelos más
# estables, como el SVM, su beneficio puede ser limitado o venir acompañado de
# sacrificios operativos importantes.

# -----------------------------------------------------------------------------
# 3.4. BAGGING REGRESIÓN LOGÍSTICA
# -----------------------------------------------------------------------------
# En este segundo bloque se analiza la aplicación de bagging sobre la
# regresión logística seleccionada anteriormente como modelo base.
#
# El objetivo es comprobar si la agregación de múltiples submuestras bootstrap
# permite mejorar la estabilidad y el rendimiento predictivo del clasificador,
# especialmente en términos de AUC, accuracy y recall, sin alejarse demasiado
# del comportamiento del modelo individual.
#
# A diferencia de los árboles, la regresión logística es un algoritmo más
# estable y con menor varianza, por lo que no cabe esperar mejoras tan
# intensas como las que suelen observarse en bagging con modelos más
# inestables. Precisamente por ello, este bloque se plantea como una
# comprobación empírica: verificar si la técnica aporta una ganancia real o si,
# por el contrario, su mejora resulta marginal frente al aumento de coste
# computacional.

print('\nBAGGING REGRESIÓN LOGÍSTICA (MODELO 2)')
nombre_b_log = 'Bagging R.Logística'

bag_log_grid = BaggingClassifier(
    estimator=modelo_log, oob_score=True, random_state=RANDOM_STATE, n_jobs=-1)

grid_bag_log = GridSearchCV(
    bag_log_grid, param_grid_bag, cv=cv, scoring=score, refit='roc_auc', 
    return_train_score=True, verbose=3, n_jobs=-1)
grid_bag_log.fit(X_train, y_train)

df_cv_bag_log = pd.DataFrame(grid_bag_log.cv_results_)
df_grid_bag_log = tabla_grid(df_cv_bag_log, parametros_bag)

grafico_parametros(df_grid_bag_log, parametros_bag, score_grid, nombre_b_log)

# ---------------------------------------------
# CANDIDATOS BAGGING REGRESIÓN LOGÍSTICA
# ---------------------------------------------
# La zona más competitiva se concentra con bootstrap=True,
# bootstrap_features=False y max_features=1.0.
# A partir de ahí, max_samples apenas cambia el rendimiento y el número de
# estimadores entra pronto en una meseta, por lo que se comparan nueve
# configuraciones representativas dentro de la región más sólida.
# Así se incluyen tanto el candidato con mejor AUC como perfiles con
# mejor equilibrio global entre AUC, accuracy y recall.
candidatos_bag_log = pd.DataFrame({
    'candidato': ['bag_log_1', 'bag_log_2', 'bag_log_3', 
                  'bag_log_4', 'bag_log_5', 'bag_log_6', 
                  'bag_log_7', 'bag_log_8', 'bag_log_9'],
    'n_estimators': [50, 50, 100,
                     100, 50, 100,
                     100, 100, 100 ],
    'max_samples': [1.0, 0.85, 1.0,
                    0.7, 0.7, 0.85, 
                    1.0, 0.85, 0.85],
    'max_features': [1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 
                    0.7, 1.0, 0.7],
    'bootstrap': [True]*9,
    'bootstrap_features': [False]*9
    })
seleccion_bag_log = candidatos_bag_log.merge(
    df_cv_bag_log, left_on=parametros_bag, right_on=[
        'param_n_estimators', 'param_max_samples', 'param_max_features',
        'param_bootstrap', 'param_bootstrap_features'], how='left')

boxplot_candidatos(seleccion_bag_log, score, nombre_b_log)

# ---------------------------------------------
# SELECCIÓN BAGGING REGRESIÓN LOGÍSTICA
# ---------------------------------------------
# Se selecciona como candidato del bloque Bagging de Regresión Logística el
# modelo bag_log_5, con n_estimators=50, max_samples=0.7, max_features=1.0,
# bootstrap=True y bootstrap_features=False.
#
# La elección se justifica porque, dentro de un bloque donde las diferencias
# entre configuraciones son muy pequeñas, presenta un rendimiento competitivo
# en las métricas principales, con una AUC media de validación cruzada de
# 0.8251, una accuracy media de 0.7751 y un recall medio de 0.7642. Estas
# cifras lo sitúan muy próximo a las mejores alternativas del bloque.
#
# En este contexto, se prioriza una solución más parsimoniosa, ya que
# configuraciones con mayor número de estimadores o mayor complejidad no
# aportan una mejora suficientemente clara del rendimiento. Por tanto,
# bag_log_5 se considera una opción estable, razonable y metodológicamente
# adecuada como candidato representativo del bloque Bagging de Regresión
# Logística.
param_bag_log ={
    'estimator': modelo_log,
    'n_estimators': 50,
    'max_samples': 0.7,
    'max_features': 1.0,
    'bootstrap': True,
    'bootstrap_features': False
    }

bagging_log = BaggingClassifier(**param_bag_log, oob_score=True, 
                                random_state=RANDOM_STATE, n_jobs=-1)
resultados_bag_log = evaluar_modelo(
    bagging_log, nombre_b_log, X_train, X_test, y_train, y_test,
    **param_bag_log)

matrices_confusion(y_test, resultados_bag_log['y_pred_test'], nombre_b_log)
heatmap_classification_reports(y_test, resultados_bag_log['y_pred_test'],
                               nombre_b_log)

sobreajuste(resultados_bag_log['resumen'])

# -----------------------------------------------------------------------------
# 3.5. COMPARACIÓN REGRESIÓN LOGÍSTICA VS BAGGING REGRESIÓN LOGÍSTICA
# -----------------------------------------------------------------------------
print('\nCOMPARACIÓN REGRESIÓN LOGÍSTICA VS BAGGING REGRESIÓN LOGÍSTICA')
df_log_vs_bag = pd.concat(
    [resultados_log['resumen'], resultados_bag_log['resumen']], 
    ignore_index=True)
print(df_log_vs_bag.round(4).to_string(index=False))

grafico_comparacion_metricas(df_log_vs_bag, [nombre_log, nombre_b_log])

matrices_confusion(
    y_test, [resultados_log['y_pred_test'], resultados_bag_log['y_pred_test']],
                   [nombre_log, nombre_b_log])
heatmap_classification_reports(
    y_test, [resultados_log['y_pred_test'], resultados_bag_log['y_pred_test']], 
    [nombre_log, nombre_b_log])

curva_auc_roc_comparativa(
    y_test, [resultados_log['y_score_test'], resultados_bag_log['y_score_test']],
    [nombre_log, nombre_b_log])

df_sobre_log_bag = sobreajuste(df_log_vs_bag)
print(df_sobre_log_bag.round(4).to_string(index=False))

# La comparación entre la Regresión Logística individual y su versión Bagging
# muestra una mejora ligera, pero no suficientemente intensa como para
# considerarla un salto real de rendimiento.
#
# En concreto, Bagging R.Logística mejora ligeramente la AUC test
# (0.8490 frente a 0.8481), la accuracy test (0.7782 frente a 0.7684), el
# recall (0.7801 frente a 0.7730), la precision (0.4661 frente a 0.4523), el
# f1-score de la clase positiva (0.5836 frente a 0.5707), el balanced
# accuracy y la specificity. Además, reduce ligeramente tanto los falsos
# positivos (132 -> 126) como los falsos negativos (32 -> 31).
#
# Sin embargo, estas mejoras son muy pequeñas y la curva ROC muestra un
# comportamiento prácticamente equivalente entre ambos modelos. Por ello, el
# bagging no aporta aquí una ventaja suficientemente clara como para justificar
# de forma concluyente el aumento de complejidad computacional.
#
# En consecuencia, Bagging R.Logística puede interpretarse como una variante
# algo mejorada y algo más estable, pero su ganancia práctica es limitada y
# no supone una mejora realmente relevante frente a la regresión logística
# individual.

# -----------------------------------------------------------------------------
# 3.6. BAGGING ÁRBOL
# -----------------------------------------------------------------------------
# En este tercer bloque se estudia la aplicación de bagging sobre el árbol de
# decisión seleccionado previamente como estimador base.
#
# El interés de este contraste es especialmente alto en modelos tipo árbol,
# ya que son clasificadores relativamente inestables y sensibles a pequeñas
# variaciones en la muestra de entrenamiento. En este contexto, el bagging
# puede reducir varianza y mejorar la generalización mediante el remuestreo
# bootstrap y la agregación de múltiples árboles.
#
# El objetivo es comprobar si esa reducción de varianza se traduce en mejoras
# reales sobre el árbol individual, tanto en capacidad discriminante como en
# rendimiento operativo sobre la clase positiva.

print('\nBAGGING ÁRBOL (MODELO 3)')
nombre_b_arbol = 'Bagging Árbol'

bag_arbol_grid = BaggingClassifier(
    estimator= modelo_arbol, oob_score=True, random_state=RANDOM_STATE)
grid_bag_arbol = GridSearchCV(
    bag_arbol_grid, param_grid_bag, cv=cv, scoring=score, refit='roc_auc', 
    return_train_score=True, verbose=3, n_jobs=-1)
grid_bag_arbol.fit(X_train, y_train)

df_cv_bag_arbol = pd.DataFrame(grid_bag_arbol.cv_results_)
df_grid_bag_arbol = tabla_grid(df_cv_bag_arbol, parametros_bag)

grafico_parametros(df_grid_bag_arbol, parametros_bag, score_grid, nombre_b_arbol)

# HEATMAP DEL BAGGING (Identificar sectores)
param_bag_arbol_heat = {
    'max_features': [0.6, 0.7, 0.85, 1.0],
    'bootstrap': [True],
    'bootstrap_features': [False]
    }
heatmap_params(df_grid_bag_arbol, nombre_b_arbol, 'n_estimators',
               'max_samples', score_grid, param_bag_arbol_heat)

# ---------------------------------------------
# CANDIDATOS BAGGING ÁRBOL
# ---------------------------------------------
# La zona más competitiva del grid se concentra con bootstrap=True,
# bootstrap_features=False y valores intermedios de max_features,
# especialmente alrededor de 0.70.
#
# A partir de esa región se seleccionan nueve configuraciones representativas
# para comparar distintos perfiles: algunos más orientados a maximizar AUC,
# otros con mayor recall y otros más equilibrados entre rendimiento global y
# parsimonia.
#
# Con ello se evita decidir únicamente por el mejor valor puntual de una sola
# métrica y se analiza también la estabilidad del bagging ante pequeños cambios
# en n_estimators, max_samples y max_features.

candidatos_bag_arbol = pd.DataFrame({
    'candidato': ['bag_arb_1', 'bag_arb_2', 'bag_arb_3',
                  'bag_arb_4', 'bag_arb_5', 'bag_arb_6', 
                  'bag_arb_7', 'bag_arb_8', 'bag_arb_9'],
    'n_estimators': [100, 100, 150,
                     150, 150, 100,
                     100, 100, 100],
    'max_samples': [1.00, 0.70, 1.00,
                    0.70, 0.85, 0.85,
                    0.60, 0.70, 0.85],
    'max_features': [0.70, 0.70, 0.70,
                     0.70, 0.60, 0.70,
                     0.70, 0.60, 0.60],
    'bootstrap': [True]*9,
    'bootstrap_features': [False]*9
    })
seleccion_bag_arbol = candidatos_bag_arbol.merge(
    df_cv_bag_arbol, left_on=parametros_bag, right_on=[
        'param_n_estimators', 'param_max_samples', 'param_max_features',
        'param_bootstrap', 'param_bootstrap_features'], how='left')

boxplot_candidatos(seleccion_bag_arbol, score, nombre_b_arbol)

# ---------------------------------------------
# SELECCIÓN BAGGING ÁRBOL
# ---------------------------------------------
# Se selecciona como candidato final del bloque Bagging Árbol el modelo
# bag_arb_6, con n_estimators=100, max_samples=0.85, max_features=0.70,
# bootstrap=True y bootstrap_features=False.
#
# La elección se basa en que ofrece uno de los perfiles más equilibrados del
# bloque. En validación cruzada alcanza una AUC media de 0.9127, una accuracy
# media de 0.9169 y un recall medio de 0.8352, situándose muy cerca de los
# mejores candidatos en las tres métricas principales de forma simultánea.
#
# Aunque bag_arb_1 obtiene la mayor AUC media del bloque (0.9140) y bag_arb_3
# presenta un comportamiento muy próximo con 150 estimadores, ambos perfiles no
# muestran una ventaja suficientemente clara en el conjunto del rendimiento que
# justifique priorizarlos frente a una solución más contenida.
#
# Por otra parte, bag_arb_2 y bag_arb_4 logran recalls ligeramente superiores
# (0.8369), pero lo hacen con una accuracy media inferior o sin una mejora real
# de AUC que compense claramente ese cambio de perfil.
#
# En consecuencia, bag_arb_6 se considera la opción más sólida y defendible
# como candidato representativo del bloque: mantiene una AUC muy alta, una
# accuracy prácticamente en la zona superior del grid y un recall competitivo,
# con 100 estimadores y una configuración más parsimoniosa que otras
# alternativas muy próximas.

param_bag_arbol={
    'estimator': modelo_arbol,
    'n_estimators': 100,
    'max_samples': 0.85,
    'max_features': 0.70,
    'bootstrap': True,
    'bootstrap_features': False
    }
bagging_arbol = BaggingClassifier(
    **param_bag_arbol, oob_score= True, random_state= RANDOM_STATE, n_jobs=-1)
resultados_bag_arbol = evaluar_modelo(
    bagging_arbol, nombre_b_arbol, X_train, X_test, y_train, y_test,
    **param_bag_arbol)

matrices_confusion(y_test, resultados_bag_arbol['y_pred_test'], nombre_b_arbol)
heatmap_classification_reports(y_test, resultados_bag_arbol['y_pred_test'],
                               nombre_b_arbol)

sobreajuste(resultados_bag_arbol['resumen'])

visualiza_arbol(
    resultados_bag_arbol['modelo'].estimators_[0],
    X_train.columns[resultados_bag_arbol['modelo'].estimators_features_[0]],
    nombre_b_arbol)

# -----------------------------------------------------------------------------
# 3.7. COMPARACIÓN ÁRBOL DE DECISIÓN VS BAGGING ÁRBOL
# -----------------------------------------------------------------------------
print('\nCOMPARACIÓN ÁRBOL DE DECISIÓN VS BAGGING ÁRBOL')
df_arbol_vs_bag = pd.concat(
    [resultados_arbol['resumen'], resultados_bag_arbol['resumen']],
    ignore_index=True)
print(df_arbol_vs_bag.round(4).to_string(index=False))

grafico_comparacion_metricas(df_arbol_vs_bag, [nombre_arbol, nombre_b_arbol])

matrices_confusion(
    y_test, [resultados_arbol['y_pred_test'], 
             resultados_bag_arbol['y_pred_test']],
    [nombre_arbol, nombre_b_arbol])
heatmap_classification_reports(
    y_test, [resultados_arbol['y_pred_test'], 
             resultados_bag_arbol['y_pred_test']],
    [nombre_arbol, nombre_b_arbol])

curva_auc_roc_comparativa(
    y_test, [resultados_arbol['y_score_test'], 
             resultados_bag_arbol['y_score_test']],
    [nombre_arbol, nombre_b_arbol])

df_sobre_arbol_bag = sobreajuste(df_arbol_vs_bag)
print(df_sobre_arbol_bag.round(4).to_string(index=False))

# La comparación entre el árbol de decisión individual y su versión Bagging
# muestra que, en este caso, el ensemble sí aporta una mejora clara y
# consistente.
#
# El Bagging Árbol incrementa de forma apreciable la AUC test
# (0.9187 -> 0.9427), la accuracy test (0.9068 -> 0.9181), el recall
# (0.8298 -> 0.8582), la precision (0.7358 -> 0.7610), el f1-score de la
# clase positiva (0.7800 -> 0.8067), el balanced accuracy y la specificity.
# Además, reduce simultáneamente los falsos positivos (42 -> 38) y los falsos
# negativos (24 -> 20), mejorando tanto la clasificación de la clase negativa
# como la detección de clientes con riesgo real de abandono.
#
# El heatmap del classification report confirma que la mejora no se concentra
# en una única métrica aislada, sino que afecta al conjunto del comportamiento
# predictivo del modelo. De forma coherente con ello, la curva ROC del Bagging
# Árbol se sitúa por encima de la del árbol individual en gran parte del
# recorrido, lo que respalda su mayor capacidad discriminante.
#
# Desde el punto de vista de generalización, el gap train-test en accuracy
# disminuye respecto al árbol base (0.0140 -> 0.0073), lo que sugiere una mejor
# estabilidad fuera de muestra. El gap ligeramente negativo en AUC
# (0.0009 -> -0.0072, con auc_test > auc_train en el bagging) no se interpreta
# como un problema, sino como un resultado compatible con la reducción de
# varianza propia del bagging y con la variabilidad muestral de una única
# partición train/test.
#
# En conjunto, el Bagging Árbol sí parece justificar su mayor complejidad
# computacional y se perfila como el ensemble más sólido de los evaluados hasta
# este punto. No solo mejora el rendimiento medio del árbol base, sino que
# además corrige una parte importante de su varianza, dando lugar a un modelo
# más robusto, más estable y más útil desde el punto de vista operativo.

# =============================================================================
# 4. AJUSTE BAGGING ÁRBOL
# =============================================================================
# En este bloque se realiza un ajuste secuencial del ensemble basado en árbol.
#
# La lógica de trabajo es la siguiente:
# 1. Primero se reajusta el árbol de decisión seleccionado como estimador base,
#    ya que el rendimiento del bagging depende en parte de la calidad y del 
#    perfil del árbol individual sobre el que se construye.
# 2. Después se comprueba cómo se comporta el bagging al sustituir el árbol
#    base original por el árbol ajustado, manteniendo inicialmente la
#    parametrización del ensemble ya seleccionada.
# 3. Por último, se reajusta también el propio bagging para verificar si el
#    cambio en el estimador base modifica la mejor zona de la rejilla.
#
# Este contraste adicional resulta útil porque Bagging Árbol es uno de los
# candidatos más sólidos del bloque ensemble y puede tener interés estratégico
# para la fase posterior de stacking y comparación global.

print('\nAJUSTE BAGGING ÁRBOL')
# -----------------------------------------------------------------------------
# 4.1. VALORES INICIALES (modelos seleccionados)
# -----------------------------------------------------------------------------
# Se almacenan como referencia el árbol individual y el bagging seleccionados
# en el bloque anterior. Ambos se utilizarán como punto de partida y como
# benchmark para valorar si el ajuste posterior aporta una mejora real o si,
# por el contrario, solo introduce complejidad adicional sin ganancia clara.

candidato_arbol_0 = pd.DataFrame({
    'candidato': ['arb_a_0'],
    'criterion': ['entropy'],
    'max_depth': [5],
    'min_samples_split': [20],
    'min_samples_leaf': [20],
    'class_weight': ['balanced']
    })

resultados_bag_arbol_0 = resultados_bag_arbol
candidato_bag_arbol_0 = pd.DataFrame({
    'candidato': ['bag_arb_a_0'],
    'n_estimators': [100],
    'max_samples': [0.85],
    'max_features': [0.7],
    'bootstrap': [True],
    'bootstrap_features': [False]
    })
# -------------------------------------------------------------------------
# 4.2. AJUSTE DEL ESTIMADOR ÁRBOL DE DECISIÓN
# -------------------------------------------------------------------------
# Se plantea un ajuste alrededor de la solución previamente elegida.
# En lugar de rehacer una búsqueda amplia, se explora una zona cercana al
# óptimo anterior, centrando la rejilla en:
# - criterion='entropy'
# - class_weight='balanced'
# - max_depth próximo a 5
# - min_samples_leaf en valores cercanos al ya seleccionado
#
# Esta estrategia permite afinar el árbol sin disparar el coste computacional y
# facilita una lectura más clara de la estabilidad del modelo en una región ya
# identificada como competitiva.

print('\nAJUSTE DEL ESTIMADOR ÁRBOL DE DECISIÓN')
nombre_arbol_a = 'Ajuste Árbol de decisiones'
lista_arb_leaf = [int(i) for i in np.arange(20, 40, 1)]
param_grid_arbol_a = {
    'criterion': ['entropy'],
    'max_depth': [4, 5, 6],
    'min_samples_split': [20],
    'min_samples_leaf': lista_arb_leaf,
    'class_weight': ['balanced']
    }
grid_arbol_a = GridSearchCV(
    arbol_d, param_grid_arbol_a, cv=cv, scoring=score, refit='roc_auc', 
    return_train_score=True, verbose=3, n_jobs=-1)
grid_arbol_a.fit(X_train, y_train)

print('\nMejores parámetros:')
print(grid_arbol_a.best_params_)
print('Mejor AUC CV:', grid_arbol_a.best_score_)

df_cv_arbol_a = pd.DataFrame(grid_arbol_a.cv_results_)
df_grid_arbol_a = tabla_grid(df_cv_arbol_a, parametros_arbol)
grafico_parametros(df_grid_arbol_a, parametros_arbol , score_grid, nombre_arbol_a)

# HEATMAP DE APOYO VISUAL PARA LOCALIZAR REGIONES
# El heatmap permite identificar con mayor claridad qué combinaciones de
# max_depth y min_samples_leaf concentran la zona más sólida del ajuste.
# Así se complementa la lectura de la tabla resumen y de los gráficos de
# evolución por parámetro.

param_arbol_a_heat={
    'criterion': ['entropy'],
    'min_samples_split': [20],
    'class_weight': ['balanced']
    }
heatmap_params(df_grid_arbol_a, nombre_arbol_a, 'max_depth',
               'min_samples_leaf', score_grid, param_arbol_a_heat)

# ---------------------------------------------
# CANDIDATOS AJUSTE ARBOL
# ---------------------------------------------
# A partir del grid, se seleccionan varios candidatos situados en la zona más 
# competitiva del ajuste. Se incluyen:
# - el árbol original como referencia,
# - configuraciones muy próximas al óptimo,
# - y algunos perfiles algo más profundos o más regularizados,
#   para comprobar si la mejora observada en CV se mantiene también en test.
#
# La idea no es elegir automáticamente el primero del ranking, sino comparar un
# conjunto corto de alternativas plausibles y valorar su equilibrio real entre
# AUC, accuracy, recall y generalización.

candidatos_arbol_a = pd.DataFrame({
    'candidato': ['arb_a_1', 'arb_a_2', 'arb_a_3', 
                  'arb_a_4', 'arb_a_5', 'arb_a_6', 
                  'arb_a_7', 'arb_a_8'],
    'criterion': ['entropy']*8,
    'max_depth': [5, 5, 5,
                  5, 5, 5,
                  6, 6],
    'min_samples_split': [20]*8,
    'min_samples_leaf': [29, 30, 32,
                         24, 27, 25,
                         30, 32],
    'class_weight': ['balanced']*8
    })
candidatos_arbol_a = pd.concat([candidato_arbol_0, candidatos_arbol_a],
                               ignore_index=True)
seleccion_arbol_a = candidatos_arbol_a.merge(
    df_cv_arbol_a, left_on=parametros_arbol, right_on=[
        'param_criterion', 'param_max_depth', 'param_min_samples_split',
        'param_min_samples_leaf', 'param_class_weight'], how='left')

# ---------------------------------------------
# EVALUACIÓN Y COMPARACIÓN DE LOS CANDIDATOS AL AJUSTE
# ---------------------------------------------
# En esta fase la decisión no se apoya solo en la tabla del GridSearchCV.
# Dado que los candidatos son ya muy próximos entre sí, se completa el análisis
# comparándolos directamente en test mediante:
# - matrices de confusión,
# - classification reports,
# - comparación global de métricas,
# - curvas ROC,
# - y análisis train-test gap.
#
# Este paso permite comprobar si pequeñas diferencias observadas en validación
# cruzada se traducen realmente en una mejora útil fuera de muestra.

boxplot_candidatos(seleccion_arbol_a, score, nombre_arbol_a)

params_arbol_a = candidatos_arbol_a.copy()
params_arbol_a = list(params_arbol_a.itertuples(index=False,name=None))
predicciones_arbol_a = []
scores_arbol_a = []
resumen_arbol_a = []
nombres_arbol_a = []

for n, c, d, s, l, cw in params_arbol_a:
    modelo_for = DecisionTreeClassifier(
        criterion=c, max_depth=d, min_samples_split= s, min_samples_leaf=l,
        class_weight=cw, random_state=RANDOM_STATE)
    resultados_modelo_for = evaluar_modelo(
        modelo_for, n, X_train, X_test, y_train, y_test, criterion= c, 
        max_depth=d, min_samples_split=s, min_samples_leaf=l, class_weight=cw)
    nombre_modelo = n
    predicciones_arbol_a.append(resultados_modelo_for['y_pred_test'])
    scores_arbol_a.append(resultados_modelo_for['y_score_test'])
    nombres_arbol_a.append(nombre_modelo)
    resumen_arbol_a.append(resultados_modelo_for['resumen'])
    
df_resumen_arbol_a = pd.concat(resumen_arbol_a, ignore_index=True)
matrices_confusion(y_test, predicciones_arbol_a, nombres_arbol_a)
heatmap_classification_reports(y_test, predicciones_arbol_a, nombres_arbol_a)
df_sobre_arbol_a = sobreajuste(df_resumen_arbol_a)
print(df_sobre_arbol_a.round(4).to_string(index=False))
 
grafico_comparacion_metricas(df_resumen_arbol_a, nombre_arbol_a )
curva_auc_roc_comparativa(y_test, scores_arbol_a, nombres_arbol_a)

# ---------------------------------------------
# ÁRBOL DE DECISIÓN AJUSTADO
# ---------------------------------------------
# Tras el ajuste del árbol, se selecciona como modelo final la configuración 
# arb_a_3, con:
# - criterion='entropy'
# - max_depth=5
# - min_samples_split=20
# - min_samples_leaf=32
# - class_weight='balanced'
#
# La elección se justifica porque ofrece un perfil especialmente equilibrado
# dentro del bloque ajustado. Sin necesidad de maximizar de forma aislada una
# única métrica, mantiene una AUC alta, alcanza una accuracy competitiva y
# conserva un recall suficientemente elevado, lo que permite un compromiso
# sólido entre discriminación, acierto global y capacidad de detección de la
# clase positiva.
#
# Además, el modelo mantiene un comportamiento razonable en generalización, sin
# mostrar señales de sobreajuste marcado frente a otros candidatos cercanos.
# Por ello, se considera la opción más robusta como árbol individual ajustado y
# como nuevo estimador base para contrastar el rendimiento del bagging.

param_arbol_final={
    'criterion': 'entropy',
    'max_depth': 5,
    'min_samples_split': 20,
    'min_samples_leaf': 32,
    'class_weight': 'balanced'
    }
modelo_arbol_final = DecisionTreeClassifier(**param_arbol_final, 
                                            random_state=RANDOM_STATE)
resultados_arbol_final = evaluar_modelo(
    modelo_arbol_final, nombre_arbol, X_train, X_test, y_train, y_test,
    **param_arbol_final)

matrices_confusion(
            y_test, resultados_arbol_final['y_pred_test'], nombre_arbol_a)
visualiza_arbol(modelo_arbol_final, X_train.columns, nombre_arbol_a)

# -------------------------------------------------------------------------
# 4.3. BAGGING CON ESTIMADOR ÁRBOL AJUSTADO
# -------------------------------------------------------------------------
# Una vez ajustado el árbol individual, se construye un nuevo bagging tomando
# ese árbol como estimador base. El objetivo es comprobar si la mejora del
# estimador individual se traslada también al ensemble o si, por el contrario,
# el bagging ya absorbía gran parte de esa mejora con el árbol original.
nombre_b_arbol_1 = 'Bagging Árbol (ajustado)'

bag_arbol_1_grid = BaggingClassifier(
    estimator= modelo_arbol_final, oob_score=True, random_state=RANDOM_STATE)
grid_bag_arbol_1 = GridSearchCV(
    bag_arbol_1_grid, param_grid_bag, cv=cv, scoring=score, refit='roc_auc', 
    return_train_score=True, verbose=3, n_jobs=-1)
grid_bag_arbol_1.fit(X_train, y_train)

df_cv_bag_arbol_1 = pd.DataFrame(grid_bag_arbol_1.cv_results_)
df_grid_bag_arbol_1 = tabla_grid(df_cv_bag_arbol_1, parametros_bag)

grafico_parametros(df_grid_bag_arbol_1, parametros_bag, score_grid, 
                   nombre_b_arbol_1)

# ---------------------------------------------
# CANDIDATOS BAGGING CON ESTIMADOR ÁRBOL AJUSTADO
# ---------------------------------------------
# Se seleccionan varias configuraciones situadas en la zona más competitiva del
# nuevo grid de bagging. Como en apartados anteriores, no se decide solo con el
# ranking del GridSearchCV: se eligen candidatos cercanos entre sí para
# contrastar después su comportamiento en test y valorar si el reajuste del
# árbol base modifica realmente la mejor configuración del ensemble.

candidatos_bag_arbol_1 = pd.DataFrame({
    'candidato': ['bag_1_arb_1', 'bag_1_arb_2', 'bag_1_arb_3',
                  'bag_1_arb_4', 'bag_1_arb_5', 'bag_1_arb_6'],
    'n_estimators': [100, 100, 150,
                     100, 100, 100],
    'max_samples': [1.00, 1.00, 1.00,
                    0.85, 0.60, 0.70],
    'max_features': [0.70, 0.60, 0.70,
                     0.70, 0.70, 0.70],
    'bootstrap': [True]*6,
    'bootstrap_features': [False]*6
    })
seleccion_bag_arbol_1 = candidatos_bag_arbol_1.merge(
    df_cv_bag_arbol_1, left_on=parametros_bag, right_on=[
        'param_n_estimators', 'param_max_samples', 'param_max_features',
        'param_bootstrap', 'param_bootstrap_features'], how='left')

boxplot_candidatos(seleccion_bag_arbol_1, score, nombre_b_arbol_1)

# ---------------------------------------------
# BAGGING CON ESTIMADOR ÁRBOL AJUSTADO
# ---------------------------------------------
# Se selecciona como candidato final del bloque el modelo bag_1_arb_4, con:
# - n_estimators=100
# - max_samples=0.85
# - max_features=0.70
# - bootstrap=True
# - bootstrap_features=False
#
# Esta elección mantiene coherencia con la zona más sólida del grid y permite
# comparar el nuevo ensemble en condiciones muy similares a las del bagging
# previo. Así se aísla mejor el efecto de haber cambiado el estimador base,
# evitando que la comparación quede distorsionada por una parametrización
# completamente distinta del bagging.

param_bag_arbol_1={
    'estimator': modelo_arbol_final,
    'n_estimators': 100,
    'max_samples': 0.85,
    'max_features': 0.70,
    'bootstrap': True,
    'bootstrap_features': False
    }
bagging_arbol_1 = BaggingClassifier(
    **param_bag_arbol_1, oob_score= True, random_state= RANDOM_STATE, n_jobs=-1)
resultados_bag_arbol_1 = evaluar_modelo(
    bagging_arbol_1, nombre_b_arbol_1, X_train, X_test, y_train, y_test,
    **param_bag_arbol_1)

matrices_confusion(y_test, resultados_bag_arbol_1['y_pred_test'], 
                   nombre_b_arbol_1)
heatmap_classification_reports(y_test, resultados_bag_arbol_1['y_pred_test'],
                               nombre_b_arbol_1)

sobreajuste(resultados_bag_arbol_1['resumen'])

# -----------------------------------------------------------------------------
# 4.4. COMPARACIÓN: BAGGING CON ESTIMADOR ORIGINAL VS AJUSTADO
# -----------------------------------------------------------------------------
# En esta comparación final se contrasta si el reajuste del árbol base aporta
# una mejora real al ensemble o si el bagging original ya capturaba
# suficientemente bien el patrón del problema.
#
# El interés aquí no está solo en verificar si cambia alguna métrica concreta,
# sino en determinar si el nuevo bagging mejora el equilibrio global entre AUC,
# accuracy, recall, precision, f1 y generalización.

df_bag_vs_bag = pd.concat(
    [resultados_bag_arbol_1['resumen'], resultados_bag_arbol['resumen']], 
    ignore_index=True)
matrices_confusion(
    y_test, [resultados_bag_arbol_0['y_pred_test'],
             resultados_bag_arbol_1['y_pred_test']],
    [nombre_b_arbol, 'Bagging con Árbol Final Ajustado'])
heatmap_classification_reports(
    y_test, [resultados_bag_arbol_0['y_pred_test'], 
             resultados_bag_arbol_1['y_pred_test']],
    [nombre_b_arbol, 'Bagging con Árbol Final Ajustado'])

sobreajuste(df_bag_vs_bag)

# La variante de bagging construida con el árbol ajustado como estimador base
# mejora ligeramente la sensibilidad del ensemble, al reducir los falsos
# negativos de 20 a 18 y aumentar el recall de la clase positiva.
#
# Sin embargo, esta ganancia se obtiene a costa de un incremento de falsos
# positivos (38 -> 47), lo que reduce la precision de la clase positiva y no
# mejora el f1-score. Además, el rendimiento global del modelo empeora
# ligeramente, ya que descienden la AUC test y la accuracy test respecto al
# bagging previo con el estimador original.
#
# En consecuencia, aunque el árbol ajustado aporta un perfil algo más sensible,
# no mejora el equilibrio global del ensemble. Por ello, no se considera que
# su uso como estimador base justifique sustituir al bagging anterior, que
# sigue ofreciendo un comportamiento más sólido y competitivo en conjunto.

# -------------------------------------------------------------------------
# 4.5. AJUSTE DEL BAGGING ÁRBOL
# -------------------------------------------------------------------------
# Tras comparar el Bagging Árbol original con la variante construida a partir
# del árbol ajustado, se observa que esta última no mejora el comportamiento
# global del ensemble. Aunque el bagging con árbol ajustado incrementa
# ligeramente la sensibilidad en algunos puntos del análisis, empeora el
# equilibrio general entre AUC, accuracy, precision y control del error.
#
# Por ello, el ajuste del bagging se realiza finalmente tomando como estimador 
# base el árbol originalmente seleccionado en el bloque anterior, que había 
# mostrado un rendimiento más sólido dentro del ensemble.
#
# El objetivo de este apartado es afinar los hiperparámetros del Bagging Árbol 
# alrededor de la mejor configuración ya detectada, comprobando si pequeñas 
# variaciones en n_estimators, max_samples y max_features aportan una mejora 
# real o si el modelo previo ya se situaba en una zona estable de rendimiento.

print('\nAJUSTE DEL BAGGING ÁRBOL')
nombre_b_a_arbol = 'Ajuste Bagging de Árbol'

# Se construye una rejilla alrededor de la solución previamente elegida para 
# comprobar si pequeñas variaciones en n_estimators, max_samples y
# max_features mejoran realmente el rendimiento del ensemble.

lista_est = [int(i) for i in np.arange(80, 110, 2)]
lista_sample = [round(i, 2) for i in np.arange(0.80, 0.90, 0.01)]
lista_feat = [round(i, 2) for i in np.arange(0.65, 0.75, 0.02)]
param_grid_bag_a_arbol = {
    'n_estimators': lista_est,
    'max_samples': lista_sample,
    'max_features': lista_feat,
    'bootstrap': [True],
    'bootstrap_features': [False]
    }
bag_a_arbol_grid = BaggingClassifier(
    estimator=modelo_arbol, oob_score=True, random_state=RANDOM_STATE, n_jobs=-1)

grid_bag_a_arbol = GridSearchCV(
    bag_a_arbol_grid, param_grid_bag_a_arbol, cv=cv, scoring=score,
    refit='roc_auc', return_train_score=True, verbose=3, n_jobs=-1)
grid_bag_a_arbol.fit(X_train, y_train)
print('\nMejores parámetros ajuste Bagging Árbol:')
print(grid_bag_a_arbol.best_params_)
print('Mejor AUC CV:', grid_bag_a_arbol.best_score_)

df_cv_bag_a_arbol = pd.DataFrame(grid_bag_a_arbol.cv_results_)
df_grid_bag_a_arbol = tabla_grid(df_cv_bag_a_arbol, parametros_bag)

grafico_parametros(df_grid_bag_a_arbol, parametros_bag, score_grid, 
                   nombre_b_a_arbol)

# Heatmap (idea de las regiones no valores)
param_bag_a_arbol_heat = {
    'max_features': [0.65, 0.67, 0.7],
    'bootstrap': [True],
    'bootstrap_features': [False]
    }
heatmap_params(df_grid_bag_a_arbol, nombre_b_a_arbol, 'n_estimators',
               'max_samples', score_grid, param_bag_a_arbol_heat)
# ---------------------------------------------
# CANDIDATOS AJUSTE DEL BAGGING ÁRBOL
# ---------------------------------------------
# A partir del ajuste se seleccionan varias configuraciones situadas en la zona
# más sólida de la rejilla. Se incorpora también como referencia el bagging
# original previamente seleccionado (bag_arb_a_0), con el fin de comprobar si
# el ajuste aporta una mejora real o si la solución inicial ya era 
# suficientemente buena.
#
# Se comparan candidatos muy próximos entre sí, variando de forma moderada
# n_estimators, max_samples y max_features, para analizar si pequeñas mejoras
# en validación cruzada se mantienen también en test y si compensan desde el
# punto de vista operativo.

candidatos_bag_a_arbol = pd.DataFrame({
    'candidato': ['bag_arb_a_1', 'bag_arb_a_2', 'bag_arb_a_3',
                  'bag_arb_a_4', 'bag_arb_a_5', 'bag_arb_a_6', 
                  'bag_arb_a_7', 'bag_arb_a_8'],
    'n_estimators': [80, 80, 86, 
                     104, 90, 100, 
                     94, 96],
    'max_samples': [0.88, 0.87, 0.88, 
                    0.84, 0.88, 0.88, 
                    0.85, 0.85],
    'max_features': [0.65]*8,
    'bootstrap': [True]*8,
    'bootstrap_features': [False]*8
    })

seleccion_bag_a_arbol = candidatos_bag_a_arbol.merge(
    df_cv_bag_a_arbol, left_on=parametros_bag, right_on=[
        'param_n_estimators', 'param_max_samples', 'param_max_features', 
              'param_bootstrap', 'param_bootstrap_features'], how='left')
seleccion_bag_a_arbol_0 = candidato_bag_arbol_0.merge(
    df_cv_bag_arbol, left_on=parametros_bag, right_on=[
        'param_n_estimators', 'param_max_samples', 'param_max_features', 
              'param_bootstrap', 'param_bootstrap_features'], how='left')
seleccion_bag_a_arbol = pd.concat(
    [seleccion_bag_a_arbol_0, seleccion_bag_a_arbol], ignore_index=True)
candidatos_bag_a_arbol = pd.concat(
    [candidato_bag_arbol_0, candidatos_bag_a_arbol], ignore_index=True)

# ---------------------------------------------
# EVALUACIÓN Y COMPARACIÓN DEL AJUSTE DEL BAGGING ÁRBOL
# ---------------------------------------------
# Como en otros bloques del proyecto, la decisión final no se apoya únicamente
# en la tabla del GridSearchCV. Dado que los candidatos se encuentran en una
# zona paramétrica muy próxima, se completa el análisis evaluando cada uno de
# ellos directamente en test.
#
# De este modo, se comprueba si las ligeras diferencias observadas en
# validación cruzada se traducen realmente en mejoras útiles fuera de muestra
# y si se mantienen en términos de AUC, accuracy, recall, matrices de
# confusión, generalización y comportamiento OOB.

boxplot_candidatos(seleccion_bag_a_arbol, score, nombre_b_a_arbol)

# Evaluación y comparación del ajuste:
params_bag_a_arbol = candidatos_bag_a_arbol.copy()
params_bag_a_arbol = list(params_bag_a_arbol.itertuples(index=False,name=None))
predicciones_bag_a_arbol = []
scores_bag_a_arbol = []
resumen_bag_a_arbol = []
nombres_bag_a_arbol = []
modelos_bag_a_arbol = {}

for n, e, s, f, bs, bf in params_bag_a_arbol:
    modelo_for = BaggingClassifier(
        estimator= modelo_arbol, n_estimators= e, max_samples= s,
        max_features=f, bootstrap= bs, bootstrap_features= bf,
        oob_score= True, random_state= RANDOM_STATE)
    resultados_modelo_for = evaluar_modelo(
        modelo_for, n, X_train, X_test, y_train, y_test,
        n_estimators= e, max_samples= s, max_features= f,
        bootstrap= bs, bootstrap_features= bf)
    nombre_modelo = f'{nombre_b_a_arbol}: {n}'
    predicciones_bag_a_arbol.append(resultados_modelo_for['y_pred_test'])
    scores_bag_a_arbol.append(resultados_modelo_for['y_score_test'])
    nombres_bag_a_arbol.append(nombre_modelo)
    resumen_bag_a_arbol.append(resultados_modelo_for['resumen'])
    modelos_bag_a_arbol[nombre_modelo] = modelo_for
    
df_compara_bag_a_arbol = pd.concat(resumen_bag_a_arbol, ignore_index=True)

matrices_confusion(y_test, predicciones_bag_a_arbol, nombres_bag_a_arbol)
heatmap_classification_reports(y_test, predicciones_bag_a_arbol,
                               nombres_bag_a_arbol)

df_sobre_bag_a_arbol = sobreajuste(df_compara_bag_a_arbol)
print(df_sobre_bag_a_arbol.round(4).to_string(index=False)) 
 
grafico_comparacion_metricas(df_compara_bag_a_arbol, nombre_b_a_arbol)

curva_auc_roc_comparativa(y_test, scores_bag_a_arbol, nombres_bag_a_arbol)

# TABLA OOB DE LOS BAGGING
df_oob_bagging_arbol = tabla_oob_bagging(modelos_bag_a_arbol)
print('\nComparación OOB - Modelos bagging')
print(df_oob_bagging_arbol.round(4).to_string(index=False))

# ---------------------------------------------
# SELECCIÓN FINAL DEL AJUSTE DEL BAGGING ÁRBOL
# ---------------------------------------------
# Tras revisar la tabla del GridSearchCV, la gráfica paramétrica, los boxplots,
# la comparación final entre candidatos y la tabla OOB, se selecciona como
# ajuste final del Bagging Árbol la configuración:
# n_estimators=90, max_samples=0.88, max_features=0.65,
# bootstrap=True y bootstrap_features=False.
#
# La elección se justifica porque este candidato ofrece el mejor equilibrio
# global dentro del bloque ajustado. En test alcanza la mayor AUC del ajuste
# (0.9447), se sitúa en la zona más alta en accuracy (0.9209), mantiene un
# recall elevado (0.8723) y logra un f1-score de la clase positiva muy
# competitivo (0.8146), sin mostrar un deterioro apreciable en precision ni en
# specificity.
#
# Además, frente a configuraciones muy próximas como bag_arb_a_6, que empata en
# accuracy, recall, precision y f1-score, bag_arb_a_5 presenta una AUC test
# ligeramente superior (0.9447 frente a 0.9433), lo que refuerza su elección
# como opción final del ajuste.
#
# También mejora ligeramente al bagging original bag_arb_a_0, incrementando la
# AUC test (0.9427 -> 0.9447), la accuracy test (0.9181 -> 0.9209), el recall
# (0.8582 -> 0.8723) y el f1-score de la clase positiva (0.8067 -> 0.8146),
# manteniendo además el mismo número de falsos positivos y reduciendo los
# falsos negativos de 20 a 18.
#
# Aunque su valor OOB no es el más alto del bloque, la decisión final prioriza
# el comportamiento en test y el equilibrio global de métricas, ya que ese es
# el criterio principal de selección en esta fase del proyecto.
#
# En conjunto, esta configuración se considera la opción más robusta, estable
# y operativamente útil como ajuste final del Bagging Árbol.

param_bag_arbol_final={
    'n_estimators': 90,
    'max_samples': 0.88,
    'max_features': 0.65,
    'bootstrap': True,
    'bootstrap_features': False
    }

param_arbol_estimador = param_arbol.copy()

modelo_arbol_est = DecisionTreeClassifier(**param_arbol_estimador, 
                                          random_state=RANDOM_STATE)
bagging_arbol_final = BaggingClassifier(
    estimator=modelo_arbol_est, **param_bag_arbol_final, oob_score=True,
    random_state=RANDOM_STATE, n_jobs=-1)

resultados_bag_arbol_final = evaluar_modelo(
    bagging_arbol_final, nombre_b_arbol, X_train, X_test, y_train, y_test,
    **param_bag_arbol_final)

matrices_confusion(y_test, resultados_bag_arbol_final['y_pred_test'], 
                   nombre_b_arbol)

# El ajuste confirma que el Bagging Árbol original ya se encontraba en una
# zona muy sólida del espacio paramétrico, pero permite mejorar ligeramente su
# rendimiento mediante una configuración próxima más refinada.
#
# En consecuencia, el modelo final del bloque bagging pasa a ser esta versión
# ajustada del Bagging Árbol, que se adopta como ensemble de referencia para
# los apartados posteriores de stacking y comparación global.

# ---------------------------------------------
# INFORMACIÓN ADICIONAL DEL BAGGING DE ÁRBOL
# ---------------------------------------------
# Una vez fijada la configuración final del Bagging Árbol, se incorporan tres
# análisis complementarios:
# 1. curva de aprendizaje, para comprobar cómo evoluciona la generalización al
#    aumentar el tamaño muestral;
# 2. importancia de variables por permutación, para identificar las señales más
#    influyentes del ensemble final;
# 3. visualización de uno de los árboles individuales del bagging, con el fin
#    de aportar una referencia estructural del tipo de reglas que aprende el
#    modelo.
#
# Estos análisis no modifican la selección final, pero enriquecen la
# interpretación del ensemble y facilitan su lectura dentro de la memoria del
# proyecto.

df_curva_bag_arbol = curvas_aprendizaje(
    {nombre_b_arbol: bagging_arbol_final}, X_train, y_train, 
    cv=cv, scoring='roc_auc')
print(df_curva_bag_arbol.round(4).to_string(index=False))

df_importancia_bag_arbol = importancia_variables(
    {nombre_b_arbol: bagging_arbol_final}, X_test, y_test, scoring='roc_auc',
    n_repeats=10, top_n=10)
print(df_importancia_bag_arbol.round(4).to_string(index=False))

# Accedemos al primer árbol del Bagging (el estimador 0)
visualiza_arbol(
    resultados_bag_arbol_final['modelo'].estimators_[0],
    X_train.columns[resultados_bag_arbol_final['modelo'].estimators_features_[0]],
    nombre_b_arbol)

# =============================================================================
# 5. COMPARACIÓN DE LOS BAGGING CON DISTINTO ESTIMADOR
# =============================================================================
# -------------------------------------------------------------------------
# 5.1. COMPARACIÓN ENTRE LOS TRES BAGGING SELECCIONADOS
# -------------------------------------------------------------------------
# En este apartado se comparan los tres modelos bagging finales seleccionados:
# - Bagging SVM RBF
# - Bagging Regresión Logística
# - Bagging Árbol
#
# El objetivo aquí no es repetir el análisis ya realizado frente a sus modelos
# base, sino obtener una visión global entre los tres ensembles resultantes.
# De este modo, se puede identificar cuál de ellos ofrece el mejor equilibrio
# entre capacidad discriminante, rendimiento de clasificación, estabilidad y
# utilidad práctica dentro del bloque bagging.
print('\nCOMPARACIÓN ENTRE LOS TRES BAGGING SELECCIONADOS')

# Reunir resúmenes
df_bagging_finales = pd.concat(
    [resultados_bag_svm['resumen'], resultados_bag_log['resumen'],
     resultados_bag_arbol_final['resumen']], ignore_index=True)
print('\nResumen comparativo bagging:')
print(df_bagging_finales.round(4).to_string(index=False))

predicciones_bagging_3 = [
    resultados_bag_svm['y_pred_test'],
    resultados_bag_log['y_pred_test'],
    resultados_bag_arbol_final['y_pred_test']
    ]
nombres_bagging_3 = [nombre_b_svm, nombre_b_log, nombre_b_arbol]

scores_bagging_3 = [
        resultados_bag_svm['y_score_test'],
        resultados_bag_log['y_score_test'],
        resultados_bag_arbol_final['y_score_test']
        ]
modelos_bagging_3 = {
    nombre_b_svm: bagging_svm,
    nombre_b_log: bagging_log,
    nombre_b_arbol: bagging_arbol_final
    }

# Gráfico comparativo de métricas
grafico_comparacion_metricas(df_bagging_finales, nombres_bagging_3)
# Matrices de confusión
matrices_confusion(y_test, predicciones_bagging_3, nombres_bagging_3)
# Heatmaps classification report
heatmap_classification_reports(y_test, predicciones_bagging_3, nombres_bagging_3)
# Curvas ROC comparativas
curva_auc_roc_comparativa(y_test, scores_bagging_3, nombres_bagging_3)
# Sobreajuste train-test
df_sobre_bagging_3 = sobreajuste(df_bagging_finales)
print(df_sobre_bagging_3.round(4).to_string(index=False))
# Tabla OOB comparación de los bagging
df_oob_bagging_3 = tabla_oob_bagging(modelos_bagging_3)
print(df_oob_bagging_3.round(4).to_string(index=False))
# Importancia de variables 
df_importancia_bagging_3 = importancia_variables(
    modelos_bagging_3, X_test, y_test, scoring='roc_auc',
    n_repeats=10, top_n=10)
print(df_importancia_bagging_3.round(4).to_string(index=False))

# Comparación final entre los modelos bagging:
# El Bagging Árbol se consolida como el ensemble ganador de este bloque, ya que
# es el que ofrece el mejor rendimiento global en el conjunto de test.
# Presenta la AUC test más alta (0.9447), la accuracy test más elevada
# (0.9209), el recall más alto (0.8723), el mejor f1-score de la clase
# positiva (0.8146) y también el mejor oob_score (0.9137), superando en el
# conjunto global tanto al Bagging SVM RBF como al Bagging de Regresión
# Logística.
#
# Esta superioridad no se limita a una única métrica aislada, sino que se
# mantiene de forma coherente en los principales indicadores de discriminación,
# clasificación y robustez. Además, desde el punto de vista de la
# generalización, el Bagging Árbol muestra un comportamiento especialmente
# sólido, con un gap de accuracy reducido (0.0055) y un gap de AUC ligeramente
# negativo (auc_test > auc_train), resultado compatible con la reducción de
# varianza propia del bagging y con la variabilidad muestral de una única
# partición train/test.
#
# Frente a ello, el Bagging SVM RBF destaca por una precision alta (0.7895) y
# una specificity muy elevada (0.9577), además de reducir claramente los falsos
# positivos. Sin embargo, ese perfil resulta excesivamente conservador, ya que
# lo consigue a costa de un recall mucho más bajo (0.6383) y de un aumento muy
# importante de falsos negativos (51), lo que penaliza su utilidad operativa.
#
# Por su parte, el Bagging de Regresión Logística muestra un comportamiento más
# equilibrado que el Bagging SVM RBF en sensibilidad, pero queda claramente por
# detrás en capacidad predictiva global, tanto en AUC como en accuracy,
# precision, f1-score y oob_score.
#
# En conjunto, la evidencia empírica confirma que el Bagging Árbol no solo es
# el mejor de los tres modelos evaluados, sino también el ensemble que aporta
# la combinación más equilibrada de rendimiento, estabilidad y utilidad
# práctica. Por ello, se considera el bagging de referencia para el cierre de
# este bloque y para su posible incorporación al stacking posterior.

# =============================================================================
# 6. MODELOS STACKING - JUSTIFICACIÓN METODOLÓGICA DEL BLOQUE STACKING
# =============================================================================
# -------------------------------------------------------------------------
# 6.1 PREPARACIÓN MODELOS STACKING
# -------------------------------------------------------------------------
# En este bloque se mantiene la misma partición train/test utilizada en el
# resto del proyecto, con el fin de asegurar la comparabilidad directa de los
# resultados. Asimismo, se reutiliza la matriz X_train/X_test ya preprocesada,
# evitando introducir cambios adicionales en el pipeline que puedan distorsionar
# la interpretación de las diferencias entre modelos.
#
# El planteamiento del stacking consiste en combinar clasificadores de
# naturalezas distintas para aprovechar fortalezas complementarias y reducir la
# correlación entre errores. La lógica del bloque no es acumular modelos
# similares, sino construir combinaciones heterogéneas que aporten señales
# predictivas diferentes al meta-modelo.
#
# En particular, se incorporan:
# - SVM RBF, como modelo no lineal de referencia con buena capacidad
#   discriminante global.
# - Regresión logística, como contraste lineal, interpretable y estable.
# - Bagging Árbol, como ensemble consolidado dentro del proyecto, capaz de
#   capturar reglas por regiones y reducir la varianza del árbol individual.
# - Random Forest y XGBoost, como modelos adicionales de tipo arbóreo/ensemble
#   para ampliar el contraste metodológico del stacking.
#
# La finalidad de este bloque no es abrir un nuevo proceso completo de ajuste
# para Random Forest o XGBoost, sino utilizarlos como modelos de contraste
# dentro del metaensamblado y comprobar si esa diversidad adicional mejora de
# forma real el rendimiento del stacking frente a combinaciones más continuistas
# con los modelos ya consolidados en el proyecto.

print('\nSTACKING - PLANTEAMIENTO GENERAL')
# SVM RBF final del apartado 2
svm_stack = modelo_svm
# Regresión logística final del bloque bagging
log_stack = modelo_log
# Bagging Árbol final fijado para el proyecto
bag_arbol_stack = bagging_arbol_final

# -------------------------------------------------------------------------
# 6.2. MODELOS PARA CONTRASTE ADICIONAL
# ---------------------------------------------
# RANDOM FOREST
# -------------------------------------------------------------------------
# Random Forest se incorpora como modelo de contraste adicional dentro del
# stacking para comprobar cómo responde el metaensamblado cuando, en lugar de
# apoyarse solo en Bagging Árbol como representante de la familia arbórea, se
# añade otra variante ensemble basada también en árboles pero con una lógica de
# construcción distinta.
#
# Mientras que Bagging Árbol parte de un árbol base optimizado y reduce su
# varianza mediante remuestreo bootstrap, Random Forest introduce además una
# aleatorización explícita en la selección de variables en cada división.
# Esto lo convierte en un contraste metodológicamente interesante, ya que puede
# aportar una señal menos correlacionada con la del bagging ya seleccionado.
#
# El objetivo, por tanto, no es sustituir el papel del Bagging Árbol dentro del
# proyecto, sino verificar si esta segunda familia arbórea añade diversidad
# funcional útil al stacking o si, por el contrario, su aportación resulta
# redundante frente al ensemble ya consolidado.

print('\nRANDOM FOREST - MODELO DE CONTRASTE 1')
# Random Forest como contraste adicional dentro del stacking
nombre_rf = 'Random Forest'
parametros_rf = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                 'min_samples_leaf', 'max_features', 'max_samples', 
                 'class_weight']
param_grid_rf = {
    'n_estimators': [100, 150, 200],
    'criterion': ['entropy'],
    'max_depth': [5, 7, 10],
    'min_samples_split': [20, 30, 40],
    'min_samples_leaf': [10, 15, 20],
    'max_features': ['sqrt', 0.7],
    'max_samples': [0.7, 0.8, 0.9],
    'class_weight': ['balanced']
    }
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

grid_rf = GridSearchCV(
    estimator=rf, param_grid=param_grid_rf, cv=cv, scoring=score,
    refit='roc_auc', return_train_score=True, verbose=3, n_jobs=-1)
grid_rf.fit(X_train, y_train)
print('\nMejores parámetros:')
print(grid_rf.best_params_)
print('Mejor AUC CV:', grid_rf.best_score_)

df_cv_rf = pd.DataFrame(grid_rf.cv_results_)
df_grid_rf = tabla_grid(df_cv_rf, parametros_rf)

grafico_parametros(df_grid_rf, parametros_rf, score_grid, nombre_rf)
# Heatmaps RandomForest
params_rf_heat = {
    'n_estimators': [100, 150],
    'criterion': ['entropy'],
    'max_depth': [5, 7, 10],
    'max_features': [0.7],
    'max_samples': [0.8],
    'class_weight': ['balanced']
    }
heatmap_params(df_grid_rf, nombre_rf, 'min_samples_leaf', 'min_samples_split',
               score_grid, params_rf_heat)
    
# ---------------------------------------------
# CANDIDATOS RANDOM FOREST
# ---------------------------------------------
# Se seleccionan 9 candidatos situados en la zona más sólida del ajuste,
# ya que esta rama concentra las configuraciones con mejor equilibrio entre
# AUC, accuracy, recall y estabilidad en validación cruzada.
#
# En términos generales, predomina una estructura bastante consistente:
# criterion='entropy', profundidades moderadas, class_weight='balanced' y
# valores de max_features cercanos a 0.7, lo que sugiere una región del
# espacio paramétrico especialmente robusta para este problema.
#
# Dentro de esa meseta superior se incorporan candidatos con pequeñas
# variaciones en n_estimators, min_samples_split, min_samples_leaf,
# max_features y max_samples, con el objetivo de comparar perfiles muy
# próximos entre sí y valorar qué combinación ofrece la mejor relación entre
# capacidad predictiva, regularización y generalización fuera de muestra.
#
# Este contraste resulta suficiente para seleccionar un Random Forest sólido
# como candidato de apoyo al stacking, sin necesidad de desarrollar en esta
# fase un bloque autónomo tan extenso como el realizado con los modelos
# principales del proyecto.

candidatos_rf = pd.DataFrame({
    'candidato': ['rf_1', 'rf_2', 'rf_3',
                  'rf_4', 'rf_5', 'rf_6',
                  'rf_7', 'rf_8', 'rf_9'],
    'n_estimators': [100, 100, 100, 
                     150, 100, 100,
                     150, 100, 150],
    'criterion': ['entropy'] * 9,
    'max_depth': [7, 7, 7,
                  7, 7, 7, 
                  7, 7, 7],
    'min_samples_split': [40, 30, 20,
                          20, 40, 20,
                          30, 20, 20],
    'min_samples_leaf': [10, 10, 15,
                         10, 10, 15,
                         10, 20, 10],
    'max_features': ['0.7', '0.7', '0.7',
                     '0.7', '0.7', '0.7',
                     '0.7', 'sqrt', '0.7'],
    'max_samples': [0.9, 0.7, 0.9,
                    0.7, 0.9, 0.8,
                    0.9, 0.8, 0.8],
    'class_weight': ['balanced'] * 9
    })
df_cv_rf['param_max_features'] = df_cv_rf['param_max_features'].astype(str)
df_cv_rf['param_class_weight'] = df_cv_rf['param_class_weight'].astype(str)

seleccion_rf = candidatos_rf.merge(
    df_cv_rf, left_on=parametros_rf, right_on=[
        'param_n_estimators', 'param_criterion', 'param_max_depth', 
        'param_min_samples_split', 'param_min_samples_leaf',
        'param_max_features', 'param_max_samples', 'param_class_weight'],
    how='left')

# ---------------------------------------------
# COMPARACIÓN Y EVALUACIÓN CANDIDATOS RANDOM FOREST
# ---------------------------------------------
boxplot_candidatos(seleccion_rf, score, nombre_rf)

candidatos_rf['max_features'] = pd.to_numeric(
    candidatos_rf['max_features'], errors='coerce'
    ).fillna(candidatos_rf['max_features'])
param_rf_cand = candidatos_rf.copy()
param_rf_cand = list(param_rf_cand.itertuples(index=False,name=None))
predicciones_rf_cand = []
scores_rf_cand = []
resumen_rf_cand = []
nombres_rf_cand = []

for n, est, c, d, sp, l, f, ms, cw in param_rf_cand:
    modelo_for = RandomForestClassifier(
        n_estimators=est, criterion=c, max_depth=d, 
        min_samples_split=sp, min_samples_leaf=l, max_features=f, 
        max_samples=ms, class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1)
    resultados_modelo_for = evaluar_modelo(
        modelo_for, n, X_train, X_test, y_train, y_test,
        n_estimators= est, criterion=c, max_depth=d, min_samples_split=sp, 
        min_samples_leaf=l, max_features=f, max_samples=ms, class_weight=cw)
    nombre_modelo = n
    predicciones_rf_cand.append(resultados_modelo_for['y_pred_test'])
    scores_rf_cand.append(resultados_modelo_for['y_score_test'])
    nombres_rf_cand.append(nombre_modelo)
    resumen_rf_cand.append(resultados_modelo_for['resumen'])
    
df_compara_rf = pd.concat(resumen_rf_cand, ignore_index=True)

matrices_confusion(y_test, predicciones_rf_cand, nombres_rf_cand)
heatmap_classification_reports(y_test, predicciones_rf_cand, nombres_rf_cand)

df_sobre_rf_candi = sobreajuste(df_compara_rf)
print(df_sobre_rf_candi.round(4).to_string(index=False))  

grafico_comparacion_metricas(df_compara_rf, f'Candidatos {nombre_rf}')
curva_auc_roc_comparativa(y_test, scores_rf_cand, nombres_rf_cand)

# ---------------------------------------------
# SELECCIÓN RANDOM FOREST
# ---------------------------------------------
# Se selecciona el candidato rf_9 como modelo final de Random Forest.
#
# Aunque no presenta la AUC test más alta de forma absoluta, su valor
# (0.9453) es prácticamente equivalente al del mejor candidato por AUC,
# por lo que la diferencia en capacidad discriminante es mínima.
#
# La elección se justifica porque rf_9 ofrece el mejor comportamiento
# global en test desde el punto de vista operativo: alcanza la mayor
# accuracy (0.9350), el mayor recall entre los candidatos más sólidos,
# una precision también muy alta (0.8146) y, en consecuencia, el mejor
# f1-score de la clase positiva.
#
# Además, su matriz de confusión es la más equilibrada del grupo fuerte,
# con 18 falsos negativos y 28 falsos positivos, lo que implica una mejora
# simultánea en la detección de clientes con riesgo real y en el control
# de clasificaciones erróneas sobre la clase negativa.
#
# Frente a rf_7, que obtiene una AUC ligeramente superior, rf_9 mejora
# el rendimiento práctico de clasificación con más verdaderos positivos,
# más verdaderos negativos y menos errores totales. Frente a rf_4, mejora
# recall y f1 positivo manteniendo una precision muy competitiva.
#
# Por tanto, rf_9 se considera la opción más robusta y útil para este
# problema, al combinar una discriminación muy alta con el mejor equilibrio
# entre accuracy, recall, precision y utilidad operativa.
param_rf_stack = {
    'n_estimators': 150,
    'criterion': 'entropy',
    'max_depth': 7,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 0.7,
    'max_samples': 0.8,
    'class_weight': 'balanced'
    }
modelo_rf_stack = RandomForestClassifier(**param_rf_stack, random_state=RANDOM_STATE,
                                         n_jobs=-1)

resultados_rf_stack = evaluar_modelo(modelo_rf_stack, nombre_rf, X_train, 
                                     X_test, y_train, y_test, **param_rf_stack)
print(resultados_rf_stack['resumen'].round(4).to_string(index=False))

matrices_confusion(y_test, resultados_rf_stack['y_pred_test'], nombre_rf)
heatmap_classification_reports(y_test, resultados_rf_stack['y_pred_test'],
                               nombre_rf)

sobreajuste(resultados_rf_stack['resumen'])

# -------------------------------------------------------------------------
# 6.3. MODELOS PARA CONTRASTE ADICIONAL (RF Y XGBOOST)
# ---------------------------------------------
# XGBOOST
# -------------------------------------------------------------------------
# XGBoost se incorpora como segundo modelo de contraste dentro del stacking con
# una finalidad metodológica clara: comprobar cómo responde el metaensamblado
# al introducir una familia boosting frente a los enfoques bagging ya
# trabajados en el proyecto.
#
# Mientras que Bagging Árbol y Random Forest se apoyan en mecanismos de
# reducción de varianza mediante agregación de múltiples árboles, XGBoost sigue
# una lógica distinta: construir secuencialmente árboles que corrigen errores
# previos, reforzando así el aprendizaje sobre patrones difíciles.
#
# Esta diferencia lo convierte en un contraste especialmente relevante en el
# contexto del stacking. No se trata solo de añadir otro modelo arbóreo, sino
# de comprobar si una señal boosting, más orientada a la corrección progresiva
# del error, aporta valor adicional frente a la familia bagging y mejora la
# diversidad funcional del meta-modelo.
#
# Por ello, el interés de este bloque no está únicamente en evaluar XGBoost
# como modelo individual, sino en verificar si su incorporación permite
# enriquecer el stacking con una perspectiva complementaria a la ya aportada
# por Bagging Árbol.

print('\nXGBOOST - MODELO DE CONTRASTE 2')
nombre_xg = 'XGBoost'
parametros_xgb = ['n_estimators', 'learning_rate', 'gamma', 'max_depth', 
                  'subsample', 'colsample_bytree', 'min_child_weight']
# Peso de la clase positiva según TRAIN
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
scale_pos_weight = n_neg/n_pos if n_pos > 0 else 1.0

param_grid_xgb = {
    'n_estimators': [100, 125],
    'learning_rate': [0.01, 0.03, 0.05],
    'gamma': [2],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'min_child_weight': [10, 12, 15]
    }
xgb = XGBClassifier(
    booster='gbtree', objective='binary:logistic', eval_metric='logloss',
    tree_method='hist', scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
    n_jobs=-1)

grid_xgb = GridSearchCV(
    estimator=xgb, param_grid=param_grid_xgb, cv=cv, scoring=score,
    refit='roc_auc', return_train_score=True, verbose=3, n_jobs=-1)
grid_xgb.fit(X_train, y_train)
print('\nMejores parámetros XGBoost:')
print(grid_xgb.best_params_)
print('Mejor AUC CV:', grid_xgb.best_score_)

df_cv_xgb = pd.DataFrame(grid_xgb.cv_results_)
df_grid_xgb = tabla_grid(df_cv_xgb, parametros_xgb)

grafico_parametros(df_grid_xgb, parametros_xgb, score_grid, nombre_xg)

colsample_bytree = [0.6, 0.7, 0.8]
for bt in colsample_bytree:
    params_xgb_heat = {
        'gamma': [2],
        'learning_rate': [0.03],
        'max_depth': [3, 5],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [bt]
        }
    heatmap_params(df_grid_xgb, nombre_xg, 'n_estimators', 'min_child_weight',
                   score_grid, params_xgb_heat)

# ---------------------------------------------
# CANDIDATOS XGBOOST
# ---------------------------------------------
# Se seleccionan 9 candidatos situados en la zona alta del grid final, con el
# objetivo de comparar no solo los mejores valores de AUC CV, sino también
# perfiles distintos de complejidad, equilibrio y potencial de generalización.
#
# La selección combina:
# - los modelos líderes en AUC CV;
# - configuraciones muy equilibradas entre AUC, accuracy y recall;
# - perfiles más conservadores con mayor regularización;
# - y alternativas algo más flexibles, para comprobar si una mayor complejidad
#   mejora realmente el rendimiento final.
#
# Así, la comparación final permitirá decidir con criterio entre candidatos muy
# próximos en validación cruzada, incorporando además el análisis de
# generalización real en train/test y su posible utilidad posterior dentro del
# stacking como representante de la familia boosting.

candidatos_xgb = pd.DataFrame({
    'candidato': ['xgb_1', 'xgb_2', 'xgb_3', 
                  'xgb_4', 'xgb_5', 'xgb_6', 
                  'xgb_7', 'xgb_8', 'xgb_9'],
    'n_estimators': [100, 125, 100,
                     100, 100, 125,
                     100, 100, 100],
    'learning_rate': [0.03, 0.03, 0.03,
                      0.03, 0.03, 0.05,
                      0.05, 0.05, 0.03],
    'gamma': [2]*9,
    'max_depth': [5, 4, 4,
                  5, 5, 4,
                  4, 5, 4],
    'subsample': [0.7, 0.8, 0.8,
                  0.8, 0.7, 0.8,
                  0.8, 0.8, 0.7],
    'colsample_bytree': [0.7, 0.7, 0.7,
                         0.7, 0.7, 0.6,
                         0.6, 0.6, 0.7],
    'min_child_weight': [12, 15, 15,
                         15, 10, 10,
                         12, 15, 12]
    })

seleccion_xgb = candidatos_xgb.merge(
    df_cv_xgb, left_on=parametros_xgb, right_on=[
        'param_n_estimators', 'param_learning_rate', 'param_gamma', 
        'param_max_depth', 'param_subsample', 'param_colsample_bytree',
        'param_min_child_weight'], how='left')

# ---------------------------------------------
# COMPARACIÓN Y EVALUACIÓN DE CANDIDATOS XGBOOST
# ---------------------------------------------
boxplot_candidatos(seleccion_xgb, score, 'XGBoost')

param_xgb_cand = candidatos_xgb.copy()
param_xgb_cand = list(param_xgb_cand.itertuples(index=False,name=None))
predicciones_xgb_cand = []
scores_xgb_cand = []
resumen_xgb_cand = []
nombres_xgb_cand = []

for n, est, lr, g, d, sub, cs, mcw in param_xgb_cand:
    modelo_for = XGBClassifier(
        booster='gbtree', objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', n_estimators=est, learning_rate=lr, gamma=g, 
        max_depth=d, subsample=sub, colsample_bytree=cs, min_child_weight=mcw, 
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1)
    resultados_modelo_for = evaluar_modelo(
        modelo_for, n, X_train, X_test, y_train, y_test,
        booster='gbtree', objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', n_estimators=est, learning_rate=lr, gamma=g, 
        max_depth=d, subsample=sub, colsample_bytree=cs, min_child_weight=mcw, 
        scale_pos_weight=scale_pos_weight)
    nombre_modelo = n
    predicciones_xgb_cand.append(resultados_modelo_for['y_pred_test'])
    scores_xgb_cand.append(resultados_modelo_for['y_score_test'])
    nombres_xgb_cand.append(nombre_modelo)
    resumen_xgb_cand.append(resultados_modelo_for['resumen'])
    
df_compara_xgb = pd.concat(resumen_xgb_cand, ignore_index=True)

matrices_confusion(y_test, predicciones_xgb_cand, nombres_xgb_cand)
heatmap_classification_reports(y_test, predicciones_xgb_cand, nombres_xgb_cand)

df_sobre_xgb = sobreajuste(df_compara_xgb)
print(df_sobre_xgb.round(4).to_string(index=False))

grafico_comparacion_metricas(df_compara_xgb, f'Candidatos {nombre_xg}')
curva_auc_roc_comparativa(y_test, scores_xgb_cand, nombres_xgb_cand)


# Tras el análisis comparativo de los candidatos finales de XGBoost, se toma
# xgb_7 como configuración de referencia para continuar el bloque.
#
# Aunque xgb_5 presenta una matriz de confusión algo más competitiva en algunos
# indicadores puntuales y un comportamiento también muy sólido, xgb_7 ofrece en
# este punto el perfil global más equilibrado entre rendimiento predictivo y
# capacidad de generalización. En particular, alcanza una AUC test de 0.9478
# con un sobreajuste muy contenido (gap_auc = 0.0047 y gap_accuracy = 0.0048),
# claramente inferior al de otros candidatos del bloque.
#
# Esta estabilidad resulta especialmente valiosa porque en esta fase no interesa
# únicamente el mejor ajuste aparente sobre una partición concreta, sino un
# modelo boosting robusto, estable y transferible, capaz de aportar valor tanto
# en el stacking como en la comparación final del proyecto frente a otras
# alternativas ensemble, especialmente el bagging.
#
# Desde el punto de vista operativo, xgb_7 mantiene además un rendimiento
# competitivo en accuracy, recall y balanced accuracy, sin mostrar señales
# relevantes de sobreentrenamiento. Por ello, se adopta como punto de partida
# más razonable para representar la familia boosting dentro del trabajo.
#
# No obstante, la comparación con xgb_5 sigue siendo lo bastante ajustada como
# para no dar por cerrado el bloque en este punto. Por ese motivo, a
# continuación se realizará un ajuste centrado en la zona de ambos candidatos, 
# para comprobar si existe una configuración intermedia o próxima que permita 
# conservar la estabilidad observada en xgb_7 y, al mismo tiempo, recoger parte
# del mejor comportamiento operativo mostrado por xgb_5.
#
# ---------------------------------------------
# AJUSTE XGBOOST
# ---------------------------------------------
# Tras la primera selección de candidatos, se realiza un ajuste adicional
# centrado en la zona más prometedora del bloque XGBoost. El objetivo no es
# reabrir una búsqueda amplia, sino comprobar si en el entorno de los mejores
# perfiles detectados existe una configuración más equilibrada entre capacidad
# discriminante, comportamiento operativo y generalización.
#
# Este ajuste resulta especialmente útil porque XGBoost no solo se valora aquí
# como modelo individual de contraste, sino también como posible representante
# de la familia boosting dentro del stacking.

print('\nXGBOOST - AJUSTE')
nombre_xg_a = 'Ajuste XGBoost'

param_grid_xgb_a = {
    'n_estimators': [100],
    'learning_rate': [0.02, 0.03, 0.04, 0.05],
    'gamma': [2],
    'max_depth': [4, 5],
    'subsample': [0.7, 0.75, 0.8],
    'colsample_bytree': [0.6, 0.65, 0.7],
    'min_child_weight': [10, 11, 12]
    }
xgb_a = XGBClassifier(
    booster='gbtree', objective='binary:logistic', eval_metric='logloss',
    tree_method='hist', scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
    n_jobs=-1)

grid_xgb_a = GridSearchCV(
    estimator=xgb_a, param_grid=param_grid_xgb_a, cv=cv, scoring=score,
    refit='roc_auc', return_train_score=True, verbose=3, n_jobs=-1)
grid_xgb_a.fit(X_train, y_train)

df_cv_xgb_a = pd.DataFrame(grid_xgb_a.cv_results_)
df_grid_xgb_a = tabla_grid(df_cv_xgb_a, parametros_xgb)

grafico_parametros(df_grid_xgb_a, parametros_xgb, score_grid, nombre_xg_a)

candidatos_xgb_a = pd.DataFrame({
    'candidato': ['xgb_a_1', 'xgb_a_2', 'xgb_a_3', 
                  'xgb_a_4', 'xgb_a_5', 'xgb_a_6', 
                  'xgb_a_7', 'xgb_a_8', 'xgb_a_9'],
    'n_estimators': [100, 100, 100,
                     100, 100, 100,
                     100, 100, 100],
    'learning_rate': [0.03, 0.05, 0.05,
                      0.04, 0.04, 0.04,
                      0.05, 0.05, 0.02],
    'gamma': [2]*9,
    'max_depth': [5, 4, 4,
                  4, 4, 4,
                  4, 4, 5],
    'subsample': [0.7, 0.8, 0.75,
                  0.75, 0.7, 0.7,
                  0.7, 0.7, 0.7],
    'colsample_bytree': [0.7, 0.6, 0.7,
                         0.7, 0.7, 0.7,
                         0.6, 0.7, 0.6],
    'min_child_weight': [10, 12, 12,
                         11, 11, 12,
                         10, 10, 10]
    })

seleccion_xgb_a = candidatos_xgb_a.merge(
    df_cv_xgb_a, left_on=parametros_xgb, right_on=[
        'param_n_estimators', 'param_learning_rate', 'param_gamma', 
        'param_max_depth', 'param_subsample', 'param_colsample_bytree',
        'param_min_child_weight'], how='left')

boxplot_candidatos(seleccion_xgb_a, score, 'XGBoost')

param_xgb_a_cand = candidatos_xgb_a.copy()
param_xgb_a_cand = list(param_xgb_a_cand.itertuples(index=False,name=None))
predicciones_xgb_a_cand = []
scores_xgb_a_cand = []
resumen_xgb_a_cand = []
nombres_xgb_a_cand = []

for n, est, lr, g, d, sub, cs, mcw in param_xgb_a_cand:
    modelo_for = XGBClassifier(
        booster='gbtree', objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', n_estimators=est, learning_rate=lr, gamma=g, 
        max_depth=d, subsample=sub, colsample_bytree=cs, min_child_weight=mcw, 
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1)
    resultados_modelo_for = evaluar_modelo(
        modelo_for, n, X_train, X_test, y_train, y_test,
        booster='gbtree', objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', n_estimators=est, learning_rate=lr, gamma=g, 
        max_depth=d, subsample=sub, colsample_bytree=cs, min_child_weight=mcw, 
        scale_pos_weight=scale_pos_weight)
    nombre_modelo = n
    predicciones_xgb_a_cand.append(resultados_modelo_for['y_pred_test'])
    scores_xgb_a_cand.append(resultados_modelo_for['y_score_test'])
    nombres_xgb_a_cand.append(nombre_modelo)
    resumen_xgb_a_cand.append(resultados_modelo_for['resumen'])
    
df_compara_xgb_a = pd.concat(resumen_xgb_a_cand, ignore_index=True)

matrices_confusion(y_test, predicciones_xgb_a_cand, nombres_xgb_a_cand)
heatmap_classification_reports(y_test, predicciones_xgb_a_cand, nombres_xgb_a_cand)

df_sobre_xgb_a = sobreajuste(df_compara_xgb_a)
print(df_sobre_xgb_a.round(4).to_string(index=False))
grafico_comparacion_metricas(df_compara_xgb_a, nombre_xg_a)

# ---------------------------------------------
# SELECCIÓN XGBOOST
# ---------------------------------------------
# Tras comparar los nueve candidatos ajustados, se selecciona xgb_a_1 como
# modelo XGBoost de referencia para el proyecto.
#
# Esta elección se mantiene tanto para su evaluación como modelo individual
# como para su incorporación posterior al bloque stacking, ya que es el
# candidato que ofrece el mejor rendimiento global y el perfil más útil
# dentro del conjunto analizado.
#
# ---------------------------------------------
# 1. SELECCIÓN DEL MEJOR XGBOOST: xgb_a_1
# ---------------------------------------------
# Se selecciona xgb_a_1 como mejor modelo XGBoost.
#
# Justificación:
# xgb_a_1 no presenta la mayor AUC test absoluta, pero su valor de 0.9474 sigue
# siendo muy alto y la diferencia respecto al mejor candidato en AUC es mínima.
# A cambio, ofrece el mejor comportamiento global en la clasificación final
# sobre el conjunto de test, que es el criterio más importante cuando el modelo
# se valora como clasificador completo y utilizable en términos prácticos.
#
# En concreto, xgb_a_1 destaca por:
# - mayor accuracy test del grupo: 0.9322
# - mayor precision de la clase positiva: 0.7888
# - mayor f1-score de la clase positiva: 0.8411
# - mayor specificity: 0.9400
# - menor número de falsos positivos: 34
# - recall alto: 0.9007
# - mismos falsos negativos que la mayoría de los mejores candidatos: 14
#
# Esto significa que xgb_a_1 mantiene una capacidad de detección de clientes
# con riesgo real de abandono muy alta. Es decir, detecta correctamente a los 
# clientes con riesgo real de abandono y, además, clasifica mejor a los 
# clientes sin riesgo, reduciendo los falsos positivos respecto a otros 
# candidatos con recall similar.
#
# Frente a candidatos como xgb_a_3, xgb_a_5, xgb_a_6, xgb_a_7, xgb_a_8 o
# xgb_a_9, mantiene el mismo recall y los mismos FN=14, pero consigue menos FP,
# lo que mejora accuracy, precision y f1-score de la clase positiva. Frente a
# xgb_a_2 y xgb_a_4, además de igualar o superar el resto de métricas, mejora
# la detección de la clase positiva al reducir los falsos negativos de 16 a 14.
#
# Es cierto que xgb_a_1 presenta un gap de AUC algo mayor que otros candidatos
# (gap_auc = 0.0149), por lo que existe una ligera señal de ajuste más fuerte.
# Sin embargo, ese comportamiento no se traduce en un peor rendimiento práctico
# en test. De hecho, sigue siendo el candidato más sólido en la clasificación
# final, con el mejor equilibrio entre detección de positivos reales y control
# de falsos positivos.
#
# Por tanto, xgb_a_1 no solo se considera el mejor XGBoost como modelo
# individual, sino también la opción más adecuada para representar la familia
# boosting dentro del stacking, al aportar una señal predictiva de gran
# calidad y con utilidad práctica demostrada en test.
#
# ---------------------------------------------
# 2. USO DE xgb_a_1 EN EL STACKING
# ---------------------------------------------
# Una vez seleccionado xgb_a_1 como mejor XGBoost del bloque individual, este
# mismo modelo se incorpora al stacking como representante de la familia
# boosting.
#
# La decisión es coherente con la lógica general del proyecto: si un modelo
# individual destaca por su capacidad discriminante y, además, por su calidad
# en la clasificación final, resulta razonable utilizar esa misma versión como
# modelo base dentro del ensemble, ya que aporta una señal predictiva fuerte y
# competitiva al meta-modelo.
#
# En este caso, xgb_a_1 combina:
# - AUC test muy alta: 0.9474
# - accuracy test más alta del bloque: 0.9322
# - precision y f1 de la clase positiva superiores al resto de candidatos
# - recall alto y estable: 0.9007
# - muy buen control de falsos positivos
#
# Por tanto, xgb_a_1 no solo se considera el mejor XGBoost como modelo
# individual, sino también la opción más adecuada para representar esta familia
# dentro del stacking, al aportar una señal predictiva de gran calidad y con
# utilidad práctica demostrada en test.
# ---------------------------------------------

param_xgb_final = {
    'n_estimators': 100,
    'learning_rate': 0.03,
    'gamma': 2,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 10
    }
modelo_xgb_final = XGBClassifier(
    booster='gbtree', objective='binary:logistic', eval_metric='logloss',
    tree_method='hist', **param_xgb_final, scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE, n_jobs=-1)

resultados_xgb_final = evaluar_modelo(
    modelo_xgb_final, nombre_xg, X_train, X_test, y_train, y_test,
    **param_xgb_final)

param_xgb_stack = param_xgb_final
modelo_xgb_stack = modelo_xgb_final
resultados_xgb_stack = resultados_xgb_final

matrices_confusion(y_test, resultados_xgb_final['y_pred_test'], nombre_xg)
heatmap_classification_reports(y_test, resultados_xgb_final['y_pred_test'], 
                               nombre_xg)

# =============================================================================
# 7. DESARROLLO STACKING
# =============================================================================
# -------------------------------------------------------------------------
# 7.1. ESTIMADORES BASE DE CANDIDATOS STACKING
# -------------------------------------------------------------------------
# CONFIGURACIÓN DE LOS CANDIDATOS STACKING
# - En este bloque se definen las combinaciones concretas de modelos base que
#   serán comparadas dentro del stacking.
# - stack_1 representa la opción principal del proyecto, al integrar los tres
#   modelos de referencia ya consolidados en bloques previos: SVM RBF,
#   Regresión Logística y Bagging Árbol.
# - stack_2 sustituye el componente arbóreo principal por Random Forest, con
#   el objetivo de comprobar si una variante arbórea alternativa, distinta del
#   bagging ya seleccionado, mejora el comportamiento conjunto del apilado.
# - stack_3 incorpora XGBoost como modelo de contraste adicional, para valorar
#   si una alternativa boosting aporta una mejora real frente a las
#   combinaciones más continuistas basadas en enfoques tipo bagging.
#
# En todos los casos se utiliza como meta-modelo una Regresión Logística L2
# con class_weight='balanced', al tratarse de una opción estable, interpretable
# y adecuada para combinar las salidas de los estimadores base.
#
# Inicialmente se fija passthrough=False para que la comparación entre
# candidatos se realice sobre una estructura homogénea, en la que el
# meta-modelo aprenda únicamente a partir de las predicciones generadas por
# los modelos base.
print('\nESTIMADORES BASE DE CANDIDATOS DE STACKING')
modelos_stack_1 = [(nombre_svm, svm_stack),
                   (nombre_log, log_stack),
                   (nombre_b_arbol, bag_arbol_stack)]

modelos_stack_2 = [(nombre_svm, svm_stack),
                   (nombre_log, log_stack),
                   (nombre_rf, modelo_rf_stack)]

modelos_stack_3 = [(nombre_svm, svm_stack),
                   (nombre_log, log_stack),
                   (nombre_xg, modelo_xgb_stack)]

# ---------------------------------------------
# 7.2. META-MODELO BASE
# ---------------------------------------------
# Se adopta una Regresión Logística como meta-modelo inicial del stacking por
# su carácter parsimonioso, estable e interpretable.
#
# En este contexto, el meta-modelo no trabaja directamente con las variables
# originales, sino con las salidas generadas por los modelos base, por lo que
# interesa una solución sencilla que aprenda a ponderar dichas predicciones sin
# añadir una complejidad excesiva al ensamblado.
#
# La elección de una penalización L2 permite regularizar el ajuste y reducir el
# riesgo de que el meta-modelo sobrerreaccione ante pequeñas diferencias entre
# estimadores base, favoreciendo así una combinación más robusta.
#
# Además, el uso de class_weight='balanced' resulta coherente con el objetivo
# del problema, ya que ayuda a no infravalorar la clase positiva en un contexto
# donde interesa mantener capacidad de detección de clientes con riesgo real de
# abandono.
#
# En definitiva, la Regresión Logística actúa aquí como un agregador lineal
# regularizado, adecuado para evaluar de forma limpia si la combinación de
# modelos base aporta una mejora real de generalización.
meta_base_stack = LogisticRegression(
    penalty='l2', C=1.0, solver='liblinear', class_weight='balanced',
    max_iter=5000, random_state=RANDOM_STATE)

# -------------------------------------------------------------------------
# 7.3. MODELOS DE LOS CANDIDATOS STACKING
# -------------------------------------------------------------------------
# Una vez fijados los estimadores base y el meta-modelo inicial, se construyen
# tres configuraciones de stacking que recogen las principales alternativas
# metodológicas del bloque:
# - una combinación principal con los modelos ya consolidados en el proyecto,
# - un contraste con Random Forest como variante arbórea alternativa,
# - y un contraste con XGBoost como representante de la familia boosting.
#
# El objetivo no es generar muchas combinaciones arbitrarias, sino comparar
# estructuras de stacking suficientemente diferenciadas como para evaluar la
# diversidad entre modelos base resulta más útil para el meta-modelo.

# stack_1: stacking principal del proyecto
stack_1 = StackingClassifier(
    estimators=modelos_stack_1, final_estimator=meta_base_stack,
    stack_method='predict_proba', passthrough=False, cv=cv, n_jobs=-1)

# stack_2: contraste añadiendo Random Forest
stack_2 = StackingClassifier(
    estimators=modelos_stack_2, final_estimator=meta_base_stack,
    stack_method='predict_proba', passthrough=False, cv=cv, n_jobs=-1)

# stack_3: contraste añadiendo XGBoost
stack_3 = StackingClassifier(
    estimators=modelos_stack_3, final_estimator=meta_base_stack,
    stack_method='predict_proba', passthrough=False, cv=cv, n_jobs=-1)

modelos_stack = {
    'Stacking 1': stack_1,
    'Stacking 2': stack_2,
    'Stacking 3': stack_3
    }

carac_stack = {}

for nombre, modelo in modelos_stack.items():
    print(f'\nAjuste inicial de {nombre}')
    modelo.fit(X_train, y_train)
    carac_stack[nombre] = caracteristicas_stacking(
        modelo, X_train, y_train, nombre)

print(carac_stack['Stacking 1']['info_meta'].round(4).to_string(index=False))
print(carac_stack['Stacking 2']['info_meta'].round(4).to_string(index=False))
print(carac_stack['Stacking 3']['info_meta'].round(4).to_string(index=False))

# -------------------------------------------------------------------------
# 7.4. AJUSTE DEL META-MODELO EN LOS STACKING
# -------------------------------------------------------------------------
# La inspección visual de las características metaaprendidas confirma que las
# salidas de los modelos base ya separan razonablemente bien ambas clases.
# SVM RBF y los modelos arbóreos concentran la mayor capacidad discriminante,
# mientras que la Regresión Logística aporta una señal más estable y lineal,
# complementaria dentro del ensamblado.
#
# Los gráficos sugieren relaciones mayoritariamente monótonas entre las
# predicciones base, por lo que no parece necesario introducir un meta-modelo
# más complejo en esta fase.
#
# Por ello, el ajuste en esta fase se limita al grado de regularización del
# meta-modelo, con el fin de comprobar qué configuración de stacking alcanza
# el mejor equilibrio entre AUC, accuracy y recall sin añadir complejidad
# innecesaria en la capa superior.
print('\nAJUSTE META-MODELO STACKING')
nombre_st = 'Stacking (R.Log.)'
parametros_meta = ['final_estimator__penalty', 'final_estimator__C',
                   'passthrough']
param_grid_meta = {
    'final_estimator__penalty': ['l1', 'l2'],
    'final_estimator__C': [0.1, 0.5, 1, 2, 5, 10],
    'passthrough': [False]
    }

resumen_grid_stack = {}
resumen_cv_stack = {}

for nombre, modelo in modelos_stack.items():
    print(f'\nAjuste {nombre}')
    grid_stack = GridSearchCV(
        estimator=modelo, param_grid=param_grid_meta, cv=cv, scoring=score,
        refit='roc_auc', return_train_score=True, verbose=3, n_jobs=-1)    
    grid_stack.fit(X_train, y_train)
    print(f'\nMejores parámetros {nombre}:')
    print(grid_stack.best_params_)
    print('Mejor AUC CV:', grid_stack.best_score_)
    
    df_cv_stack = pd.DataFrame(grid_stack.cv_results_)
    if 'param_final_estimator__penalty' in df_cv_stack.columns:
        df_cv_stack['param_final_estimator__penalty'] = df_cv_stack[
            'param_final_estimator__penalty'].fillna('None').astype(str)
    df_grid_stack = tabla_grid(df_cv_stack, parametros_meta)
   
    grafico_parametros(df_grid_stack, parametros_meta, score_grid, 
                       f'{nombre_st} - {nombre}')
    resumen_grid_stack[nombre] = df_grid_stack.copy()
    resumen_cv_stack[nombre] = df_cv_stack.copy()

df_grid_stack_completo = pd.concat(
    resumen_grid_stack, names=['modelo']).reset_index(level=0)
df_grid_stack_completo = df_grid_stack_completo.sort_values(
    'auc_cv', ascending=False).reset_index(drop=True)

# -------------------------------------------------------------------------
# 7.5. CANDIDATOS DEL META-MODELO STACKING
# -------------------------------------------------------------------------
# A partir del ajuste del meta-modelo se realiza una primera criba de 12
# candidatos finales, con el objetivo de reducir el bloque a una selección
# manejable sin perder representatividad de las zonas más competitivas del
# grid.
#
# Esta primera selección no busca todavía fijar el stacking ganador, sino
# conservar los perfiles que, por validación cruzada y por comportamiento
# preliminar en train/test, merecen una revisión manual más detallada.
#
# La lógica de la selección es la siguiente:
#
# - Stacking 3 recibe mayor peso en esta primera criba porque domina la parte
#   alta del ranking en AUC test y también ocupa una zona muy fuerte del ajuste
#   en validación cruzada. Se conservan cinco variantes, ya que es la familia
#   con mayor potencial discriminante. No obstante, también presenta el
#   sobreajuste más visible, por lo que interesa comprobar después si esa
#   ventaja en AUC compensa realmente en términos prácticos.
#
# - Stacking 2 se mantiene como bloque muy competitivo porque, aunque queda
#   ligeramente por detrás de Stacking 3 en AUC, ofrece una accuracy test más
#   sólida y un comportamiento muy estable en varias configuraciones. Ese
#   patrón sugiere un equilibrio muy robusto entre capacidad predictiva y
#   control del error, por lo que se incluyen cuatro variantes para contrastar
#   si esta familia puede acabar siendo preferible en la comparación final por
#   equilibrio global.
#
# - Stacking 1 se conserva solo con tres variantes, ya que queda algo por
#   detrás en rendimiento medio, pero sigue siendo un contraste importante por
#   su menor complejidad estructural y por mostrar un ajuste bastante más
#   contenido. Aun así, se mantienen sus configuraciones más competitivas como
#   referencia útil para comprobar si una solución más sencilla y parsimoniosa
#   puede seguir resultando competitiva en fases posteriores.
#
# Dentro de cada familia se han priorizado:
# - configuraciones situadas en la zona alta de AUC CV y AUC test,
# - perfiles con distinta penalización (L1 y L2),
# - valores de C representativos (medios y altos),
# - y combinaciones suficientemente diferenciadas entre sí para evitar
#   quedarnos solo con máximos puntuales demasiado redundantes.
#
# Así, la comparación posterior podrá centrarse en candidatos realmente serios,
# combinando:
# - capacidad discriminante,
# - estabilidad entre train y test,
# - calidad del ajuste,
# - y equilibrio global entre AUC, accuracy y recall.
#
# Con esta primera criba se preservan los perfiles realmente competitivos del
# bloque, de modo que la comparación posterior pueda centrarse ya en candidatos
# con opciones reales de convertirse en stacking final del proyecto.
# -------------------------------------------------------------------------
candidatos_stack = pd.DataFrame({
    'candidato': ['stk_1', 'stk_2', 'stk_3',
                  'stk_4', 'stk_5', 'stk_6',
                  'stk_7', 'stk_8', 'stk_9',
                  'stk_10', 'stk_11', 'stk_12'],
    
    'stacking': ['stack_1', 'stack_1', 'stack_2',
                 'stack_2', 'stack_2', 'stack_2',
                 'stack_2', 'stack_3', 'stack_3',
                 'stack_3', 'stack_3', 'stack_3'],
    
    'final_estimator__penalty': ['l1', 'l1', 'l2',
                                 'l1', 'l1', 'l2',
                                 'l2', 'l1', 'l2',
                                 'l2', 'l2', 'l2'],
    
    'final_estimator__C': [2.0, 5.0, 1.0,
                           2.0, 1.0, 0.5,
                           2.0, 0.5, 1.0,
                           0.1, 0.5, 2],
    
    'passthrough': [False] * 12
})

sel_stack_1 = candidatos_stack[
    candidatos_stack['stacking'] == 'stack_1'].merge(
        resumen_cv_stack['Stacking 1'], left_on=parametros_meta, right_on=[
            'param_final_estimator__penalty', 'param_final_estimator__C',
            'param_passthrough'],
        how='left')

sel_stack_2 = candidatos_stack[
    candidatos_stack['stacking'] == 'stack_2'].merge(
        resumen_cv_stack['Stacking 2'], left_on=parametros_meta, right_on=[
            'param_final_estimator__penalty', 'param_final_estimator__C',
            'param_passthrough'],
    how='left')

sel_stack_3 = candidatos_stack[
    candidatos_stack['stacking'] == 'stack_3'].merge(
        resumen_cv_stack['Stacking 3'], left_on=parametros_meta, right_on=[
            'param_final_estimator__penalty', 'param_final_estimator__C',
            'param_passthrough'],
    how='left')

seleccion_stack = pd.concat([sel_stack_1, sel_stack_2, sel_stack_3],
                            ignore_index=True)

# -------------------------------------------------------------------------
# 7.6. COMPARACIÓN DE LOS CANDIDATOS
# -------------------------------------------------------------------------
# En esta fase se comparan directamente los candidatos finales del bloque
# stacking sobre el conjunto de test.
#
# La finalidad ya no es observar únicamente su comportamiento medio en
# validación cruzada, sino comprobar cómo se traducen esas diferencias en una
# clasificación final real, atendiendo de forma conjunta a:
# - AUC y accuracy,
# - recall, precision y f1-score,
# - matrices de confusión,
# - curvas ROC,
# - y gaps train-test.
#
# Este análisis permite seleccionar el stacking final con una lógica coherente
# con la seguida en el resto del proyecto: primero se filtra por CV y después
# se confirma la decisión mediante evaluación directa fuera de muestra.

boxplot_candidatos(seleccion_stack, score, nombre_st)

param_stack_cand = candidatos_stack.copy()
param_stack_cand = list(param_stack_cand.itertuples(index=False,name=None))
predicciones_stack_cand = []
scores_stack_cand = []
resumen_stack_cand = []
nombres_stack_cand = []
modelos_stack_cand = {}

for cand, st, p, c, pt in param_stack_cand:
    meta_for = LogisticRegression(
        solver='liblinear', class_weight='balanced', max_iter=5000,
        random_state=RANDOM_STATE, penalty=p, C=c)
    if st == 'stack_1':
       modelo_for = StackingClassifier(
            estimators=modelos_stack_1, final_estimator=meta_for,
            stack_method='predict_proba', passthrough=pt, cv=cv, n_jobs=-1)
    elif st == 'stack_2':
        modelo_for = StackingClassifier(
            estimators=modelos_stack_2, final_estimator=meta_for,
            stack_method='predict_proba', passthrough=pt, cv=cv, n_jobs=-1)
    else:
        modelo_for = StackingClassifier(
            estimators=modelos_stack_3, final_estimator=meta_for,
            stack_method='predict_proba', passthrough=pt, cv=cv, n_jobs=-1)
    
    nombre_modelo = f'{nombre_st}: {cand}/{st}'
    resultados_modelo_for = evaluar_modelo(
        modelo_for, nombre_modelo, X_train, X_test, y_train, y_test,
        final_estimator__penalty=p, final_estimator__C=c,
        stack_method='predict_proba', passthrough=pt)    
    predicciones_stack_cand.append(resultados_modelo_for['y_pred_test'])
    scores_stack_cand.append(resultados_modelo_for['y_score_test'])
    nombres_stack_cand.append(nombre_modelo)
    resumen_stack_cand.append(resultados_modelo_for['resumen'])
    modelos_stack_cand[nombre_modelo] = modelo_for
    
df_compara_stack = pd.concat(resumen_stack_cand, ignore_index=True)

matrices_confusion(y_test, predicciones_stack_cand, nombres_stack_cand)
heatmap_classification_reports(y_test, predicciones_stack_cand,
                               nombres_stack_cand)

df_sobre_stack = sobreajuste(df_compara_stack)
print(df_sobre_stack.round(4).to_string(index=False))

grafico_comparacion_metricas(df_compara_stack, f'Candidatos {nombre_st}')

curva_auc_roc_comparativa(y_test, scores_stack_cand, nombres_stack_cand)

# -------------------------------------------------------------------------
# 7.7. SELECCIÓN FINAL DEL META-MODELO STACKING
# -------------------------------------------------------------------------
# En esta fase la decisión final se adopta de forma provisional a partir del
# análisis conjunto de métricas en test, matrices de confusión, curvas ROC y
# gaps train-test, quedando pendiente únicamente su contraste final mediante
# curvas de aprendizaje.
#
# La decisión se apoya en que, dentro de los 12 candidatos finales, es el
# modelo que ofrece el mejor equilibrio global entre capacidad discriminante,
# rendimiento en clasificación y utilidad práctica. En concreto, alcanza una
# AUC test de 0.9466, la más alta dentro del grupo realmente competitivo de
# stack_2 y stack_3, y la combina con una accuracy test de 0.9280, también
# superior a la de los mejores perfiles de stack_2. Además, mantiene valores
# muy sólidos de precision, f1, f1_macro, balanced_accuracy y specificity.
#
# Un aspecto especialmente relevante es que la comparación ya no depende solo
# de la AUC. En esta configuración final, stk_12/stack_3 no solo se sitúa en
# la zona alta en capacidad discriminante, sino que además traduce esa ventaja
# en un mejor rendimiento de clasificación sobre test. Por ello, la decisión
# final se apoya en un equilibrio más completo entre discriminación global,
# precisión y control del error.
#
# Frente a los candidatos más fuertes de stack_2, como stk_4/stack_2, la
# superioridad de stk_12/stack_3 es consistente. En concreto, mejora la
# AUC test (0.9466 frente a 0.9451), la accuracy test (0.9280 frente a 0.9266),
# la precision (0.7744 frente a 0.7697), el f1-score (0.8328 frente a 0.8301),
# el balanced_accuracy (0.9177 frente a 0.9168) y la specificity
# (0.9347 frente a 0.9330), manteniendo además el mismo recall de 0.9007.
# También presenta un sobreajuste más contenido, con gap_auc = 0.0171 frente
# a 0.0253 y gap_accuracy = 0.0041 frente a 0.0073.
#
# Dentro de la propia familia stack_3, la comparación final se concentra 
# en stk_12/stack_3, stk_9/stack_3 y stk_8/stack_3. Frente a stk_9/stack_3,
# la decisión es muy ajustada, ya que ambos mantienen prácticamente el mismo
# perfil de clasificación en test. Sin embargo, stk_12/stack_3 se impone por
# alcanzar una AUC test ligeramente superior (0.9466 frente a 0.9459),
# conservando el mismo recall, la misma accuracy test y un rendimiento global
# algo más favorable.
#
# Frente a stk_8/stack_3, la comparación resulta también muy ilustrativa.
# Aunque stk_8 presenta una precision algo más alta (0.7778), reduce el recall
# hasta 0.8936 y aumenta los falsos negativos a 15, algo menos conveniente en
# un problema donde interesa detectar correctamente a los clientes con riesgo
# de abandono. En cambio, stk_12/stack_3 mantiene recall = 0.9007, con solo
# 14 falsos negativos, y además logra una AUC test ligeramente superior
# (0.9466 frente a 0.9464).
#
# En consecuencia, stk_12/stack_3 se considera por ahora la alternativa más
# robusta y útil dentro del bloque stacking, al combinar una AUC muy alta con
# la mejor síntesis global entre accuracy, precision, f1-score, balanced
# accuracy, specificity y control razonable del sobreajuste.
#
# Para contrastar esta decisión con las curvas de aprendizaje, se seleccionan
# tres candidatos representativos, todos ellos pertenecientes a las familias
# stack_2 y stack_3, ya que stack_1 queda descartada en esta fase por mostrar
# un rendimiento global claramente inferior en accuracy, precision, f1 y
# métricas derivadas, pese a mantener una AUC competitiva.
#
# Los modelos elegidos para esta comparación son:
# - stk_12/stack_3: por ser el candidato provisionalmente seleccionado y el que
#   mejor resume el nuevo perfil ganador del bloque final.
# - stk_9/stack_3: por ser la alternativa más cercana dentro de la misma
#   familia, con un comportamiento prácticamente equivalente en clasificación
#   y una AUC test muy próxima.
# - stk_4/stack_2: por ser el mejor representante externo de stack_2 y servir
#   como contraste directo frente al liderazgo final de stack_3.
#
# Tras la comparación global de los candidatos finales, se realiza una última
# comprobación mediante curvas de aprendizaje sobre los perfiles más fuertes del
# bloque. El objetivo es verificar si la decisión provisional se mantiene al
# analizar la evolución del rendimiento y la estabilidad al aumentar el tamaño
# muestral.

modelos_stack_curva = {
    k: modelos_stack_cand[k]
    for k in [
        'Stacking (R.Log.): stk_12/stack_3',
        'Stacking (R.Log.): stk_9/stack_3',
        'Stacking (R.Log.): stk_4/stack_2'
    ]
}

curvas_selec_stack = curvas_aprendizaje(
    modelos_stack_curva, X_train, y_train, cv=cv, scoring='roc_auc')

# Las curvas de aprendizaje confirman que los tres modelos mejoran de forma
# clara al aumentar el tamaño de TRAIN y que, a partir de tamaños intermedios,
# entran en una zona de estabilización bastante definida.
#
# Entre ellos, stk_9/stack_3 presenta una ventaja muy ligera en estabilidad,
# ya que termina con el mayor score medio en test en la curva
# (0.9121 frente a 0.9119 de stk_12/stack_3) y con el menor gap final
# (0.0547 frente a 0.0555). No obstante, esa diferencia es mínima y no resulta
# suficiente para desplazar la decisión principal, ya que en el conjunto de
# test final stk_12/stack_3 sigue ofreciendo una AUC superior y el mismo
# rendimiento práctico en clasificación.
#
# Por su parte, stk_4/stack_2 muestra también una evolución favorable, pero
# termina con un score medio en test algo inferior y con un gap final más alto,
# por lo que queda un escalón por detrás de los dos mejores candidatos de
# stack_3.
#
# En consecuencia, las curvas de aprendizaje no modifican la decisión inicial,
# sino que la refuerzan: stk_12/stack_3 se mantiene como modelo final del
# bloque stacking, mientras que stk_9/stack_3 queda como alternativa muy
# próxima y especialmente destacable por su ligera ventaja en estabilidad.

params_meta_stacking = {
    'penalty': 'l2',
    'C': 2.0,
    'solver': 'liblinear',
    'class_weight': 'balanced'
    }
meta_stack_final = LogisticRegression(**params_meta_stacking, max_iter=5000,
                                      random_state=RANDOM_STATE)

params_modelo_stacking={
    'estimators': modelos_stack_3,
    'final_estimator': meta_stack_final,
    'stack_method': 'predict_proba',
    'passthrough': False
    }
modelo_stack_final = StackingClassifier(**params_modelo_stacking, cv=cv,
                                        n_jobs=-1)

resultados_stack_final = evaluar_modelo(
    modelo_stack_final, nombre_st, X_train, X_test,
    y_train, y_test, final_estimator__penalty=params_meta_stacking['penalty'],
    final_estimator__C=params_meta_stacking['C'], 
    passthrough=params_modelo_stacking['passthrough'])
print(resultados_stack_final['resumen'].round(4).to_string(index=False))

matrices_confusion(y_test, resultados_stack_final['y_pred_test'], nombre_st)
heatmap_classification_reports(y_test, resultados_stack_final['y_pred_test'],
                               nombre_st)

caracteristicas_stacking(modelo_stack_final, X_train, y_train, nombre_st)
sobreajuste(resultados_stack_final['resumen'])

# -------------------------------------------------------------------------
# 7.8. INTERPRETACIÓN FINAL DEL STACKING SELECCIONADO
# -------------------------------------------------------------------------
# El modelo final de stacking confirma un rendimiento global muy sólido.
# En el conjunto de test alcanza una AUC de 0.9467 y una accuracy de 0.9280,
# lo que lo sitúa en la zona alta del proyecto tanto en capacidad
# discriminante como en clasificación final.
#
# La matriz de confusión y el heatmap del classification report muestran que
# el modelo mantiene una capacidad alta para detectar la clase positiva sin
# deteriorar en exceso el comportamiento sobre la clase negativa, logrando un
# equilibrio competitivo entre recall, precision, f1-score y specificity.
#
# Desde el punto de vista del metaaprendizaje, el gráfico de características
# metaaprendidas permite comprobar que el meta-modelo está combinando señales
# realmente complementarias de los tres modelos base:
# - SVM RBF aporta una señal muy útil para identificar observaciones con alta
#   probabilidad de pertenecer a la clase positiva.
# - La Regresión Logística introduce una salida más progresiva y estable,
#   útil para ordenar casos intermedios.
# - XGBoost ofrece una separación especialmente informativa en la zona alta de
#   probabilidad, reforzando la discriminación en observaciones complejas.
#
# En conjunto, las tres salidas no son redundantes, sino complementarias, lo
# que justifica el uso del stacking: el meta-modelo no se limita a promediar,
# sino que aprende a combinar patrones distintos de decisión generados por los
# estimadores base.
#
# El análisis de sobreajuste también es favorable. Aunque existe una diferencia
# natural entre entrenamiento y test, esta se mantiene en niveles razonables:
# - gap_auc ≈ 0.0170
# - gap_accuracy ≈ 0.0045
#
# Esto sugiere que el modelo conserva una buena capacidad de generalización y
# que la mejora observada no se debe a un ajuste excesivamente optimista sobre
# TRAIN. En particular, el gap en accuracy es bajo, lo que refuerza la idea de
# un comportamiento estable en clasificación final.
#
# Por tanto, el stacking final puede considerarse un modelo robusto, bien
# calibrado desde el punto de vista práctico y coherente con el objetivo del
# problema, al combinar alta capacidad predictiva con un nivel de sobreajuste
# contenido y una integración efectiva de señales complementarias.
# =============================================================================
# 8. GUARDADO
# =============================================================================
print('\nGUARDADO ENSEMBLE')
bbdd_ensemble = {
    'nombre_bagging': nombre_b_arbol,
    'modelo_bagging': bagging_arbol_final,
    'parametros_bagging': param_bag_arbol_final,
    'estimador_bagging': {
        'modelo': modelo_arbol_est,
        'parametros': param_arbol_estimador
        },
    'resumen_bagging': resultados_bag_arbol_final['resumen'],  
    
    'nombre_stacking': nombre_st,
    'modelo_stacking': modelo_stack_final,
    'meta_modelo_stacking': meta_stack_final,
    'parametros_meta_modelo': params_meta_stacking,
    'parametros_stacking': params_modelo_stacking,
    'base_stacking':modelos_stack_3,
    'resumen_stacking': resultados_stack_final['resumen'],
    
    'nombre_xgboost': nombre_xg,
    'modelo_xgboost': modelo_xgb_final,
    'parametros_xgboost': param_xgb_final,
    'resumen_xgboost': resultados_xgb_final['resumen'],
    
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'columnas_svm': X_train.columns.tolist()
    }

with open(RUTA_SALIDA, 'wb') as archivo:
    pickle.dump(bbdd_ensemble, archivo)

print(f'\nArchivo guardado: {RUTA_SALIDA.resolve()}')
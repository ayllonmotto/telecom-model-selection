# -*- coding: utf-8 -*-
# =============================================================================
# 02_svm_ml.py
# Proyecto: Machine Learning 2026
#
# Objetivo:
# - Cargar la base revisada del Script 01.
# - Realizar la partición train/test.
# - Preparar los datos para el modelado con SVM.
# - Evaluar distintos kernels mediante búsqueda paramétrica.
# - Comparar manualmente candidatos representativos de cada kernel.
# - Seleccionar el mejor modelo SVM y realizar un ajuste adicional centrado en
#   la zona paramétrica más competitiva.
# - Guardar la información necesaria para los bloques posteriores de ensemble
#   y comparación global.
#
# Enfoque general:
# - Este script marca el paso desde el análisis exploratorio inicial al
#   modelado supervisado.
# - La lógica seguida no consiste en seleccionar automáticamente el mejor
#   estimador del GridSearchCV, sino en combinar validación cruzada,
#   comparación manual de candidatos, análisis de métricas, matrices de
#   confusión, curvas ROC y revisión del sobreajuste.
#
# Nota:
# - Se mantiene una única partición train/test para garantizar que la
#   comparación posterior con modelos de ensemble y con el bloque global se
#   realice sobre exactamente la misma base.
# - Como métricas principales se utilizan AUC y accuracy, por coherencia con
#   el enunciado y con el objetivo de comparar capacidad discriminante y
#   acierto global.
# - Se añade recall como tercera métrica estratégica, ya que en el problema
#   interesa detectar correctamente clientes con riesgo real de abandono y,
#   por tanto, reducir falsos negativos.
# =============================================================================
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     StratifiedKFold)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from funciones_ml import (
    RANDOM_STATE, v_cramer, heatmap_ranking_vars, tabla_grid, 
    grafico_parametros, heatmap_params, boxplot_candidatos, evaluar_modelo,
    curva_auc_roc, grafico_thresholds, matrices_confusion,
    heatmap_classification_reports, grafico_comparacion_metricas,
    curva_auc_roc_comparativa, sobreajuste)

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# RUTAS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RUTA_DATOS = BASE_DIR / 'outputs' / 'BBDD_eda.pickle'
RUTA_SALIDA = BASE_DIR / 'outputs' / 'BBDD_svm.pickle'

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        'No se encontró BBDD_eda.pickle. '
        'Ejecuta primero 01_depuracion_eda_ml.py.'
    )

# =============================================================================
# ANÁLISIS Y CONFIGURACIÓN DEL MODELO SVM
# =============================================================================
# En este bloque se recupera la base depurada en el Script 01 y se define el
# punto de partida del modelado SVM.
#
# La base cargada ya ha pasado por una primera fase de revisión estructural
# (renombrado, eliminación de duplicados, análisis de coherencia y primera
# selección de variables), por lo que aquí no se repite ese trabajo, sino que
# se continúa directamente con la preparación específica para el modelo.
#
# Además, en esta etapa se conservan todavía algunas variables que después no
# formarán parte del modelo final, porque resultan útiles para completar la
# imputación coherente de ciertos missings antes del recorte definitivo del
# conjunto de entrada.
# -----------------------------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------------------------
with open(RUTA_DATOS, 'rb') as archivo:
    bbdd_eda = pickle.load(archivo)
df = bbdd_eda['datos'].copy()
vars_cat_iniciales = bbdd_eda['vars_cat']
vars_num_iniciales = bbdd_eda['vars_num']

vars_modelo = vars_cat_iniciales + vars_num_iniciales + ['Y']
df = df[vars_modelo].copy()

print(df.head())
print('\nVariables categóricas iniciales:\n', vars_cat_iniciales)
print('\nVariables numéricas iniciales:\n', vars_num_iniciales)

# Se observa que la base recuperada contiene 4 variables categóricas iniciales,
# 15 variables numéricas y la variable objetivo Y. Entre las numéricas todavía
# aparecen las variables de minutos por tramo, que se mantienen en esta fase
# porque serán necesarias para completar la imputación previa al modelado SVM.

# =============================================================================
# 1. PARTICIÓN TRAIN / TEST
# =============================================================================
# En este primer bloque se fija la partición que servirá como referencia para
# el proyecto. A partir de este momento, cualquier decisión de selección,
# ajuste o comparación se apoyará siempre en esta misma división, con el fin de
# garantizar coherencia metodológica entre el modelo SVM, los ensembles y la
# comparación global final.
#
# La partición se realiza de forma estratificada respecto a Y para conservar en
# TRAIN y TEST la misma proporción de clientes con y sin riesgo de abandono
# observada en la base depurada. Esto resulta especialmente importante porque,
# tras eliminar duplicados, la variable objetivo deja de estar balanceada y
# pasa a presentar una distribución aproximadamente 80/20.
# -----------------------------------------------------------------------------
# 1.1. PARTICIÓN TRAIN / TEST
# -----------------------------------------------------------------------------
print('\nPARTICIÓN TRAIN / TEST')
X = df.drop(columns='Y')
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

print(f'\nDimensión X_train: {X_train.shape}')
print(f'Dimensión X_test: {X_test.shape}')
print('\nDistribución de Y en train:')
print(y_train.map({0: 'NO', 1: 'SI'}).value_counts(normalize=True))
print('\nDistribución de Y en test:')
print(y_test.map({0: 'NO', 1: 'SI'}).value_counts(normalize=True))

# La partición mantiene prácticamente intacta la distribución de la variable
# objetivo entre TRAIN y TEST. En ambos subconjuntos la clase mayoritaria
# representa alrededor del 80% y la minoritaria cerca del 20%, por lo que la
# muestra de evaluación sigue siendo representativa del problema real y permite
# comparar modelos sin introducir sesgos artificiales por desbalance distinto.
#
# Además, el tamaño de TRAIN (2828 observaciones) resulta suficiente para
# realizar validación cruzada y ajustes paramétricos con una base razonablemente
# estable, mientras que TEST (708 observaciones) conserva un volumen adecuado
# para valorar el rendimiento final de generalización.

# -----------------------------------------------------------------------------
# 1.2. REVISIÓN DE VARIABLES CATEGÓRICAS
# -----------------------------------------------------------------------------
# La revisión de variables categóricas se realiza únicamente sobre TRAIN.
# Esta decisión es clave desde el punto de vista metodológico, ya que evita
# utilizar información del conjunto de test para decidir qué variables entran
# o salen del modelo.
#
# En esta fase se evalúan tres aspectos:
# - complejidad estructural de cada variable (número de categorías y dummies),
# - fragmentación de la muestra en categorías poco frecuentes,
# - y asociación preliminar con la variable objetivo mediante V de Cramer.
#
# El objetivo no es hacer una selección automática, sino confirmar si las
# conclusiones obtenidas en el análisis exploratorio del Script 01 se mantienen
# cuando se observa únicamente el conjunto de entrenamiento.

print('\nRevisión variables categóricas en TRAIN')
cat_train = []

for var in vars_cat_iniciales:
    frec_abs = X_train[var].value_counts(dropna=False)
    frec_rel = X_train[var].value_counts(normalize=True, dropna=False)
    n_categorias = X_train[var].nunique(dropna=True)
    n_dummies = max(n_categorias - 1, 0)
    pct_obs_cat_raras = frec_rel[frec_rel < 0.05].sum()*100
    v_y = v_cramer(X_train[var], y_train)

    cat_train.append({
        'variable': var,
        'n_categorias': n_categorias,
        'n_dummies': n_dummies,
        'freq_min_abs': frec_abs.min(),
        'freq_max_abs': frec_abs.max(),
        'pct_obs_cat_<5%': round(pct_obs_cat_raras, 2),
        'v_cramer_y': round(v_y, 4) if pd.notna(v_y) else np.nan
        })
    
df_cat_train = pd.DataFrame(cat_train).sort_values(
    ['n_dummies', 'pct_obs_cat_<5%'], 
    ascending=[False, False]).reset_index(drop=True)

print('\nResumen TRAIN de variables categóricas:')
print(df_cat_train)

# El comportamiento observado en TRAIN confirma lo ya detectado en el análisis
# previo:
# - region mantiene una cardinalidad muy alta (51 categorías), obligaría a
#   generar un número elevado de dummies y fragmenta completamente la muestra,
#   ya que el 100% de sus observaciones queda en categorías con peso inferior
#   al 5%. Aunque su asociación con Y no es nula, su coste estructural resulta
#   excesivo para un modelo SVM de este tamaño.
# - cod_area no presenta un problema de cardinalidad, pero su asociación con Y
#   es prácticamente inexistente en TRAIN (V de Cramer ≈ 0.006), por lo que su
#   aportación potencial al modelo parece muy limitada.
# - plan_inter es la variable categórica con señal más clara respecto a la
#   variable objetivo, por lo que debe conservarse.
# - buzon_voz muestra una asociación más modesta, pero sigue aportando cierta
#   información y no introduce complejidad estructural relevante.
#
# En consecuencia, se mantiene la misma decisión de trabajo que en el Script 01:
# conservar únicamente las variables categóricas binarias con mejor equilibrio
# entre señal y parsimonia, descartando las que añaden complejidad excesiva o
# escasa utilidad predictiva.
#
# Decisión variables en TRAIN:
# - region se descarta por alta cardinalidad y fuerte fragmentación.
# - cod_area se descarta por señal prácticamente nula con Y.

vars_cat_svm = ['plan_inter', 'buzon_voz']
print('\nVariables categóricas finales para SVM:')
print(vars_cat_svm)

# -----------------------------------------------------------------------------
# 1.3. REVISIÓN DE VARIABLES NUMÉRICAS
# -----------------------------------------------------------------------------
# En las variables numéricas aparece un caso particular relevante: algunas
# observaciones presentan valores perdidos en costes o minutos dentro de un
# mismo tramo horario, pese a que ambas magnitudes mantienen una relación casi
# determinista a través de la tarifa por minuto.
#
# En lugar de imputar directamente con medias o medianas generales, se opta por
# una reconstrucción más coherente desde el punto de vista del negocio:
# estimar la tarifa media por minuto en TRAIN para cada tramo y utilizarla para
# reconstruir los valores ausentes de coste o de minutos cuando la otra magnitud
# sí está disponible.
#
# Esta estrategia permite:
# - aprovechar una relación estructural real de los datos,
# - mantener coherencia entre minutos y costes,
# - y evitar fuga de información, ya que la tarifa se aprende exclusivamente en
#   TRAIN y después se aplica tanto a TRAIN como a TEST.
#
# Las variables de minutos se conservan temporalmente desde el script de
# depuración porque resultan necesarias para esta imputación dirigida. Una vez
# completado este paso, el modelo SVM se ajustará únicamente con el conjunto
# final de variables candidatas definido tras el análisis exploratorio.

vars_num_svm = vars_num_iniciales.copy()
print('\nVariables numéricas para SVM:')
print(vars_num_svm)

tramos_precio = bbdd_eda['tramos_precio']
filas_tarifas_train = []

for t, m, coste in tramos_precio:
    tarifa_train = (X_train[coste]/X_train[m].replace(0, np.nan))
    tarifa_train = tarifa_train.replace([np.inf, -np.inf], np.nan).mean()

    # TRAIN: imputar coste a partir de minutos
    id_coste_train = X_train[X_train[coste].isna() & X_train[m].notna()].index
    X_train.loc[id_coste_train, coste] = (
        X_train.loc[id_coste_train, m]*tarifa_train).round(2)

    # TRAIN: imputar minutos a partir de coste
    id_min_train = X_train[X_train[m].isna() & X_train[coste].notna()].index
    X_train.loc[id_min_train, m] = (
        X_train.loc[id_min_train, coste]/tarifa_train).round(1)

    # TEST: aplicar la tarifa aprendida en TRAIN
    id_coste_test = X_test[X_test[coste].isna() & X_test[m].notna()].index
    X_test.loc[id_coste_test, coste] = (
        X_test.loc[id_coste_test, m]*tarifa_train).round(2)

    id_min_test = X_test[X_test[m].isna() & X_test[coste].notna()].index
    X_test.loc[id_min_test, m] = (
        X_test.loc[id_min_test, coste]/tarifa_train).round(1)

    filas_tarifas_train.append({
        'tramo': t,
        'tarifa_train': tarifa_train,
        'costes_imputados_train': len(id_coste_train),
        'minutos_imputados_train': len(id_min_train),
        'costes_imputados_test': len(id_coste_test),
        'minutos_imputados_test': len(id_min_test)
        })

df_tarifas_train = pd.DataFrame(filas_tarifas_train)
print('\nResumen imputación por tarifa aprendida en TRAIN:')
print(df_tarifas_train)

# Los resultados confirman que la relación coste/minuto es muy estable también
# en TRAIN, ya que las tarifas estimadas por tramo quedan prácticamente fijadas:
# aproximadamente 0.17 en día, 0.085 en tarde, 0.045 en noche y 0.27 en
# internacional.
#
# Además, el volumen de imputaciones necesarias es reducido, lo que refuerza la
# idea de que no se está alterando de forma sustancial la estructura del
# conjunto, sino corrigiendo un número pequeño de ausencias de forma coherente
# con la lógica tarifaria del problema.
#
# En consecuencia, este paso actúa como una depuración dirigida y no como una
# transformación agresiva del dataset.
#
# Tras la imputación, las variables de minutos ya no forman parte del modelo.
# A partir de este punto, las variables numéricas candidatas/finales para SVM
# son las definidas en el script 1, que excluyen min_dia, min_tarde,
# min_noche y min_inter.

vars_num_svm = bbdd_eda['vars_num_cand'].copy()
print('\nVariables numéricas finales para SVM:')
print(vars_num_svm)

# Una vez utilizada la información de minutos para imputar, se recorta el
# dataset al conjunto final de variables del modelo SVM, formado por las dos
# variables categóricas seleccionadas y las variables numéricas candidatas
# depuradas en el Script 01.

columnas_svm = vars_cat_svm + vars_num_svm
X_train = X_train[columnas_svm].copy()
X_test = X_test[columnas_svm].copy()

# -----------------------------------------------------------------------------
# 1.4. PROCESADO DE DATOS PARA SVM
# -----------------------------------------------------------------------------
# Una vez fijadas las variables finales del modelo, se construye el bloque de
# preprocesado específico para SVM.
#
# Este paso es especialmente importante porque SVM es sensible a la escala de
# las variables y, además, no admite valores perdidos. Por ello, antes del
# ajuste se realiza:
# - imputación de variables numéricas,
# - imputación de variables categóricas,
# - escalado robusto de las numéricas,
# - y codificación One-Hot de las categóricas.
#
# Todo el proceso se ajusta en TRAIN y después se aplica a TEST, manteniendo así
# la coherencia metodológica y evitando fuga de información.

print('\nPROCESADO DE DATOS PARA SVM')
X_train_num = X_train[vars_num_svm].copy()
X_test_num = X_test[vars_num_svm].copy()

X_train_cat = X_train[vars_cat_svm].copy()
X_test_cat = X_test[vars_cat_svm].copy()

# 1. IMPUTACIÓN NUMÉRICAS
# Se utiliza IterativeImputer porque permite imputar cada variable numérica
# incompleta a partir de la información del resto, captando mejor la estructura
# conjunta de los datos que una imputación simple por media o mediana.
#
# Esta elección resulta razonable en este problema, donde varias variables
# numéricas presentan relaciones funcionales o dependencias claras entre sí.

imp_num = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)
X_train_num_imp = pd.DataFrame(imp_num.fit_transform(X_train_num), 
                           columns=vars_num_svm, index=X_train.index)
X_test_num_imp = pd.DataFrame(imp_num.transform(X_test_num), 
                          columns=vars_num_svm, index=X_test.index)

# 2. IMPUTACIÓN CATEGÓRICAS
# En las variables categóricas se aplica imputación por la moda antes de la
# codificación, ya que es una solución simple, estable y suficiente dado el bajo
# nivel de complejidad de estas variables tras la depuración previa.

imp_cat = SimpleImputer(strategy='most_frequent')
X_train_cat_imp = imp_cat.fit_transform(X_train_cat)
X_test_cat_imp = imp_cat.transform(X_test_cat)

# 3. DATAFRAME PREVIO AL COLUMNTRANSFORMER
# Se reconstruyen los conjuntos imputados en formato DataFrame para conservar
# nombres de columnas e índices antes de aplicar la transformación conjunta.

X_train_trf = pd.concat([
    pd.DataFrame(X_train_num_imp, columns=vars_num_svm, index=X_train.index),
    pd.DataFrame(X_train_cat_imp, columns=vars_cat_svm, index=X_train.index)],
    axis=1)

X_test_trf = pd.concat([
    pd.DataFrame(X_test_num_imp, columns=vars_num_svm, index=X_test.index),
    pd.DataFrame(X_test_cat_imp, columns=vars_cat_svm, index=X_test.index)],
    axis=1)

# 4. PROCESADO DE TRANSFORMACIÓN
# Se define el tratamiento específico de cada bloque de variables:
# - RobustScaler en numéricas, para reducir el impacto de posibles valores
#   extremos sin perder la comparabilidad de escalas.
# - OneHotEncoder en categóricas, generando dummies a partir de las categorías
#   observadas en TRAIN.
#
# Se utiliza drop='first' para evitar redundancia en variables binarias y
# handle_unknown='ignore' para asegurar que TEST pueda transformarse sin error
# aunque apareciera alguna categoría no observada en entrenamiento.

procesado = ColumnTransformer(
    transformers=[
        ('scaler', RobustScaler(), vars_num_svm),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False,
                              handle_unknown='ignore'), vars_cat_svm)],
    remainder='passthrough')

# 5. TRANSFORMACIÓN DATOS
X_train_proces = procesado.fit_transform(X_train_trf)
X_test_proces = procesado.transform(X_test_trf)

# 6. COLUMNAS
cat_ohe = procesado.named_transformers_['ohe'].get_feature_names_out(
    vars_cat_svm)
columnas_finales = vars_num_svm + list(cat_ohe)

# 7. DATOS FINALES
X_train_svm = pd.DataFrame(X_train_proces, columns=columnas_finales,
                           index=X_train.index)
X_test_svm = pd.DataFrame(X_test_proces, columns=columnas_finales,
                          index=X_test.index)

print(f'\nNúmero de variables numéricas: {len(vars_num_svm)}')
print(f'Número de dummies creadas: {len(cat_ohe)}')
print(f'Dimensión - X_train:{X_train_svm.shape} | X_test_svm:{X_test_svm.shape}')

# Comprobación final de missings antes del modelado
print('\nMissings tras imputación y transformación:')
print('\nX_train_svm:')
print(X_train_svm.isna().sum())
print('\nX_test_svm:')
print(X_test_svm.isna().sum())

# El resultado final confirma que el preprocesado ha dejado los datos listos
# para el ajuste de SVM:
# - no quedan valores perdidos,
# - las variables numéricas quedan escaladas de forma robusta,
# - y las categóricas se incorporan mediante dos dummies binarias.
#
# En consecuencia, el conjunto final de modelado queda formado por 13 variables:
# 11 numéricas transformadas y 2 variables indicadoras derivadas de
# plan_inter y buzon_voz.

# =============================================================================
# 2. CONFIGURACIÓN INICIAL
# =============================================================================
# ---------------------------------------------
# 2.1. CONFIGURACIÓN GENERAL DE VALIDACIÓN
# ---------------------------------------------
# Se utiliza validación cruzada estratificada para conservar en cada partición
# la misma proporción de la variable objetivo observada en la muestra. Esto es
# especialmente importante en este problema, ya que tras la depuración la clase
# positiva queda claramente menos representada que la negativa.
#
# De este modo, la comparación entre configuraciones SVM se realiza en un
# esquema de evaluación homogéneo y más estable, reduciendo el riesgo de que
# una partición concreta condicione en exceso los resultados.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------
# 2.2. MEDIDAS DE BONDAD
# ---------------------------------------------
# La evaluación del modelo no se basa en una única métrica, sino en tres
# medidas complementarias que permiten valorar el problema desde perspectivas
# distintas.
#
# Además, esta elección está especialmente justificada porque, tras la
# depuración realizada en el script anterior, la distribución de la variable
# objetivo deja de estar equilibrada y la clase positiva pasa a ser claramente
# minoritaria. Por ello, no resulta suficiente utilizar solo una medida global
# de acierto, ya que un modelo podría aparentar buen rendimiento simplemente por
# clasificar correctamente la clase mayoritaria.
#
# 1) ROC_AUC
#    Se utiliza como medida principal de capacidad discriminante. Evalúa hasta
#    qué punto el modelo es capaz de separar correctamente ambas clases a lo
#    largo de todos los posibles umbrales de decisión, y no solo en un punto de
#    corte fijo.
#
#    Su uso es especialmente adecuado en este problema porque permite comparar
#    modelos atendiendo a la calidad de la ordenación de probabilidades incluso
#    cuando existe cierto desbalanceo entre clases.
#
# 2) ACCURACY
#    Se mantiene como medida global de acierto, ya que resume la proporción de
#    observaciones correctamente clasificadas sobre el total.
#
#    No obstante, su interpretación debe hacerse con cautela, precisamente
#    porque la clase positiva es minoritaria tras la depuración. Por sí sola no
#    basta para seleccionar el mejor modelo, pero sí aporta una referencia útil
#    sobre el comportamiento global del clasificador.
#
# 3) RECALL
#    Se incorpora como tercera métrica estratégica, ya que el interés del
#    problema no es solo acertar mucho en promedio, sino detectar correctamente
#    los casos positivos.
#
#    El recall mide qué proporción de positivos reales identifica el modelo.
#    Por tanto, resulta especialmente relevante cuando interesa reducir falsos
#    negativos, es decir, casos realmente positivos que el modelo clasifica como
#    negativos.
#
# En conjunto:
# - ROC_AUC valora la capacidad general de separación entre clases.
# - ACCURACY resume el acierto total del clasificador.
# - RECALL refuerza la evaluación específica de la clase positiva.
#
# Esta combinación permite comparar los modelos con una visión más completa y
# coherente con la estructura real del problema y con la distribución final de
# la variable objetivo tras la depuración.

score = ['roc_auc', 'accuracy', 'recall']
score_grid = ['auc_cv', 'accuracy_cv', 'recall_cv']
# ---------------------------------------------
# 2.3. REVISIÓN DE VARIABLES EN TRAIN
# ---------------------------------------------
# Antes de iniciar el ajuste de los distintos kernels SVM, se realiza una
# revisión adicional de las variables finales utilizando solo TRAIN.
#
# Este bloque no redefine la depuración realizada en el script anterior, sino
# que actúa como comprobación de consistencia: permite verificar si las
# variables retenidas siguen mostrando señal relevante bajo distintos criterios
# de selección.
#
# Para ello se combinan tres enfoques complementarios:
# - Random Forest, como referencia de importancia no lineal;
# - Regresión Logística con penalización L1, como criterio de selección
#   escasa y regularizada;
# - SelectKBest con ANOVA F, como medida univariante de asociación con la
#   variable objetivo.
#
# Dado que el conjunto final de variables ya es relativamente reducido, esta
# revisión no busca necesariamente recortar más dimensiones, sino comprobar la
# estabilidad del bloque seleccionado y detectar si alguna variable quedara
# claramente desalineada respecto al resto.

# 1. RandomForest
rf_sel = RandomForestClassifier(
    n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced')
rf_sel.fit(X_train_svm, y_train)
df_rf = pd.DataFrame({
    'var': X_train_svm.columns,
    'imp_rf': rf_sel.feature_importances_
    }).sort_values('imp_rf', ascending=False).fillna(0)

top_rf = df_rf.head(12)['var'].tolist()
df_rf['top_rf'] = df_rf['var'].isin(top_rf)

# 2. Regresión Logística (L1)
rlog_sel = LogisticRegression(
    penalty='l1', solver='liblinear', class_weight='balanced', max_iter=5000,
    random_state=RANDOM_STATE)
rlog_sel.fit(X_train_svm, y_train)
coef_rl = np.abs(rlog_sel.coef_[0])
df_rl = pd.DataFrame({
    'var': X_train_svm.columns,
    'coef_l1': coef_rl,
    'selec_rl': coef_rl > 1e-5
    }).assign(
        coef_l1=lambda x: x['coef_l1'].fillna(0),
        selec_rl=lambda x: x['selec_rl'].fillna(False)
        ).sort_values('coef_l1', ascending=False)

# 3. SelectKBest (f y m_i)
kb_selec = 12
# F Clasificación
kb_f = SelectKBest(score_func=f_classif, k=kb_selec)
kb_f.fit(X_train_svm, y_train)
df_kbest_f = pd.DataFrame({
    'var': X_train_svm.columns, 
    'score_f': kb_f.scores_,
    'selec_f': kb_f.get_support()}
    ).assign(score_f=lambda x: x['score_f'].fillna(0),
             selec_f=lambda x: x['selec_f'].fillna(False)
             ).sort_values('score_f', ascending=False)

# 4. Resumen de variables
df_revision_vars = df_rf.copy()
df_revision_vars = df_revision_vars.merge(
    df_rl[['var', 'coef_l1', 'selec_rl']], on='var', how='left')
df_revision_vars = df_revision_vars.merge(
    df_kbest_f[['var', 'score_f', 'selec_f']], on='var', how='left')

df_revision_vars['n_metodos'] = (df_revision_vars['top_rf'].astype(int) +
                                 df_revision_vars['selec_rl'].astype(int) +
                                 df_revision_vars['selec_f'].astype(int))

df_revision_vars = df_revision_vars.sort_values(
    ['n_metodos', 'imp_rf', 'coef_l1', 'score_f'],
    ascending=[False, False, False, False])

print('\nRevisión de variables en TRAIN:')
print(df_revision_vars)
heatmap_ranking_vars(df_revision_vars)

# La revisión muestra una señal bastante coherente entre métodos.
# La mayoría de variables seleccionadas aparecen respaldadas por los tres
# criterios utilizados, lo que refuerza la solidez del conjunto final para el
# ajuste del SVM.
#
# Destacan especialmente coste_dia, n_att_cl y plan_inter_1, que ocupan
# posiciones altas de forma consistente. En cambio, variables como n_noche o
# buzon_voz_1 presentan un respaldo algo menor, aunque no lo bastante débil
# como para justificar su eliminación automática en esta fase.
#
# Por tanto, no se realiza una reducción adicional de variables antes del
# modelado. Se mantiene el bloque actual y se deja que la comparación entre
# kernels y configuraciones determine si alguna simplificación posterior resulta
# realmente ventajosa.

# =============================================================================
# 3. SVM LINEAL
# =============================================================================
# El kernel lineal se incorpora como primer punto de referencia dentro del
# bloque SVM por su mayor simplicidad estructural y por ofrecer una solución
# fácilmente interpretable en comparación con kernels más flexibles.
#
# Su interés en este proyecto es doble:
# - servir como modelo base de comparación frente a alternativas no lineales;
# - comprobar hasta qué punto una frontera lineal puede capturar la separación
#   entre clases con el conjunto de variables finalmente seleccionado.
#
# El ajuste se realiza sobre el parámetro C, que controla el grado de
# regularización, y sobre class_weight, con el fin de comprobar si compensar el
# desbalanceo de la clase positiva mejora la capacidad de detección del modelo.

print('\nSVM LINEAL')
nombre_l = 'SVM lineal'
parametros_lineal = ['C', 'class_weight']

param_grid_lineal = {
    'C': [0.01, 0.1, 0.5, 1, 5, 10],
    'class_weight': [None,'balanced']
    }
svm_lineal = SVC(kernel='linear')

grid_lineal = GridSearchCV(
    svm_lineal, param_grid_lineal, cv=cv, scoring=score, refit='roc_auc',
    verbose=3)
grid_lineal.fit(X_train_svm, y_train)
print('\nMejores parámetros por GridSearchCV:')
print(grid_lineal.best_params_)
print('Mejor AUC CV:', grid_lineal.best_score_)

# Resultados completos y ordenados del GridSearchCV
df_cv_lineal = pd.DataFrame(grid_lineal.cv_results_)
df_grid_lineal = tabla_grid(df_cv_lineal, parametros_lineal)
print(df_grid_lineal.to_string(index=False))

# Gráfico general del comportamiento de C
# La evolución de las métricas confirma dos ideas principales:
# 1) las configuraciones sin ponderación de clases ofrecen una accuracy alta,
#    pero a costa de un recall claramente insuficiente, lo que indica una mala
#    detección de la clase positiva;
# 2) al activar class_weight='balanced', el modelo mejora de forma muy clara su
#    utilidad práctica, elevando el recall hasta la zona 0.77-0.79 sin perder
#    en exceso capacidad discriminante global.
#
# Además, dentro del bloque balanced, el comportamiento es bastante estable para
# valores de C entre 0.1 y 10, lo que sugiere que el modelo lineal no depende
# de un ajuste extremo del parámetro para situarse en su zona alta
# de rendimiento.
grafico_parametros(df_grid_lineal, parametros_lineal, score_grid, nombre_l)

# ---------------------------------------------
# 3.1. CANDIDATOS SVM LINEAL
# ---------------------------------------------
# A partir del grid se conservan únicamente configuraciones con
# class_weight='balanced', ya que las alternativas sin ponderación quedan
# descartadas por su bajo recall, incompatible con el objetivo del problema.
#
# Se seleccionan seis candidatos con distintos valores de C para comparar de
# forma manual si la mejora del modelo lineal procede de un punto concreto del
# grid o de una zona amplia y estable del parámetro.
#
# La lógica de esta criba es:
# - incluir el mejor valor de AUC CV del grid (C=10),
# - mantener valores próximos con rendimiento casi equivalente (C=5 y C=1),
# - conservar opciones algo más regularizadas (C=0.5 y C=0.1),
# - y dejar C=0.01 como extremo inferior para comprobar hasta qué punto una
#   regularización demasiado fuerte deteriora el ajuste.
candidatos_lineal = pd.DataFrame({
    'candidato': ['lineal_1', 'lineal_2', 'lineal_3',
                  'lineal_4', 'lineal_5', 'lineal_6'],
    'C': [0.1, 10, 5, 0.5, 1, 0.01],
    'class_weight': ['balanced']*6
    })
seleccion_lineal = candidatos_lineal.merge(
    df_cv_lineal, left_on=parametros_lineal, right_on=[
        'param_C', 'param_class_weight'], how='left')

# Los boxplots permiten comprobar que, salvo el caso más regularizado
# (C=0.01), el bloque balanced presenta un comportamiento muy homogéneo en AUC,
# accuracy y recall. No aparece un candidato claramente dominante, sino una
# meseta de soluciones muy próximas entre sí.
#
# Esta lectura es importante porque evita sobrerreaccionar ante diferencias
# mínimas de validación cruzada y permite tomar la decisión final con un
# criterio más robusto, apoyado también en parsimonia.

boxplot_candidatos(seleccion_lineal, score, nombre_l)

# ---------------------------------------------
# 3.2. EVALUACIÓN SVM LINEAL
# ---------------------------------------------
# Aunque el GridSearchCV señala C=10 como mejor combinación en términos de
# AUC CV, la diferencia respecto a C=5 es prácticamente despreciable
# (0.825107 frente a 0.824943), manteniendo además ambas configuraciones el
# mismo recall CV y una accuracy CV prácticamente idéntica.
#
# Por ello, se selecciona C=5 con class_weight='balanced' como candidato final
# del kernel lineal. La elección no se basa en maximizar una cifra aislada,
# sino en priorizar una solución algo más regularizada que C=10, situada en la
# misma meseta alta de rendimiento y, por tanto, más razonable desde el punto
# de vista de parsimonia.
#
# También se descarta C=0.1, pese a presentar el mejor equilibrio medio entre
# métricas, porque su ligera mejora en recall no compensa una reducción de AUC
# y no representa con tanta claridad la zona central más estable del ajuste.
#
# En consecuencia, el modelo lineal final se define como una referencia sólida,
# sencilla y competitiva dentro del bloque SVM, aunque todavía pendiente de ser
# comparada posteriormente con kernels más flexibles.
# Elección tras grid y boxplots (similares C más sencilla)-Por parsimonia C=5
params_lineal = {
    'C': 5,
    'class_weight': 'balanced'
    }
# Modelado
modelo_lineal = SVC(kernel='linear', **params_lineal)
# Función para modelar, predicciones, score y resumen:
resultados_lineal = evaluar_modelo(
    modelo_lineal, nombre_l, X_train_svm, X_test_svm, y_train, y_test,
    **params_lineal)
# Resumen, reporte, matriz de confusión y sobreajuste
print(f'\nResumen - {nombre_l}:')
print(resultados_lineal['resumen'].round(4))
print(f'\nClassification report - {nombre_l}')
print(classification_report(
    y_test, resultados_lineal['y_pred_test'], target_names=['NO','SI'],
    zero_division=0))

matrices_confusion(
    y_test, resultados_lineal['y_pred_test'], f'{nombre_l}')
df_sobre_lineal = sobreajuste(resultados_lineal['resumen'])
print(df_sobre_lineal.round(4).to_string(index=False))

# La evaluación en test confirma que el kernel lineal es capaz de captar parte
# de la señal del problema, pero su rendimiento sigue siendo moderado para los
# objetivos del proyecto.
#
# En términos de capacidad discriminante, el modelo alcanza una AUC test de
# 0.8459, superior a su AUC train (0.8304). Este gap_auc negativo no se
# interpreta como un problema, sino como un resultado compatible con la
# variabilidad muestral de la partición y con el efecto regularizador del
# modelo lineal, que evita un ajuste excesivo sobre TRAIN.
#
# Sin embargo, la lectura operativa del modelo es menos favorable. Aunque el
# recall de la clase positiva es razonable (0.7801), el modelo genera un número
# muy elevado de falsos positivos (140), lo que reduce claramente la precision
# de la clase positiva (0.4400) y deja el f1-score en un nivel discreto
# (0.5627). En otras palabras, el modelo detecta una parte importante de los
# casos positivos, pero lo hace a costa de lanzar demasiadas falsas alarmas.
#
# La matriz de confusión refuerza esta idea: mantiene 31 falsos negativos,
# cifra aceptable para un modelo base, pero el volumen de errores sobre la clase
# negativa sigue siendo demasiado alto. Esto explica que la accuracy test
# quede en 0.7585 y que el comportamiento global del clasificador sea todavía
# insuficiente frente a lo que se espera de un modelo final.
#
# En conjunto, el SVM lineal debe interpretarse como una referencia sencilla y
# metodológicamente útil dentro del bloque SVM: ofrece una primera frontera de
# separación razonable y un ajuste estable, pero su capacidad para representar
# la estructura real del problema parece limitada. Por ello, su principal valor
# en el proyecto es actuar como benchmark frente a kernels más flexibles, que
# deberían mejorar especialmente la precision y el equilibrio global del modelo.

# ---------------------------------------------
# 3.3. CURVA ROC Y ANÁLISIS DE THRESHOLDS
# ---------------------------------------------
# Una vez fijado el modelo lineal, se analiza su comportamiento para distintos
# puntos de corte sobre la puntuación de decisión.
#
# Este análisis permite comprobar si el threshold estándar del clasificador
# (equivalente a 0 en la función de decisión) es el más adecuado desde el
# punto de vista operativo, o si existe otro punto de corte que mejore el
# equilibrio entre sensibilidad y especificidad.
#
# La curva ROC resume la capacidad discriminante del modelo a lo largo de todos
# los thresholds posibles. A partir de ella se calcula el índice de Youden,
# definido como sensibilidad + especificidad - 1, que permite localizar el
# punto de corte que maximiza conjuntamente ambas magnitudes.
#
# En este proyecto, este análisis no se utiliza para reentrenar el modelo, sino
# para estudiar cómo cambia el compromiso entre aciertos y errores al modificar
# el umbral de clasificación, especialmente sobre la clase positiva.

info_roc_l = curva_auc_roc(y_test, resultados_lineal['y_score_test'], nombre_l)
thr_youden_l = info_roc_l['youden']['threshold']
youden_l = info_roc_l['youden']['youden']

print(f'\nThreshold óptimo por Youden - {nombre_l}: {thr_youden_l:.4f}')
print(f'Índice de Youden máximo: {youden_l:.4f}')
print(f'Threshold estándar: {info_roc_l["estandar"]["threshold"]:.4f}')

# El threshold óptimo de Youden resulta más bajo que el threshold estándar.
# Esto implica que el modelo será más proclive a clasificar observaciones como
# clase positiva, lo que previsiblemente aumentará el recall pero también el
# número de falsos positivos.
#
# Por tanto, el análisis posterior permitirá comprobar si esta ganancia en
# sensibilidad compensa o no la pérdida de precisión y especificidad desde una
# perspectiva operativa.

grafico_thresholds(
    y_test,resultados_lineal['y_score_test'], nombre_l, thr_youden_l)

y_pred_youden_lineal = (
    resultados_lineal['y_score_test'] >= thr_youden_l).astype(int)

heatmap_classification_reports(
    y_test,[resultados_lineal['y_pred_test'], y_pred_youden_lineal],
    [nombre_l, f'{nombre_l}\n(thr Youden={thr_youden_l:.4f})'])
matrices_confusion(
    y_test,[resultados_lineal['y_pred_test'], y_pred_youden_lineal],
    [nombre_l, f'{nombre_l}\n(thr Youden={thr_youden_l:.4f})'])

# La comparación entre el threshold estándar y el threshold óptimo según
# Youden muestra el comportamiento esperado:
#
# - el nuevo punto de corte incrementa el recall de la clase positiva,
#   pasando aproximadamente de 0.780 a 0.872;
# - además, reduce los falsos negativos de 31 a 18, lo que mejora la capacidad
#   del modelo para detectar casos positivos;
# - sin embargo, esta mejora se consigue a costa de aumentar los falsos
#   positivos, que pasan de 140 a 176, y de reducir la precision de la clase
#   positiva de 0.440 a 0.411.
#
# En términos de equilibrio global, el cambio de threshold mejora la balanced
# accuracy, pero no aporta una mejora real en el f1-score de la clase positiva,
# que permanece prácticamente estable.
#
# Por tanto, el ajuste del threshold sí permite orientar el modelo hacia una
# detección más agresiva de la clase positiva, pero no corrige la principal
# limitación del SVM lineal: su escasa capacidad para separar bien ambas clases
# sin generar un volumen elevado de falsas alarmas.
#
# En consecuencia, este análisis resulta útil como contraste metodológico y
# como apoyo a la interpretación de la ROC, pero no cambia la conclusión
# general del bloque: el kernel lineal ofrece un rendimiento aceptable como
# referencia inicial, aunque previsiblemente será superado por kernels más
# flexibles en el equilibrio entre discriminación, recall y precisión.

# =============================================================================
# 4. SVM RBF
# =============================================================================
# El kernel RBF se incorpora como alternativa no lineal principal dentro del
# bloque SVM, ya que permite capturar relaciones más complejas entre las
# variables explicativas y la variable objetivo que el kernel lineal no siempre
# puede representar adecuadamente.
#
# Su interés en este proyecto es doble:
# 1. comprobar si una frontera de decisión no lineal mejora la capacidad de
#    separación entre ambas clases;
# 2. valorar si esa ganancia potencial se mantiene de forma razonable en
#    accuracy y recall, sin deteriorar en exceso la estabilidad del modelo.
#
# La búsqueda paramétrica se centra en tres elementos:
# - C: controla la penalización del error y, por tanto, el grado de
#   flexibilidad del clasificador;
# - gamma: determina el alcance de influencia de cada observación en la
#   frontera de decisión, siendo clave en el comportamiento del kernel RBF;
# - class_weight: permite comprobar si el reequilibrado de clases mejora la
#   detección de la clase positiva.

print('\nSVM RBF')
nombre_rbf = 'SVM RBF'
parametros_rbf = ['C', 'gamma', 'class_weight']

param_grid_rbf = {
    'C': [0.1, 0.5, 1, 2.5, 5],
    'gamma': [0.01, 0.1, 1],
    'class_weight': [None, 'balanced']
    }
svm_rbf = SVC(kernel='rbf')

grid_rbf = GridSearchCV(
    svm_rbf, param_grid_rbf, cv=cv, scoring=score, refit='roc_auc', verbose=3, 
    n_jobs=-1)
grid_rbf.fit(X_train_svm, y_train)
print('\nMejores parámetros por GridSearchCV:')
print(grid_rbf.best_params_)
print('Mejor AUC CV:', grid_rbf.best_score_)

# Resultados completos y ordenados del GridSearchCV.
# Aunque el modelo con mejor AUC CV se identifica automáticamente, la selección
# final no se basa únicamente en best_estimator_, sino en una revisión manual
# del conjunto de configuraciones más competitivas.
df_cv_rbf = pd.DataFrame(grid_rbf.cv_results_)
if 'param_class_weight' in df_cv_rbf.columns:
    df_cv_rbf['param_class_weight'] = df_cv_rbf[
        'param_class_weight'].fillna('None').astype(str)
df_grid_rbf = tabla_grid(df_cv_rbf, parametros_rbf)
print(df_grid_rbf.to_string(index=False))

# Los gráficos de evolución permiten observar la tendencia general de las
# métricas al variar los parámetros principales, mientras que los heatmaps
# ayudan a localizar visualmente las zonas más competitivas del grid y a
# detectar combinaciones estables o claramente desfavorables.
grafico_parametros(df_grid_rbf, parametros_rbf, score_grid, nombre_rbf)
# HEATMAPS SVM RBF
heatmap_params(df_grid_rbf, nombre_rbf, 'gamma', 'C', score_grid, 
               {'class_weight': ['None', 'balanced']})

# A partir de este punto se trabaja únicamente con las configuraciones
# class_weight='balanced', ya que son las únicas que mantienen un recall
# suficientemente competitivo para el objetivo del problema.
df_grid_rbf=df_grid_rbf[df_grid_rbf['class_weight']=='balanced']

# ---------------------------------------------
# 4.1. CANDIDATOS SVM RBF
# ---------------------------------------------
# La selección preliminar de candidatos no pretende recoger todo el grid, sino
# concentrarse en las configuraciones más representativas de la zona alta del
# ajuste. Para ello se priorizan combinaciones con class_weight='balanced', ya
# que las soluciones con class_weight=None muestran, en general, recalls
# demasiado bajos para el objetivo del problema, aunque en algunos casos
# mantengan una accuracy o un AUC aceptables.
#
# Se incluyen candidatos de dos zonas principales:
# - gamma = 0.1, que concentra los mejores valores globales de AUC y accuracy;
# - gamma = 0.01, que actúa como zona de contraste, con perfiles algo más
#   conservadores pero todavía competitivos en recall.

candidatos_rbf = pd.DataFrame({
    'candidato':['rbf_1', 'rbf_2', 'rbf_3', 'rbf_4',
                 'rbf_5', 'rbf_6', 'rbf_7', 'rbf_8'],
    'C': [1.0, 2.5, 0.5, 5,
          5, 0.1, 2.5, 1],
    'gamma': [0.1, 0.1, 0.1, 0.1, 
              0.01, 0.01, 0.01, 0.01],
    'class_weight': ['balanced']*8
    })
seleccion_rbf=candidatos_rbf.merge(
    df_cv_rbf,left_on=parametros_rbf, right_on=[
        'param_C', 'param_gamma', 'param_class_weight'], how='left')

# El boxplot permite comparar visualmente la distribución de AUC, accuracy y
# recall en validación cruzada para cada candidato, evitando decidir solo por
# un máximo puntual y facilitando una lectura más robusta del equilibrio entre
# métricas.

boxplot_candidatos(seleccion_rbf, score, nombre_rbf)

# ---------------------------------------------
# 4.2. EVALUACIÓN SVM RBF
# ---------------------------------------------
# El análisis conjunto de la tabla resumen, los gráficos de evolución por
# parámetro, los heatmaps y los boxplots permite extraer una conclusión clara:
# el mejor comportamiento del kernel RBF se concentra en la zona
# gamma = 0.1 con class_weight = 'balanced'.
#
# En primer lugar, se descartan como candidatas finales las configuraciones con
# class_weight = None. Aunque algunas de ellas alcanzan accuracies e incluso
# valores de AUC aceptables, lo hacen sacrificando claramente el recall sobre
# la clase positiva, lo que las hace menos adecuadas dentro del objetivo del
# problema.
#
# En segundo lugar, el parámetro gamma muestra tres comportamientos bien
# diferenciados:
# - gamma = 0.01 genera modelos más conservadores, con recalls todavía altos,
#   pero con menor capacidad discriminante global y menor accuracy;
# - gamma = 1 produce perfiles inestables o claramente desequilibrados, con
#   pérdidas importantes en accuracy y AUC;
# - gamma = 0.1 concentra la zona más sólida del ajuste, combinando los
#   mejores valores de AUC con accuracies altas y recalls todavía
#   suficientemente competitivos.
#
# Dentro de esa zona principal (gamma = 0.1, class_weight = 'balanced'),
# los candidatos más fuertes son rbf_1, rbf_2 y rbf_3:
#
# - rbf_1 = (C=1.0, gamma=0.1, class_weight='balanced')
# - rbf_2 = (C=2.5, gamma=0.1, class_weight='balanced')
# - rbf_3 = (C=0.5, gamma=0.1, class_weight='balanced')
#
# La comparación entre ellos refleja el trade-off habitual entre AUC,
# accuracy y recall:
# - rbf_1 presenta la mayor AUC media del bloque (0.8999);
# - rbf_2 logra una accuracy media superior (0.8812), pero reduce el recall
#   hasta 0.8068;
# - rbf_3 mejora ligeramente el recall (0.8475), pero pierde accuracy y
#   también algo de AUC respecto a rbf_1.
#
# Por tanto, la elección provisional recae en rbf_1, ya que no maximiza de
# forma aislada una única métrica, pero sí ofrece el mejor equilibrio global
# entre capacidad discriminante, acierto general y sensibilidad suficiente
# sobre la clase positiva.
#
# En consecuencia, en esta fase del ajuste se selecciona provisionalmente:
# rbf_1 = (C=1.0, gamma=0.1, class_weight='balanced')
#
# Este candidato se toma como punto de partida para la evaluación posterior en
# test y, en su caso, para un ajuste en esta zona del espacio paramétrico.

params_rbf = {
    'C': 1,
    'gamma': 0.1,
    'class_weight': 'balanced'
    }

# Modelado
modelo_rbf = SVC(kernel='rbf', **params_rbf)

# Ajuste y evaluación del modelo provisional seleccionado sobre TRAIN y TEST.
# A partir de este punto, el análisis deja de centrarse en validación cruzada
# y pasa a comprobar si el equilibrio observado durante el ajuste se mantiene
# en el conjunto de test.
resultados_rbf = evaluar_modelo(
    modelo_rbf, 'SVM RBF', X_train_svm, X_test_svm, y_train, y_test, 
    **params_rbf)

# Resumen, reporte, matriz de confusión y sobreajuste
print(f'\nResumen - {nombre_rbf}:')
print(resultados_rbf['resumen'].round(4))
print(f'\nClassification report - {nombre_rbf}')
print(classification_report(
    y_test, resultados_rbf['y_pred_test'], target_names=['NO','SI'],
    zero_division=0))

matrices_confusion(
    y_test, resultados_rbf['y_pred_test'], nombre_rbf)
df_sobre_rbf = sobreajuste(resultados_rbf['resumen'])
print(df_sobre_rbf.round(4).to_string(index=False))

# La evaluación en test confirma que el kernel RBF mejora claramente el
# comportamiento observado con el kernel lineal y se consolida como la
# alternativa SVM más competitiva hasta este punto del proyecto.
#
# En concreto, el modelo seleccionado alcanza una AUC test de 0.9176 y una
# accuracy test de 0.8828, manteniendo además un recall alto sobre la clase
# positiva (0.8582). Esta combinación refleja un equilibrio muy sólido entre:
# - capacidad discriminante,
# - acierto global,
# - y sensibilidad suficiente para detectar clientes con riesgo real.
#
# La matriz de confusión refuerza esta lectura: el modelo obtiene 121 verdaderos
# positivos y 20 falsos negativos, junto con 504 verdaderos negativos y 63
# falsos positivos. En comparación con el SVM lineal, el salto es claro tanto
# en capacidad discriminante como en rendimiento práctico de clasificación.
#
# Desde el punto de vista de generalización, el modelo mantiene un sobreajuste
# razonablemente contenido. El gap en AUC (0.0233) y en accuracy (0.0129)
# indica una ligera pérdida esperable al pasar de TRAIN a TEST, pero no una
# señal preocupante de deterioro. Por tanto, el candidato seleccionado en la
# fase de validación cruzada también muestra un comportamiento sólido fuera de
# muestra.
#
# En consecuencia, rbf_1 = (C=1, gamma=0.1, class_weight='balanced') se
# consolida provisionalmente como mejor modelo SVM del proyecto y pasa a ser
# el punto de partida para el ajuste posterior.

# ---------------------------------------------
# 4.3. CURVA ROC SVM RBF
# ---------------------------------------------
# La curva ROC permite comprobar si el punto de corte estándar utilizado por el
# modelo coincide con una zona operativamente razonable o si existe margen de
# mejora ajustando el threshold.

info_roc_rbf = curva_auc_roc(y_test, resultados_rbf['y_score_test'], nombre_rbf)
thr_youden_rbf = info_roc_rbf['youden']['threshold']
youden_rbf = info_roc_rbf['youden']['youden']

print(f'\nThreshold óptimo por Youden {nombre_rbf}:{thr_youden_rbf:.4f}')
print(f'Índice de Youden máximo:{youden_rbf:.4f}')
print('Threshold estándar:', info_roc_rbf['estandar']['threshold'])

# COMPARACIÓN THRESHOLDS
grafico_thresholds(
    y_test,resultados_rbf['y_score_test'], nombre_rbf, thr_youden_rbf)

y_pred_youden_rbf = (
    resultados_rbf['y_score_test'] >= thr_youden_rbf).astype(int)

print(f'\nClassification report - {nombre_rbf} con threshold Youden')
print(classification_report(
    y_test, y_pred_youden_rbf, target_names=['NO','SI'], zero_division=0))

heatmap_classification_reports(
    y_test,[resultados_rbf['y_pred_test'], y_pred_youden_rbf],
    [nombre_rbf, f'{nombre_rbf}\n(thr Youden={thr_youden_rbf:.4f})'])
matrices_confusion(
    y_test,[resultados_rbf['y_pred_test'], y_pred_youden_rbf],
    [nombre_rbf, f'{nombre_rbf}\n(thr Youden={thr_youden_rbf:.4f})'])

# En este caso, el threshold óptimo según el criterio de Youden es -0.0958,
# muy próximo al threshold estándar del clasificador (aproximadamente 0). Esto
# sugiere que el modelo ya estaba funcionando cerca de un punto de equilibrio
# razonable entre sensibilidad y especificidad.
#
# Al aplicar el threshold de Youden, el modelo mejora la detección de la clase
# positiva: los falsos negativos se reducen de 20 a 15 y el recall aumenta de
# 0.858 a 0.894. Desde el punto de vista operativo, este cambio puede resultar
# interesante si se desea priorizar la detección de clientes con riesgo real.
#
# Sin embargo, esta mejora en sensibilidad se consigue a costa de aumentar los
# falsos positivos de 63 a 72, lo que reduce la specificity y también la
# precision de la clase positiva. La accuracy desciende ligeramente y el
# f1-score apenas cambia, por lo que no puede afirmarse que el modelo mejore de
# forma global; lo que cambia es el perfil operativo del clasificador.
#
# Por tanto, en este bloque se mantiene como referencia principal el threshold
# estándar, al ofrecer un equilibrio más limpio entre recall, precision,
# specificity y accuracy. El threshold de Youden queda como alternativa
# estratégica si más adelante se quisiera priorizar expresamente la reducción
# de falsos negativos.

# =============================================================================
# 5. SVM POLY
# =============================================================================
# El kernel polinómico se incorpora como tercera familia SVM para comprobar si
# una frontera no lineal de complejidad intermedia puede mejorar al modelo
# lineal sin llegar al grado de flexibilidad del kernel RBF.
#
# Su interés metodológico está en que permite capturar interacciones y curvaturas
# entre variables mediante dos elementos clave:
# - el grado del polinomio (degree), que regula la complejidad de la frontera;
# - y gamma, que controla la influencia local de las observaciones.
#
# En este bloque se analiza si esa mayor flexibilidad aporta una mejora real en
# capacidad discriminante y rendimiento de clasificación, o si, por el
# contrario, conduce a soluciones menos estables o con mayor riesgo de
# sobreajuste.

print('\nSVM POLY')
nombre_p = 'SVM Polinómico'
parametros_poly = ['C', 'gamma', 'degree', 'class_weight']

param_grid_poly = {
    'C': [0.1, 0.5, 1, 2.5, 5],
    'gamma': [0.01, 0.1, 1],
    'degree': [2, 3],
    'class_weight': [None, 'balanced']
    }
svm_poly = SVC(kernel='poly')

grid_poly = GridSearchCV(
    svm_poly, param_grid_poly, cv=cv, scoring=score, refit='roc_auc', 
    verbose=3, n_jobs=-1)
grid_poly.fit(X_train_svm, y_train)
print('\nMejores parámetros por GridSearchCV:')
print(grid_poly.best_params_)
print('Mejor AUC CV:', grid_poly.best_score_)

# Resultados completos y ordenados del GridSearchCV
df_cv_poly = pd.DataFrame(grid_poly.cv_results_)
if 'param_class_weight' in df_cv_poly.columns:
    df_cv_poly['param_class_weight'] = df_cv_poly[
        'param_class_weight'].fillna('None').astype(str)
df_grid_poly = tabla_grid(df_cv_poly, parametros_poly)
print(df_grid_poly.to_string(index=False))

grafico_parametros(df_grid_poly, parametros_poly, score_grid, nombre_p)
# HEATMAPS SVM Polinómico
heatmap_params(df_grid_poly, nombre_p, 'gamma', 'C', score_grid, 
               {'class_weight': ['None', 'balanced'], 'degree': [2, 3]})

# ---------------------------------------------
# 5.1. CANDIDATOS SVM POLY
# ---------------------------------------------
# El análisis conjunto del grid, los gráficos paramétricos, los heatmaps y los
# boxplots muestra un patrón bastante claro en el bloque polinómico.
#
# En primer lugar, la mejor AUC media de validación cruzada corresponde a una
# configuración con class_weight=None
# (C=5, gamma=0.1, degree=3). Sin embargo, ese perfil reduce el recall hasta
# 0.5709, por lo que, aunque ordena razonablemente bien las observaciones, no
# resulta suficientemente adecuado para el objetivo del problema.
#
# Por ese motivo, la selección manual se centra en configuraciones con
# class_weight='balanced', que mantienen una capacidad de detección mucho más
# útil sobre la clase positiva.
#
# Dentro de ese subconjunto aparecen dos zonas diferenciadas:
# - degree=3 con gamma=0.1, que concentra las mejores AUC y accuracies del
#   bloque balanced;
# - degree=2 con gamma=1, que empuja algo más el recall, pero a costa de una
#   pérdida clara de AUC y de acierto global.
#
# Así, los candidatos escogidos permiten representar ambos perfiles:
# - poly_1, poly_2, poly_3 y poly_4 recogen la zona principal del degree=3,
#   gamma=0.1, que es la más fuerte en capacidad discriminante;
# - poly_5, poly_6, poly_7 y poly_8 recogen el perfil alternativo con degree=2
#   y gamma=1, útil como contraste por su mayor sensibilidad.

candidatos_poly = pd.DataFrame({
    'candidato': ['poly_1', 'poly_2', 'poly_3', 'poly_4', 
                  'poly_5', 'poly_6', 'poly_7', 'poly_8'],
    'C': [1, 0.5, 2.5, 5,
          5, 0.1, 0.5, 2.5],
    'gamma': [0.1, 0.1, 0.1, 0.1, 
              0.1, 1, 1, 1],
    'degree': [3, 3, 3, 3, 
               2, 2, 2, 2],
    'class_weight': ['balanced']*8
    })
seleccion_poly = candidatos_poly.merge(
    df_cv_poly, left_on=parametros_poly, right_on=[
        'param_C', 'param_gamma', 'param_degree', 'param_class_weight'],
    how='left')

boxplot_candidatos(seleccion_poly, score, nombre_p)

# ---------------------------------------------
# 5.2. EVALUACIÓN SVM POLY
# ---------------------------------------------
# La comparación final entre candidatos balanced confirma que la zona más sólida
# del bloque polinómico se sitúa en degree=3 con gamma=0.1.
#
# Dentro de esa zona, poly_4 (C=5) presenta medias ligeramente superiores, pero
# poly_3 (C=2.5) mantiene un rendimiento muy próximo en AUC, accuracy y recall
# con una complejidad algo más contenida. Dado que no se observan diferencias
# suficientemente amplias como para justificar una penalización más alta, se
# prioriza la opción más parsimoniosa.
#
# Por tanto, se selecciona como candidato representativo del bloque SVM
# polinómico:
# poly_3 = (C=2.5, gamma=0.1, degree=3, class_weight='balanced')
#
# Esta configuración ofrece un perfil razonablemente equilibrado en validación
# cruzada, con buena capacidad discriminante, accuracy alta y un recall
# suficiente para mantener utilidad práctica en la detección de la clase
# positiva.

params_poly = {
    'C': 2.5,
    'gamma': 0.1,
    'degree': 3,
    'class_weight': 'balanced'
    }

# Modelado
modelo_poly = SVC(kernel='poly', **params_poly)
# Función para modelar, predicciones, score y resumen:
resultados_poly = evaluar_modelo(
    modelo_poly, nombre_p, X_train_svm, X_test_svm, y_train, y_test, 
    **params_poly)

# Resumen, reporte, matriz de confusión y sobreajuste
print(f'\nResumen - {nombre_p}:')
print(resultados_poly['resumen'].round(4))
print(f'\nClassification report - {nombre_p}')
print(classification_report(
    y_test, resultados_poly['y_pred_test'], target_names=['NO','SI'],
    zero_division=0))

matrices_confusion(
    y_test, resultados_poly['y_pred_test'], f'{nombre_p}')
df_sobre_poly = sobreajuste(resultados_poly['resumen'])
print(df_sobre_poly.round(4).to_string(index=False))

# La evaluación en test muestra que el kernel polinómico alcanza un rendimiento
# global aceptable, pero queda por detrás del SVM RBF tanto en capacidad
# discriminante como en comportamiento operativo.
#
# En concreto, el modelo obtiene una AUC test de 0.8851 y una accuracy test de
# 0.8799, junto con 104 verdaderos positivos, 37 falsos negativos, 519
# verdaderos negativos y 48 falsos positivos. Esto refleja un modelo útil, pero
# menos equilibrado que el RBF en la relación entre acierto global, sensibilidad
# y control del error.
#
# Además, los gaps de generalización son relativamente amplios
# (gap_auc = 0.0596 y gap_accuracy = 0.0444), lo que sugiere un nivel de ajuste
# más fuerte sobre TRAIN que el observado en otros kernels. Por tanto, aunque
# el bloque polinómico aporta una alternativa válida de contraste, no se perfila
# como el candidato SVM más robusto del proyecto.

# ---------------------------------------------
# 5.3. CURVA ROC SVM POLY
# ---------------------------------------------
info_roc_poly = curva_auc_roc(y_test, resultados_poly['y_score_test'], nombre_p)
thr_youden_poly = info_roc_poly['youden']['threshold']
youden_poly = info_roc_poly['youden']['youden']

print(f'\nThreshold óptimo por Youden {nombre_p}:{thr_youden_poly:.4f}')
print(f'Índice de Youden máximo:{youden_poly:.4f}')
print('Threshold estándar:', info_roc_poly['estandar']['threshold'])

# COMPARACIÓN THRESHOLDS
grafico_thresholds(
    y_test,resultados_poly['y_score_test'], nombre_p, thr_youden_poly)

y_pred_youden_poly = (
    resultados_poly['y_score_test'] >= thr_youden_poly).astype(int)

heatmap_classification_reports(
    y_test,[resultados_poly['y_pred_test'], y_pred_youden_poly],
    [nombre_p, f'{nombre_p}\n(thr Youden={thr_youden_poly:.4f})'])
matrices_confusion(
    y_test,[resultados_poly['y_pred_test'], y_pred_youden_poly],
    [nombre_p, f'{nombre_p}\n(thr Youden={thr_youden_poly:.4f})'])

# De forma complementaria, se analiza el ajuste del threshold de decisión en el
# modelo SVM polinómico mediante el índice de Youden, con el fin de comprobar
# si un pequeño desplazamiento del punto de corte permite mejorar la capacidad
# de detección de la clase positiva.
#
# El cambio de threshold desde el valor estándar hasta el punto óptimo de
# Youden (thr = -0.2479) produce, efectivamente, una mejora en sensibilidad:
# el recall de la clase positiva aumenta de 0.738 a 0.794 y los falsos
# negativos se reducen de 37 a 29.
#
# Sin embargo, esta ganancia se consigue a costa de un deterioro apreciable en
# el control del error sobre la clase negativa. En concreto, los falsos
# positivos aumentan de 48 a 73, la precision de la clase positiva cae de
# 0.684 a 0.605 y la specificity desciende de forma visible. Como consecuencia,
# también empeoran la accuracy global y el f1-score de la clase positiva.
#
# Por tanto, el threshold ajustado puede interpretarse como una alternativa
# operativa si se quisiera priorizar la detección de positivos reales por
# encima del control de falsas alarmas. No obstante, no supone una mejora
# global del modelo, sino un cambio de equilibrio entre sensibilidad y
# precisión.
#
# En consecuencia, para la comparación principal entre kernels y para la
# selección final del bloque SVM, se mantiene como referencia el threshold
# estándar del modelo, dejando el punto de corte de Youden únicamente como
# contraste adicional de carácter operativo.

# =============================================================================
# 6. COMPARACIÓN DE LOS MODELOS SVM CON DISTINTO KERNEL
# =============================================================================
# En este bloque se comparan los tres kernels evaluados (lineal, RBF y
# polinómico) con el objetivo de identificar cuál ofrece el mejor comportamiento
# global dentro de la familia SVM.
#
# La finalidad de esta comparación no es cerrar todavía el modelo definitivo del
# proyecto, sino seleccionar el kernel más adecuado para continuar después con
# su ajuste específico. Es decir, aquí se decide qué tipo de SVM merece seguir
# desarrollándose en mayor profundidad, a partir de una evaluación homogénea de
# rendimiento, errores de clasificación y generalización.
#
# Para ello, se analizan conjuntamente las métricas principales en test, las
# matrices de confusión, los classification reports, las curvas ROC y el
# comportamiento train/test. De este modo, la elección del kernel ganador no se
# basa en un único indicador aislado, sino en una lectura global y coherente
# del comportamiento de cada alternativa.

print('\nCOMPARACIÓN DE LOS MODELOS SVM CON DISTINTO KERNEL')
df_comparacion_svm = pd.concat(
    [resultados_lineal['resumen'], resultados_rbf['resumen'],
     resultados_poly['resumen']], ignore_index=True)

nombres_svm = [nombre_l, nombre_rbf, nombre_p]

predicciones_svm = [resultados_lineal['y_pred_test'],
                    resultados_rbf['y_pred_test'],
                    resultados_poly['y_pred_test']]

scores_svm = [resultados_lineal['y_score_test'], 
               resultados_rbf['y_score_test'],
               resultados_poly['y_score_test']]


# ---------------------------------------------
# 6.1. RESUMEN DE LOS DISTINTOS KERNEL Y GRÁFICOS
# ---------------------------------------------
print(df_comparacion_svm.round(4).to_string(index=False))
# Comparación de las principales métricas
grafico_comparacion_metricas(df_comparacion_svm, nombres_svm)
# Matriz de confusión - comparación
matrices_confusion(y_test, predicciones_svm, nombres_svm)
# Comparación sobreajuste
df_sobre_comparacion = sobreajuste(df_comparacion_svm)
print(df_sobre_comparacion.round(4).to_string(index=False))
# Heatmap con los reportes de clasificación
heatmap_classification_reports(y_test, predicciones_svm, nombres_svm)
# Curva AUC/ROC comparativa
curva_auc_roc_comparativa(y_test, scores_svm, nombres_svm)

# Tras comparar los tres kernels evaluados, se selecciona como modelo ganador
# el SVM con kernel RBF.
#
# La elección se basa en una lectura conjunta de las métricas principales, las
# matrices de confusión, la curva ROC, los reportes de clasificación y el
# análisis de generalización. En conjunto, el kernel RBF es el que ofrece el
# mejor equilibrio entre capacidad discriminante, acierto global y detección de
# la clase positiva.
#
# En test, el modelo RBF alcanza la mayor AUC de los tres kernels (0.9176) y
# también la mayor accuracy (0.8828). Además, presenta el recall más alto sobre
# la clase positiva (0.8582), superando claramente al SVM Polinómico (0.7376)
# y al SVM lineal (0.7801). Esta diferencia es especialmente relevante en el
# contexto del problema, ya que interesa reducir falsos negativos y detectar
# correctamente a los clientes con mayor probabilidad de abandono.
#
# La matriz de confusión refuerza esta conclusión. Frente al modelo lineal, el
# SVM RBF reduce simultáneamente los falsos negativos (20 frente a 31) y los
# falsos positivos (63 frente a 140), lo que supone una mejora clara tanto en
# sensibilidad como en control del error. Frente al modelo polinómico, el RBF
# también resulta preferible: aunque el polinómico logra menos falsos positivos
# (48 frente a 63) y una specificity algo más alta (0.9153 frente a 0.8889),
# lo hace a costa de aumentar notablemente los falsos negativos (37 frente a 20)
# y reducir el recall de la clase positiva.
#
# El classification report confirma el mismo patrón. El SVM Polinómico obtiene
# una precision algo superior sobre la clase positiva (0.6842 frente a 0.6576),
# pero pierde capacidad de detección, lo que termina penalizando su utilidad
# operativa. El modelo lineal, por su parte, queda claramente por detrás en
# precision (0.4400), f1-score (0.5627) y rendimiento global. En cambio, el RBF
# combina una precision razonable con el mejor recall y el mejor f1-score de la
# clase positiva (0.7446), por lo que representa el compromiso más sólido entre
# detección y fiabilidad.
#
# La curva ROC también respalda la elección, ya que el SVM RBF domina con
# claridad al modelo lineal y se sitúa por encima del polinómico a lo largo de
# buena parte del recorrido. Esto confirma que su superioridad no depende de un
# único punto de corte, sino de una mejor capacidad global de separación entre
# ambas clases.
#
# El análisis de generalización completa la interpretación. El modelo lineal
# presenta el menor gap train-test (gap_auc = -0.0155;
# gap_accuracy = 0.0113), pero ese comportamiento se asocia a un rendimiento
# claramente inferior, por lo que apunta más a infraajuste que a una ventaja
# real.
#
# El modelo polinómico muestra el peor perfil de generalización, con los
# mayores gaps del bloque (gap_auc = 0.0596; gap_accuracy = 0.0444).
#
# El SVM RBF se sitúa en una posición más favorable: mantiene un gap moderado
# (gap_auc = 0.0233; gap_accuracy = 0.0129) y, al mismo tiempo, conserva el
# mejor rendimiento en test de los tres kernels evaluados.
#
# En consecuencia, el SVM con kernel RBF se considera el modelo más sólido y
# equilibrado del bloque SVM, por lo que se selecciona como ganador para el
# ajuste posterior y para su comparación con los modelos ensemble.

# =============================================================================
# 7. AJUSTE DEL MODELO GANADOR: SVM RBF
# =============================================================================
# Una vez identificado el kernel RBF como mejor alternativa del bloque SVM, el
# siguiente paso consiste en ajustar sus parámetros dentro de una zona más
# reducida del espacio paramétrico.
#
# La finalidad de este bloque no es volver a comparar kernels distintos, sino
# refinar el comportamiento del modelo ganador para comprobar si es posible
# mejorar su equilibrio entre capacidad discriminante, acierto global y
# detección de la clase positiva.
#
# Para ello, se construye una malla más densa alrededor de la configuración
# inicialmente seleccionada (C=1 y gamma=0.1, con class_weight='balanced'),
# manteniendo fijo el esquema de ponderación por clases. De este modo, el
# análisis se centra exclusivamente en pequeños cambios de C y gamma, que son
# los parámetros con mayor impacto práctico en el rendimiento del kernel RBF.
#
# Así, este apartado permite verificar si la configuración inicial ya se
# encontraba en una zona estable o si existen combinaciones muy próximas capaces
# de ofrecer una mejora adicional en validación cruzada y, posteriormente, en
# test.

print('\nAJUSTE DEL MODELO GANADOR: SVM RBF')
lista_c_ajuste = [round(i, 2) for i in np.arange(0.70, 1.21, 0.05)]
lista_g_ajuste = [round(i, 3) for i in np.arange(0.060, 0.111, 0.005)]
nombre_rbf_a = 'Ajuste SVM RBF'
param_grid_rbf_a = {
    'C': lista_c_ajuste,
    'gamma': lista_g_ajuste,
    'class_weight': ['balanced']
    }
svm_rbf_a = SVC(kernel='rbf')

grid_rbf_a = GridSearchCV(
    svm_rbf_a, param_grid_rbf_a, cv=cv, scoring=score, refit='roc_auc', 
    return_train_score=True, verbose=3, n_jobs=-1)
grid_rbf_a.fit(X_train_svm, y_train)
print('\nMejores parámetros del ajuste:')
print(grid_rbf_a.best_params_)
print('Mejor AUC CV:', grid_rbf_a.best_score_)

# Resultados del ajuste
df_cv_rbf_a = pd.DataFrame(grid_rbf_a.cv_results_)
if 'param_class_weight' in df_cv_rbf_a.columns:
    df_cv_rbf_a['param_class_weight'] = df_cv_rbf_a[
        'param_class_weight'].fillna('None').astype(str)
df_grid_rbf_a = tabla_grid(df_cv_rbf_a, parametros_rbf)

grafico_parametros(df_grid_rbf_a, parametros_rbf, score_grid, nombre_rbf_a)
# Heatmaps del ajuste
heatmap_params(df_grid_rbf_a, nombre_rbf_a, 'gamma', 'C', score_grid,
               {'class_weight': ['balanced']})

# ---------------------------------------------    
# 7.1. CANDIDATOS SVM RBF AJUSTADO
# ---------------------------------------------
# Se conserva el modelo base y se añaden once configuraciones destacadas del
# ajuste para compararlas de forma homogénea también en test.

candidato_rbf_0 = pd.DataFrame({
    'candidato': ['rbf_0'], 
    'C': [1.0], 
    'gamma': [0.1],
    'class_weight': ['balanced']
    })

candidatos_rbf_a = pd.DataFrame({
    'candidato': ['rbf_a_1', 'rbf_a_2', 'rbf_a_3', 'rbf_a_4', 
                  'rbf_a_5', 'rbf_a_6', 'rbf_a_7', 'rbf_a_8',
                  'rbf_a_9', 'rbf_a_10', 'rbf_a_11'],
    'C': [0.9, 0.95, 0.95, 1.05,
          0.85, 0.8, 0.9, 0.95,
          0.8, 0.85, 0.9],
    'gamma': [0.095, 0.095, 0.09, 0.095,
              0.095, 0.10, 0.085, 0.08,
              0.09, 0.085, 0.08],
    'class_weight': ['balanced']*11
    })
candidatos_rbf_a = pd.concat([candidato_rbf_0, candidatos_rbf_a],
                             ignore_index=True)
seleccion_rbf_a = candidatos_rbf_a.merge(
    df_cv_rbf_a, left_on=parametros_rbf, right_on=[
        'param_C', 'param_gamma', 'param_class_weight'], how='left')

boxplot_candidatos(seleccion_rbf_a, score, nombre_rbf_a)

# ---------------------------------------------    
# 7.2. EVALUACIÓN Y COMPARACIÓN DE LOS CANDIDATOS
# ---------------------------------------------
# En consecuencia, en esta fase del ajuste no basta con identificar el mejor
# candidato por una diferencia mínima en AUC CV. Al tratarse de modelos muy
# próximos entre sí, resulta más razonable valorar también cómo se comporta
# cada uno en test y si esa ventaja se mantiene en términos operativos.
#
# Por ello, la decisión final del ajuste RBF se fundamenta en la combinación de
# evidencia de validación cruzada y contraste externo en test, priorizando los
# candidatos que conservan una AUC muy alta y, al mismo tiempo, ofrecen el
# mejor equilibrio global entre accuracy, recall, precision, f1-score y
# generalización.

params_rbf_a = candidatos_rbf_a.copy()
params_rbf_a = list(params_rbf_a.itertuples(index=False,name=None))

predicciones_rbf_a = []
scores_rbf_a = []
resumen_rbf_a = []
nombres_rbf_a = []
modelos_curva_rbf_a = {}

for n, c, g, cw in params_rbf_a:
    modelo_for = SVC(kernel='rbf', C=c, gamma=g, class_weight=cw)
    resultados_modelo_for = evaluar_modelo(
        modelo_for, n, X_train_svm, X_test_svm, y_train, y_test, C=c, gamma=g, 
        class_weight=cw)
    nombre_modelo = f'{nombre_rbf_a}: {n}'
    predicciones_rbf_a.append(resultados_modelo_for['y_pred_test'])
    scores_rbf_a.append(resultados_modelo_for['y_score_test'])
    nombres_rbf_a.append(nombre_modelo)
    resumen_rbf_a.append(resultados_modelo_for['resumen'])
    modelos_curva_rbf_a[nombre_modelo] = modelo_for

# Comparaciones
df_resumen_rbf_a = pd.concat(resumen_rbf_a, ignore_index=True)

matrices_confusion(y_test, predicciones_rbf_a, nombres_rbf_a)

heatmap_classification_reports(y_test, predicciones_rbf_a, nombres_rbf_a)

df_sobre_rbf_a = sobreajuste(df_resumen_rbf_a)
print(df_sobre_rbf_a.round(4).to_string(index=False))  

grafico_comparacion_metricas(df_resumen_rbf_a, nombre_rbf_a)

curva_auc_roc_comparativa(y_test, scores_rbf_a, nombres_rbf_a)

# =============================================================================
# 8. MODELO FINAL SVM
# =============================================================================
# Tras el ajuste del modelo SVM RBF, se selecciona como modelo final la
# configuración rbf_a_11, con C=0.9, gamma=0.08 y class_weight='balanced'.
#
# Esta elección se justifica porque mantiene una AUC test muy alta (0.9189) y
# la misma accuracy test que el modelo base (0.8828), pero mejora la capacidad
# de detección de la clase positiva, alcanzando un recall de 0.9007 y el mejor
# f1-score de la clase positiva dentro del bloque ajustado (0.7537).
#
# Aunque presenta una ligera caída en precision y specificity respecto a
# configuraciones algo más conservadoras, esa pérdida es reducida y queda
# compensada por la mejora en sensibilidad. En este problema resulta
# especialmente relevante identificar correctamente a los clientes con mayor
# probabilidad de abandono, por lo que la reducción de falsos negativos tiene un
# valor operativo claro.
#
# Además, el análisis de sobreajuste muestra un gap train-test más contenido que
# el del modelo inicial, y la comparación global del bloque de ajuste confirma
# que rbf_a_11 se mantiene dentro del grupo de candidatos con mejor capacidad de
# generalización.
#
# Por tanto, rbf_a_11 se considera la opción más sólida y equilibrada dentro del
# ajuste del kernel RBF y se adopta como modelo final del bloque SVM.

params_rbf_final = {
    'C': 0.9,
    'gamma': 0.08,
    'class_weight': 'balanced'
    }

modelo_rbf_final = SVC(kernel='rbf', **params_rbf_final)
resultados_rbf_final = evaluar_modelo(
    modelo_rbf_final, nombre_rbf, X_train_svm, X_test_svm, y_train, y_test, 
    **params_rbf_final)

print(resultados_rbf_final['resumen'].round(4))
print(f'\nClassification report - {nombre_rbf}')
print(classification_report(
    y_test, resultados_rbf_final['y_pred_test'], target_names=['NO','SI'],
    zero_division=0))

titulo_rbf_final = f"{nombre_rbf}\n(C={params_rbf_final['C']},\
gamma={params_rbf_final['gamma']}, cw=balanced)"
matrices_confusion(
    y_test, resultados_rbf_final['y_pred_test'], titulo_rbf_final)

df_sobre_rbf = sobreajuste(resultados_rbf_final['resumen'])
print(df_sobre_rbf.round(4).to_string(index=False))


# CURVA ROC + THRESHOLD
info_roc_rbf_final = curva_auc_roc(y_test, resultados_rbf_final['y_score_test'],
                                   nombre_rbf)
thr_youden_rbf_f = info_roc_rbf_final['youden']['threshold']
youden_rbf_f = info_roc_rbf_final['youden']['youden']
grafico_thresholds(y_test, resultados_rbf_final['y_score_test'],
                   nombre_rbf, thr_youden_rbf_f)
y_pred_youden_rbf_f = (resultados_rbf_final['y_score_test'] >=
                       thr_youden_rbf_f).astype(int)

heatmap_classification_reports(
    y_test, [resultados_rbf_final['y_pred_test'], y_pred_youden_rbf_f], 
    [nombre_rbf, f'{nombre_rbf}\n(thr Youden)'])
matrices_confusion(
    y_test, [resultados_rbf_final['y_pred_test'], y_pred_youden_rbf_f], 
    [nombre_rbf, f'{nombre_rbf}\n(thr Youden)'])
# Modelo final para bagging, ensemble y comparación global posterior
modelo_rbf = modelo_rbf_final

# Actualización de la tabla resumen final con el RBF ajustado
df_comparacion_svm_modelos = pd.concat(
    [resultados_lineal['resumen'], resultados_rbf['resumen'], 
     resultados_poly['resumen'], resultados_rbf_final['resumen']],
    ignore_index=True)

# El modelo final SVM RBF con C=0.9 y gamma=0.08 confirma la mejora observada
# en el bloque de ajuste. En test alcanza una AUC de 0.9189, ligeramente
# superior a la del modelo RBF inicial, y mantiene la misma accuracy global
# (0.8828), lo que indica que la mejora no se consigue sacrificando el
# rendimiento general del clasificador.
#
# Desde el punto de vista operativo, el resultado es favorable: el modelo final
# reduce los falsos negativos hasta 14 y eleva el recall de la clase positiva
# hasta 0.9007, mejorando así la capacidad de detección de clientes con mayor
# probabilidad de abandono. Aunque esta mejora viene acompañada de un aumento
# moderado de falsos positivos, el equilibrio global sigue siendo sólido, como
# refleja también el f1-score de la clase positiva (0.754), el mejor del bloque
# SVM ajustado.
#
# Además, el análisis de generalización resulta especialmente positivo. El gap
# en accuracy es ligeramente negativo (-0.0016), lo que indica que el
# rendimiento en test iguala e incluso supera levemente al de train, mientras
# que el gap en AUC (0.0117) es reducido. En conjunto, esto sugiere un modelo
# estable, con muy buen comportamiento fuera de muestra y sin señales relevantes
# de sobreajuste.
#
# Por otra parte, el análisis del threshold confirma que el punto de corte
# óptimo según Youden (0.0012) es prácticamente coincidente con el threshold
# estándar de decisión. En consecuencia, las predicciones, la matriz de
# confusión y el classification report no cambian en la práctica. Esta
# coincidencia refuerza la robustez del modelo final, ya que indica que no es
# necesario reajustar el punto de corte para obtener una mejora operativa
# adicional.
#
# En definitiva, la configuración final C=0.9, gamma=0.08 y
# class_weight='balanced' se adopta como modelo definitivo del bloque SVM y
# será la referencia para los apartados posteriores de bagging, stacking y
# comparación global.

# =============================================================================
# 9. COMPARACIÓN SVM RBF FINAL: ORIGINAL vs TOP10 RF vs TOP10 SKB
# =============================================================================
# Como comprobación complementaria, se analiza si una reducción simple del
# número de variables puede mantener el rendimiento del modelo final RBF o
# mejorar alguna de sus métricas principales.
#
# Para ello se construyen dos versiones reducidas del modelo: una con las
# 10 variables más relevantes según Random Forest y otra con las 10 variables
# mejor puntuadas por SelectKBest. Ambas se comparan frente al modelo original
# con el mismo clasificador y los mismos hiperparámetros finales.

print('\nCOMPARACIÓN SVM RBF FINAL: ORIGINAL vs TOP10 RF vs TOP10 SKB')
# Selección de variables SOLO en TRAIN
# Top 10 por Random Forest
top10_rf = df_rf.sort_values('imp_rf', ascending=False)['var'].head(10).tolist()

# Top 10 por SelectKBest (F-clasificación)
top10_skb = df_kbest_f.sort_values(
    'score_f', ascending=False)['var'].head(10).tolist()

print('\nTop 10 variables por RF:')
print(top10_rf)
print('\nTop 10 variables por SelectKBest (F):')
print(top10_skb)

# Construcción de matrices reducidas
X_train_rbf = X_train_svm.copy()
X_test_rbf = X_test_svm.copy()
X_train_rbf_rf10 = X_train_svm[top10_rf].copy()
X_test_rbf_rf10 = X_test_svm[top10_rf].copy()
X_train_rbf_skb10 = X_train_svm[top10_skb].copy()
X_test_rbf_skb10 = X_test_svm[top10_skb].copy()

print('\nDimensiones comparación:')
print(f'{nombre_rbf} Original: {X_train_rbf.shape}|{X_test_rbf.shape}')
print(f'{nombre_rbf}-RF_top10: {X_train_rbf_rf10.shape}|{X_test_rbf_rf10.shape}')
print(f'{nombre_rbf}-SKB_top10: {X_train_rbf_skb10.shape}|{X_test_rbf_skb10.shape}')

# Modelado con las selecciones
modelo_rbf_original = SVC(kernel='rbf', **params_rbf_final)
modelo_rbf_rf10 = SVC(kernel='rbf', **params_rbf_final)
modelo_rbf_skb10 = SVC(kernel='rbf', **params_rbf_final)

# Evaluación y comparación
resultados_rbf_original = evaluar_modelo(
    modelo_rbf_original, nombre_rbf, X_train_rbf, X_test_rbf, y_train, y_test,
    **params_rbf_final, n_vars=X_train_rbf.shape[1])
resultados_rbf_rf10 = evaluar_modelo(
    modelo_rbf_rf10, f'{nombre_rbf}-RF_top10', X_train_rbf_rf10, 
    X_test_rbf_rf10, y_train, y_test, **params_rbf_final,
    n_vars=X_train_rbf_rf10.shape[1])
resultados_rbf_skb10 = evaluar_modelo(
    modelo_rbf_skb10, f'{nombre_rbf}-SKB_top10', X_train_rbf_skb10,
    X_test_rbf_skb10, y_train, y_test, **params_rbf_final,
    n_vars=X_train_rbf_skb10.shape[1])

df_comparacion_seleccion_v = pd.concat(
    [resultados_rbf_original['resumen'], resultados_rbf_rf10['resumen'], 
     resultados_rbf_skb10['resumen']], ignore_index=True)
print(df_comparacion_seleccion_v.round(4).to_string(index=False))

predicciones_seleccion_v = [resultados_rbf_original['y_pred_test'],
                            resultados_rbf_rf10['y_pred_test'],
                            resultados_rbf_skb10['y_pred_test']]

scores_seleccion_v = [resultados_rbf_original['y_score_test'],
                      resultados_rbf_rf10['y_score_test'],
                      resultados_rbf_skb10['y_score_test']]

nombres_seleccion_v = [nombre_rbf, f'{nombre_rbf}-RF_top10',
                       f'{nombre_rbf}-SKB_top10']

grafico_comparacion_metricas(df_comparacion_seleccion_v, nombres_seleccion_v)

matrices_confusion(y_test, predicciones_seleccion_v, nombres_seleccion_v)
heatmap_classification_reports(y_test, predicciones_seleccion_v,
                               nombres_seleccion_v)

df_sobre_seleccion_v = sobreajuste(df_comparacion_seleccion_v)
print(df_sobre_seleccion_v.round(4).to_string(index=False))

# Este bloque se incorpora como contraste adicional para comprobar si una
# reducción simple del espacio de variables puede mantener —o incluso mejorar—
# el comportamiento del modelo final SVM RBF.
#
# Para ello se comparan tres versiones del mismo clasificador:
# - el modelo RBF original con las 13 variables finales,
# - una versión reducida con las 10 variables más importantes según Random Forest,
# - y una versión reducida con las 10 variables mejor valoradas por SelectKBest.
#
# El objetivo no es reabrir todo el proceso de selección de variables, sino
# comprobar si una reducción moderada de dimensionalidad permite simplificar el
# modelo sin deteriorar su rendimiento, o incluso mejorar alguna dimensión
# concreta del problema.
#
# Los resultados permiten distinguir dos comportamientos claramente diferentes.
#
# En primer lugar, la versión SVM RBF-RF_top10 queda descartada como alternativa
# final. Aunque reduce el número de variables de 13 a 10, lo hace empeorando de
# forma apreciable el rendimiento global del modelo: disminuyen la AUC test
# (0.9189 -> 0.9157), la accuracy test (0.8828 -> 0.8573), la precision
# (0.6480 -> 0.5952), el f1-score de la clase positiva (0.7537 -> 0.7123) y la
# specificity (0.8783 -> 0.8501). Además, aumentan los falsos positivos
# (69 -> 85) y también los falsos negativos (14 -> 16). Por tanto, esta
# reducción no compensa y se descarta con claridad.
#
# En segundo lugar, la versión SVM RBF-SKB_top10 sí constituye un contraste más
# interesante. Con solo 10 variables mejora la AUC test (0.9274 frente a
# 0.9189), incrementa ligeramente el recall (0.9078 frente a 0.9007) y reduce
# los falsos negativos (13 frente a 14), lo que indica una capacidad algo mayor
# para detectar la clase positiva.
#
# Sin embargo, esta mejora se obtiene a costa de un deterioro en varias métricas
# complementarias: baja la accuracy test (0.8743 frente a 0.8828), la precision
# de la clase positiva (0.6275 frente a 0.6480), el f1-score (0.7420 frente a
# 0.7537) y la specificity (0.8660 frente a 0.8783), además de aumentar los
# falsos positivos (76 frente a 69). En otras palabras, el modelo SKB_top10
# detecta un positivo más, pero genera también siete falsos positivos
# adicionales.
#
# Por ello, aunque SVM RBF-SKB_top10 puede considerarse una alternativa
# razonable si se quisiera priorizar todavía más la sensibilidad del modelo, no
# supera de forma concluyente al SVM RBF original en equilibrio global.
#
# En consecuencia, se mantiene como opción preferida el modelo SVM RBF original,
# ya que sigue ofreciendo el compromiso más sólido entre capacidad
# discriminante, accuracy, precision, f1-score, control del error y utilidad
# operativa general. La reducción por SelectKBest queda, no obstante, como un
# contraste metodológicamente interesante, al mostrar que una simplificación
# moderada del modelo puede mejorar AUC y recall, aunque sin llegar a justificar
# un cambio de modelo final.

# =============================================================================
# 10. CONCLUSIÓN FINAL DEL SCRIPT SVM
# =============================================================================
# El desarrollo de este script permite establecer una conclusión clara:
# tras comparar los kernels lineal, RBF y polinómico, y realizar después un
# ajuste específico sobre el kernel ganador, el modelo seleccionado como mejor
# alternativa SVM es el SVM RBF con parámetros:
# C = 0.9, gamma = 0.08 y class_weight = 'balanced'.
#
# La elección no se ha basado en una única métrica aislada, sino en una lectura
# conjunta de la validación cruzada, el comportamiento en test, las matrices de
# confusión, los reportes de clasificación, la curva ROC y el análisis de
# generalización. En ese conjunto de evidencias, el modelo RBF ajustado es el
# que ofrece el compromiso más sólido entre capacidad discriminante, acierto
# global y detección de la clase positiva.
#
# Además, las comparaciones complementarias con versiones reducidas del modelo
# muestran que simplificar el espacio de variables no mejora de forma
# concluyente el equilibrio global del clasificador. En particular, la reducción
# basada en Random Forest empeora claramente el rendimiento, mientras que la
# versión basada en SelectKBest mejora ligeramente algunas dimensiones
# concretas, pero a costa de aumentar falsos positivos y perder robustez en el
# balance general de métricas.
#
# En consecuencia, este script deja fijado como modelo SVM de referencia para
# los siguientes bloques el SVM RBF ajustado, que será el utilizado como punto
# de partida en los apartados posteriores de bagging, stacking y comparación
# global final.
print('\nCONCLUSIÓN FINAL DEL SCRIPT SVM')
print('Modelo SVM seleccionado: SVM RBF')
print("Parámetros finales: C=0.9, gamma=0.08, class_weight='balanced'")

# =============================================================================
# 11. GUARDADO EN PICKLE
# =============================================================================
print('\nGUARDADO SVM')

bbdd_svm = {
    'nombre_modelo_ganador': nombre_rbf,
    'modelo_rbf': modelo_rbf,
    'parametros_rbf': params_rbf_final,
    
    'X_train_svm': X_train_svm,
    'X_test_svm': X_test_svm,
    'y_train': y_train,
    'y_test': y_test,
    'columnas_svm': X_train_svm.columns.tolist(),
    'var_categoricas_svm': vars_cat_svm,
    'var_numericas_svm': vars_num_svm,
    
    'df_resumen_rbf': resultados_rbf_final['resumen'],
    'df_comparacion_svm': df_comparacion_svm_modelos,
    }

with open(RUTA_SALIDA, 'wb') as archivo:
    pickle.dump(bbdd_svm, archivo)

print(f'\nArchivo guardado: {RUTA_SALIDA.resolve()}')
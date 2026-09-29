# -*- coding: utf-8 -*-
# =============================================================================
# 04_comparacion_ml.py
# Proyecto: Machine Learning 2026
#
# Objetivo:
# - Cargar los modelos finales seleccionados en los scripts 02 y 03.
# - Verificar que todos trabajan sobre la misma partición train/test.
# - Evaluar de forma homogénea los cuatro modelos finales:
#       1. SVM RBF
#       2. Bagging Árbol
#       3. XGBoost
#       4. Meta-modelo Stacking
# - Preparar todos los objetos necesarios para la comparación global.
#
# Nota:
# - Se parte exclusivamente de los modelos ya seleccionados manualmente.
# - La comparación se realizará sobre la misma partición fija del proyecto.
# =============================================================================
from pathlib import Path
import pickle
import warnings

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from funciones_ml import (
    RANDOM_STATE, resumen_global, evaluar_modelo, grafico_comparacion_metricas,
    matrices_confusion, heatmap_classification_reports, curvas_validacion,
    curva_auc_roc_comparativa, importancia_variables,
    grafico_ranking, heatmap_metricas
    )

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# RUTAS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RUTA_SVM = BASE_DIR / 'outputs' / 'BBDD_svm.pickle'
RUTA_ENSEMBLE = BASE_DIR / 'outputs' / 'BBDD_ensemble.pickle'
RUTA_SALIDA = BASE_DIR / 'outputs' / 'BBDD_comparacion_global.pickle'

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

if not RUTA_SVM.exists():
    raise FileNotFoundError(
        'No se encontró BBDD_svm.pickle. '
        'Ejecuta primero 02_svm_ml.py.'
    )

if not RUTA_ENSEMBLE.exists():
    raise FileNotFoundError(
        'No se encontró BBDD_ensemble.pickle. '
        'Ejecuta primero 03_ensemble_ml.py.'
    )

# =============================================================================
# 1. CARGA DE DATOS Y MODELOS
# =============================================================================
# En esta primera parte no se construyen modelos nuevos ni se reabre ningún
# proceso de ajuste. El objetivo es recuperar exactamente los clasificadores
# finales ya seleccionados en los scripts anteriores y comprobar que todos
# ellos se apoyan sobre la misma partición train/test y sobre el mismo conjunto
# de variables.
#
# Esta verificación es imprescindible para que la comparación global posterior
# sea metodológicamente válida. Si los modelos se evaluaran sobre particiones
# distintas o con matrices de entrada no equivalentes, cualquier diferencia en
# las métricas podría deberse al cambio de muestra y no al comportamiento real
# del clasificador.
#
# Por tanto, este bloque actúa como un control de consistencia previo a la
# comparación final del proyecto: asegura que SVM RBF, Bagging Árbol, XGBoost
# y Stacking parten exactamente del mismo escenario experimental y que sus
# resultados pueden interpretarse de forma homogénea y directamente comparable.
print('\nCARGA DE DATOS Y MODELOS')

with open(RUTA_SVM, 'rb') as archivo:
    bbdd_svm = pickle.load(archivo)

with open(RUTA_ENSEMBLE, 'rb') as archivo:
    bbdd_ensemble = pickle.load(archivo)

# ---------------------------------------------
# 1.1. OBJETOS DEL SCRIPT 02 - SVM
# ---------------------------------------------
nombre_rbf = bbdd_svm['nombre_modelo_ganador']
modelo_rbf = bbdd_svm['modelo_rbf']
parametros_rbf = bbdd_svm['parametros_rbf']

X_train_svm = bbdd_svm['X_train_svm'].copy()
X_test_svm = bbdd_svm['X_test_svm'].copy()
y_train_svm = bbdd_svm['y_train'].copy()
y_test_svm = bbdd_svm['y_test'].copy()
columnas_svm = bbdd_svm['columnas_svm']

# ---------------------------------------------
# 1.2. OBJETOS DEL SCRIPT 03 - ENSEMBLE
# ---------------------------------------------
nombre_bag = bbdd_ensemble['nombre_bagging']
modelo_bag = bbdd_ensemble['modelo_bagging']
parametros_bag = bbdd_ensemble['parametros_bagging']

nombre_st = bbdd_ensemble['nombre_stacking']
modelo_st = bbdd_ensemble['modelo_stacking']
parametros_st = bbdd_ensemble['parametros_stacking']
parametros_meta_st = bbdd_ensemble['parametros_meta_modelo']

nombre_xgb = bbdd_ensemble['nombre_xgboost']
modelo_xgb = bbdd_ensemble['modelo_xgboost']
parametros_xgb = bbdd_ensemble['parametros_xgboost']

X_train_ens = bbdd_ensemble['X_train'].copy()
X_test_ens = bbdd_ensemble['X_test'].copy()
y_train_ens = bbdd_ensemble['y_train'].copy()
y_test_ens = bbdd_ensemble['y_test'].copy()
columnas_ens = bbdd_ensemble['columnas_svm']

# ---------------------------------------------
# 1.3. COMPROBACIÓN DE DATOS
# ---------------------------------------------
print('\nCOMPROBACIÓN DE DATOS')

comparacion = [('X_train', X_train_svm, X_train_ens),
               ('X_test', X_test_svm, X_test_ens),
               ('y_train', y_train_svm, y_train_ens),
               ('y_test', y_test_svm, y_test_ens),
               ('Columnas', columnas_svm, columnas_ens)]
resultados = []

for nombre, obj_svm, obj_ens in comparacion:
    if hasattr(obj_svm, 'equals'):
        bien = obj_svm.equals(obj_ens)  # Dataframe
    else:
        bien = (obj_svm == obj_ens)     # Lista
    estado = 'OK' if bien else 'ERROR'
    resultados.append({'Comparación': nombre, 'Estado': estado})

df_comparacion = pd.DataFrame(resultados)

print(df_comparacion)

if (df_comparacion['Estado'] != 'OK').any():
    raise ValueError(
        'Los modelos no comparten exactamente la misma partición '
        'TRAIN/TEST o el mismo conjunto de variables.'
    )
    
X_train = X_train_svm.copy()
X_test = X_test_svm.copy()
y_train = y_train_svm.copy()
y_test = y_test_svm.copy()

# =============================================================================
# 2. CONFIGURACIÓN GENERAL
# =============================================================================
# Una vez comprobada la consistencia de los datos y de los objetos cargados,
# se fija la configuración general del bloque comparativo.
#
# En particular, se mantiene la misma validación cruzada estratificada empleada
# a lo largo del proyecto, con el fin de conservar la coherencia metodológica
# entre los scripts previos y esta comparación global. Además, se organizan en
# estructuras comunes los modelos finales seleccionados, sus parámetros y las
# métricas que servirán como base para el análisis conjunto.
#
# De este modo, los apartados siguientes podrán centrarse ya en la comparación
# crítica del rendimiento, sin necesidad de introducir ajustes adicionales ni
# cambios de criterio respecto a lo ya cerrado en los bloques anteriores.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

modelos_seleccionados = {
    nombre_rbf: modelo_rbf,
    nombre_bag: modelo_bag,
    nombre_xgb: modelo_xgb,
    nombre_st: modelo_st
    }

parametros_modelos = {
    nombre_rbf: parametros_rbf,
    nombre_bag: parametros_bag,
    nombre_xgb: parametros_xgb,
    nombre_st: {**parametros_st, **parametros_meta_st}
    }

metricas_analisis = ['auc_test', 'accuracy_test', 'recall', 'precision',
                     'specificity', 'fp', 'fn', 'gap_auc', 'gap_accuracy']

# =============================================================================
# 3. EVALUACIÓN DE LOS MODELOS Y CREACIÓN DE OBJETOS PARA LAS COMPARACIONES
# =============================================================================
# Este bloque actúa como punto de arranque del análisis comparativo final.
# Su finalidad no es todavía extraer conclusiones detalladas, sino evaluar de
# forma homogénea los cuatro modelos seleccionados y dejar preparados todos los
# objetos necesarios para los apartados posteriores.
#
# En concreto, aquí se centraliza:
# - la evaluación sobre TRAIN y TEST de cada clasificador final,
# - la construcción de una tabla resumen común,
# - y la organización de predicciones y scores que servirán como base para las
#   comparaciones visuales y métricas de los bloques siguientes.
#
# De este modo, el análisis posterior se apoya en una única estructura de
# resultados, evitando repetir evaluaciones y garantizando que todas las
# comparaciones parten de exactamente la misma información.
# -----------------------------------------------------------------------------
# 3.1. EVALUACIÓN DE LOS MODELOS
# -----------------------------------------------------------------------------
# Se evalúa cada uno de los modelos finales sobre los conjuntos de TRAIN y TEST
# utilizando la misma función de evaluación. Así se obtiene, para todos ellos,
# un bloque homogéneo de métricas, predicciones y probabilidades.
#
# Aunque en este punto todavía no se desarrolla la interpretación detallada,
# esta evaluación unificada es necesaria para asegurar que la comparación
# posterior entre modelos sea consistente y directamente comparable.
#
# Además, se conservan tanto las salidas a umbral fijo como los scores
# probabilísticos, ya que ambos serán necesarios en apartados posteriores:
# matrices de confusión, curvas ROC, análisis de sobreajuste, ranking global y
# demás visualizaciones comparativas.

print('\nEVALUACIÓN DE LOS MODELOS')
resultados_modelos = []

for n, modelo in modelos_seleccionados.items():
    params = parametros_modelos.get(n, {})
    res = evaluar_modelo(modelo, n, X_train, X_test, y_train, y_test, **params)
    resultados_modelos.append(res)

# -----------------------------------------------------------------------------
# 3.2. RESUMEN DE LOS MODELOS
# -----------------------------------------------------------------------------
# A partir de las evaluaciones individuales se construye una tabla resumen
# global que reúne, en un único objeto, las métricas principales de los cuatro
# modelos finales.
#
# Esta tabla no sustituye al análisis detallado posterior, pero sí proporciona
# una primera visión sintética del rendimiento conjunto y servirá como base
# para ordenar, filtrar y contrastar los modelos a lo largo del script.

print('\nRESUMEN GLOBAL DE LOS MODELOS')
df_resumen_global = resumen_global(resultados_modelos)
print(df_resumen_global.round(4).to_string(index=False))

# -----------------------------------------------------------------------------
# 3.3. CREACIÓN DE OBJETOS PARA LAS COMPARACIONES
# -----------------------------------------------------------------------------
# Una vez evaluados los modelos, se reorganizan sus resultados en estructuras
# comunes para facilitar el trabajo de los apartados posteriores.
#
# En particular, se separan:
# - las predicciones finales sobre TEST;
# - y los scores probabilísticos sobre TEST.
#
# Esta preparación permite alimentar de forma directa las funciones de
# comparación sin tener que reevaluar modelos ni reconstruir objetos en cada
# bloque.

print('\nCREACIÓN DE OBJETOS PARA LAS COMPARACIONES')
nombres_modelos = list(modelos_seleccionados.keys())

resultados_nombres = {}
for r in resultados_modelos:
    nombre = r['resumen']['modelo'].iloc[0]
    resultados_nombres[nombre]=r

predicciones_test = []
scores_test = []

for n in nombres_modelos:
    res = resultados_nombres[n]    
    predicciones_test.append(res['y_pred_test'])
    scores_test.append(res['y_score_test'])

# =============================================================================
# 4. COMPARACIÓN GLOBAL DE MODELOS
# =============================================================================
# Este bloque tiene como finalidad comparar de forma homogénea los modelos
# finales seleccionados en los apartados anteriores, manteniendo exactamente la
# misma partición train/test y el mismo conjunto de variables. De este modo,
# cualquier diferencia observada en las métricas puede atribuirse al modelo y no
# a cambios en los datos o en el proceso de validación.
#
# La comparación no se plantea como una nueva búsqueda automática de modelos,
# sino como una evaluación crítica y observacional de alternativas ya cerradas
# metodológicamente. Por ello, no se utilizan best_estimator_ ni reglas
# automáticas de selección final, sino que se analizan de forma conjunta las
# métricas principales, los errores de clasificación, la estabilidad y la
# complejidad relativa de cada enfoque.
# -----------------------------------------------------------------------------
# 4.1. COMPARACIÓN GLOBAL DE MÉTRICAS
# -----------------------------------------------------------------------------
# Este apartado permite obtener una primera visión del rendimiento de
# los modelos finales con las métricas principales del proyecto:
# AUC, accuracy, recall, precision, f1, f1_macro, balanced_accuracy y
# specificity.
#
# Su utilidad es doble:
# 1. comparar de forma homogénea el rendimiento de los modelos finales;
# 2. detectar rápidamente si existe un modelo líder o si, por el
#    contrario, aparecen trade-offs entre capacidad discriminante, acierto
#    global y rendimiento sobre la clase positiva.
#
# Se utilizan estas métricas porque son las que se han venido empleando a lo
# largo del proyecto y permiten mantener la coherencia metodológica entre el
# análisis individual de cada modelo y la comparación final.
print('\nCOMPARACIÓN GLOBAL DE MÉTRICAS')
grafico_comparacion_metricas(df_resumen_global, nombres_modelos)

# La comparación global de métricas muestra un patrón bastante claro. XGBoost
# se sitúa como el modelo más sólido del bloque final, ya que ocupa la zona
# más alta o prácticamente más alta en casi todas las métricas principales del
# proyecto. Destaca especialmente en AUC test, accuracy test, precision,
# f1-score, balanced accuracy y specificity, lo que indica que combina muy bien
# capacidad discriminante, acierto global y control de falsas alarmas.
#
# El stacking con meta-modelo de regresión logística aparece como la alternativa
# más cercana a XGBoost. Su rendimiento es muy competitivo y mantiene valores
# muy próximos en AUC, accuracy y métricas derivadas, aunque queda ligeramente
# por detrás en precision, f1 y specificity. En consecuencia, se confirma como
# un ensemble potente, pero sin llegar a superar al mejor XGBoost individual en
# el equilibrio global del problema.
#
# El Bagging Árbol presenta un comportamiento intermedio. Mejora claramente al
# SVM RBF en la mayor parte de métricas y ofrece un perfil bastante equilibrado,
# pero queda por debajo de XGBoost y del stacking tanto en capacidad
# discriminante como en rendimiento global sobre test.
#
# Por su parte, el SVM RBF conserva un recall alto, alineado con el objetivo de
# detectar clientes con riesgo real de abandono, pero sufre una caída más clara
# en precision, f1, balanced accuracy y specificity. Esto sugiere que mantiene
# una buena capacidad de detección, pero a costa de generar más falsas alarmas,
# lo que penaliza su utilidad práctica frente a las alternativas ensemble.
#
# En conjunto, esta primera comparación sugiere que la jerarquía global de los
# modelos queda encabezada por XGBoost, seguido muy de cerca por el stacking,
# con Bagging Árbol en una posición intermedia y SVM RBF como la alternativa
# menos competitiva dentro del bloque final. No obstante, esta lectura debe
# completarse con los apartados siguientes, donde se analizarán también los
# errores de clasificación, el sobreajuste y la estabilidad relativa de cada
# enfoque.


# -----------------------------------------------------------------------------
# 4.2. MATRICES DE CONFUSIÓN Y HEATMAPS DE CLASSIFICATION REPORT
# -----------------------------------------------------------------------------
# Las matrices de confusión permiten traducir las métricas agregadas a errores
# concretos de clasificación. En un problema de riesgo de abandono, esta 
# lectura es especialmente relevante, ya que no tiene el mismo significado 
# confundir un cliente que se quedará con uno que realmente abandonará, 
# que cometer el error inverso.
#
# Por ello, este apartado complementa la comparación de métricas mostrando de
# forma directa cuántos verdaderos positivos, verdaderos negativos, falsos
# positivos y falsos negativos produce cada modelo. Esta información es clave
# para interpretar la utilidad operativa real de los clasificadores.
#
# Los heatmaps facilitan además una comparación visual rápida entre modelos en
# precision, recall y f1-score de cada clase, lo que ayuda a valorar el
# equilibrio entre sensibilidad y control del error, ampliando la información
# de la matriz de confusión y permitiendo analizar con más detalle el 
# comportamiento de cada modelo sobre ambas clases.

print('\nMATRICES DE CONFUSIÓN')
matrices_confusion(y_test, predicciones_test, nombres_modelos)

print('\nHEATMAPS CLASSIFICATION REPORT')
heatmap_classification_reports(y_test, predicciones_test, nombres_modelos)

# La lectura conjunta de las matrices de confusión y de los heatmaps del
# classification report permite aterrizar la comparación global en términos
# operativos, observando no solo qué modelo obtiene mejores métricas agregadas,
# sino también cómo se reparten sus errores de clasificación.
#
# En primer lugar, el SVM RBF mantiene una capacidad de detección alta sobre la
# clase positiva, con 127 verdaderos positivos y solo 14 falsos negativos. Sin
# embargo, este resultado se consigue a costa de un exceso claro de falsos
# positivos (69), muy por encima del resto de modelos. Esto explica su baja
# precision en la clase positiva (0.648), así como su peor f1-score positivo
# (0.754) y su menor specificity. En términos prácticos, es un modelo agresivo:
# detecta bien el riesgo de abandono, pero lanza demasiadas falsas alarmas.
#
# El Bagging Árbol corrige gran parte de ese problema, reduciendo de forma
# notable los falsos positivos hasta 38. No obstante, esa mejora viene
# acompañada de una pérdida en la detección de clientes con riesgo real, ya que
# aumenta los falsos negativos hasta 18 y reduce el recall de la clase positiva
# a 0.872. Por tanto, presenta un perfil más conservador: mejora bastante frente
# al SVM en control del error, pero pierde sensibilidad sobre la clase de interés.
#
# XGBoost vuelve a mostrar la matriz de confusión más sólida del bloque final.
# Mantiene un nivel de detección muy alto sobre la clase positiva, con 127
# verdaderos positivos y solo 14 falsos negativos, y además consigue el menor
# número de falsos positivos del conjunto (34). Esta combinación se traduce en
# la mejor precision de la clase positiva (0.789), el mejor f1-score positivo
# (0.841) y la mayor specificity, confirmando que es el modelo más equilibrado
# entre capacidad de detección y control de falsas alarmas.
#
# El stacking con meta-modelo de regresión logística presenta un comportamiento
# muy próximo al de XGBoost. Mantiene 127 verdaderos positivos y 14 falsos
# negativos, por lo que conserva el mismo recall de la clase positiva (0.901),
# con 37 falsos positivos. Esto sitúa su precision positiva en 0.774 y su
# f1-score positivo en 0.833, quedando muy cerca de XGBoost y claramente por
# delante de Bagging Árbol y SVM RBF.
#
# En conjunto, este apartado refuerza con bastante claridad la jerarquía final
# de modelos. XGBoost sigue siendo la referencia principal, al combinar máxima
# detección de riesgo de abandono con el mejor control de falsos positivos. El stacking se
# sitúa inmediatamente después, con un comportamiento muy competitivo y una
# pérdida muy pequeña respecto al mejor modelo individual. Bagging Árbol ocupa
# una posición intermedia, penalizado por su menor recall, mientras que SVM RBF
# queda como la alternativa más débil del bloque final debido a su elevado número
# de falsos positivos.

# -----------------------------------------------------------------------------
# 4.3. CURVAS ROC COMPARATIVAS
# -----------------------------------------------------------------------------
# Las curvas ROC se incluyen para comprobar si la superioridad de un modelo es 
# estable o solo aparente.
#
# Mientras que la accuracy o el recall dependen de una decisión concreta de
# clasificación, la curva ROC muestra cómo se comporta cada modelo a lo largo de
# distintos umbrales posibles. Por ello, este apartado resulta especialmente
# útil para reforzar la lectura de la AUC test y comprobar si la superioridad de
# un modelo es estable o solo puntual.
#
# En un problema con cierto desbalanceo entre clases, la AUC sigue siendo una de
# las medidas más sólidas para comparar capacidad de separación entre clientes
# con y sin abandono.

print('\nCURVAS ROC COMPARATIVAS')
curva_auc_roc_comparativa(y_test, scores_test, nombres_modelos)

# La comparación de las curvas ROC confirma, de forma visual, la jerarquía ya
# observada en las métricas globales. XGBoost presenta la mayor capacidad
# discriminante del bloque final, con una AUC test de 0.9474, situándose como
# el modelo que mejor separa, en términos globales, a los clientes con riesgo 
# de abandono de los que no abandonan.
#
# El stacking con meta-modelo de regresión logística se mantiene muy próximo a
# XGBoost, con una AUC test de 0.9467. La cercanía entre ambas curvas sugiere
# que el stacking conserva una capacidad de ordenación probabilística muy alta,
# aunque sin llegar a superar al mejor modelo individual. En este sentido, se
# confirma como una alternativa muy competitiva, pero ligeramente por detrás de
# XGBoost en discriminación global.
#
# El Bagging Árbol también ofrece una curva ROC sólida, con una AUC test de
# 0.9447, aunque se observa algo más alejado de los dos mejores modelos en la
# zona de falsos positivos bajos, precisamente la región más relevante cuando se
# busca mantener una buena detección sin disparar falsas alarmas. Por tanto, su
# rendimiento discriminante es alto, pero algo menos fino que el de XGBoost y
# stacking.
#
# El SVM RBF queda claramente descolgado respecto al resto, con una AUC test de
# 0.9189. Aunque sigue mostrando una capacidad predictiva razonable, su curva se
# sitúa sistemáticamente por debajo de las demás, lo que indica una menor
# capacidad para ordenar correctamente los casos positivos y negativos a lo
# largo de distintos umbrales.
#
# En conjunto, este apartado refuerza la conclusión de que XGBoost es el modelo
# más sólido del bloque final en términos de capacidad discriminante global,
# seguido muy de cerca por el stacking. Bagging Árbol mantiene un comportamiento
# competitivo, aunque algo inferior, mientras que SVM RBF queda como la opción
# más débil en este criterio de comparación.

# -----------------------------------------------------------------------------
# 4.4. IMPORTANCIA DE VARIABLES EN LOS MODELOS
# -----------------------------------------------------------------------------
# Se comprueba como influyen las diferentes variables, se incorpora como bloque
# complementario de interpretación.
#
# Este análisis no se utiliza para seleccionar variables ni para reabrir el
# proceso de modelado, sino para enriquecer la lectura final del proyecto. Su
# interés radica en comprobar si distintos enfoques predictivos tienden a apoyarse
# en señales similares o si, por el contrario, explotan patrones diferentes del
# problema.
#
# Además, al calcularse con el mismo conjunto X_test / y_test y con la misma
# métrica (roc_auc), se garantiza una comparación homogénea entre modelos.
print('\nIMPORTANCIA DE VARIABLES EN LOS MODELOS')

df_importancia_vars = importancia_variables(
    modelos_seleccionados, X_test, y_test, scoring='roc_auc')

# ---------------------------------------
# TOP5 DE VARIABLES
# ---------------------------------------
top_vars = (
    df_importancia_vars.sort_values(['modelo', 'importancia_media'], 
                                   ascending=[True, False]).groupby('modelo')
    .head(5).reset_index(drop=True))
print('\nTop 5 variables por modelo:')
print(top_vars.round(4).to_string(index=False))

# La comparación de importancias confirma una idea bastante consistente a lo
# largo de todo el proyecto: los modelos más competitivos se apoyan, en gran
# medida, en un núcleo común de variables. En los cuatro enfoques destacan de
# forma muy clara coste_dia, n_att_cl y plan_inter_1, que aparecen
# sistemáticamente en las primeras posiciones y con una pérdida de ROC_AUC muy
# superior al resto al ser permutadas.
#
# Esta coincidencia resulta relevante porque refuerza la estabilidad
# interpretativa del problema: aunque cambie el algoritmo, la señal predictiva
# principal sigue concentrándose en un mismo bloque de variables. Por tanto, no
# parece que el buen rendimiento de XGBoost, Bagging o Stacking dependa de
# patrones completamente distintos, sino de explotar con distinta eficacia una
# estructura informativa bastante compartida.
#
# Entre esas variables comunes, coste_dia y n_att_cl sobresalen con mucha
# claridad sobre el resto en todos los modelos. Esto sugiere que la intensidad
# de uso y el comportamiento asociado a la atención al cliente constituyen el
# núcleo más discriminante para separar clientes con y sin riesgo de abandono.
# En un segundo escalón aparece plan_inter_1, que también mantiene una 
# contribución muy estable y relevante en los cuatro enfoques.
#
# A partir de ahí sí se observan pequeños matices entre modelos. SVM RBF
# concentra más peso en las variables líderes y cae con mayor rapidez a partir
# de la cuarta o quinta posición, lo que encaja con un patrón algo más apoyado
# en un subconjunto reducido de señales muy fuertes. En cambio, Bagging Árbol,
# XGBoost y Stacking muestran una distribución algo más equilibrada en la parte
# media del ranking, incorporando con más claridad variables como coste_tarde o
# coste_inter como apoyo complementario.
#
# El caso del stacking es especialmente interesante, porque sus variables clave
# reproducen en gran medida el patrón de los mejores modelos individuales. Esto
# es coherente con su propia naturaleza: el meta-modelo no descubre una lógica
# completamente nueva, sino que combina y reorganiza señales ya valiosas
# aprendidas por los modelos base. En ese sentido, la cercanía del ranking de
# stacking al de XGBoost y Bagging refuerza la idea de que el ensemble final se
# construye sobre una base predictiva coherente y no sobre dependencias
# espurias.
#
# En conjunto, este bloque aporta una lectura final útil: los mejores modelos
# no solo compiten en métricas muy próximas, sino que además coinciden en
# señalar prácticamente las mismas variables como principales responsables de su
# capacidad discriminante. Esto añade solidez a la interpretación global del
# proyecto.
#
# Debe recordarse, no obstante, que estas importancias por permutación miden
# pérdida de rendimiento predictivo al desordenar una variable, no causalidad.
# Además, cuando existen variables relacionadas entre sí, parte de su
# importancia puede repartirse entre varias columnas, por lo que la lectura debe
# hacerse en clave predictiva e interpretativa, no causal.

# -----------------------------------------------------------------------------
# 4.5. CURVAS DE VALIDACIÓN
# -----------------------------------------------------------------------------
# Las curvas de validación se utilizan aquí como herramienta de apoyo visual
# para analizar la estabilidad local de algunos parámetros relevantes en los
# modelos ya seleccionados.
#
# No se emplean para volver a optimizar ni para sustituir la selección realizada
# en los scripts anteriores, sino para comprobar si los valores finalmente
# elegidos se sitúan en zonas estables del espacio paramétrico o si, por el
# contrario, se apoyan en máximos demasiado frágiles.

print('\nCURVAS DE VALIDACIÓN')

rango_rbf = {
    'C': [0.8, 0.9, 1.0, 1.1, 1.2],
    'gamma': [0.06, 0.07, 0.08, 0.09, 0.10]
    }
rango_bag = {
    'n_estimators': [80, 90, 100, 110],
    'max_samples': [0.82, 0.85, 0.88, 0.90],
    'max_features': [0.65, 0.67, 0.69, 0.71]
    }
rango_xgb = {
    'n_estimators': [80, 100, 120, 140],
    'learning_rate': [0.02, 0.03, 0.04, 0.05],
    'min_child_weight': [8, 9, 10, 11, 12]
    }
rango_st = {
    'final_estimator__C': [1.5, 1.75, 2, 2.25, 2.5],
    }

rangos_validacion = {
    nombre_rbf: rango_rbf,
    nombre_bag:rango_bag,
    nombre_xgb: rango_xgb,
    nombre_st: rango_st        
    }

resultados_val = curvas_validacion(
    modelos_seleccionados, rangos_validacion, X_train, y_train, cv, 'roc_auc')

print('\nResumen curva validación por modelo')
print(resultados_val)

# La lectura conjunta de las curvas de validación refuerza una idea importante:
# los modelos finales no parecen apoyarse, en general, en valores paramétricos
# extremadamente frágiles, sino en zonas razonablemente estables del espacio de
# búsqueda. Esto es relevante porque aporta solidez a la comparación global
# posterior y sugiere que el rendimiento observado en test no depende de un
# ajuste puntual difícil de reproducir.
#
# En SVM RBF, tanto C como gamma muestran una evolución suave: al aumentar el
# parámetro, la curva de train mejora de forma progresiva, mientras que la de
# validación cruzada apenas crece y termina entrando en una zona casi plana.
# Esto sugiere que el modelo gana capacidad de ajuste en entrenamiento más
# rápido de lo que mejora en validación, por lo que no conviene interpretar los
# valores más altos como una ganancia real clara. En consecuencia, el ajuste
# final puede considerarse situado en una zona competitiva y estable, sin
# depender de un máximo excesivamente agudo.
#
# En Bagging Árbol, las curvas son especialmente tranquilizadoras desde el punto
# de vista de la robustez. El número de estimadores apenas modifica la validación
# cruzada y, de hecho, aumentarlo no aporta una mejora real. Del mismo modo,
# max_samples y max_features presentan oscilaciones pequeñas, con una meseta
# bastante amplia en la que varios valores ofrecen resultados muy similares.
# Esto encaja con la conclusión alcanzada en el script anterior: el bagging se
# mueve en una zona paramétrica estable y la selección final responde más a un
# criterio de equilibrio y parsimonia que a una búsqueda de máximos puntuales.
#
# XGBoost es el modelo que muestra la señal más clara de sensibilidad al ajuste.
# En n_estimators y learning_rate se observa un patrón bastante nítido: la curva
# de train crece con fuerza mientras que la validación cruzada se estanca o
# incluso empeora ligeramente. Esto sugiere que, a partir de cierto nivel de
# complejidad, el modelo tiende a capturar mejor el entrenamiento sin traducir
# esa ganancia en una mejora equivalente de generalización. Por tanto, estas
# curvas respaldan una lectura prudente del ajuste final: el modelo seleccionado
# no debe interpretarse como el que maximiza la validación cruzada local en cada
# parámetro aislado, sino como el que mejor equilibrio ofrece en el rendimiento
# final sobre test.
#
# En min_child_weight, sin embargo, el comportamiento de XGBoost es bastante más
# estable. Las diferencias en validación cruzada son reducidas a lo largo del
# rango analizado, lo que indica que este parámetro tiene un efecto más suave en
# la zona explorada y que el valor finalmente utilizado se sitúa dentro de un
# entorno competitivo.
#
# El meta-modelo de Stacking (R.Log.) presenta el patrón más estable de todos.
# La curva asociada a final_estimator__C es prácticamente plana tanto en train
# como en validación cruzada, lo que indica que pequeñas variaciones en la
# regularización del meta-modelo apenas alteran su capacidad discriminante. Esta
# estabilidad es coherente con la lógica del stacking: buena parte de la señal
# ya viene determinada por los modelos base, y el meta-modelo actúa más como
# combinador de probabilidades que como una fuente autónoma de complejidad.
#
# En conjunto, este bloque permite extraer dos conclusiones útiles. La primera
# es que Bagging Árbol y Stacking se apoyan en zonas paramétricas especialmente
# estables, lo que refuerza su robustez comparativa. La segunda es que XGBoost y,
# en menor medida, SVM RBF, muestran una sensibilidad algo mayor a ciertos
# parámetros, aunque sin que ello invalide la selección final realizada. Por
# tanto, las curvas de validación no reabren el proceso de ajuste, pero sí
# ayudan a contextualizar qué modelos parecen descansar sobre una base más
# estable y cuáles dependen en mayor medida de un equilibrio entre complejidad 
# y generalización.
#
# -----------------------------------------------------------------------------
# 4.6. RANKING COMPARATIVO POR MÉTRICAS Y ERRORES
# -----------------------------------------------------------------------------
# Este apartado resume la comparación final en un panel visual ordenado por
# criterios de rendimiento, error y generalización.
#
# Su objetivo no es automatizar la elección del mejor modelo, sino facilitar una
# lectura comparativa clara y rápida de los principales indicadores empleados en
# la decisión final. Por ello, se combinan métricas donde "más alto es mejor"
# (como AUC, accuracy o recall) con errores y gaps donde "más bajo es mejor"
# (como falsos positivos, falsos negativos o gap train-test).
#
# El diseño en varias filas permite además separar visualmente:
# - rendimiento predictivo principal,
# - comportamiento operativo sobre errores de clasificación,
# - estabilidad y generalización.
#
# De este modo, el ranking actúa como una síntesis visual del bloque de
# comparación global, pero sin sustituir la interpretación crítica posterior.
print('\nRANKING COMPARATIVO GLOBAL')
ranking = grafico_ranking(df_resumen_global, metricas_analisis)
print(ranking.round(4).to_string(index=False))

# La lectura del ranking comparativo confirma, de forma muy sintética, la
# jerarquía que ya se venía intuyendo en los apartados anteriores.
#
# XGBoost ocupa la posición más fuerte en el balance global del bloque final.
# Lidera las métricas principales de rendimiento predictivo (AUC test,
# accuracy test, precision y specificity) y también presenta el menor número
# de falsos positivos. Además, mantiene el mismo recall que los modelos más
# competitivos, por lo que no sacrifica capacidad de detección de clientes con
# posible riesgo de abandono para conseguir esa mejora en precisión. En 
# conjunto, esto refuerza la idea de que XGBoost es el modelo más completo 
# cuando se busca maximizar rendimiento operativo global.
#
# Stacking (R.Log.) aparece de forma consistente como segunda alternativa del
# ranking. Aunque queda ligeramente por detrás de XGBoost en las métricas de
# rendimiento puro, mantiene un comportamiento muy competitivo y cercano en la
# mayoría de los indicadores relevantes. De hecho, su distancia respecto al 
# primer puesto es pequeña y su perfil resulta especialmente sólido por 
# combinar una AUC muy alta, accuracy elevada, buen control de falsos positivos
# y el mismo nivel de recall que XGBoost y SVM RBF. Por ello, puede 
# interpretarse como la segunda opción más robusta del bloque de candidatos.
#
# Bagging Árbol se sitúa en una posición intermedia. Su principal fortaleza
# aparece en la generalización medida mediante gap_auc, donde obtiene el mejor
# resultado del conjunto. Sin embargo, ese mejor comportamiento relativo en
# estabilidad no termina de traducirse en una ventaja competitiva global,
# ya que pierde claramente frente a XGBoost y Stacking en recall, falsos
# negativos y métricas derivadas de la detección de la clase positiva. En otras
# palabras, es un modelo razonablemente sólido y estable, pero menos eficaz
# desde el punto de vista operativo del problema.
#
# SVM RBF queda en la posición más débil del ranking global. Aunque conserva un
# recall alto y comparte con XGBoost y Stacking el menor número de falsos
# negativos, sufre una pérdida importante en precision, specificity y falsos
# positivos, lo que penaliza su utilidad práctica. También presenta la AUC test
# y la accuracy test más bajas del conjunto. Por tanto, su principal fortaleza
# está en no dejar escapar potenciales clientes con riesgo de abandono, pero 
# lo hace a costa de generar más falsas alarmas que el resto de alternativas.
#
# En conjunto, el ranking no debe interpretarse como una regla automática de
# decisión, sino como una síntesis visual del equilibrio entre rendimiento,
# errores y generalización. Desde esa perspectiva, la comparación final apunta a
# una conclusión clara: XGBoost ofrece el perfil más fuerte del bloque, Stacking
# se consolida como la alternativa más cercana y competitiva, Bagging Árbol
# queda como opción intermedia con buen comportamiento en estabilidad, y SVM RBF
# aparece como el modelo menos equilibrado para el objetivo concreto del
# problema.

# -----------------------------------------------------------------------------
# 4.7. HEATMAP COMPARATIVO DE MÉTRICAS
# -----------------------------------------------------------------------------
# Este gráfico actúa como un consolidado visual de alto nivel que permite
# identificar patrones de comportamiento entre modelos sin necesidad de 
# analizar tablas densas de resultados.
#
# Al normalizar todas las métricas en una escala común (0 a 1) e invertir el
# sentido de los errores (fp, fn y gaps), el heatmap ofrece una lectura 
# intuitiva: a mayor intensidad de color, mejor es el rendimiento relativo 
# del modelo en esa dimensión específica.
#
# Su función principal es facilitar la detección de fortalezas y debilidades
# cruzadas, permitiendo visualizar de un solo vistazo si un modelo destaca 
# de forma equilibrada en todas las áreas o si su rendimiento se apoya
# únicamente en métricas puntuales.

print('\nHEATMAP COMPARATIVO GLOBAL')
heatmap_metricas(df_resumen_global)

# La lectura del heatmap comparativo refuerza de forma muy clara las
# conclusiones que ya se venían observando en los apartados anteriores.
#
# XGBoost aparece como el modelo más sólido en la visión global del bloque.
# Presenta la mejor intensidad relativa en la mayoría de métricas principales,
# especialmente en auc_test, accuracy_test, precision, f1, f1_macro,
# balanced_accuracy, specificity y control de falsos positivos. Además,
# mantiene también un comportamiento competitivo en recall y falsos negativos,
# por lo que su fortaleza no se apoya en una única dimensión, sino en un
# rendimiento alto y bastante equilibrado.
#
# Stacking (R.Log.) se sitúa como la alternativa más cercana a XGBoost. Aunque
# queda ligeramente por detrás en varias métricas de rendimiento puro, conserva
# un perfil muy competitivo y homogéneo, con valores altos en auc_test,
# accuracy_test, recall, precision y métricas derivadas. En la representación
# normalizada se aprecia, por tanto, como un modelo muy equilibrado, con pocas
# debilidades claras y un comportamiento global muy próximo al del mejor modelo.
#
# Bagging Árbol ocupa una posición intermedia. El heatmap muestra que mantiene
# un rendimiento razonable en bastantes dimensiones, pero sin llegar al nivel
# de XGBoost ni de Stacking en las métricas más importantes para el problema.
# Su principal punto fuerte vuelve a aparecer en la estabilidad relativa,
# especialmente en gap_auc, donde destaca frente al resto. Sin embargo, esa
# ventaja no compensa completamente su peor comportamiento en recall, fn y
# algunas métricas asociadas a la detección de la clase positiva.
#
# SVM RBF, por su parte, presenta el perfil más débil dentro de la comparación
# global. Aunque conserva un buen comportamiento relativo en recall, fn y
# gap_accuracy, el heatmap evidencia que pierde claramente terreno en auc_test,
# accuracy_test, precision, f1, f1_macro, balanced_accuracy, specificity y
# control de falsos positivos. En consecuencia, su rendimiento resulta menos
# equilibrado y menos competitivo desde el punto de vista operativo.
#
# En conjunto, esta visualización permite confirmar que no solo importa qué
# modelo lidera una métrica aislada, sino también la consistencia de su
# comportamiento a través de todas las dimensiones evaluadas. Desde esa
# perspectiva, el heatmap consolida la misma jerarquía general ya detectada:
# XGBoost como opción más fuerte, Stacking como alternativa muy próxima,
# Bagging Árbol como modelo intermedio con cierta fortaleza en estabilidad, y
# SVM RBF como la opción menos equilibrada del bloque final.

# =============================================================================
# 5. INTERPRETACIÓN FINAL
# =============================================================================
# En esta etapa se trasciende el análisis puramente estadístico para integrar
# factores de gestión y viabilidad operativa. La decisión final de un modelo
# que pretende detectar el riesgo de abandono no puede depender únicamente del 
# AUC, sino que debe considerar:
# - el impacto operativo de los errores (coste de falsos negativos),
# - la interpretabilidad del modelo (necesaria para acciones comerciales),
# - y el coste computacional asociado a su puesta en producción.
#
# Por ello, se construye una tabla maestra que cruza las métricas técnicas 
# críticas con indicadores de complejidad y explicabilidad. Este enfoque 
# multidimensional asegura que el modelo seleccionado no sea solo el más 
# preciso en el test, sino el más útil y sostenible para la organización.
print('\nDECISIÓN FINAL')

info_gestion = pd.DataFrame({
    'modelo': [nombre_rbf, nombre_bag, nombre_xgb, nombre_st],
    'complejidad': ['Media', 'Media', 'Alta', 'Alta'],
    'explicabilidad': ['Media', 'Media', 'Baja', 'Baja'],
    'coste_computacional': ['Medio', 'Medio', 'Alto', 'Alto']
    })

cols_decision = ['modelo', 'auc_test', 'recall', 'fp', 'fn', 'gap_auc']

tabla_decision = (
    df_resumen_global[cols_decision].merge(info_gestion, on='modelo')
    .sort_values(['recall', 'auc_test'], ascending=[False, False]))

print('\nResumen: Rendimiento vs Operativa vs Coste Computacional')
print(tabla_decision.round(4).to_string(index=False))

# La tabla de decisión final permite integrar dos planos complementarios:
# por un lado, el rendimiento predictivo real sobre test; por otro, la
# viabilidad operativa del modelo en términos de complejidad,
# explicabilidad y coste computacional.
#
# La comparación muestra que XGBoost y Stacking (R.Log.) forman el grupo
# claramente dominante desde el punto de vista predictivo. Ambos alcanzan
# el mayor recall del bloque (0.9007), lo que implica que detectan el mismo
# porcentaje de clientes con riesgo real de abandono y, además, ambos
# minimizan los falsos negativos (fn = 14), aspecto especialmente relevante
# en este problema, donde no identificar a un cliente que realmente va a
# abandonar resulta más costoso que generar una acción preventiva innecesaria.
#
# Entre estos dos modelos de alto rendimiento, XGBoost presenta una ventaja
# ligera pero consistente frente al stacking: obtiene el mejor auc_test
# (0.9474 frente a 0.9467), reduce ligeramente los falsos positivos
# (34 frente a 37) y mantiene también un gap_auc algo más contenido,
# lo que sugiere una generalización algo más sólida. Por tanto, si la
# prioridad principal es maximizar la capacidad predictiva global del
# sistema, XGBoost se posiciona como la mejor alternativa final.
#
# SVM RBF conserva un recall idéntico al de los dos mejores modelos y
# comparte también el mismo número de falsos negativos (14), lo que a
# primera vista lo mantiene como opción competitiva. Sin embargo, su auc_test
# es claramente inferior y, sobre todo, incrementa de forma muy notable los
# falsos positivos (69), lo que implicaría activar muchas más acciones
# comerciales innecesarias. En consecuencia, aunque su coste computacional y
# su explicabilidad son más favorables, su eficiencia operativa global queda
# por debajo de XGBoost y Stacking.
#
# Bagging Árbol, por su parte, presenta una complejidad media y una
# explicabilidad razonable, pero pierde ventaja precisamente en el criterio
# más sensible del problema: reduce su recall hasta 0.8723 y aumenta los
# falsos negativos a 18. Esto significa que dejaría escapar más clientes con
# riesgo real de abandono, lo que penaliza su utilidad práctica a pesar de
# su menor coste.
#
# En conclusión, la decisión final razonada es seleccionar XGBoost como
# modelo ganador del proyecto. Es el modelo que ofrece la mejor combinación
# entre discriminación global, capacidad de detección de la clase positiva y
# control simultáneo de errores. Stacking (R.Log.) queda como alternativa
# muy próxima y técnicamente solvente, mientras que SVM RBF puede
# interpretarse como una opción de compromiso si en un contexto real se
# priorizara una solución más simple y algo más interpretable. Bagging Árbol,
# aun siendo un modelo competitivo, queda por detrás debido a su peor
# comportamiento en recall y falsos negativos.

# =============================================================================
# 6. SELECCIÓN FINAL RAZONADA
# =============================================================================
# La elección final no se realiza por ranking automático ni por una única
# métrica, sino por observación conjunta del rendimiento, los errores de
# clasificación, la estabilidad y la complejidad relativa del modelo
print('\nSELECCIÓN FINAL RAZONADA')

# Resultados de la observación crítica
mejor_modelo_global = 'XGBoost'
mejor_modelo_operativo = 'XGBoost'

print(f'\n-> Ganador por Potencia Predictiva (AUC): {mejor_modelo_global}')
print(f'-> Ganador por Perfil Operativo Global: {mejor_modelo_operativo}')

# -----------------------------------------------------------------------------
# 6.1. INTERPRETACIÓN FINAL DE LA COMPARACIÓN
# -----------------------------------------------------------------------------
# La comparación global confirma que no existe un dominio absoluto de todos
# los modelos en todos los criterios, pero sí una alternativa que sobresale
# con mayor claridad en el equilibrio conjunto entre rendimiento predictivo
# y utilidad práctica.
#
# XGBoost obtiene el mejor AUC test del conjunto, lo que indica la mayor
# capacidad global de discriminación entre clientes con y sin riesgo de
# abandono. Además, comparte el mejor recall observado (0.9007) y el menor
# número de falsos negativos (fn = 14), por lo que también se sitúa en la
# zona más favorable desde el punto de vista operativo: detecta el mismo
# volumen de clientes en riesgo real que los modelos más sensibles, pero con
# una capacidad de separación global superior.
#
# Frente a Stacking (R.Log.), la ventaja de XGBoost es pequeña pero
# consistente. Ambos modelos presentan el mismo recall y los mismos falsos
# negativos, pero XGBoost logra un auc_test ligeramente superior y reduce
# también los falsos positivos. Esto implica una política de retención algo
# más eficiente, al mantener la capacidad de detección sin incrementar
# innecesariamente las intervenciones comerciales.
#
# SVM RBF mantiene también el mismo recall y los mismos falsos negativos,
# pero a costa de un número de falsos positivos claramente más elevado. Por
# tanto, aunque sigue siendo un modelo útil y competitivo, su rendimiento
# operativo real queda penalizado por una menor precisión práctica en la
# identificación de clientes problemáticos.
#
# Bagging Árbol, aun mostrando un comportamiento sólido y una complejidad
# moderada, queda por detrás al reducir el recall y aumentar los falsos
# negativos. Dado que en este problema el coste de no identificar un cliente
# que realmente va a abandonar es especialmente relevante, esta pérdida de
# sensibilidad limita su candidatura final.
#
# En consecuencia, la selección final razonada recae en XGBoost tanto por
# potencia predictiva como por perfil operativo. No solo lidera la capacidad
# global de discriminación, sino que además mantiene el nivel óptimo de
# detección de la clase positiva con un control de errores más eficiente que
# sus principales competidores. Por ello, se considera el modelo más sólido
# y equilibrado del proyecto.

print('\nConclusión: XGBoost se selecciona como modelo final por ofrecer el \
mejor equilibrio global entre AUC, capacidad de detección del riesgo de \
abandono y eficiencia operativa.')

# -----------------------------------------------------------------------------
# 6.2. CONCLUSIÓN FINAL
# -----------------------------------------------------------------------------
# El desarrollo completo del proyecto permite extraer una conclusión sólida:
# la comparación final no se ha apoyado en una única métrica aislada, sino en
# una evaluación progresiva, coherente y multidimensional de los modelos,
# considerando de forma conjunta la capacidad discriminante, el rendimiento de
# clasificación, los errores operativos, la generalización y la complejidad
# relativa de cada enfoque.
#
# Esta forma de trabajo refuerza la validez de la decisión adoptada, ya que el
# modelo finalmente seleccionado no sobresale solo en una comparación puntual
# sobre el conjunto de test, sino que mantiene un comportamiento competitivo y
# consistente a lo largo de todo el proceso de análisis.
#
# Además, el proyecto pone de manifiesto que pequeñas diferencias en AUC no
# implican necesariamente una superioridad automática desde el punto de vista
# práctico. En un problema que trata de predecir el riesgo de abandono, resulta
# imprescindible complementar la lectura de la capacidad discriminante con 
# métricas como recall, precision, falsos negativos, falsos positivos y 
# estabilidad train-test, ya que el valor real del modelo depende de cómo 
# transforma su potencia predictiva en decisiones útiles para la organización.
#
# En este contexto, XGBoost se consolida como la alternativa más completa del
# proyecto, al combinar una AUC test muy alta con una detección eficaz de la
# clase positiva y un control del error especialmente favorable frente al resto
# de modelos comparados.
#
# Conviene señalar, además, que XGBoost no formaba parte del núcleo inicial del
# trabajo en los mismos términos que SVM, bagging y stacking, sino que se
# incorporó durante el desarrollo del bloque ensemble como modelo de contraste.
# Su inclusión permitió añadir una referencia boosting frente al enfoque
# bagging ya analizado, enriqueciendo así la comparación entre familias de
# modelos ensemble.
#
# Aunque su presencia no respondía a un requisito principal independiente del
# enunciado, su evaluación resultó metodológicamente muy útil, ya que permitió
# comprobar si una estrategia boosting podía aportar una mejora real tanto en
# capacidad discriminante como en rendimiento operativo. De hecho, lejos de
# quedar como un contraste secundario, terminó mostrando el comportamiento más
# sólido del proyecto en el equilibrio global entre AUC, precision, control del
# error y utilidad práctica.
#
# Como posibles líneas futuras de mejora, el trabajo podría ampliarse mediante:
# - calibración de probabilidades y revisión del punto de corte óptimo según
#   costes de negocio;
# - análisis coste-beneficio de campañas de retención a partir de FP y FN;
# - explicación local de predicciones con técnicas interpretativas específicas;
# - y validación adicional sobre nuevas particiones o datos temporales, en un
#   escenario más próximo a producción.
#
# En definitiva, el proyecto no solo permite identificar el modelo con mejor
# rendimiento global, sino también justificar de forma razonada su elección
# desde una perspectiva metodológica, técnica y operativa.

# =============================================================================
# 7. GUARDADO PARA LA PARTE FINAL DE INTERPRETACIÓN
# =============================================================================
print('\nGUARDADO')

bbdd_comparacion_global = {
    'modelos_evaluados': nombres_modelos,
    'df_resumen_global': df_resumen_global,
    'predicciones_test_global': predicciones_test,
    'scores_test_global': scores_test,
    'df_perm_global': df_importancia_vars,
    'top_perm_global': top_vars,
    'resultados_validacion': resultados_val,
    }

with open(RUTA_SALIDA, 'wb') as archivo:
    pickle.dump(bbdd_comparacion_global, archivo)

print(f'\nArchivo guardado: {RUTA_SALIDA.resolve()}')
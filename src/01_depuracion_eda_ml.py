# -*- coding: utf-8 -*-
# =============================================================================
# 01_depuracion_eda_ml.py
# Proyecto: Machine Learning 2026
#
# Objetivo:
# - Cargar la base de datos original.
# - Auditar su estructura, valores perdidos y registros duplicados.
# - Realizar una depuración inicial sin imputar ni escalar variables.
# - Analizar la coherencia interna de variables y relaciones básicas con la
#   variable objetivo.
# - Desarrollar el análisis exploratorio de datos (EDA) para apoyar la
#   selección de variables candidatas.
# - Guardar la base depurada en formato pickle para los scripts posteriores.
#
# Nota:
# - Las transformaciones que aprendan de los datos, como imputación,
#   codificación y escalado, se realizarán más adelante dentro de pipelines
#   ajustados exclusivamente con TRAIN.
# =============================================================================
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from funciones_ml import (
    v_cramer, matriz_corr, histogramas, resumen_categoricas,
    analisis_categoricas, boxplot_numericas, histogramas_densidad_y)

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# RUTAS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RUTA_DATOS = BASE_DIR / 'data' / 'BBDD_ML_TAREA.csv'
RUTA_SALIDA = BASE_DIR / 'outputs' / 'BBDD_eda.pickle'

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        'No se encontró BBDD_ML_TAREA.csv. '
        'Coloca el dataset en la carpeta data/ del proyecto.'
    )

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------


# =============================================================================
# DEPURACIÓN INICIAL
# =============================================================================
# -----------------------------------------------------------------------------
# 1.1. CARGA DE DATOS
# -----------------------------------------------------------------------------
# En primer lugar se carga la base de datos original en formato CSV y se
# realiza una comprobación inicial de su dimensión y aspecto general.
#
# Este paso permite verificar que la lectura del fichero se ha realizado
# correctamente, confirmar el número de observaciones y variables disponibles,
# y obtener una primera visión del formato de los datos antes de iniciar
# cualquier proceso de depuración.
#
# La base original contiene 9200 registros y 21 variables, lo que constituye
# el punto de partida del trabajo. En esta fase todavía se conserva la
# estructura bruta del fichero, sin aplicar ninguna modificación sobre nombres,
# duplicados o posibles incoherencias.

print('\nCARGA DE DATOS')
datos_input = pd.read_csv(RUTA_DATOS)
print('Ruta del CSV:', RUTA_DATOS)
print('Cantidad de datos:', datos_input.shape)
print(datos_input.head())

# -----------------------------------------------------------------------------
# 1.2. RENOMBRADO
# -----------------------------------------------------------------------------
# A continuación se renombran las variables originales para facilitar la
# interpretación del dataset y hacer más legible el análisis posterior.
#
# Este cambio no altera el contenido de la base, pero sí mejora de forma
# importante la claridad del script, ya que sustituye etiquetas genéricas
# (V1, V2, ..., V20) por nombres descriptivos asociados al significado de
# cada variable.
#
# De este modo, el resto del proceso de depuración, análisis exploratorio y
# modelado puede desarrollarse con una nomenclatura coherente y mucho más
# interpretable.

print('\nRENOMBRADO')
renombrado = {
    'V1': 'region',
    'V2': 'antiguedad',
    'V3': 'cod_area',
    'V4': 'tlf',
    'V5': 'plan_inter',
    'V6': 'buzon_voz',
    'V7': 'n_mns_voz',
    'V8': 'min_dia',
    'V9': 'n_dia',
    'V10': 'coste_dia',
    'V11': 'min_tarde',
    'V12': 'n_tarde',
    'V13': 'coste_tarde',
    'V14': 'min_noche',
    'V15': 'n_noche',
    'V16': 'coste_noche',
    'V17': 'min_inter',
    'V18': 'n_inter',
    'V19': 'coste_inter',
    'V20': 'n_att_cl',
    'Y': 'Y'
    }
datos_input = datos_input.rename(columns=renombrado)
print('\nVariables renombradas:')
print(datos_input.columns.tolist())

# -----------------------------------------------------------------------------
# 1.3. VISIÓN GENERAL DE LA BASE DE DATOS
# -----------------------------------------------------------------------------
# Una vez renombradas las variables, se realiza una auditoría general de la
# base para identificar su estructura, cardinalidad, presencia de valores
# perdidos y posibles valores anómalos.
#
# Esta revisión permite detectar desde el inicio varios aspectos relevantes
# para la depuración posterior:
# - coexistencia de variables numéricas continuas, discretas y categóricas
#   codificadas como enteros;
# - presencia de valores perdidos en algunas variables concretas, sobre todo
#   en cod_area y en varias variables de costes o consumo;
# - alta cardinalidad en variables como tlf, lo que ya anticipa su posible
#   papel como identificador;
# - y equilibrio aparente de la variable objetivo en la base original
#   (4600 casos de Y=0 y 4600 de Y=1).
#
# Esta última observación es especialmente importante, ya que más adelante se
# comprobará si dicho equilibrio responde al comportamiento real del problema
# o si está condicionado por la existencia de registros duplicados.

print('\nVISIÓN GENERAL DE LA BASE DE DATOS')
resumen = pd.DataFrame({
    'tipo': datos_input.dtypes.astype(str),
    'n_unicos': datos_input.nunique(dropna=True),
    'n_missing': datos_input.isna().sum(),
    'pct_missing': round(datos_input.isna().mean() * 100, 3),
    'negativos': (datos_input.select_dtypes(include=[np.number]) < 0).sum()
    })

describe_num = datos_input.describe().T
resumen = resumen.join(describe_num, how='left')

print('\nResumen de variables:')
print(resumen)
print('\nDistribución de Y:')
print(datos_input['Y'].value_counts())

# -----------------------------------------------------------------------------
# 1.4. DUPLICADOS EXACTOS Y PRIMERA DEPURACIÓN DE REGISTROS
# -----------------------------------------------------------------------------
# Antes de continuar con el análisis, se comprueba la existencia de registros
# duplicados exactos, ya que su presencia puede distorsionar tanto la
# distribución de la variable objetivo como la evaluación posterior de los
# modelos.
#
# El resultado muestra un volumen muy elevado de duplicados exactos
# (5662 observaciones), por lo que su eliminación constituye una depuración
# imprescindible y no una decisión secundaria.
#
# Tras eliminar estos registros repetidos, la base se reduce a 3538 filas.
# Además, la distribución de la variable objetivo cambia de forma muy acusada:
# se pasa de un aparente equilibrio 50/50 en la base original a una estructura
# claramente desbalanceada, con 2832 casos de Y=0 frente a 706 de Y=1.
#
# Este hallazgo es clave para el proyecto, ya que indica que la base original
# ofrecía una imagen artificialmente equilibrada del problema. Por tanto, la
# eliminación de duplicados no solo limpia la muestra, sino que permite
# recuperar una representación mucho más realista del fenómeno a modelizar.

print('\nDUPLICADOS')
n_duplicados = datos_input.duplicated().sum()
print(f'\nDuplicados exactos: {n_duplicados}')

df = datos_input.drop_duplicates(keep='first').copy()
filas_totales = df.shape[0]
print(f'\nFilas sin duplicados: {filas_totales}')
print('\nDistribución de Y en la base sin duplicados exactos:')
print(df['Y'].value_counts())

# Con esta primera depuración queda corregido uno de los principales problemas
# estructurales del dataset. A partir de este punto, el análisis se realiza
# sobre una base más representativa, lo que permite revisar con mayor 
# fiabilidad la coherencia de las variables y su posible utilidad para el 
# modelado.

# -----------------------------------------------------------------------------
# 1.5. ANÁLISIS INICIAL DE LAS VARIABLES
# -----------------------------------------------------------------------------
# Tras la eliminación de duplicados exactos, se realiza una primera revisión
# funcional de algunas variables clave para comprobar su papel dentro del
# dataset y detectar posibles incoherencias estructurales.
#
# En este punto interesa especialmente revisar:
# - si cod_area puede tratarse como variable categórica discreta;
# - si tlf actúa en la práctica como identificador de cliente;
# - y si existe coherencia lógica entre la variable de buzón de voz y el número
#   de mensajes de voz registrados.

print('\nANÁLISIS DE LAS VARIABLES')

# cod_area se transforma a entero con soporte para valores perdidos, ya que
# toma un número muy reducido de códigos posibles y su interpretación es más
# coherente como variable categórica que como magnitud continua.

df['cod_area'] = df['cod_area'].round().astype('Int64')

# -------------------------------
# REVISIÓN DE tlf COMO POSIBLE IDENTIFICADOR
# -------------------------------
# La variable tlf presenta una cardinalidad muy alta, por lo que desde el inicio
# apunta a un posible papel identificador. Se comprueba cuántos valores únicos
# contiene y si todavía quedan clientes repetidos tras eliminar los duplicados
# exactos.
print('\nValores únicos de tlf:', df['tlf'].nunique(dropna=True))
print('Clientes duplicados en tlf:', df['tlf'].duplicated().sum())

df_tlf_duplicados = df[
    df.duplicated(subset='tlf', keep=False)].sort_values('tlf')
print('\nFilas con tlf duplicado:')
print(df_tlf_duplicados)

# El resultado muestra dos teléfonos repetidos, correspondientes a cuatro
# registros. Para evitar que un mismo cliente quede representado más de una
# vez, se aplica un criterio uniforme de depuración: para cada teléfono se
# conserva la fila con menor número de valores ausentes y, en caso de empate,
# la primera aparición.
#
# Esta depuración evita que un mismo cliente quede representado más de una vez
# en la base final, lo que podría introducir sesgos y aumentar artificialmente
# la información disponible en fases posteriores del análisis o del modelado.

df['tlf_nan'] = df.isna().sum(axis=1)
df = df.sort_values(['tlf', 'tlf_nan']).drop_duplicates(
    subset='tlf', keep='first').drop(columns='tlf_nan').sort_index().copy()
print('\nFilas con tlf duplicado tras depuración:',
      df['tlf'].duplicated().sum())
print('\nFilas eliminadas:', filas_totales - df.shape[0])
print('\nFilas tras depuración de tlf:', df.shape)

# -------------------------------
# COMPROBACIÓN DE COHERENCIA: BUZÓN DE VOZ Y MENSAJES
# -------------------------------
# Se revisa la consistencia lógica entre buzon_voz y n_mns_voz. En particular,
# no debería haber clientes con buzon_voz = 0 y, al mismo tiempo, con un número
# positivo de mensajes de voz.
#
# La ausencia de casos incoherentes en esta comprobación refuerza la calidad
# interna de ambas variables y permite mantenerlas sin necesidad de correcciones
# adicionales en esta fase.

df_error_buzon = df[(df['buzon_voz'] == 0) & (df['n_mns_voz'] > 0)]
print(f'\nErrores en datos de buzón de voz: {df_error_buzon.shape[0]}')

# -------------------------------
# CLASIFICACIÓN INICIAL DE VARIABLES
# -------------------------------
# Finalmente, se separan de forma preliminar las variables categóricas y
# numéricas para organizar el análisis exploratorio posterior. Esta división
# servirá como base para estudiar distribuciones, relaciones con la variable
# objetivo y posibles decisiones de depuración o transformación.

vars_categoricas = ['region', 'cod_area', 'plan_inter', 'buzon_voz']
vars_numericas = [
    'antiguedad', 'n_mns_voz', 'min_dia', 'n_dia', 'coste_dia', 'min_tarde',
    'n_tarde', 'coste_tarde', 'min_noche', 'n_noche', 'coste_noche', 
    'min_inter', 'n_inter', 'coste_inter', 'n_att_cl']

# Con esta revisión queda confirmado que tlf funciona de facto como un
# identificador de cliente, por lo que no tendrá interés predictivo directo
# y deberá tratarse con cautela en el resto del análisis. Además, no se
# detectan incoherencias entre buzon_voz y n_mns_voz, lo que permite continuar
# con una base ya depurada en sus principales aspectos estructurales.

# -----------------------------------------------------------------------------
# 1.6. ANÁLISIS DEL CONJUNTO: LLAMADAS, MINUTOS, COSTES Y COSTE/MINUTO
# -----------------------------------------------------------------------------
# En este bloque se estudia la relación entre número de llamadas, minutos y
# coste en cada tramo horario (día, tarde, noche e internacional).
#
# El objetivo es doble:
# - comprobar si el coste por minuto se mantiene prácticamente constante, lo que
#   permitiría utilizar esa relación como criterio de imputación;
# - y detectar incoherencias entre llamadas, minutos y costes que puedan revelar
#   errores de registro o variables derivadas de otras.
#
# Esta revisión es especialmente importante porque varias variables de coste
# podrían estar calculadas directamente a partir de los minutos, por lo que
# conviene verificar primero si esa dependencia es realmente estructural antes
# de incorporarlas al análisis predictivo.

print('\nANÁLISIS DE LLAMADAS, MINUTOS, COSTES Y COSTE/MINUTO')
filas_valores = []
filas_cero = []
tramos_consumo = [
    ('dia', 'n_dia', 'min_dia', 'coste_dia'),
    ('tarde', 'n_tarde', 'min_tarde', 'coste_tarde'),
    ('noche', 'n_noche', 'min_noche', 'coste_noche'),
    ('inter', 'n_inter', 'min_inter', 'coste_inter')
]

# -------------------------------
# RESUMEN DEL COSTE POR MINUTO
# -------------------------------
# Se calcula el coste medio por minuto en cada tramo, junto con su dispersión
# y rango observado. Si la desviación es prácticamente nula, podrá asumirse que
# el coste está determinado por una tarifa fija y que, por tanto, coste y
# minutos contienen información casi redundante.

for t, llamadas, m, coste in tramos_consumo:
    coste_minuto = (df[coste] / df[m].replace(0, np.nan))
    coste_minuto = coste_minuto.replace([np.inf, -np.inf], np.nan)
    filas_valores.append({
        'tramo': t,
        'media_llamadas': df[llamadas].mean(),
        'media_minutos': df[m].mean(),
        'media_coste': df[coste].mean(),
        'coste_minuto_medio': coste_minuto.mean(),
        'coste_minuto_std': coste_minuto.std(),
        'coste_minuto_min': coste_minuto.min(),
        'coste_minuto_max': coste_minuto.max()
        })

df_coste_minuto = pd.DataFrame(filas_valores)
tarifas = df_coste_minuto.set_index('tramo')['coste_minuto_medio']
print('\nResumen por tramo:')
print(df_coste_minuto)

# La tabla confirma que el coste por minuto es extraordinariamente estable en
# los cuatro tramos, con desviaciones mínimas y rangos muy estrechos. Esto
# respalda la idea de que las variables de coste se generan prácticamente como
# una transformación lineal de los minutos, por lo que podrán utilizarse como
# apoyo para imputar valores perdidos y, más adelante, deberán revisarse con
# cautela por posible redundancia predictiva.

# ----------------------------
# COMPROBACIÓN DE COHERENCIA ENTRE LLAMADAS, MINUTOS Y COSTES
# ----------------------------
# Se revisan combinaciones lógicamente problemáticas, como por ejemplo:
# - llamadas iguales a cero con minutos o costes positivos;
# - minutos iguales a cero con coste positivo;
# - coste igual a cero con minutos positivos;
# - o minutos iguales a cero con llamadas positivas.
#
# Esta comprobación permite localizar qué tramos presentan inconsistencias
# relevantes y ayuda a decidir después cómo imputar o depurar cada variable.

for t, llamadas, m, coste in tramos_consumo:
    n = df[llamadas].fillna(0)
    m = df[m].fillna(0)
    c = df[coste].fillna(0)

    n0 = ((n == 0) & ((m != 0) | (c != 0))).sum()
    min0_coste = ((m == 0) & (c != 0)).sum()
    coste0_min = ((c == 0) & (m != 0)).sum()
    min0_n = ((m == 0) & (n != 0)).sum()
    filas_cero.append({
        'tramo': t,
        'n_0_y_min_o_coste_no_0': n0,
        'min__0_y_coste_no_0': min0_coste,
        'coste_0_y_min_no_0': coste0_min,
        'min_0_y_n_no_0': min0_n
        })

df_cero = pd.DataFrame(filas_cero)
print('\nResumen coherencia de llamadas-minutos-costes:')
print(df_cero)

# Los resultados muestran que día y tarde presentan una estructura muy limpia,
# mientras que las incidencias se concentran sobre todo en noche e internacional.
# En particular, aparecen casos con coste igual a cero pese a existir minutos
# registrados, así como algunas combinaciones incoherentes entre número de
# llamadas y minutos. Esto sugiere que la lógica de generación de estas
# variables no es completamente homogénea en todos los tramos y que la
# imputación posterior deberá apoyarse en reglas consistentes con la tarifa
# observada.

# -----------------------------
# CORRELACIÓN ENTRE LLAMADAS, MINUTOS Y COSTES
# -----------------------------
# Finalmente, se analiza la correlación entre llamadas, minutos y costes para
# comprobar hasta qué punto las variables de coste son esencialmente derivadas
# de los minutos y si el número de llamadas aporta una señal distinta.
#
# Este análisis no se utiliza aún para eliminar variables, pero sí para dejar
# documentada una posible redundancia estructural que deberá tenerse en cuenta
# en la selección final de predictores.

coste_min = df[['n_dia', 'min_dia', 'coste_dia', 'n_tarde', 'min_tarde', 
                'coste_tarde', 'n_noche', 'min_noche', 'coste_noche','n_inter',
                'min_inter', 'coste_inter']].copy()
corr_coste_min = coste_min.corr(numeric_only=True)
matriz_corr(
    corr_coste_min, titulo='llamadas, minutos y costes')

# La matriz de correlación confirma de forma muy clara que, en cada tramo,
# minutos y coste están correlacionados prácticamente de forma perfecta
# (correlación ≈ 1). En cambio, el número de llamadas apenas se relaciona con
# minutos y costes. Esto refuerza dos ideas importantes:
# - los costes son variables casi derivadas de los minutos;
# - y el número de llamadas aporta una dimensión distinta del comportamiento
#   del cliente, menos ligada al volumen temporal de uso.

# En conjunto, este análisis sugiere que las variables de coste no deben
# interpretarse como fuentes independientes de información, sino como
# transformaciones casi directas de los minutos consumidos. Por ello, serán
# útiles para apoyar la imputación, pero más adelante habrá que valorar si
# merece la pena conservar simultáneamente minutos y costes en el modelado,
# ya que pueden introducir redundancia muy alta.
#
# Además, las pequeñas incoherencias detectadas en los tramos de noche e
# internacional justifican una revisión específica en la fase de depuración,
# especialmente cuando aparezcan combinaciones imposibles entre llamadas,
# minutos y coste.

# -----------------------------------------------------------------------------
# 1.7. ANÁLISIS GRÁFICO INICIAL
# -----------------------------------------------------------------------------
# Tras la revisión tabular previa, se realiza una primera inspección visual de
# las variables numéricas. Este bloque permite complementar el análisis
# descriptivo inicial mediante:
# - un boxplot conjunto, útil para detectar dispersión, diferencias de escala y
#   posibles valores extremos;
# - y un conjunto de histogramas, orientado a estudiar la forma de las
#   distribuciones, su simetría y su concentración.
#
# Esta revisión no se utiliza todavía para eliminar variables, pero sí para
# identificar patrones relevantes de cara a la imputación, el tratamiento de
# atípicos y la selección posterior de predictores.

print('\nANÁLISIS GRÁFICO INICIAL')

# -------------------------------
# BOXPLOT CONJUNTO DE VARIABLES NUMÉRICAS
# -------------------------------
# El boxplot conjunto permite detectar de forma rápida:
# - diferencias importantes de escala entre variables;
# - posibles valores extremos;
# - y tramos con mayor dispersión.
#
# Debe interpretarse con cautela, ya que al combinar variables con unidades
# muy distintas en una sola figura, algunas distribuciones pueden quedar
# visualmente comprimidas. Por ello, este gráfico se utiliza como visión
# general inicial y no como diagnóstico definitivo de atípicos.
plt.figure(figsize=(16, 8))

sns.boxplot(data=df[vars_numericas], palette='Set2')

plt.title('Boxplot conjunto de variables numéricas', fontsize=16, weight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------
# HISTOGRAMAS DE VARIABLES NUMÉRICAS SELECCIONADAS
# ---------------------------------
# A continuación se representan histogramas de un conjunto representativo de
# variables numéricas para estudiar la forma de sus distribuciones.
#
# Se incluyen variables de antigüedad, uso por tramos horarios, actividad
# internacional y atención al cliente, ya que son dimensiones relevantes del
# comportamiento del cliente y, además, permiten comprobar si las variables
# siguen patrones aproximadamente simétricos o si presentan asimetrías que
# puedan influir en la depuración y en el modelado posterior.
var_hist = ['antiguedad', 'min_dia', 'n_dia', 'min_tarde', 'n_tarde', 
            'min_noche', 'n_noche', 'min_inter', 'n_inter', 'n_att_cl']
histogramas(df, var_hist, columnas=2, 
            titulo='Distribución inicial de variables numéricas seleccionadas')

# En conjunto, los histogramas muestran que la mayoría de variables de uso
# general (antigüedad, minutos y número de llamadas en día, tarde y noche)
# presentan distribuciones bastante regulares, unimodales y relativamente
# centradas, sin indicios de comportamientos claramente anómalos.
#
# En cambio, las variables n_inter y n_att_cl muestran una estructura más
# discreta y asimétrica, con fuerte concentración en valores bajos y cola
# hacia la derecha. Esto es coherente con su naturaleza: no todos los clientes
# realizan muchas llamadas internacionales ni contactan con frecuencia con
# atención al cliente.
#
# Por tanto, esta primera inspección sugiere que no todas las variables deben
# tratarse igual en la depuración: mientras que varias variables de uso parecen
# bastante estables, otras presentan distribuciones más sesgadas y discretas,
# lo que convendrá tener en cuenta más adelante al interpretar atípicos y
# al seleccionar predictores.

# -----------------------------------------------------------------------------
# 1.8. DEPURACIÓN BASE PARA EDA Y MODELADO
# -----------------------------------------------------------------------------
# Una vez revisada la estructura general de la base, los duplicados, el papel
# identificativo de tlf y la coherencia básica de las variables, se construye
# una primera versión depurada del conjunto de datos para continuar con el
# análisis exploratorio y el modelado.
#
# En esta fase no se pretende todavía cerrar la selección final de variables,
# sino definir una base de trabajo limpia y coherente sobre la que poder:
# - estudiar relaciones con la variable objetivo;
# - analizar valores perdidos;
# - revisar la utilidad real de las variables geográficas;
# - y preparar posteriormente la depuración específica para el modelado.
#
# Por ello, la depuración aquí es deliberadamente conservadora:
# - se elimina tlf por su carácter identificativo y por no aportar valor
#   predictivo generalizable;
# - se mantienen region y cod_area para comprobar si la localización aporta
#   información relevante;
# - y se conservan las variables de coste, aunque algunas puedan resultar
#   redundantes más adelante, porque en este punto todavía son útiles para
#   analizar coherencia interna e imputar valores perdidos.
#
# El resultado es una base depurada mínima, adecuada para continuar el EDA sin
# perder información potencialmente valiosa de forma prematura.
print('\nDEPURACIÓN BASE PARA EDA Y MODELADO')

var_eliminar = ['tlf']
print('\nVariables eliminadas en esta fase:')
print(var_eliminar)

df = df.drop(columns=var_eliminar).copy()
print(f'\nDimensión de la base tras depuración inicial: {df.shape}')
print('\nVariables categóricas propuestas:')
print(vars_categoricas)
print('\nVariables numéricas propuestas:')
print(vars_numericas)
print('\nMissing en la base depurada:')
print(df.loc[:, df.isna().sum() > 0].isna().sum())

# Tras esta depuración base, el conjunto queda en 3536 observaciones y 20
# variables, manteniendo por ahora tanto las variables estructurales del cliente
# como las relacionadas con actividad, minutos, costes y atención al cliente.
#
# Los valores perdidos quedan concentrados en un bloque reducido de variables
# (cod_area, coste_dia, min_noche, n_noche y coste_inter), lo que confirma que
# el problema de missing no está extendido a toda la base, sino localizado en
# algunos campos concretos que podrán revisarse con más detalle en los apartados
# siguientes.
#
# En consecuencia, esta versión del dataset puede considerarse ya una base
# operativa válida para el análisis exploratorio, aunque todavía no definitiva
# desde el punto de vista del modelado final.

# =============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
# En esta segunda parte se desarrolla el análisis exploratorio de datos
# (Exploratory Data Analysis, EDA) sobre la base ya depurada.
#
# El objetivo no es todavía ajustar modelos, sino comprender mejor el
# comportamiento de la variable objetivo y de los predictores, detectar patrones
# relevantes, revisar posibles desequilibrios y obtener evidencias que ayuden a
# justificar después la selección de variables y las decisiones de modelado.
#
# Este bloque resulta especialmente importante porque, tras eliminar duplicados y
# depurar registros repetidos por identificador, la estructura de la base cambia
# respecto al fichero original. Por ello, antes de estudiar relaciones entre
# variables, conviene empezar revisando cómo queda distribuida la variable
# objetivo en la muestra final de trabajo.
#
# -----------------------------------------------------------------------------
# 2.1. VISIÓN GENERAL DE LA VARIABLE OBJETIVO
# -----------------------------------------------------------------------------
# Este apartado permite comprobar la frecuencia y proporción de la variable Y
# tras la depuración previa del conjunto de datos.
#
# Su revisión es fundamental porque el equilibrio o desequilibrio entre clases
# condiciona directamente la interpretación de las métricas, la estrategia de
# validación y el tipo de ajustes que podrán ser necesarios más adelante
# (ponderación de clases, atención especial al recall, análisis de falsos
# negativos, etc.).
#
# Además, este análisis sirve para confirmar si la depuración de duplicados ha
# alterado la distribución inicial de la variable objetivo, algo especialmente
# relevante en este proyecto.

print('\nVISIÓN GENERAL DE LA VARIABLE OBJETIVO')
df_y = pd.DataFrame({
    'frecuencia': df['Y'].value_counts().sort_index(),
    'proporcion': df['Y'].value_counts(normalize=True).sort_index()
    })
print('\nDistribución de Y:')
print(df_y)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Y')
plt.title('Distribución de la variable Y', fontsize=16, weight='bold')
plt.xlabel('Clase de Y')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Tras la depuración, la variable objetivo presenta un claro desbalanceo:
# aproximadamente el 80.1% de los casos pertenecen a la clase 0, frente al
# 19.9% de la clase 1.
#
# Este resultado contrasta con la distribución perfectamente equilibrada del
# fichero original y confirma que una parte importante del equilibrio inicial
# estaba influida por la presencia de registros duplicados exactos. Por tanto,
# la base finalmente utilizada representa un escenario más realista y más
# exigente desde el punto de vista predictivo.
#
# Desde una perspectiva metodológica, este desbalanceo obliga a interpretar con
# cautela métricas como la accuracy, ya que un buen resultado global podría
# ocultar un peor comportamiento sobre la clase minoritaria. En consecuencia,
# durante el modelado será especialmente importante complementar la lectura de
# la accuracy con métricas como AUC, recall, precision, f1-score,
# balanced_accuracy y el análisis de la matriz de confusión.
#
# En términos de negocio, la clase 1 pasa a comportarse como la clase de mayor
# interés estratégico, por lo que detectar correctamente estos casos será una
# prioridad durante la comparación de modelos.

# -----------------------------------------------------------------------------
# 2.2. REVISIÓN DE MISSINGS EN MINUTOS Y COSTES
# -----------------------------------------------------------------------------
# En este apartado se revisan específicamente los valores ausentes en variables
# de minutos y costes, ya que entre ambas existe una relación prácticamente
# determinista dentro de cada tramo horario.
#
# No obstante, aunque esta relación permita una imputación técnicamente muy
# razonable, dicha imputación aprende un parámetro a partir de los datos
# (la tarifa coste/minuto). Para evitar fuga de información, este cálculo no se
# realiza aquí sobre la base completa, sino más adelante en el script de
# modelado, estimando la tarifa únicamente con TRAIN y aplicándola después a
# TRAIN y TEST.
#
# Por tanto, en este bloque no se imputa todavía ningún valor: únicamente se
# cuantifica el alcance del problema y se identifican los casos potencialmente
# recuperables para el modelado posterior.

print('\nREVISIÓN DE MISSINGS EN MINUTOS Y COSTES')

tramos_precio = [
    ('dia', 'min_dia', 'coste_dia'),
    ('tarde', 'min_tarde', 'coste_tarde'),
    ('noche', 'min_noche', 'coste_noche'),
    ('inter', 'min_inter', 'coste_inter')
]

filas_revision = []
for t, m, coste in tramos_precio:
    id_coste = df[df[coste].isna() & df[m].notna()].index
    id_min = df[df[m].isna() & df[coste].notna()].index
    filas_revision.append({
        'tramo': t,
        'n_costes_imputacion': len(id_coste),
        'n_minutos_imputacion': len(id_min)
    })

df_revision_imputacion = pd.DataFrame(filas_revision)
print('\nResumen de costes y minutos imputables:')
print(df_revision_imputacion)

# La revisión confirma que el problema de missings en minutos y costes es
# acotado y se concentra en unos pocos tramos concretos.
#
# En particular:
# - en el tramo de día aparecen 13 costes ausentes con minutos disponibles;
# - en el tramo de noche aparecen 25 minutos ausentes con coste disponible;
# - en el tramo internacional aparecen 16 costes ausentes con minutos
#   disponibles;
# - en el tramo de tarde no se detectan valores imputables de este tipo.
#
# Este patrón es coherente con el análisis previo de tarifas, donde se observó
# que la relación entre minutos y coste es prácticamente determinista dentro de
# cada tramo. Por ello, los valores ausentes identificados podrán recuperarse
# más adelante de forma justificada a partir de la tarifa aprendida en TRAIN,
# evitando introducir imputaciones arbitrarias.
#
# Además, el volumen total de casos afectados es reducido respecto al tamaño de
# la muestra, por lo que no parece tratarse de un problema estructural grave,
# sino de incidencias puntuales de registro.
#
# En consecuencia, en esta fase del EDA no se modifican todavía estos valores,
# pero queda claramente identificado qué parte de los missings podrá resolverse
# posteriormente mediante una imputación guiada por la relación minutos-coste,
# siempre aprendida exclusivamente sobre TRAIN para evitar fuga de información.

# -----------------------------------------------------------------------------
# 2.3. ANÁLISIS DE VARIABLES CATEGÓRICAS RESPECTO A Y
# -----------------------------------------------------------------------------
# En este bloque se estudian las variables categóricas desde tres perspectivas
# complementarias:
# 1. su relación directa con la variable objetivo, observando la tasa de Y=1 en
#    cada categoría;
# 2. la intensidad global de asociación con Y mediante la V de Cramer;
# 3. su complejidad práctica para el modelado, valorando número de categorías,
#    número potencial de dummies y peso de categorías minoritarias.
#
# Esta revisión es importante porque una variable categórica no solo debe mostrar
# relación con la respuesta, sino también hacerlo de una forma utilizable. Una
# señal aparentemente interesante puede perder valor si está muy fragmentada en
# muchas categorías pequeñas o si obliga a introducir demasiadas dummies para un
# aporte predictivo limitado.

print('\nANÁLISIS VARIABLES CATEGÓRICAS')
filas_cat = []
for var in vars_categoricas:
    tabla = df.groupby(var)['Y'].agg(
        frecuencia='count', casos_y1='sum', tasa_y1='mean').reset_index()
    tabla.columns = ['valor', 'frecuencia', 'casos_y1', 'tasa_y1']
    tabla.insert(0, 'variable', var)
    filas_cat.append(tabla)

df_resumen_cat = pd.concat(filas_cat, ignore_index=True)
print('\nResumen de variables categóricas:')
print(df_resumen_cat)
resumen_categoricas(df, vars_categoricas)

# Matriz de V de Cramer
var_cramer = vars_categoricas + ['Y']
matriz_vcramer = pd.DataFrame(index=var_cramer, columns=var_cramer)
for var1 in var_cramer:
    for var2 in var_cramer:
        if var1 == var2:
            matriz_vcramer.loc[var1, var2] = 1.0
        else:
            matriz_vcramer.loc[var1, var2] = v_cramer(df[var1], df[var2])
matriz_vcramer = matriz_vcramer.astype(float)
matriz_corr(matriz_vcramer, 'V de Cramer')

print('\nComplejidad variables categóricas')
complejidad_cat = []

for var in vars_categoricas:
    frec_abs = df[var].value_counts(dropna=False)
    frec_rel = df[var].value_counts(normalize=True, dropna=False)
    n_categorias = df[var].nunique(dropna=True)
    n_dummies = max(n_categorias - 1, 0)
    n_cat_muy_min = (frec_rel < 0.01).sum()
    n_cat_min = (frec_rel < 0.05).sum()
    pct_cat_muy_raras = frec_rel[frec_rel < 0.01].sum()*100
    pct_cat_raras = frec_rel[frec_rel < 0.05].sum()*100
    v_y = v_cramer(df[var], df['Y'])

    complejidad_cat.append({
        'variable': var,
        'n_categorias': n_categorias,
        'n_dummies': n_dummies,
        'freq_min_abs': frec_abs.min(),
        'freq_max_abs': frec_abs.max(),
        'n_cat_<1%': n_cat_muy_min,
        'n_cat_<5%': n_cat_min,
        'pct_obs_cat_<1%': round(pct_cat_muy_raras, 2),
        'pct_obs_cat_<5%': round(pct_cat_raras, 2),
        'v_cramer_y': round(v_y, 4) if pd.notna(v_y) else np.nan
        })

df_complejidad_cat = pd.DataFrame(
    complejidad_cat).sort_values(
        ['n_dummies', 'pct_obs_cat_<5%'], ascending=[False, False]
        ).reset_index(drop=True)

print('\nResumen de complejidad de variables categóricas:')
print(df_complejidad_cat)

# El análisis de complejidad resume hasta qué punto cada variable categórica
# puede incorporarse al modelo sin penalizar en exceso la parsimonia del diseño.
# En particular, ayuda a detectar variables con demasiadas categorías o con
# categorías muy minoritarias, que podrían generar muchas dummies para un aporte
# predictivo reducido.

analisis_categoricas(df_complejidad_cat)

# La exploración categórica permite extraer varias conclusiones iniciales.
#
# En primer lugar, plan_inter es la variable categórica con mayor relación con
# la respuesta. La tasa de Y=1 pasa de 0.1606 cuando no existe plan
# internacional a 0.5155 cuando sí existe, y además presenta la V de Cramer más
# alta frente a Y (0.2759). Por tanto, desde esta primera revisión se perfila
# como una variable claramente relevante para el modelado.
#
# En segundo lugar, buzon_voz también muestra señal útil, aunque más moderada.
# Los clientes con buzón de voz presentan una tasa de Y=1 bastante menor
# (0.1116) que los que no lo tienen (0.2300), y su asociación con la variable
# objetivo alcanza una V de Cramer de 0.1289. Esto sugiere que puede aportar
# información explicativa, aunque con menos fuerza que plan_inter.
#
# En cambio, cod_area ofrece muy poca capacidad discriminante. Las tasas de Y=1
# son muy parecidas entre sus tres categorías (en torno al 20%) y su asociación
# con Y es prácticamente nula (V de Cramer = 0.0131). Además, su complejidad es
# muy baja, por lo que no parece una variable problemática, pero tampoco una
# fuente clara de información predictiva.
#
# El caso de region es distinto: sí aparece cierta heterogeneidad en las tasas
# de Y=1 entre categorías y su V de Cramer con Y (0.1603) supera a la de
# cod_area y buzon_voz, pero esta señal queda muy fragmentada. La variable tiene
# 51 categorías, generaría 50 dummies y, además, el 100% de las observaciones se
# distribuye en categorías con peso inferior al 5% cada una. Esto implica una
# complejidad muy alta para una señal que, aunque existente, no parece
# especialmente sólida ni parsimoniosa.
#
# En consecuencia, la lectura conjunta de asociación y complejidad sugiere una
# primera jerarquía clara:
# - plan_inter destaca como la categórica más informativa;
# - buzon_voz parece útil como apoyo;
# - cod_area muestra una contribución muy débil;
# - y region, aunque no es irrelevante, plantea dudas por su elevada
#   fragmentación y su coste estructural en forma de dummies.
#
# Esta evidencia será especialmente útil en la fase posterior de selección de
# variables, donde no solo importará la fuerza de asociación con Y, sino también
# la estabilidad y la eficiencia del conjunto final de predictores.

# -----------------------------------------------------------------------------
# 2.4. ANÁLISIS DE VARIABLES NUMÉRICAS RESPECTO A Y
# -----------------------------------------------------------------------------
# En este bloque se estudian las variables numéricas desde tres enfoques
# complementarios:
# 1. comparación de estadísticos descriptivos según el valor de Y;
# 2. análisis visual mediante boxplots e histogramas de densidad;
# 3. medida de asociación lineal con Y a través de la correlación de Pearson.
#
# El objetivo no es decidir todavía de forma definitiva qué variables se
# conservarán en el modelo, sino identificar qué señales parecen más útiles,
# qué variables muestran separación entre clases y cuáles aportan poca
# información discriminante en esta fase exploratoria.
#
# Además, este análisis resulta especialmente importante en este problema porque
# varias variables están ligadas entre sí por construcción, como ocurre con
# minutos y costes dentro de cada tramo. Por ello, la lectura de este bloque
# debe combinar capacidad explicativa con cautela frente a posibles redundancias.

print('\nANÁLISIS DE VARIABLES NUMÉRICAS RESPECTO A Y')
filas_num = []
for var in vars_numericas:
    for valor in sorted(df['Y'].dropna().unique()):
        sub = df.loc[df['Y'] == valor, var]
        filas_num.append({
            'variable': var,
            'Y': valor,
            'media': sub.mean(),
            'mediana': sub.median(),
            'std': sub.std()
            })
        
df_resumen_num = pd.DataFrame(filas_num)
print('\nResumen de variables numéricas por Y:')
print(df_resumen_num)

# Los boxplots permiten comprobar si existen desplazamientos en mediana,
# dispersión o presencia de atípicos entre clientes con Y=0 y Y=1.
# Los histogramas de densidad complementan esta visión mostrando el grado de
# solapamiento real entre ambas clases, algo clave para valorar la utilidad
# discriminante de cada variable.

boxplot_numericas(df, vars_numericas)

# Histogramas de densidad 
histogramas_densidad_y(df, vars_numericas)

# Correlación de las variables numéricas
df_corr_y = df[['Y'] + vars_numericas].corr(method='pearson')['Y']
print('\nCorrelación de Pearson con Y:')
print(df_corr_y)

plt.figure(figsize=(8, 6))
df_corr_y.drop('Y').plot(kind='bar')
plt.title('Correlación de Pearson entre variables numéricas e Y', fontsize=16,
          weight='bold')
plt.ylabel('Correlación')
plt.tight_layout()
plt.show()

# La exploración numérica permite identificar una estructura bastante clara.
#
# En primer lugar, n_att_cl aparece como la variable numérica con mayor relación
# lineal con Y (r = 0.2414). Los clientes con Y=1 presentan, en promedio, un
# número sensiblemente mayor de llamadas al servicio de atención al cliente
# (2.25 frente a 1.43), lo que sugiere una señal consistente y con sentido
# operativo.
#
# También destacan min_dia y coste_dia, con correlaciones muy similares
# (r ≈ 0.234), algo esperable porque coste_dia deriva prácticamente de min_dia.
# En ambas variables se observa un desplazamiento claro hacia valores más altos
# en la clase Y=1, lo que indica que el uso diurno parece estar asociado a una
# mayor probabilidad de abandono.
#
# En un segundo nivel aparecen min_tarde y coste_tarde (r ≈ 0.109), así como
# min_inter y coste_inter (r ≈ 0.078). Su relación con Y es más moderada, pero
# sigue siendo apreciable y coherente en medias, medianas y distribuciones.
#
# En cambio, varias variables muestran una señal muy limitada. Es el caso de
# antiguedad, n_dia y n_noche, cuyas correlaciones con Y son casi nulas, y
# también de n_tarde, cuyo signo es ligeramente negativo pero con magnitud muy
# pequeña. Estas variables, al menos en esta fase exploratoria, no parecen
# especialmente prometedoras como predictores individuales.
#
# Merece mención aparte n_mns_voz, que presenta una correlación negativa
# moderada (r = -0.1143). Los clientes con Y=1 tienden a registrar menos
# mensajes de voz que los clientes con Y=0, por lo que podría aportar
# información útil en sentido inverso.
#
# En conjunto, la evidencia gráfica y numérica sugiere que las señales más
# fuertes del bloque numérico se concentran en:
# - n_att_cl,
# - min_dia / coste_dia,
# - min_tarde / coste_tarde,
# - y, en menor medida, min_inter / coste_inter.
#
# No obstante, esta lectura debe matizarse con una idea importante: varias de
# estas variables no son independientes entre sí, sino prácticamente
# equivalentes desde el punto de vista informativo, especialmente las parejas
# minutos-coste dentro de cada tramo. Por ello, en la fase posterior de
# selección de variables no bastará con identificar qué variables se relacionan
# más con Y, sino también cuáles aportan información nueva sin introducir
# redundancia innecesaria.

# -----------------------------------------------------------------------------
# 2.5. REVISIÓN Y SELECCIÓN DE VARIABLES CANDIDATAS PARA MODELADO
# -----------------------------------------------------------------------------
# En este punto se realiza una primera depuración orientada ya al modelado.
# El objetivo no es cerrar todavía la selección definitiva de variables, sino
# construir un bloque inicial razonable de predictores candidatos para la fase
# posterior de entrenamiento.
#
# Para ello se combinan tres criterios complementarios:
# 1. criterio estructural, eliminando variables problemáticas por diseño;
# 2. criterio de redundancia, evitando incorporar señales prácticamente
#    equivalentes;
# 3. criterio de utilidad exploratoria, conservando variables que, aun sin ser
#    definitivas, muestran potencial suficiente para ser evaluadas después en
#    el ajuste del modelo.
#
# Esta criba inicial resulta especialmente importante en SVM, donde una
# representación excesivamente grande o ruidosa puede perjudicar tanto la
# estabilidad del ajuste como la interpretación posterior de los resultados.

print('\nREVISIÓN Y SELECCIÓN DE VARIABLES CANDIDATAS PARA MODELADO')

# En las numéricas se parte ya de una selección reducida respecto al bloque
# exploratorio completo. En particular, no se incorporan simultáneamente
# minutos y costes dentro de cada tramo, ya que esas parejas contienen
# información casi equivalente por construcción. Se conserva una sola
# representación por tramo para evitar redundancia innecesaria.
# -----------------------------
# 2.5.1. VARIABLES
# -----------------------------
vars_cat_0 = ['region', 'cod_area', 'plan_inter', 'buzon_voz']
vars_num_0 = ['antiguedad', 'n_mns_voz', 'n_dia', 'coste_dia', 'n_tarde',
              'coste_tarde', 'n_noche', 'coste_noche', 'n_inter', 'coste_inter',
              'n_att_cl']
print('\nVariables categóricas candidatas:')
print(vars_cat_0)
print('\nVariables numéricas candidatas:')
print(vars_num_0)
# -----------------------------
# 2.5.2. REDUNDANCIA ENTRE NUMÉRICAS
# -----------------------------
corr_num = df[vars_num_0].corr(method='pearson').abs()
matriz_corr(corr_num, 'Correlación entre variables numéricas candidatas')

# La matriz de correlación confirma que, una vez retiradas las variables
# redundantes más evidentes, las numéricas candidatas presentan correlaciones
# lineales muy bajas entre sí. Por tanto, no se detectan problemas relevantes
# de colinealidad dentro de este bloque inicial y la selección puede apoyarse
# más en criterios de utilidad y complejidad que en restricciones puramente
# estadísticas.

# -----------------------------
# 2.5.3. VARIABLES CANDIDATAS SELECCIONADAS
# -----------------------------
print('\nVARIABLES CANDIDATAS DE MODELADO')

# A la vista del análisis exploratorio previo, la propuesta inicial para el
# modelado queda acotada del siguiente modo:
#
# - En las variables categóricas se descarta region, no porque carezca por
#   completo de relación con Y, sino porque su cardinalidad es demasiado alta
#   para esta fase del trabajo. Generaría un número muy elevado de dummies,
#   fragmentando la muestra en muchas categorías pequeñas y aumentando la
#   complejidad del modelo sin una evidencia clara de ganancia proporcional.
#
# - También se descarta cod_area, ya que su complejidad es baja pero su
#   asociación con Y es prácticamente nula, tanto en la comparación de tasas
#   como en la V de Cramer. En consecuencia, no parece una variable prioritaria
#   para el ajuste inicial.
#
# - Se conservan plan_inter y buzon_voz porque combinan baja complejidad,
#   codificación sencilla y una relación apreciable con la variable objetivo,
#   especialmente en el caso de plan_inter.
#
# - En las variables numéricas se mantienen las candidatas que no muestran
#   colinealidad alta entre sí tras la reducción previa y que, además, permiten
#   conservar señales de uso, gasto y comportamiento del cliente en distintos
#   tramos horarios.
#
# - Esta selección sigue siendo una criba inicial y deliberadamente amplia:
#   algunas variables pueden acabar mostrando una aportación limitada en el
#   ajuste final, pero se mantienen aquí para que sea el modelado posterior el
#   que confirme su utilidad real en combinación con el resto.

vars_cat_cand = ['plan_inter', 'buzon_voz']
vars_num_cand = ['antiguedad', 'n_mns_voz', 'n_dia', 'coste_dia', 'n_tarde',
                 'coste_tarde', 'n_noche', 'coste_noche', 'n_inter', 
                 'coste_inter', 'n_att_cl']

print('\nVariables categóricas candidatas para SVM:')
print(vars_cat_cand)
print('\nVariables numéricas candidatas para SVM:')
print(vars_num_cand)

# Se opta por mantener las variables de coste frente a sus equivalentes en
# minutos cuando ambas aportan una señal muy similar, ya que permiten una
# lectura más directa desde el punto de vista de negocio y reducen la
# redundancia del bloque numérico.

# =============================================================================
# GUARDADO EN PICKLE PARA SVM
# =============================================================================
print('\nGUARDADO')

bbdd_eda = {
    'vars_num': vars_numericas,
    'vars_cat': vars_categoricas,
    'vars_num_cand': vars_num_cand,
    'vars_cat_cand': vars_cat_cand,
    'tramos_precio': tramos_precio,
    'tarifas': tarifas,
    'datos': df
    }

with open(RUTA_SALIDA, 'wb') as archivo:
    pickle.dump(bbdd_eda, archivo)

print(f'\nArchivo guardado: {RUTA_SALIDA.resolve()}')
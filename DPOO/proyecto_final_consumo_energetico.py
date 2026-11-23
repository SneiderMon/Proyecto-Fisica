# -*- coding: utf-8 -*-
import csv
import datetime
import math
import os
import random
import re
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS ---
TAMANOS = [500, 1000, 2000, 3000, 4000, 6000]
REPETICIONES = 3
TIEMPO_MIN = 3.0
T_REPOSO = 10.0
PAUSA = 1.0
ESPERA_INICIO = 15.0
SEMILLA = 42

MEDIR_AHORA = True  # True para que mida en tu PC y cree los CSV. (Si ya tienes los CSV, pon False)
ARCHIVO_MARCAS = "marcas_tiempo.csv"
ARCHIVO_HWINFO = "hwinfo.csv"
COLUMNA_POTENCIA = None

P_REPOSO_SUPUESTO = 5.0
P_ALG_SUPUESTO = 20.0
POTENCIA_MAX_W = 25.0

# Estilos básicos para gráficas
COLOR = {"Insertion Sort": "#2a78d6", "Merge Sort": "#eb6834"}
MARCA = {"Insertion Sort": "o", "Merge Sort": "s"}
GRIS = "#8a8a86"

# --- 2. ALGORITMOS ---
def insertion_sort(lista):
    a = list(lista)
    comparaciones = 0
    for i in range(1, len(a)):
        clave = a[i]
        j = i - 1
        while j >= 0:
            comparaciones += 1
            if a[j] > clave:
                a[j + 1] = a[j]
                j -= 1
            else:
                break
        a[j + 1] = clave
    return a, comparaciones

def merge_sort(lista):
    n = len(lista)
    if n <= 1:
        return list(lista), 0
    mitad = n // 2
    izq, comp_izq = merge_sort(lista[:mitad])
    der, comp_der = merge_sort(lista[mitad:])
    resultado = []
    i = 0
    j = 0
    comparaciones = comp_izq + comp_der
    while i < len(izq) and j < len(der):
        comparaciones += 1
        if izq[i] <= der[j]:
            resultado.append(izq[i])
            i += 1
        else:
            resultado.append(der[j])
            j += 1
    resultado.extend(izq[i:])
    resultado.extend(der[j:])
    return resultado, comparaciones

algoritmos = {"Insertion Sort": insertion_sort, "Merge Sort": merge_sort}

def lista_aleatoria(n, semilla=SEMILLA):
    rng = random.Random(semilla)
    return [rng.random() for _ in range(n)]

# --- 3. HERRAMIENTAS NUMÉRICAS ---
def trapecio(t, p):
    total = 0.0
    for i in range(len(t) - 1):
        total += 0.5 * (p[i] + p[i + 1]) * (t[i + 1] - t[i])
    return total

def trapecio_acumulado(t, p):
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    pedazos = 0.5 * (p[1:] + p[:-1]) * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(pedazos)])

def interpolar(x, xs, ys):
    i = int(np.searchsorted(xs, x))
    if i <= 0: return ys[0]
    if i >= len(xs): return ys[-1]
    x0, x1 = xs[i - 1], xs[i]
    if x1 == x0: return ys[i]
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - x0) / (x1 - x0)

def ajustar_constante(f_n, y):
    f = np.asarray(f_n, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sum(f * y) / np.sum(f * f)

def pendiente_loglog(x, y):
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    return np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2)

def f_ins(n): return np.asarray(n, dtype=float) ** 2
def f_merge(n): return np.asarray(n, dtype=float) * np.log2(np.asarray(n, dtype=float))

def estilo(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

resumen = {}

# --- 4. MEDICIÓN ---
def segundos_del_dia():
    ahora = datetime.datetime.now()
    return ahora.hour * 3600 + ahora.minute * 60 + ahora.second + ahora.microsecond / 1e6

def calcular_K(algoritmo, datos):
    t0 = time.perf_counter()
    algoritmo(datos)
    t_una = time.perf_counter() - t0
    return max(1, math.ceil(TIEMPO_MIN / t_una))

def medir_todo(espera):
    filas = []
    if espera > 0:
        print(f"Inicien el registro en HWiNFO. La medición empieza en {espera:.0f} s...")
        time.sleep(espera)

    for alg in algoritmos.values():
        alg(lista_aleatoria(500))

    t0 = segundos_del_dia()
    time.sleep(T_REPOSO)
    t1 = segundos_del_dia()
    filas.append(["reposo", "", 0, 0, 0, t0, t1, t1 - t0])
    
    datos_por_n = {n: lista_aleatoria(n) for n in TAMANOS}
    K_por_caso = {}
    for nombre, alg in algoritmos.items():
        for n in TAMANOS:
            K_por_caso[(nombre, n)] = calcular_K(alg, datos_por_n[n])

    for rep in range(REPETICIONES):
        for n in TAMANOS:
            for nombre, alg in algoritmos.items():
                K = K_por_caso[(nombre, n)]
                datos = datos_por_n[n]
                time.sleep(PAUSA)
                t0 = segundos_del_dia()
                p0 = time.perf_counter()
                for _ in range(K):
                    alg(datos)
                duracion = time.perf_counter() - p0
                t1 = segundos_del_dia()
                filas.append(["algoritmo", nombre, n, rep, K, t0, t1, duracion])

    time.sleep(3)

    with open(ARCHIVO_MARCAS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "algoritmo", "n", "rep", "K", "t_inicio_s", "t_fin_s", "duracion_s"])
        w.writerows(filas)
    print(f"Marcas de tiempo guardadas en {ARCHIVO_MARCAS}")

if MEDIR_AHORA or not os.path.exists(ARCHIVO_MARCAS):
    medir_todo(ESPERA_INICIO if MEDIR_AHORA else 0)

# --- 5. PROCESAMIENTO DE DATOS ---
def a_segundos(texto):
    t = texto.strip().lower()
    m = re.search(r"(\d+):(\d+):(\d+(?:[.,]\d+)?)", t)
    if m is None: return None
    h = int(m.group(1))
    minutos = int(m.group(2))
    s = float(m.group(3).replace(",", "."))
    resto = t[m.end():]
    if "p" in resto and h < 12: h += 12
    if "a" in resto and h == 12: h = 0
    return h * 3600 + minutos * 60 + s

def buscar_columna_potencia(encabezado):
    if COLUMNA_POTENCIA is not None: return encabezado.index(COLUMNA_POTENCIA)
    for clave in ["package power", "potencia total de cpu", "ppt", "cpu power"]:
        for i, nombre in enumerate(encabezado):
            if clave in nombre.lower() and "[w]" in nombre.lower():
                return i
    return None

def leer_hwinfo(ruta):
    with open(ruta, encoding="latin-1", newline="") as f:
        filas = list(csv.reader(f))
    encabezado = [c.replace("ï»¿", "").strip() for c in filas[0]]
    col_t = None
    for i, nombre in enumerate(encabezado):
        if nombre.lower() in ("time", "hora"): col_t = i
    col_p = buscar_columna_potencia(encabezado)

    if col_t is None or col_p is None: return None, None, None

    tiempos, potencias = [], []
    for fila in filas[1:]:
        if len(fila) <= max(col_t, col_p): continue
        t = a_segundos(fila[col_t])
        try: p = float(fila[col_p].replace(",", "."))
        except ValueError: continue
        if t is None: continue
        tiempos.append(t)
        potencias.append(p)
    return np.array(tiempos), np.array(potencias), encabezado[col_p]

def leer_marcas():
    marcas = []
    with open(ARCHIVO_MARCAS) as f:
        for fila in csv.DictReader(f):
            marcas.append({"tipo": fila["tipo"], "algoritmo": fila["algoritmo"], "n": int(fila["n"]),
                           "rep": int(fila["rep"]), "K": int(fila["K"]),
                           "t0": float(fila["t_inicio_s"]), "t1": float(fila["t_fin_s"]),
                           "duracion": float(fila["duracion_s"])})
    return marcas

marcas = leer_marcas()
reposo = [m for m in marcas if m["tipo"] == "reposo"][0]
corridas = [m for m in marcas if m["tipo"] == "algoritmo"]
tamanos_medidos = sorted({c["n"] for c in corridas})

MODO_ENERGIA = "estimada"
t_hw = p_hw = None
columna_usada = None
if os.path.exists(ARCHIVO_HWINFO):
    t_hw, p_hw, columna_usada = leer_hwinfo(ARCHIVO_HWINFO)
    if t_hw is not None and len(t_hw) > 5:
        t_inicio_total = min(reposo["t0"], min(c["t0"] for c in corridas))
        t_fin_total = max(c["t1"] for c in corridas)
        if not (t_hw.min() > t_inicio_total or t_hw.max() < t_fin_total):
            MODO_ENERGIA = "medida"

def ventana_potencia(t0, t1):
    dentro = (t_hw > t0) & (t_hw < t1)
    t = np.concatenate([[t0], t_hw[dentro], [t1]])
    p = np.concatenate([[interpolar(t0, t_hw, p_hw)], p_hw[dentro], [interpolar(t1, t_hw, p_hw)]])
    return t, p

if MODO_ENERGIA == "medida":
    t_r, p_r = ventana_potencia(reposo["t0"], reposo["t1"])
    P_REPOSO = trapecio(t_r, p_r) / (reposo["t1"] - reposo["t0"])
else:
    P_REPOSO = P_REPOSO_SUPUESTO

for c in corridas:
    if MODO_ENERGIA == "medida":
        t, p = ventana_potencia(c["t0"], c["t1"])
        energia_total = trapecio(t, p - P_REPOSO)
    else:
        energia_total = P_ALG_SUPUESTO * c["duracion"]
    c["tiempo"] = c["duracion"] / c["K"]
    c["energia"] = energia_total / c["K"]

resultados = {}
for nombre in algoritmos:
    resultados[nombre] = []
    for n in tamanos_medidos:
        grupo = [c for c in corridas if c["algoritmo"] == nombre and c["n"] == n]
        tiempos = np.array([c["tiempo"] for c in grupo])
        energias = np.array([c["energia"] for c in grupo])
        resultados[nombre].append({
            "n": n, "K": grupo[0]["K"],
            "tiempo": float(tiempos.mean()), "tiempo_std": float(tiempos.std()),
            "energia": float(energias.mean()), "energia_std": float(energias.std()),
            "potencia": float(energias.mean() / tiempos.mean()),
        })

with open("mediciones.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["algoritmo", "n", "K", "tiempo_s", "tiempo_std_s", "energia_J", "energia_std_J", "potencia_alg_W", "modo_energia"])
    for nombre in algoritmos:
        for r in resultados[nombre]:
            w.writerow([nombre, r["n"], r["K"], r["tiempo"], r["tiempo_std"], r["energia"], r["energia_std"], r["potencia"], MODO_ENERGIA])
print("Tabla guardada en mediciones.csv")

# --- 6. GRÁFICAS (Solo las importantes para la app) ---
fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
for nombre, f in [("Insertion Sort", f_ins), ("Merge Sort", f_merge)]:
    rs = resultados[nombre]
    n = np.array([r["n"] for r in rs])
    t = np.array([r["tiempo"] for r in rs])
    E = np.array([r["energia"] for r in rs])

    ejes[0].plot(n, ajustar_constante(f(n), t) * f(n), "-", color=COLOR[nombre], linewidth=1.5, label=f"{nombre} (ajuste)")
    ejes[1].plot(n, ajustar_constante(f(n), E) * f(n), "-", color=COLOR[nombre], linewidth=1.5, label=f"{nombre} (ajuste)")
    ejes[0].errorbar(n, t, yerr=[r["tiempo_std"] for r in rs], fmt=MARCA[nombre], color=COLOR[nombre], capsize=3, label=f"{nombre} (medido)")
    ejes[1].errorbar(n, E, yerr=[r["energia_std"] for r in rs], fmt=MARCA[nombre], color=COLOR[nombre], capsize=3, label=f"{nombre} (medido)")

for eje in ejes:
    eje.set_xscale("log")
    eje.set_yscale("log")
estilo(ejes[0], "Tiempo de ejecución vs. n", "n", "Tiempo (s)")
estilo(ejes[1], "Energía de ejecución vs. n", "n", "Energía (J)")
ejes[0].legend()
ejes[1].legend()
plt.tight_layout()
plt.show()

# --- 7. CÁLCULO DEL CRUCE (10x) ---
N_CAL = 3
modelo = {}
for nombre, f in [("Insertion Sort", f_ins), ("Merge Sort", f_merge)]:
    rs = resultados[nombre]
    n = np.array([r["n"] for r in rs])
    E = np.array([r["energia"] for r in rs])
    k = ajustar_constante(f(n[:N_CAL]), E[:N_CAL])
    modelo[nombre] = k

n_grande = np.arange(2, 200001)
razon = (modelo["Insertion Sort"] * f_ins(n_grande)) / (modelo["Merge Sort"] * f_merge(n_grande))
indices = np.where(razon >= 10)[0]

if len(indices) > 0:
    n_10 = int(n_grande[indices[0]])
    print(f"\nSegún el modelo, Insertion Sort gasta 10 veces o más que Merge Sort desde n = {n_10}")

plt.figure(figsize=(6.5, 4.2))
plt.loglog(n_grande, razon, "-", color="#0b0b0b", linewidth=1.5, label="Modelo")
n_med = np.array(tamanos_medidos)
razon_med = np.array([a["energia"] / b["energia"] for a, b in zip(resultados["Insertion Sort"], resultados["Merge Sort"])])
plt.loglog(n_med, razon_med, "o", color=COLOR["Insertion Sort"], markersize=7, label="Medido")
plt.axhline(10, color=GRIS, linestyle="--", label="Razón = 10")
estilo(plt.gca(), "Cuántas veces más energía gasta Insertion Sort", "n (tamaño de la lista)", "E_insertion / E_merge (veces)")
plt.legend()
plt.tight_layout()
plt.show()
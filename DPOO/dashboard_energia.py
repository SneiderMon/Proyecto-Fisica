# -*- coding: utf-8 -*-
"""
Dashboard interactivo: traduce la energia medida (J) a unidades tangibles.
Lee mediciones.csv generado por proyecto_final_consumo_energetico.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

ARCHIVO = "mediciones.csv"
if not os.path.exists(ARCHIVO):
    raise FileNotFoundError(
        f"No encontre {ARCHIVO}. Correlo en la misma carpeta del notebook principal."
    )

df = pd.read_csv(ARCHIVO)

# --- Constantes de conversion (editables) ---
J_POR_KWH   = 3.6e6        # 1 kWh = 3.6e6 J
J_POR_CARGA = 4.0e4        # ~11 Wh por carga de celular (aprox)
COSTO_KWH   = 800.0        # COP por kWh (aprox Colombia 2025)
CO2_KG_KWH  = 0.164        # kg CO2 por kWh (factor red Colombia ~2024)

# --- Datos por algoritmo ---
def energia_de(algoritmo, n):
    fila = df[(df["algoritmo"] == algoritmo) & (df["n"] == n)]
    if fila.empty:
        return np.nan
    return float(fila["energia_J"].iloc[0])

tamanos = sorted(df["n"].unique())
algoritmos = ["Insertion Sort", "Merge Sort"]

# --- Widgets ---
slider_n = widgets.SelectionSlider(
    options=tamanos, value=tamanos[0],
    description="n:", continuous_update=True,
)
slider_listas = widgets.IntSlider(
    value=1_000_000, min=1, max=100_000_000, step=100_000,
    description="Listas:", continuous_update=False,
)
slider_costo = widgets.FloatSlider(
    value=COSTO_KWH, min=100, max=2000, step=50,
    description="COP/kWh:", continuous_update=False,
)
slider_co2 = widgets.FloatSlider(
    value=CO2_KG_KWH, min=0.0, max=0.5, step=0.01,
    description="kg CO2/kWh:", continuous_update=False,
)

salida = widgets.Output()

def actualizar(change=None):
    with salida:
        clear_output(wait=True)
        n = slider_n.value
        L = slider_listas.value
        costo = slider_costo.value
        co2 = slider_co2.value

        E_ins = energia_de("Insertion Sort", n)
        E_mer = energia_de("Merge Sort", n)
        if np.isnan(E_ins) or np.isnan(E_mer):
            print("No hay datos para ese n.")
            return

        E_ins_tot = E_ins * L
        E_mer_tot = E_mer * L

        def fmt(E):
            kwh = E / J_POR_KWH
            cargas = E / J_POR_CARGA
            pesos = kwh * costo
            gramos = kwh * co2 * 1000.0
            return (f"{E:,.0f} J  =  {kwh:.6f} kWh  |  "
                    f"{cargas:.3f} cargas de celular  |  "
                    f"{pesos:,.2f} COP  |  "
                    f"{gramos:.3f} g CO2")

        print(f"n = {n}, listas = {L:,}")
        print(f"Insertion Sort: {fmt(E_ins_tot)}")
        print(f"Merge Sort:     {fmt(E_mer_tot)}")
        print(f"Razon Insertion / Merge: {E_ins / E_mer:.1f} x")

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(["Insertion Sort", "Merge Sort"],
               [E_ins_tot, E_mer_tot],
               color=["#2a78d6", "#eb6834"])
        ax.set_ylabel("Energia total (J)")
        ax.set_title(f"Energia para {L:,} listas de n = {n}")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.show()

slider_n.observe(actualizar, names="value")
slider_listas.observe(actualizar, names="value")
slider_costo.observe(actualizar, names="value")
slider_co2.observe(actualizar, names="value")

panel = widgets.VBox([
    widgets.HBox([slider_n, slider_listas]),
    widgets.HBox([slider_costo, slider_co2]),
    salida,
])
display(panel)
actualizar()
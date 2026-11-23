# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display, HTML, clear_output


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = "mediciones.csv"

if not os.path.exists(ARCHIVO):
    raise FileNotFoundError(
        f"No encontré {ARCHIVO}. Ejecuta primero el programa principal."
    )

df = pd.read_csv(ARCHIVO)

J_POR_KWH = 3.6e6
J_POR_CARGA = 4.0e4

COSTO_KWH = 800.0
CO2_KG_KWH = 0.164

ALGORITMOS = ["Insertion Sort", "Merge Sort"]

COLORES = {
    "Insertion Sort": "#4c9aff",
    "Merge Sort": "#ff8a4c"
}

# Colores coherentes con app.py

BG = "#14161b"
PANEL = "#1b1e25"
CARD = "#232730"
CARD2 = "#2c313c"

TEXT = "#e9ebf0"
MUTED = "#9aa1af"
GRID = "#3a404d"


# ============================================================
# DATOS
# ============================================================

def energia_de(algoritmo, n):

    fila = df[
        (df["algoritmo"] == algoritmo) &
        (df["n"] == n)
    ]

    if fila.empty:
        return np.nan

    return float(fila["energia_J"].iloc[0])


def tiempo_de(algoritmo, n):

    if "tiempo_s" not in df.columns:
        return np.nan

    fila = df[
        (df["algoritmo"] == algoritmo) &
        (df["n"] == n)
    ]

    if fila.empty:
        return np.nan

    return float(fila["tiempo_s"].iloc[0])


# ============================================================
# CONVERSIONES
# ============================================================

def convertir(E, costo, co2):

    kwh = E / J_POR_KWH

    cargas = E / J_POR_CARGA

    pesos = kwh * costo

    emisiones = kwh * co2 * 1000

    return {
        "J": E,
        "kWh": kwh,
        "cargas": cargas,
        "COP": pesos,
        "CO2": emisiones
    }


# ============================================================
# TARJETAS VISUALES
# ============================================================

def tarjeta(algoritmo, resultados):

    color = COLORES[algoritmo]

    return f"""
    <div style="
        background:{CARD};
        border-radius:14px;
        padding:20px;
        border-top:5px solid {color};
        min-width:250px;
        flex:1;
    ">

        <h3 style="
            color:{color};
            margin-top:0;
            font-size:17px;
        ">
            {algoritmo}
        </h3>

        <div style="
            font-size:29px;
            font-weight:bold;
            color:{TEXT};
            margin-bottom:5px;
        ">
            {resultados['J']:,.2f} J
        </div>

        <div style="
            color:{MUTED};
            margin-bottom:18px;
        ">
            Energía consumida
        </div>

        <hr style="border-color:{GRID};">

        <p style="color:{TEXT};">
            ⚡ <b>{resultados['kWh']:.6f}</b> kWh
        </p>

        <p style="color:{TEXT};">
            📱 <b>{resultados['cargas']:.3f}</b> cargas de celular
        </p>

        <p style="color:{TEXT};">
            💰 <b>{resultados['COP']:,.2f}</b> COP
        </p>

        <p style="color:{TEXT};">
            🌱 <b>{resultados['CO2']:.4f}</b> g CO₂
        </p>

    </div>
    """


# ============================================================
# WIDGETS
# ============================================================

tamanos = sorted(df["n"].unique())

slider_n = widgets.SelectionSlider(
    options=tamanos,
    value=tamanos[0],
    description="Tamaño:",
    continuous_update=True,
    style={"description_width": "initial"}
)

slider_listas = widgets.IntSlider(
    value=1_000_000,
    min=1,
    max=100_000_000,
    step=100_000,
    description="Listas:",
    continuous_update=False,
    style={"description_width": "initial"}
)

slider_costo = widgets.FloatSlider(
    value=COSTO_KWH,
    min=100,
    max=2000,
    step=50,
    description="COP/kWh:",
    continuous_update=False,
    style={"description_width": "initial"}
)

slider_co2 = widgets.FloatSlider(
    value=CO2_KG_KWH,
    min=0,
    max=0.5,
    step=0.01,
    description="CO₂/kWh:",
    continuous_update=False,
    style={"description_width": "initial"}
)


# ============================================================
# SALIDAS
# ============================================================

salida_tarjetas = widgets.HTML()

salida_ventaja = widgets.HTML()

salida_comparacion = widgets.Output()

salida_impacto = widgets.Output()

salida_equivalencias = widgets.Output()


# ============================================================
# ACTUALIZACIÓN
# ============================================================

def actualizar(change=None):

    n = slider_n.value
    L = slider_listas.value

    costo = slider_costo.value
    co2 = slider_co2.value

    E_ins = energia_de("Insertion Sort", n)
    E_mer = energia_de("Merge Sort", n)

    if np.isnan(E_ins) or np.isnan(E_mer):

        salida_tarjetas.value = "No hay datos disponibles."
        return

    E_ins *= L
    E_mer *= L

    datos = {
        "Insertion Sort": convertir(E_ins, costo, co2),
        "Merge Sort": convertir(E_mer, costo, co2)
    }

    # ---------------- TARJETAS ----------------

    salida_tarjetas.value = f"""
    <div style="
        display:flex;
        gap:16px;
        flex-wrap:wrap;
        background:{PANEL};
        padding:15px;
        border-radius:14px;
    ">

        {tarjeta("Insertion Sort", datos["Insertion Sort"])}

        {tarjeta("Merge Sort", datos["Merge Sort"])}

    </div>
    """

    # ---------------- VENTAJA ----------------

    if E_mer > 0:

        razon = E_ins / E_mer

        if E_ins > E_mer:

            mejor = "Merge Sort"
            ahorro = (1 - E_mer / E_ins) * 100

        else:

            mejor = "Insertion Sort"
            ahorro = (1 - E_ins / E_mer) * 100

        salida_ventaja.value = f"""
        <div style="
            background:{CARD};
            padding:18px;
            border-radius:12px;
            border-left:5px solid #4ade9a;
        ">

            <h3 style="color:#4ade9a;margin-top:0;">
                🏆 VENTAJA ENERGÉTICA
            </h3>

            <div style="color:{TEXT};font-size:17px;">

                <b>{mejor}</b> es el algoritmo más eficiente
                para este tamaño de lista.

            </div>

            <p style="color:{MUTED};">

                Diferencia energética:
                <b>{razon:.2f} veces</b> entre ambos algoritmos.

                <br>

                Ahorro energético:
                <b>{ahorro:.2f}%</b> respecto al algoritmo de mayor consumo.

            </p>

        </div>
        """

    # ---------------- COMPARACIÓN ----------------

    with salida_comparacion:

        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=(9, 4))

        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)

        valores = [E_ins, E_mer]

        barras = ax.bar(
            ALGORITMOS,
            valores,
            color=[COLORES[a] for a in ALGORITMOS],
            width=0.5
        )

        ax.set_yscale("log")

        ax.set_ylabel("Energía total (J)", color=TEXT)

        ax.set_title(
            f"Consumo energético para {L:,} listas de n = {n:,}",
            color=TEXT,
            fontsize=13,
            pad=18
        )

        ax.tick_params(colors=MUTED)

        for spine in ax.spines.values():
            spine.set_color(GRID)

        ax.grid(axis="y", color=GRID, linestyle="--", alpha=0.5)

        ax.set_axisbelow(True)

        for barra, valor in zip(barras, valores):

            ax.annotate(
                f"{valor:,.2f} J",
                (barra.get_x() + barra.get_width()/2, valor),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color=TEXT,
                fontweight="bold"
            )

        plt.tight_layout()
        plt.show()

    # ---------------- IMPACTO AMBIENTAL ----------------

    with salida_impacto:

        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=(9, 4))

        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)

        valores = [
            datos[a]["CO2"] for a in ALGORITMOS
        ]

        ax.bar(
            ALGORITMOS,
            valores,
            color=[COLORES[a] for a in ALGORITMOS],
            width=0.5
        )

        ax.set_title(
            "Emisiones de CO₂ estimadas",
            color=TEXT,
            fontsize=13
        )

        ax.set_ylabel("Gramos de CO₂", color=TEXT)

        ax.tick_params(colors=MUTED)

        for spine in ax.spines.values():
            spine.set_color(GRID)

        ax.grid(axis="y", color=GRID, linestyle="--", alpha=0.5)

        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.show()

    # ---------------- EQUIVALENCIAS ----------------

    with salida_equivalencias:

        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=(9, 4))

        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)

        categorias = [
            "Cargas de celular",
            "Costo (COP)"
        ]

        ins = [
            datos["Insertion Sort"]["cargas"],
            datos["Insertion Sort"]["COP"]
        ]

        mer = [
            datos["Merge Sort"]["cargas"],
            datos["Merge Sort"]["COP"]
        ]

        x = np.arange(len(categorias))
        ancho = 0.35

        ax.bar(
            x - ancho/2,
            ins,
            ancho,
            label="Insertion Sort",
            color=COLORES["Insertion Sort"]
        )

        ax.bar(
            x + ancho/2,
            mer,
            ancho,
            label="Merge Sort",
            color=COLORES["Merge Sort"]
        )

        ax.set_xticks(x)
        ax.set_xticklabels(categorias, color=TEXT)

        ax.set_title(
            "Equivalencias del consumo energético",
            color=TEXT,
            fontsize=13
        )

        ax.tick_params(colors=MUTED)

        ax.legend(
            facecolor=CARD2,
            edgecolor=GRID,
            labelcolor=TEXT
        )

        ax.grid(axis="y", color=GRID, linestyle="--", alpha=0.5)

        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.show()


# ============================================================
# EVENTOS
# ============================================================

for widget in [
    slider_n,
    slider_listas,
    slider_costo,
    slider_co2
]:

    widget.observe(actualizar, names="value")


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

display(HTML(f"""

<div style="
    background:{BG};
    padding:20px;
    border-radius:14px;
    margin-bottom:12px;
">

    <h1 style="
        color:{TEXT};
        margin:0;
        font-size:26px;
    ">
        ⚡ ANALIZADOR ENERGÉTICO
    </h1>

    <p style="color:{MUTED};margin-bottom:0;">

        El costo energético de las decisiones algorítmicas

    </p>

</div>

"""))


# Panel de configuración

configuracion = widgets.VBox([

    widgets.HTML(
        f"<h3 style='color:{TEXT}'>Configuración del experimento</h3>"
    ),

    widgets.HBox([
        slider_n,
        slider_listas
    ]),

    widgets.HBox([
        slider_costo,
        slider_co2
    ])

])

# Pestañas

pestanas = widgets.Tab()

pestanas.children = [

    salida_comparacion,
    salida_impacto,
    salida_equivalencias

]

pestanas.set_title(0, "📊 Comparación")
pestanas.set_title(1, "🌱 Impacto ambiental")
pestanas.set_title(2, "📱 Equivalencias")

panel = widgets.VBox([

    configuracion,

    salida_tarjetas,

    salida_ventaja,

    pestanas

])

panel.layout = widgets.Layout(
    width="100%",
    gap="15px"
)

display(panel)

actualizar()
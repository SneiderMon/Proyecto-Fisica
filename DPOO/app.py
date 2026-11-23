# -*- coding: utf-8 -*-
"""
Analizador energético: Insertion Sort vs Merge Sort

Lee mediciones.csv (generado por el script de medición) y traduce la energía
medida en julios a cosas que se entienden: cargas de celular, pesos (COP),
gramos de CO2 y tiempo total.

Dependencias:  pip install customtkinter pandas numpy matplotlib
Uso:           python app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ARCHIVO_DEFECTO = "mediciones.csv"
ALGS = ["Insertion Sort", "Merge Sort"]
J_POR_KWH = 3.6e6
N_CAL = 3                 # puntos usados para ajustar el modelo (igual que el script de medición)
N_MAX_MODELO = 200_000    # hasta qué n se busca el cruce de 10x

# Valores por defecto de los parámetros editables
DEF_LISTAS = 1_000_000
DEF_COSTO = 800.0         # COP por kWh
DEF_CO2 = 0.164           # kg CO2 por kWh (red de Colombia)
DEF_WH = 11.1             # batería de un celular, en Wh (~40 000 J)

# Paleta: azul = Insertion, naranja = Merge (igual que las gráficas del script de medición)
BG = "#14161b"
PANEL = "#1b1e25"
CARD = "#232730"
CARD2 = "#2c313c"
TEXT = "#e9ebf0"
MUTED = "#9aa1af"
GRID = "#3a404d"
OK = "#4ade9a"
WARN = "#fbbf24"
ERR = "#ff6b6b"
COLORES = {"Insertion Sort": "#4c9aff", "Merge Sort": "#ff8a4c"}
MARCAS = {"Insertion Sort": "o", "Merge Sort": "s"}


# ---------------------------------------------------------------------------
# Formato de números
# ---------------------------------------------------------------------------
def fmt_energia(j):
    a = abs(j)
    for limite, div, unidad in ((1e9, 1e9, "GJ"), (1e6, 1e6, "MJ"), (1e3, 1e3, "kJ"),
                                (1, 1, "J"), (1e-3, 1e-3, "mJ")):
        if a >= limite:
            return f"{j / div:,.2f} {unidad}"
    return f"{j * 1e6:,.2f} µJ"


def fmt_tiempo(s):
    if s >= 3600:
        return f"{s / 3600:,.2f} h"
    if s >= 60:
        return f"{s / 60:,.2f} min"
    if s >= 1:
        return f"{s:,.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:,.2f} ms"
    return f"{s * 1e6:,.2f} µs"


def fmt_co2(g):
    if g >= 1e6:
        return f"{g / 1e6:,.2f} t"
    if g >= 1e3:
        return f"{g / 1e3:,.2f} kg"
    if g >= 1:
        return f"{g:,.2f} g"
    return f"{g * 1e3:,.2f} mg"


def fmt_cop(p):
    if p >= 1000:
        return f"$ {p:,.0f} COP"
    if p >= 1:
        return f"$ {p:,.2f} COP"
    return f"$ {p:.4f} COP"


def fmt_cargas(c):
    if c >= 100:
        return f"{c:,.0f}"
    if c >= 1:
        return f"{c:,.2f}"
    return f"{c:.4f}"


def fmt_kwh(k):
    if k >= 1:
        return f"{k:,.2f} kWh"
    return f"{k:.6f} kWh"


# ---------------------------------------------------------------------------
# Datos y modelo (sin interfaz, para poder probarlos por separado)
# ---------------------------------------------------------------------------
class Datos:
    """Mediciones validadas, ordenadas por n y listas para graficar."""

    def __init__(self, ruta):
        df = pd.read_csv(ruta)
        requeridas = {"algoritmo", "n", "tiempo_s", "energia_J"}
        faltan = requeridas - set(df.columns)
        if faltan:
            raise ValueError("Al CSV le faltan las columnas: " + ", ".join(sorted(faltan)))

        self.modo = "desconocido"
        if "modo_energia" in df.columns and df["modo_energia"].notna().any():
            self.modo = str(df["modo_energia"].dropna().iloc[0]).strip().lower()

        num = [c for c in ("tiempo_s", "tiempo_std_s", "energia_J", "energia_std_J") if c in df.columns]
        df["algoritmo"] = df["algoritmo"].astype(str).str.strip()
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
        for c in num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["n", "tiempo_s", "energia_J"])
        df = df[df["algoritmo"].isin(ALGS)]
        df["n"] = df["n"].astype(int)
        df = df.groupby(["algoritmo", "n"], as_index=False)[num].mean()

        por_alg = {a: set(df.loc[df["algoritmo"] == a, "n"]) for a in ALGS}
        for a in ALGS:
            if not por_alg[a]:
                raise ValueError(f"No hay mediciones de {a} en el archivo.")
        comunes = sorted(por_alg[ALGS[0]] & por_alg[ALGS[1]])
        if not comunes:
            raise ValueError("Insertion Sort y Merge Sort no comparten ningún tamaño n.")

        self.ruta = ruta
        self.tamanos = comunes
        self.n = np.array(comunes, dtype=float)
        self.S = {}
        for a in ALGS:
            sub = df[df["algoritmo"] == a].set_index("n").loc[comunes]
            cero = np.zeros(len(comunes))
            self.S[a] = {
                "t": sub["tiempo_s"].to_numpy(float),
                "E": sub["energia_J"].to_numpy(float),
                "tstd": sub["tiempo_std_s"].to_numpy(float) if "tiempo_std_s" in sub else cero,
                "Estd": sub["energia_std_J"].to_numpy(float) if "energia_std_J" in sub else cero,
            }
        self._ajustar_modelo()

    def _ajustar_modelo(self):
        """E = k·n² (Insertion) y E = k·n·log2(n) (Merge), con los primeros N_CAL puntos."""
        self.modelo = None
        self.n_10x = None
        m = min(N_CAL, len(self.n))
        n = self.n[:m]
        fi, fm = n ** 2, n * np.log2(n)
        Ei, Em = self.S[ALGS[0]]["E"][:m], self.S[ALGS[1]]["E"][:m]
        k_i = np.sum(fi * Ei) / np.sum(fi * fi)
        k_m = np.sum(fm * Em) / np.sum(fm * fm)
        if not (np.isfinite(k_i) and np.isfinite(k_m)) or k_i <= 0 or k_m <= 0:
            return
        self.modelo = (k_i, k_m)
        grande = np.arange(2, N_MAX_MODELO + 1, dtype=float)
        razon = (k_i / k_m) * grande / np.log2(grande)
        idx = np.where(razon >= 10)[0]
        if len(idx):
            self.n_10x = int(grande[idx[0]])

    def razon_modelo(self, n):
        k_i, k_m = self.modelo
        n = np.asarray(n, dtype=float)
        return (k_i / k_m) * n / np.log2(n)

    def razon_medida(self):
        Ei, Em = self.S[ALGS[0]]["E"], self.S[ALGS[1]]["E"]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((Ei > 0) & (Em > 0), Ei / Em, np.nan)
        return r


def calcular(datos, idx, listas, costo, co2, wh):
    """Convierte la energía de `listas` listas de tamaño n = tamanos[idx] a unidades tangibles."""
    j_carga = wh * 3600.0
    out = {}
    for a in ALGS:
        E = datos.S[a]["E"][idx] * listas
        T = datos.S[a]["t"][idx] * listas
        kwh = E / J_POR_KWH
        out[a] = {
            "E": E, "T": T, "kwh": kwh,
            "cargas": E / j_carga,
            "cop": kwh * costo,
            "co2_g": kwh * co2 * 1000.0,
        }
    return out


# ---------------------------------------------------------------------------
# Gráficas (reciben una Figure, no dependen de Tk)
# ---------------------------------------------------------------------------
def estilar(ax, titulo, xl="", yl=""):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(titulo, color=TEXT, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel(xl, color=MUTED, fontsize=9)
    ax.set_ylabel(yl, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def leyenda(ax, **kw):
    leg = ax.legend(facecolor=CARD2, edgecolor=GRID, fontsize=9, **kw)
    for t in leg.get_texts():
        t.set_color(TEXT)


def ejes_n_log(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def sin_datos(fig, mensaje):
    fig.clear()
    fig.set_facecolor(CARD)
    ax = fig.add_subplot(111)
    ax.set_facecolor(CARD)
    ax.axis("off")
    ax.text(0.5, 0.5, mensaje, ha="center", va="center", color=MUTED, fontsize=12, wrap=True)


def dibujar_comparacion(fig, calc, listas, n, escala_log):
    fig.clear()
    fig.set_facecolor(CARD)
    axs = fig.subplots(1, 2)
    paneles = [
        (axs[0], "E", "Energía total", fmt_energia, "Julios (J)"),
        (axs[1], "T", "Tiempo total", fmt_tiempo, "Segundos (s)"),
    ]
    for ax, clave, titulo, ff, ylabel in paneles:
        v = np.array([calc[a][clave] for a in ALGS], dtype=float)
        estilar(ax, titulo, "", ylabel)
        pos = v[v > 0]
        if escala_log and pos.size:
            ax.set_yscale("log")
        y = np.where(v > 0, v, np.nan) if (escala_log and pos.size) else v
        ax.bar(range(2), y, color=[COLORES[a] for a in ALGS], width=0.55, zorder=3)
        ax.set_xticks(range(2))
        ax.set_xticklabels(ALGS, color=TEXT, fontsize=10)
        if pos.size:
            if escala_log:
                ax.set_ylim(pos.min() / 6, pos.max() * 8)
            else:
                ax.set_ylim(0, pos.max() * 1.2)
        for i, val in enumerate(v):
            if val > 0 or not escala_log:
                ax.annotate(ff(val), (i, val if val > 0 else 0), xytext=(0, 6),
                            textcoords="offset points", ha="center", color=TEXT,
                            fontsize=10, fontweight="bold")
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{listas:,} listas de n = {n:,}", color=MUTED, fontsize=10)


def dibujar_curvas(fig, datos, idx):
    fig.clear()
    fig.set_facecolor(CARD)
    axs = fig.subplots(1, 2)
    n = datos.n
    paneles = [
        (axs[0], "t", "tstd", "Tiempo por lista", "Segundos (s)"),
        (axs[1], "E", "Estd", "Energía por lista", "Julios (J)"),
    ]
    for ax, clave, std, titulo, ylabel in paneles:
        estilar(ax, titulo, "n (tamaño de la lista)", ylabel)
        ax.set_yscale("log")
        for a in ALGS:
            y, e = datos.S[a][clave], datos.S[a][std]
            m = y > 0
            if m.any():
                ax.errorbar(n[m], y[m], yerr=e[m], marker=MARCAS[a], color=COLORES[a], lw=2,
                            markersize=6, capsize=3, label=a, zorder=3)
            if y[idx] > 0:
                ax.scatter([n[idx]], [y[idx]], s=170, facecolors="none", edgecolors=TEXT,
                           linewidths=2, zorder=5)
        ax.axvline(n[idx], color=MUTED, linestyle=":", linewidth=1)
        ejes_n_log(ax, n)
        leyenda(ax, loc="upper left")


def dibujar_razon(fig, datos, idx):
    fig.clear()
    fig.set_facecolor(CARD)
    ax = fig.add_subplot(111)
    estilar(ax, "Cuántas veces más energía gasta Insertion Sort que Merge Sort",
            "n (tamaño de la lista)", "E Insertion / E Merge (veces)")
    n = datos.n
    r_med = datos.razon_medida()
    ax.set_xscale("log")
    ax.set_yscale("log")
    if datos.modelo is not None:
        n_plot = np.logspace(np.log10(max(2, n[0])), np.log10(max(N_MAX_MODELO, n[-1] * 2)), 400)
        ax.plot(n_plot, datos.razon_modelo(n_plot), color=TEXT, lw=1.8, label="Modelo (n² contra n·log n)")
    m = np.isfinite(r_med)
    if m.any():
        ax.plot(n[m], r_med[m], "o", color=COLORES[ALGS[0]], markersize=8, label="Medido", zorder=4)
    ax.axhline(10, color=MUTED, linestyle="--", linewidth=1, label="Razón = 10")
    if np.isfinite(r_med[idx]):
        ax.scatter([n[idx]], [r_med[idx]], s=220, facecolors="none", edgecolors=TEXT, linewidths=2, zorder=5)
    if datos.n_10x:
        ax.axvline(datos.n_10x, color=WARN, linestyle=":", linewidth=1.2)
        ax.annotate(f"10× desde n ≈ {datos.n_10x:,}", (datos.n_10x, 10), xytext=(8, -18),
                    textcoords="offset points", color=WARN, fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    leyenda(ax, loc="upper left")


# ===========================================================================
# INTERFAZ
# ===========================================================================
import customtkinter as ctk                      # noqa: E402
from tkinter import filedialog, messagebox       # noqa: E402

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def parsear_entero(texto):
    """Acepta 1000000, 1.000.000, 1,000,000, 100k, 2m."""
    t = texto.strip().lower().replace(",", "").replace(".", "").replace(" ", "").replace("_", "")
    mult = 1
    if t.endswith("k"):
        mult, t = 1_000, t[:-1]
    elif t.endswith("m"):
        mult, t = 1_000_000, t[:-1]
    return int(t) * mult


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Analizador energético · Insertion Sort vs Merge Sort")
        self.geometry("1380x880")
        self.minsize(1200, 780)
        self.configure(fg_color=BG)

        self.datos = None
        self.listas = DEF_LISTAS
        self._job = None
        self._ultimo = None
        self.figs, self.canvases = {}, {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._crear_panel()
        self._crear_principal()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self._sincronizar_etiquetas()
        self.after(150, self._carga_inicial)

    # ------------------------------------------------------------------ panel izquierdo
    def _seccion(self, parent, texto):
        ctk.CTkLabel(parent, text=texto, font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=TEXT).pack(anchor="w", padx=18, pady=(20, 2))

    def _fila_slider(self, parent, titulo, desde, hasta, pasos, comando=None):
        fila = ctk.CTkFrame(parent, fg_color="transparent")
        fila.pack(fill="x", padx=18, pady=(8, 0))
        top = ctk.CTkFrame(fila, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=titulo, text_color=MUTED).pack(side="left")
        val = ctk.CTkLabel(top, text="", font=ctk.CTkFont(weight="bold"), text_color=TEXT)
        val.pack(side="right")
        sl = ctk.CTkSlider(fila, from_=desde, to=hasta, number_of_steps=pasos, height=16,
                           command=comando or (lambda _v: self._cambio()))
        sl.pack(fill="x", pady=(6, 0))
        return sl, val

    def _crear_panel(self):
        panel = ctk.CTkScrollableFrame(self, width=310, fg_color=PANEL, corner_radius=0)
        panel.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(panel, text="Energía de algoritmos", font=ctk.CTkFont(size=21, weight="bold"),
                     text_color=TEXT).pack(anchor="w", padx=18, pady=(22, 0))
        ctk.CTkLabel(panel, text="Cuánto cuesta ordenar, en cosas que se entienden",
                     text_color=MUTED, wraplength=260, justify="left").pack(anchor="w", padx=18, pady=(2, 0))

        # Datos
        self._seccion(panel, "Datos")
        self.lbl_archivo = ctk.CTkLabel(panel, text="Ningún archivo cargado", text_color=MUTED,
                                        wraplength=270, justify="left")
        self.lbl_archivo.pack(anchor="w", padx=18, pady=(2, 6))
        fila = ctk.CTkFrame(panel, fg_color="transparent")
        fila.pack(fill="x", padx=18)
        ctk.CTkButton(fila, text="Abrir CSV…", width=120, command=self.abrir_csv).pack(side="left")
        ctk.CTkButton(fila, text="Recargar", width=120, fg_color=CARD2, hover_color=GRID,
                      command=self.recargar).pack(side="right")

        # Tamaño
        self._seccion(panel, "Tamaño de la lista")
        self.sl_n, self.lbl_n = self._fila_slider(panel, "n", 0, 1, 1)

        # Cantidad de listas
        self._seccion(panel, "Cantidad de listas a ordenar")
        self.sl_listas, self.lbl_listas_dummy = self._fila_slider(
            panel, "Escala (10^x)", 0, 9, 90, comando=self._desde_slider_listas)
        self.lbl_listas_dummy.pack_forget()
        self.sl_listas.set(float(np.log10(self.listas)))
        self.var_listas = ctk.StringVar(value=f"{self.listas:,}")
        self.ent_listas = ctk.CTkEntry(panel, textvariable=self.var_listas, justify="right")
        self.ent_listas.pack(fill="x", padx=18, pady=(10, 0))
        self.ent_listas.bind("<Return>", lambda _e: self._confirmar_listas())
        self.ent_listas.bind("<FocusOut>", lambda _e: self._confirmar_listas())
        ctk.CTkLabel(panel, text="Escribe un número o usa 100k, 2m", text_color=MUTED,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=18, pady=(2, 0))
        presets = ctk.CTkFrame(panel, fg_color="transparent")
        presets.pack(fill="x", padx=18, pady=(8, 0))
        for texto, valor in (("1 mil", 1_000), ("100 mil", 100_000), ("1 M", 1_000_000), ("100 M", 100_000_000)):
            ctk.CTkButton(presets, text=texto, width=58, height=26, fg_color=CARD2, hover_color=GRID,
                          command=lambda v=valor: self._poner_listas(v)).pack(side="left", padx=(0, 5))

        # Parámetros
        self._seccion(panel, "Parámetros de conversión")
        self.sl_costo, self.lbl_costo = self._fila_slider(panel, "Costo de la energía", 100, 2000, 38)
        self.sl_co2, self.lbl_co2 = self._fila_slider(panel, "Emisiones de la red", 0.0, 0.5, 500)
        self.sl_wh, self.lbl_wh = self._fila_slider(panel, "Batería del celular", 5, 20, 150)
        self._valores_defecto()

        ctk.CTkButton(panel, text="Restablecer valores", fg_color="transparent", border_width=1,
                      border_color=GRID, hover_color=CARD2, text_color=TEXT,
                      command=self._restablecer).pack(fill="x", padx=18, pady=(22, 6))
        ctk.CTkLabel(panel, text="1 kWh = 3,6 × 10⁶ J", text_color=MUTED,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=18, pady=(2, 18))

    # ------------------------------------------------------------------ zona principal
    def _crear_principal(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=22, pady=18)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(3, weight=1)

        # Encabezado
        enc = ctk.CTkFrame(main, fg_color="transparent")
        enc.grid(row=0, column=0, sticky="ew")
        enc.grid_columnconfigure(0, weight=1)
        izq = ctk.CTkFrame(enc, fg_color="transparent")
        izq.grid(row=0, column=0, sticky="w")
        self.lbl_titulo = ctk.CTkLabel(izq, text="Sin datos", font=ctk.CTkFont(size=26, weight="bold"),
                                       text_color=TEXT)
        self.lbl_titulo.pack(anchor="w")
        self.lbl_sub = ctk.CTkLabel(izq, text="Abre un archivo mediciones.csv para empezar", text_color=MUTED)
        self.lbl_sub.pack(anchor="w")
        der = ctk.CTkFrame(enc, fg_color="transparent")
        der.grid(row=0, column=1, sticky="e")
        self.badge = ctk.CTkLabel(der, text="", corner_radius=12, fg_color=CARD2, text_color=MUTED,
                                  font=ctk.CTkFont(size=12, weight="bold"), height=26)
        self.badge.pack(side="left", padx=(0, 12))
        ctk.CTkButton(der, text="Exportar gráfica", width=130, fg_color=CARD2, hover_color=GRID,
                      command=self.exportar_png).pack(side="left", padx=(0, 8))
        ctk.CTkButton(der, text="Exportar tabla", width=120, fg_color=CARD2, hover_color=GRID,
                      command=self.exportar_csv).pack(side="left")

        # Tarjetas
        tarjetas = ctk.CTkFrame(main, fg_color="transparent")
        tarjetas.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        tarjetas.grid_columnconfigure((0, 1), weight=1, uniform="cards")
        self.kpi = {}
        for i, a in enumerate(ALGS):
            self.kpi[a] = self._crear_tarjeta(tarjetas, a)
            self.kpi[a]["frame"].grid(row=0, column=i, sticky="nsew", padx=(0, 8) if i == 0 else (8, 0))

        # Veredicto
        self.banner = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12)
        self.banner.grid(row=2, column=0, sticky="ew", pady=(12, 12))
        self.lbl_veredicto = ctk.CTkLabel(self.banner, text="", text_color=TEXT, wraplength=900,
                                          justify="left", font=ctk.CTkFont(size=14))
        self.lbl_veredicto.pack(anchor="w", padx=18, pady=12)

        # Gráficas
        self.tabs = ctk.CTkTabview(main, fg_color=CARD, corner_radius=14,
                                   segmented_button_selected_color="#3b6fd4",
                                   command=lambda: self.after(30, self._redibujar_actual))
        self.tabs.grid(row=3, column=0, sticky="nsew")
        for nombre in ("Comparación", "Curvas por n", "Cruce 10×"):
            tab = self.tabs.add(nombre)
            if nombre == "Comparación":
                barra = ctk.CTkFrame(tab, fg_color="transparent")
                barra.pack(fill="x")
                self.sw_log = ctk.CTkSwitch(barra, text="Escala logarítmica", command=self._cambio)
                self.sw_log.select()
                self.sw_log.pack(side="right", padx=8)
            fig = Figure(figsize=(8, 3.6), facecolor=CARD, layout="constrained")
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().configure(highlightthickness=0, bg=CARD)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figs[nombre], self.canvases[nombre] = fig, canvas
            sin_datos(fig, "Abre un archivo mediciones.csv para ver las gráficas")

    def _crear_tarjeta(self, parent, alg):
        f = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14)
        ctk.CTkFrame(f, height=5, fg_color=COLORES[alg], corner_radius=3).pack(fill="x", padx=18, pady=(14, 0))
        ctk.CTkLabel(f, text=alg, font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORES[alg]).pack(anchor="w", padx=18, pady=(8, 0))
        lbl = {"frame": f}
        lbl["E"] = ctk.CTkLabel(f, text="—", font=ctk.CTkFont(size=30, weight="bold"), text_color=TEXT)
        lbl["E"].pack(anchor="w", padx=18)
        lbl["kwh"] = ctk.CTkLabel(f, text="", text_color=MUTED)
        lbl["kwh"].pack(anchor="w", padx=18, pady=(0, 6))
        for clave, texto in (("T", "Tiempo total"), ("cargas", "Cargas de celular"),
                             ("cop", "Costo"), ("co2", "CO₂ emitido")):
            fila = ctk.CTkFrame(f, fg_color="transparent")
            fila.pack(fill="x", padx=18, pady=2)
            ctk.CTkLabel(fila, text=texto, text_color=MUTED).pack(side="left")
            lbl[clave] = ctk.CTkLabel(fila, text="—", text_color=TEXT, font=ctk.CTkFont(weight="bold"))
            lbl[clave].pack(side="right")
        ctk.CTkFrame(f, height=8, fg_color="transparent").pack()
        return lbl

    # ------------------------------------------------------------------ valores y eventos
    def _valores_defecto(self):
        self.sl_costo.set(DEF_COSTO)
        self.sl_co2.set(DEF_CO2)
        self.sl_wh.set(DEF_WH)

    def _restablecer(self):
        self._valores_defecto()
        self._poner_listas(DEF_LISTAS)

    def _poner_listas(self, valor):
        self.listas = int(min(max(valor, 1), 1_000_000_000))
        self.var_listas.set(f"{self.listas:,}")
        self.sl_listas.set(float(np.log10(self.listas)))
        self._cambio()

    def _confirmar_listas(self):
        try:
            valor = parsear_entero(self.var_listas.get())
            if valor < 1:
                raise ValueError
        except ValueError:
            self.var_listas.set(f"{self.listas:,}")
            return
        self._poner_listas(valor)

    def _sincronizar_etiquetas(self):
        self.lbl_costo.configure(text=f"{self.sl_costo.get():,.0f} COP/kWh")
        self.lbl_co2.configure(text=f"{self.sl_co2.get():.3f} kg CO₂/kWh")
        self.lbl_wh.configure(text=f"{self.sl_wh.get():.1f} Wh")
        if self.datos is not None:
            self.lbl_n.configure(text=f"{self._n_actual():,}")

    def _indice_n(self):
        if self.datos is None:
            return 0
        return int(min(max(round(self.sl_n.get()), 0), len(self.datos.tamanos) - 1))

    def _n_actual(self):
        return self.datos.tamanos[self._indice_n()]

    def _desde_slider_listas(self, exp):
        self.listas = int(float(f"{10 ** exp:.3g}"))
        self.var_listas.set(f"{self.listas:,}")
        self._cambio()

    def _cambio(self):
        self._sincronizar_etiquetas()
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(40, self.refrescar)

    # ------------------------------------------------------------------ carga de datos
    def _carga_inicial(self):
        base = os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv and sys.argv[0] else __file__))
        for ruta in (os.path.join(os.getcwd(), ARCHIVO_DEFECTO), os.path.join(base, ARCHIVO_DEFECTO)):
            if os.path.exists(ruta):
                self.cargar(ruta)
                return
        self.lbl_sub.configure(text=f"No encontré {ARCHIVO_DEFECTO}. Usa «Abrir CSV…» o corre primero el script de medición.",
                               text_color=WARN)

    def abrir_csv(self):
        ruta = filedialog.askopenfilename(title="Selecciona mediciones.csv",
                                          filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if ruta:
            self.cargar(ruta)

    def recargar(self):
        if self.datos is None:
            self.abrir_csv()
        else:
            self.cargar(self.datos.ruta)

    def cargar(self, ruta):
        try:
            nuevos = Datos(ruta)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("No se pudo cargar el archivo", f"{os.path.basename(ruta)}\n\n{e}")
            return
        n_previo = self._n_actual() if self.datos is not None else None
        self.datos = nuevos
        tam = nuevos.tamanos
        self.sl_n.configure(from_=0, to=max(1, len(tam) - 1), number_of_steps=max(1, len(tam) - 1))
        self.sl_n.set(tam.index(n_previo) if n_previo in tam else 0)
        self.lbl_archivo.configure(text=os.path.basename(ruta), text_color=TEXT)
        if nuevos.modo == "medida":
            self.badge.configure(text="  ● Energía medida con HWiNFO  ", text_color=OK)
        elif nuevos.modo == "estimada":
            self.badge.configure(text="  ● Energía estimada (sin HWiNFO)  ", text_color=WARN)
        else:
            self.badge.configure(text="  ● Origen de la energía desconocido  ", text_color=MUTED)
        self._sincronizar_etiquetas()
        self.refrescar()

    # ------------------------------------------------------------------ actualización
    def refrescar(self):
        self._job = None
        if self.datos is None:
            return
        d = self.datos
        idx = self._indice_n()
        n = d.tamanos[idx]
        costo, co2, wh = self.sl_costo.get(), self.sl_co2.get(), self.sl_wh.get()
        calc = calcular(d, idx, self.listas, costo, co2, wh)
        self._ultimo = {"calc": calc, "n": n, "listas": self.listas, "costo": costo, "co2": co2, "wh": wh}

        self.lbl_titulo.configure(text="Insertion Sort vs Merge Sort")
        self.lbl_sub.configure(text=f"Listas de n = {n:,}  ·  {self.listas:,} listas", text_color=MUTED)

        for a in ALGS:
            c, k = calc[a], self.kpi[a]
            k["E"].configure(text=fmt_energia(c["E"]))
            k["kwh"].configure(text=fmt_kwh(c["kwh"]))
            k["T"].configure(text=fmt_tiempo(c["T"]))
            k["cargas"].configure(text=fmt_cargas(c["cargas"]))
            k["cop"].configure(text=fmt_cop(c["cop"]))
            k["co2"].configure(text=fmt_co2(c["co2_g"]))

        Ei, Em = calc[ALGS[0]]["E"], calc[ALGS[1]]["E"]
        if Ei > 0 and Em > 0:
            mayor, menor = (ALGS[0], ALGS[1]) if Ei >= Em else (ALGS[1], ALGS[0])
            alto, bajo = calc[mayor], calc[menor]
            razon = alto["E"] / bajo["E"]
            dif = {k: alto[k] - bajo[k] for k in ("E", "cargas", "cop", "co2_g")}
            self.lbl_veredicto.configure(
                text=(f"{mayor} gasta {razon:,.1f}× más energía que {menor}. "
                      f"Usar {menor} ahorra {fmt_energia(dif['E'])}: {fmt_cargas(dif['cargas'])} cargas de celular, "
                      f"{fmt_cop(dif['cop'])} y {fmt_co2(dif['co2_g'])} de CO₂."))
            self.banner.configure(fg_color="#1d3a2f")
        else:
            self.lbl_veredicto.configure(
                text="La energía medida para este n es cero o negativa (el consumo quedó por debajo del reposo). "
                     "Repite la medición con más repeticiones o un n mayor.")
            self.banner.configure(fg_color="#4a3a1a")

        self._redibujar_todo()

    def _dibujar(self, nombre):
        d, u = self.datos, self._ultimo
        if d is None or u is None:
            return
        fig = self.figs[nombre]
        idx = self._indice_n()
        if nombre == "Comparación":
            dibujar_comparacion(fig, u["calc"], u["listas"], u["n"], bool(self.sw_log.get()))
        elif nombre == "Curvas por n":
            dibujar_curvas(fig, d, idx)
        else:
            if d.modelo is None:
                sin_datos(fig, "No se pudo ajustar el modelo: la energía de los primeros tamaños es cero o negativa.")
            else:
                dibujar_razon(fig, d, idx)
        self.canvases[nombre].draw_idle()

    def _redibujar_actual(self):
        self._dibujar(self.tabs.get())

    def _redibujar_todo(self):
        for nombre in self.figs:
            self._dibujar(nombre)

    # ------------------------------------------------------------------ exportar
    def exportar_png(self):
        if self.datos is None:
            messagebox.showinfo("Exportar gráfica", "Primero carga un archivo de mediciones.")
            return
        nombre = self.tabs.get()
        ruta = filedialog.asksaveasfilename(title="Guardar gráfica", defaultextension=".png",
                                            initialfile=f"grafica_{nombre.replace(' ', '_').lower()}.png",
                                            filetypes=[("PNG", "*.png")])
        if ruta:
            try:
                self.figs[nombre].savefig(ruta, dpi=200, facecolor=CARD)
            except OSError as e:
                messagebox.showerror("No se pudo guardar", str(e))

    def exportar_csv(self):
        if self._ultimo is None:
            messagebox.showinfo("Exportar tabla", "Primero carga un archivo de mediciones.")
            return
        ruta = filedialog.asksaveasfilename(title="Guardar tabla", defaultextension=".csv",
                                            initialfile="comparacion_energia.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not ruta:
            return
        u = self._ultimo
        filas = []
        for a in ALGS:
            c = u["calc"][a]
            filas.append({
                "algoritmo": a, "n": u["n"], "listas": u["listas"],
                "energia_J": c["E"], "energia_kWh": c["kwh"], "tiempo_total_s": c["T"],
                "cargas_celular": c["cargas"], "costo_COP": c["cop"], "co2_g": c["co2_g"],
                "COP_por_kWh": u["costo"], "kgCO2_por_kWh": u["co2"], "bateria_Wh": u["wh"],
            })
        try:
            pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8-sig")
        except OSError as e:
            messagebox.showerror("No se pudo guardar", str(e))

    def _cerrar(self):
        if self._job is not None:
            self.after_cancel(self._job)
        self.quit()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()

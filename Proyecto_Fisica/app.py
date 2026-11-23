
# -*- coding: utf-8 -*-
"""
Analizador energético: Insertion Sort vs Merge Sort
Mide con HWiNFO (un botón) y traduce julios a cargas de celular, COP, CO2 y tiempo.
 
Dependencias:  pip install customtkinter pandas numpy matplotlib
Archivos:      app.py + motor.py en la misma carpeta
Uso:           python app.py
"""
import os
import queue
import sys
import threading
import time
 
import numpy as np
import pandas as pd
import matplotlib
 
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
 
from motor import (ALGS, ConfigMedicion, Medidor, MedicionCancelada, Datos, Parametros,
                   calcular, diagnostico_hwinfo, equivalencias, presupuesto_a_julios)
 
# ---------------------------------------------------------------------------
ARCHIVO_DEFECTO = "mediciones.csv"
N_MAX_MODELO = 200_000
DEF_LISTAS, DEF_COSTO, DEF_CO2, DEF_WH = 1_000_000, 800.0, 0.164, 11.1
PRESETS = {"Rápida": "rapida", "Estándar": "estandar", "Precisa": "precisa"}
DURACION = {"Rápida": "≈ 2 min", "Estándar": "≈ 6 min", "Precisa": "≈ 15 min"}
UNIDADES = ["COP", "kWh", "J", "cargas", "gCO2"]
 
BG, PANEL, CARD, CARD2 = "#14161b", "#1b1e25", "#232730", "#2c313c"
TEXT, MUTED, GRID = "#e9ebf0", "#9aa1af", "#3a404d"
OK, WARN, ERR = "#4ade9a", "#fbbf24", "#ff6b6b"
COLORES = {"Insertion Sort": "#4c9aff", "Merge Sort": "#ff8a4c"}
MARCAS = {"Insertion Sort": "o", "Merge Sort": "s"}
 
 
# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------
def fmt_energia(j):
    a = abs(j)
    for lim, div, u in ((1e9, 1e9, "GJ"), (1e6, 1e6, "MJ"), (1e3, 1e3, "kJ"), (1, 1, "J"), (1e-3, 1e-3, "mJ")):
        if a >= lim:
            return f"{j / div:,.2f} {u}"
    return f"{j * 1e6:,.2f} µJ"
 
 
def fmt_tiempo(s):
    if s >= 86400:
        return f"{s / 86400:,.2f} días"
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
    return f"{c:,.0f}" if c >= 100 else (f"{c:,.2f}" if c >= 1 else f"{c:.4f}")
 
 
def fmt_kwh(k):
    return f"{k:,.2f} kWh" if k >= 1 else f"{k:.6f} kWh"
 
 
def fmt_horas(h):
    if h >= 24:
        return f"{h / 24:,.1f} días"
    return f"{h:,.1f} h" if h >= 1 else f"{h * 60:,.1f} min"
 
 
def parsear_numero(texto):
    """Acepta 1000000, 1.000.000, 1,5, 100k, 2m, 3.5"""
    t = texto.strip().lower().replace(" ", "").replace("_", "")
    mult = 1
    if t.endswith("k"):
        mult, t = 1_000, t[:-1]
    elif t.endswith("m"):
        mult, t = 1_000_000, t[:-1]
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".") if t.count(",") == 1 and len(t.split(",")[1]) != 3 else t.replace(",", "")
    elif t.count(".") > 1 or (t.count(".") == 1 and len(t.split(".")[1]) == 3 and mult == 1):
        t = t.replace(".", "")
    return float(t) * mult
 
 
# ---------------------------------------------------------------------------
# Gráficas
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
 
 
def leyenda(ax, handles=None, **kw):
    leg = ax.legend(handles=handles, facecolor=CARD2, edgecolor=GRID, fontsize=9, **kw) if handles else \
        ax.legend(facecolor=CARD2, edgecolor=GRID, fontsize=9, **kw)
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
    for ax, clave, titulo, ff, yl in ((axs[0], "E", "Energía total", fmt_energia, "Julios (J)"),
                                      (axs[1], "T", "Tiempo total", fmt_tiempo, "Segundos (s)")):
        v = np.array([calc[a][clave] for a in ALGS], dtype=float)
        estilar(ax, titulo, "", yl)
        pos = v[v > 0]
        usa_log = escala_log and pos.size
        if usa_log:
            ax.set_yscale("log")
        y = np.where(v > 0, v, np.nan) if usa_log else v
        ax.bar(range(2), y, color=[COLORES[a] for a in ALGS], width=0.55, zorder=3)
        ax.set_xticks(range(2))
        ax.set_xticklabels(ALGS, color=TEXT, fontsize=10)
        if pos.size:
            ax.set_ylim((pos.min() / 6, pos.max() * 8) if usa_log else (0, pos.max() * 1.2))
        for i, val in enumerate(v):
            if val > 0 or not usa_log:
                ax.annotate(ff(val), (i, max(val, 0)), xytext=(0, 6), textcoords="offset points",
                            ha="center", color=TEXT, fontsize=10, fontweight="bold")
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{listas:,} listas de n = {n:,}", color=MUTED, fontsize=10)
 
 
def dibujar_curvas(fig, datos, idx):
    fig.clear()
    fig.set_facecolor(CARD)
    axs = fig.subplots(1, 2)
    n = datos.n
    for ax, clave, std, titulo, yl in ((axs[0], "t", "tstd", "Tiempo por lista", "Segundos (s)"),
                                       (axs[1], "E", "Estd", "Energía por lista", "Julios (J)")):
        estilar(ax, titulo, "n (tamaño de la lista)", yl)
        ax.set_yscale("log")
        for a in ALGS:
            y, e = datos.S[a][clave], datos.S[a][std]
            m = y > 0
            if m.any():
                ax.errorbar(n[m], y[m], yerr=e[m], marker=MARCAS[a], color=COLORES[a], lw=2,
                            markersize=6, capsize=3, label=a, zorder=3)
            if y[idx] > 0:
                ax.scatter([n[idx]], [y[idx]], s=170, facecolors="none", edgecolors=TEXT, linewidths=2, zorder=5)
        ax.axvline(n[idx], color=MUTED, linestyle=":", linewidth=1)
        ejes_n_log(ax, n)
        leyenda(ax, loc="upper left")
 
 
def dibujar_razon(fig, datos, idx):
    fig.clear()
    fig.set_facecolor(CARD)
    ax = fig.add_subplot(111)
    estilar(ax, "Cuántas veces más energía gasta Insertion Sort que Merge Sort",
            "n (tamaño de la lista)", "E Insertion / E Merge (veces)")
    n, r_med = datos.n, datos.razon_medida()
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
 
 
def dibujar_traza(fig, traza, ventanas, p_reposo=None, en_vivo=False):
    fig.clear()
    fig.set_facecolor(CARD)
    if traza is None or len(traza) < 2:
        sin_datos(fig, "Aquí aparece la potencia del CPU medida con HWiNFO.\n"
                       "Inicia una medición, o carga un CSV que tenga traza_potencia.csv al lado.")
        return
    ax = fig.add_subplot(111)
    estilar(ax, "Potencia del CPU durante la medición" + (" (en vivo)" if en_vivo else ""),
            "Tiempo (s)", "Potencia (W)")
    ax.step(traza[:, 0], traza[:, 1], where="pre", color=TEXT, lw=1.1, zorder=3)
    handles = []
    if ventanas is not None:
        for _, v in ventanas.iterrows():
            alg = v["algoritmo"] if isinstance(v["algoritmo"], str) else ""
            ax.axvspan(v["t0"], v["t1"], color=COLORES.get(alg, MUTED), alpha=0.20 if alg else 0.10, lw=0)
        handles = [Patch(color=COLORES[a], alpha=0.5, label=a) for a in ALGS] + \
                  [Patch(color=MUTED, alpha=0.3, label="Reposo / pausas")]
    if p_reposo:
        ax.axhline(p_reposo, color=WARN, linestyle="--", linewidth=1)
        handles.append(plt_line(WARN, f"Reposo ≈ {p_reposo:.1f} W"))
    if handles:
        leyenda(ax, handles=handles, loc="upper left")
 
 
def plt_line(color, label):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=color, linestyle="--", label=label)
 
 
# ===========================================================================
# INTERFAZ
# ===========================================================================
import customtkinter as ctk                      # noqa: E402
from tkinter import filedialog, messagebox       # noqa: E402
 
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
 
 
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Analizador energético · Insertion Sort vs Merge Sort")
        self.geometry("1420x900")
        self.minsize(1240, 800)
        self.configure(fg_color=BG)
 
        self.datos = None
        self.listas = DEF_LISTAS
        self._job = None
        self._ultimo = None
        self.figs, self.canvases = {}, {}
        self.carpeta = os.path.dirname(os.path.abspath(sys.argv[0] if sys.argv and sys.argv[0] else __file__))
        # medición
        self.midiendo = False
        self.cola = queue.Queue()
        self.cancelar = threading.Event()
        self.live_t, self.live_p, self._t_live0, self._ult_dibujo = [], [], 0.0, 0.0
 
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
 
        # --- Medición
        self._seccion(panel, "Medir en este computador")
        self.seg_preset = ctk.CTkSegmentedButton(panel, values=list(PRESETS), command=self._cambio_preset)
        self.seg_preset.set("Estándar")
        self.seg_preset.pack(fill="x", padx=18, pady=(4, 0))
        self.lbl_duracion = ctk.CTkLabel(panel, text=f"Duración {DURACION['Estándar']} · no uses el PC mientras mide",
                                         text_color=MUTED, font=ctk.CTkFont(size=11), wraplength=270, justify="left")
        self.lbl_duracion.pack(anchor="w", padx=18, pady=(2, 6))
        self.menu_sensor = ctk.CTkOptionMenu(panel, values=["Sensor: automático"], fg_color=CARD2,
                                             button_color=CARD2, button_hover_color=GRID)
        self.menu_sensor.pack(fill="x", padx=18)
        fila = ctk.CTkFrame(panel, fg_color="transparent")
        fila.pack(fill="x", padx=18, pady=(8, 0))
        self.btn_verificar = ctk.CTkButton(fila, text="Verificar HWiNFO", width=130, fg_color=CARD2,
                                           hover_color=GRID, command=self.verificar_hwinfo)
        self.btn_verificar.pack(side="left")
        self.btn_cancelar = ctk.CTkButton(fila, text="Cancelar", width=110, fg_color="transparent",
                                          border_width=1, border_color=GRID, hover_color=CARD2,
                                          state="disabled", command=self.cancelar_medicion)
        self.btn_cancelar.pack(side="right")
        self.btn_medir = ctk.CTkButton(panel, text="▶  Iniciar medición", height=38, fg_color="#2f9e6b",
                                       hover_color="#26825a", font=ctk.CTkFont(size=14, weight="bold"),
                                       command=self.iniciar_medicion)
        self.btn_medir.pack(fill="x", padx=18, pady=(10, 0))
        self.barra = ctk.CTkProgressBar(panel, height=10)
        self.barra.set(0)
        self.barra.pack(fill="x", padx=18, pady=(12, 0))
        self.lbl_prog = ctk.CTkLabel(panel, text="Listo para medir", text_color=MUTED, wraplength=270,
                                     justify="left")
        self.lbl_prog.pack(anchor="w", padx=18, pady=(4, 0))
        self.lbl_vatios = ctk.CTkLabel(panel, text="", font=ctk.CTkFont(size=20, weight="bold"), text_color=OK)
        self.lbl_vatios.pack(anchor="w", padx=18)
 
        # --- Datos
        self._seccion(panel, "Datos")
        self.lbl_archivo = ctk.CTkLabel(panel, text="Ningún archivo cargado", text_color=MUTED,
                                        wraplength=270, justify="left")
        self.lbl_archivo.pack(anchor="w", padx=18, pady=(2, 6))
        fila = ctk.CTkFrame(panel, fg_color="transparent")
        fila.pack(fill="x", padx=18)
        ctk.CTkButton(fila, text="Abrir CSV…", width=120, command=self.abrir_csv).pack(side="left")
        ctk.CTkButton(fila, text="Recargar", width=120, fg_color=CARD2, hover_color=GRID,
                      command=self.recargar).pack(side="right")
 
        # --- Tamaño
        self._seccion(panel, "Tamaño de la lista")
        self.sl_n, self.lbl_n = self._fila_slider(panel, "n", 0, 1, 1)
 
        # --- Cantidad de listas
        self._seccion(panel, "Cantidad de listas a ordenar")
        self.sl_listas, dummy = self._fila_slider(panel, "Escala (10^x)", 0, 9, 90, comando=self._desde_slider_listas)
        dummy.pack_forget()
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
 
        # --- Parámetros
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
 
        enc = ctk.CTkFrame(main, fg_color="transparent")
        enc.grid(row=0, column=0, sticky="ew")
        enc.grid_columnconfigure(0, weight=1)
        izq = ctk.CTkFrame(enc, fg_color="transparent")
        izq.grid(row=0, column=0, sticky="w")
        self.lbl_titulo = ctk.CTkLabel(izq, text="Sin datos", font=ctk.CTkFont(size=26, weight="bold"),
                                       text_color=TEXT)
        self.lbl_titulo.pack(anchor="w")
        self.lbl_sub = ctk.CTkLabel(izq, text="Mide en este computador o abre un archivo mediciones.csv",
                                    text_color=MUTED)
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
 
        tarjetas = ctk.CTkFrame(main, fg_color="transparent")
        tarjetas.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        tarjetas.grid_columnconfigure((0, 1), weight=1, uniform="cards")
        self.kpi = {}
        for i, a in enumerate(ALGS):
            self.kpi[a] = self._crear_tarjeta(tarjetas, a)
            self.kpi[a]["frame"].grid(row=0, column=i, sticky="nsew", padx=(0, 8) if i == 0 else (8, 0))
 
        self.banner = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12)
        self.banner.grid(row=2, column=0, sticky="ew", pady=(12, 12))
        self.lbl_veredicto = ctk.CTkLabel(self.banner, text="", text_color=TEXT, wraplength=900,
                                          justify="left", font=ctk.CTkFont(size=14))
        self.lbl_veredicto.pack(anchor="w", padx=18, pady=(12, 2))
        self.lbl_equiv = ctk.CTkLabel(self.banner, text="", text_color=MUTED, wraplength=900, justify="left")
        self.lbl_equiv.pack(anchor="w", padx=18, pady=(0, 12))
 
        self.tabs = ctk.CTkTabview(main, fg_color=CARD, corner_radius=14,
                                   segmented_button_selected_color="#3b6fd4",
                                   command=lambda: self.after(30, self._redibujar_actual))
        self.tabs.grid(row=3, column=0, sticky="nsew")
        for nombre in ("Comparación", "Curvas por n", "Cruce 10×", "Potencia", "Presupuesto"):
            tab = self.tabs.add(nombre)
            if nombre == "Presupuesto":
                self._crear_tab_presupuesto(tab)
                continue
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
            sin_datos(fig, "Inicia una medición o abre un archivo mediciones.csv para ver las gráficas")
 
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
 
    def _crear_tab_presupuesto(self, tab):
        ctk.CTkLabel(tab, text="¿Cabe esta tarea en mi presupuesto de energía?",
                     font=ctk.CTkFont(size=15, weight="bold"), text_color=TEXT).pack(anchor="w", padx=10, pady=(6, 0))
        ctk.CTkLabel(tab, text="Usa el n y la cantidad de listas del panel izquierdo. Predicción con el modelo ajustado.",
                     text_color=MUTED).pack(anchor="w", padx=10)
        fila = ctk.CTkFrame(tab, fg_color="transparent")
        fila.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(fila, text="Presupuesto", text_color=MUTED).pack(side="left")
        self.var_pres = ctk.StringVar(value="1000")
        ent = ctk.CTkEntry(fila, textvariable=self.var_pres, width=130, justify="right")
        ent.pack(side="left", padx=8)
        ent.bind("<Return>", lambda _e: self._cambio())
        ent.bind("<FocusOut>", lambda _e: self._cambio())
        self.var_unidad = ctk.StringVar(value="COP")
        ctk.CTkOptionMenu(fila, values=UNIDADES, variable=self.var_unidad, width=100, fg_color=CARD2,
                          button_color=CARD2, button_hover_color=GRID,
                          command=lambda _v: self._cambio()).pack(side="left")
        self.lbl_pres_j = ctk.CTkLabel(fila, text="", text_color=MUTED)
        self.lbl_pres_j.pack(side="left", padx=12)
 
        cuerpo = ctk.CTkFrame(tab, fg_color="transparent")
        cuerpo.pack(fill="x", padx=6)
        cuerpo.grid_columnconfigure((0, 1), weight=1, uniform="p")
        self.pres = {}
        for i, a in enumerate(ALGS):
            f = ctk.CTkFrame(cuerpo, fg_color=CARD2, corner_radius=12)
            f.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            ctk.CTkLabel(f, text=a, font=ctk.CTkFont(size=14, weight="bold"),
                         text_color=COLORES[a]).pack(anchor="w", padx=14, pady=(10, 0))
            d = {"estado": ctk.CTkLabel(f, text="—", font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT)}
            d["estado"].pack(anchor="w", padx=14)
            for clave, texto in (("uso", "Usa del presupuesto"), ("listas", "Máx. listas con este n"),
                                 ("nmax", "Máx. n con estas listas")):
                r = ctk.CTkFrame(f, fg_color="transparent")
                r.pack(fill="x", padx=14, pady=2)
                ctk.CTkLabel(r, text=texto, text_color=MUTED).pack(side="left")
                d[clave] = ctk.CTkLabel(r, text="—", text_color=TEXT, font=ctk.CTkFont(weight="bold"))
                d[clave].pack(side="right")
            ctk.CTkFrame(f, height=8, fg_color="transparent").pack()
            self.pres[a] = d
        self.lbl_pres_rec = ctk.CTkLabel(tab, text="", text_color=TEXT, wraplength=900, justify="left",
                                         font=ctk.CTkFont(size=14))
        self.lbl_pres_rec.pack(anchor="w", padx=10, pady=(10, 0))
        self.lbl_pres_aviso = ctk.CTkLabel(tab, text="", text_color=WARN, wraplength=900, justify="left")
        self.lbl_pres_aviso.pack(anchor="w", padx=10, pady=(4, 0))
 
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
            valor = int(parsear_numero(self.var_listas.get()))
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
 
    def _cambio_preset(self, valor):
        self.lbl_duracion.configure(text=f"Duración {DURACION[valor]} · no uses el PC mientras mide")
 
    # ------------------------------------------------------------------ medición
    def verificar_hwinfo(self):
        ok, msg, sensores = diagnostico_hwinfo()
        if ok:
            self.menu_sensor.configure(values=["Sensor: automático"] + sensores)
            self.lbl_prog.configure(text="✔ " + msg, text_color=OK)
        else:
            self.lbl_prog.configure(text="✘ " + msg, text_color=WARN)
 
    def iniciar_medicion(self):
        if self.midiendo:
            return
        ok, msg, _ = diagnostico_hwinfo()
        if not ok:
            seguir = messagebox.askyesno(
                "HWiNFO no detectado",
                msg + "\n\n¿Medir de todos modos con energía ESTIMADA (potencia supuesta)?\n"
                      "Los resultados no serán energía real.")
            if not seguir:
                return
        cfg = ConfigMedicion.preset(PRESETS[self.seg_preset.get()])
        sel = self.menu_sensor.get()
        cfg.sensor = sel.split(" · ", 1)[1] if " · " in sel else None
        cfg.carpeta = self.carpeta
 
        self.cancelar = threading.Event()
        self.cola = queue.Queue()
        self.live_t, self.live_p, self._t_live0 = [], [], time.perf_counter()
        self._set_midiendo(True)
        self.barra.set(0)
        self.lbl_prog.configure(text="Iniciando…", text_color=MUTED)
 
        def trabajo():
            try:
                r = \
                    Medidor(cfg, self.cola.put, self.cancelar).ejecutar()
                self.cola.put({"fase": "fin", "resultado": r})
            except MedicionCancelada:
                self.cola.put({"fase": "cancelada"})
            except Exception as e:  # noqa: BLE001
                self.cola.put({"fase": "error", "mensaje": str(e)})
 
        threading.Thread(target=trabajo, daemon=True).start()
        self.after(200, self._sondear)
 
    def cancelar_medicion(self):
        self.cancelar.set()
        self.lbl_prog.configure(text="Cancelando…", text_color=WARN)
 
    def _set_midiendo(self, activo):
        self.midiendo = activo
        self.btn_medir.configure(state="disabled" if activo else "normal")
        self.btn_verificar.configure(state="disabled" if activo else "normal")
        self.btn_cancelar.configure(state="normal" if activo else "disabled")
        self.seg_preset.configure(state="disabled" if activo else "normal")
        if not activo:
            self.lbl_vatios.configure(text="")
 
    def _sondear(self):
        try:
            while True:
                self._procesar_mensaje(self.cola.get_nowait())
        except queue.Empty:
            pass
        if self.midiendo:
            self.after(200, self._sondear)
 
    def _procesar_mensaje(self, m):
        fase = m.get("fase")
        if fase == "fin":
            self._set_midiendo(False)
            r = m["resultado"]
            self.barra.set(1)
            self.lbl_prog.configure(text="✔ Medición terminada", text_color=OK)
            self.cargar(r.ruta_csv)
            self.tabs.set("Potencia")
            self._redibujar_actual()
            if r.advertencias:
                messagebox.showwarning("Revisa esto", "\n\n".join(r.advertencias))
        elif fase == "cancelada":
            self._set_midiendo(False)
            self.barra.set(0)
            self.lbl_prog.configure(text="Medición cancelada", text_color=WARN)
        elif fase == "error":
            self._set_midiendo(False)
            self.barra.set(0)
            self.lbl_prog.configure(text="Error en la medición", text_color=ERR)
            messagebox.showerror("Error en la medición", m.get("mensaje", ""))
        else:
            self.barra.set(m["fraccion"])
            rest = m["restante_s"]
            self.lbl_prog.configure(text=f'{m["mensaje"]}\nFaltan ≈ {fmt_tiempo(rest)}', text_color=MUTED)
            if m.get("potencia_w") is not None:
                self.lbl_vatios.configure(text=f'{m["potencia_w"]:.1f} W')
                self.live_t.append(time.perf_counter() - self._t_live0)
                self.live_p.append(m["potencia_w"])
                if self.tabs.get() == "Potencia" and time.perf_counter() - self._ult_dibujo > 1.0:
                    self._ult_dibujo = time.perf_counter()
                    dibujar_traza(self.figs["Potencia"], np.column_stack([self.live_t, self.live_p]),
                                  None, en_vivo=True)
                    self.canvases["Potencia"].draw_idle()
 
    # ------------------------------------------------------------------ carga de datos
    def _carga_inicial(self):
        for ruta in (os.path.join(os.getcwd(), ARCHIVO_DEFECTO), os.path.join(self.carpeta, ARCHIVO_DEFECTO)):
            if os.path.exists(ruta):
                self.cargar(ruta)
                return
        self.lbl_sub.configure(text="Aún no hay mediciones. Abre HWiNFO y pulsa «Iniciar medición».",
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
    def _params(self):
        return Parametros(self.sl_costo.get(), self.sl_co2.get(), self.sl_wh.get())
 
    def refrescar(self):
        self._job = None
        if self.datos is None:
            return
        d, idx = self.datos, self._indice_n()
        n, p = d.tamanos[idx], self._params()
        calc = calcular(d, idx, self.listas, p.costo_kwh, p.co2_kg_kwh, p.wh_celular)
        self._ultimo = {"calc": calc, "n": n, "listas": self.listas, "costo": p.costo_kwh,
                        "co2": p.co2_kg_kwh, "wh": p.wh_celular}
 
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
            eq = equivalencias(dif["E"], dif["co2_g"])
            self.lbl_equiv.configure(
                text=(f"Lo que se ahorra equivale a un bombillo LED encendido {fmt_horas(eq['horas_bombillo_led_9W'])}, "
                      f"una TV encendida {fmt_horas(eq['horas_tv_100W'])} y "
                      f"{eq['km_en_carro']:,.1f} km recorridos en carro."))
            self.banner.configure(fg_color="#1d3a2f")
        else:
            self.lbl_veredicto.configure(
                text="La energía medida para este n es cero o negativa (el consumo quedó por debajo del reposo). "
                     "Repite la medición con el preset «Precisa» o un n mayor.")
            self.lbl_equiv.configure(text="")
            self.banner.configure(fg_color="#4a3a1a")
 
        self._actualizar_presupuesto(n, p)
        self._redibujar_todo()
 
    def _actualizar_presupuesto(self, n, p):
        m = self.datos.pred
        try:
            valor = parsear_numero(self.var_pres.get())
            if valor <= 0:
                raise ValueError
        except ValueError:
            self.lbl_pres_rec.configure(text="Escribe un presupuesto válido (número positivo).")
            return
        if not m.valido:
            self.lbl_pres_rec.configure(text="No se pudo ajustar el modelo con estos datos.")
            return
        J = presupuesto_a_julios(valor, self.var_unidad.get(), p)
        self.lbl_pres_j.configure(text=f"= {fmt_energia(J)}")
        ev = m.evaluar(n, self.listas, J)
        for a in ALGS:
            e, w = ev["por_algoritmo"][a], self.pres[a]
            w["estado"].configure(text="✔ Cabe" if e["cabe"] else "✘ No cabe", text_color=OK if e["cabe"] else ERR)
            w["uso"].configure(text=f'{e["uso_pct"]:,.1f} %')
            w["listas"].configure(text=f"{m.max_listas(a, n, J):,}")
            nm = m.max_n(a, self.listas, J)
            w["nmax"].configure(text="> 10 000 000 000" if nm >= 1e10 else f"{nm:,}")
        self.lbl_pres_rec.configure(text=ev["recomendacion"])
        self.lbl_pres_aviso.configure(
            text="⚠ n está muy fuera del rango medido: la predicción es una extrapolación." if ev["extrapolando"] else "")
 
    def _dibujar(self, nombre):
        d, u = self.datos, self._ultimo
        if nombre not in self.figs:
            return
        fig = self.figs[nombre]
        if nombre == "Potencia":
            if self.midiendo and self.live_t:
                return
            if d is None or d.traza is None:
                dibujar_traza(fig, None, None)
            else:
                dibujar_traza(fig, d.traza, d.ventanas, d.p_reposo)
        elif d is None or u is None:
            return
        elif nombre == "Comparación":
            dibujar_comparacion(fig, u["calc"], u["listas"], u["n"], bool(self.sw_log.get()))
        elif nombre == "Curvas por n":
            dibujar_curvas(fig, d, self._indice_n())
        else:
            if d.modelo is None:
                sin_datos(fig, "No se pudo ajustar el modelo: la energía de los primeros tamaños es cero o negativa.")
            else:
                dibujar_razon(fig, d, self._indice_n())
        self.canvases[nombre].draw_idle()
 
    def _redibujar_actual(self):
        self._dibujar(self.tabs.get())
 
    def _redibujar_todo(self):
        for nombre in self.figs:
            self._dibujar(nombre)
 
    # ------------------------------------------------------------------ exportar
    def exportar_png(self):
        nombre = self.tabs.get()
        if self.datos is None or nombre not in self.figs:
            messagebox.showinfo("Exportar gráfica", "Carga datos y elige una pestaña con gráfica.")
            return
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
                                            initialfile="comparacion_energia.csv", filetypes=[("CSV", "*.csv")])
        if not ruta:
            return
        u = self._ultimo
        filas = [{"algoritmo": a, "n": u["n"], "listas": u["listas"], "energia_J": u["calc"][a]["E"],
                  "energia_kWh": u["calc"][a]["kwh"], "tiempo_total_s": u["calc"][a]["T"],
                  "cargas_celular": u["calc"][a]["cargas"], "costo_COP": u["calc"][a]["cop"],
                  "co2_g": u["calc"][a]["co2_g"], "COP_por_kWh": u["costo"],
                  "kgCO2_por_kWh": u["co2"], "bateria_Wh": u["wh"]} for a in ALGS]
        try:
            pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8-sig")
        except OSError as e:
            messagebox.showerror("No se pudo guardar", str(e))
 
    def _cerrar(self):
        self.cancelar.set()
        if self._job is not None:
            self.after_cancel(self._job)
        self.quit()
        self.destroy()
 
 
if __name__ == "__main__":
    App().mainloop()
# -*- coding: utf-8 -*-
"""
motor.py  ·  Lógica del proyecto "El costo energético de las decisiones algorítmicas"

Todo lo que NO es interfaz vive aquí:
  1. Algoritmos (Insertion Sort, Merge Sort)
  2. Lector de HWiNFO por memoria compartida (sin CSV, sin sincronizar relojes)
  3. Medidor: corre el experimento completo y entrega energía por lista
  4. Datos: lee/valida mediciones.csv (drop-in del app.py actual)
  5. Modelo predictivo: E(n), cruces, presupuesto energético
  6. Conversión a unidades tangibles (kWh, cargas de celular, COP, CO2, equivalencias)

Requisitos del usuario para medir (una sola vez):
  - Abrir HWiNFO en modo "Sensors-only"
  - Settings ▸ General ▸ marcar "Shared Memory Support"
  - (recomendado) Polling period = 500 ms

Uso por consola:
  python motor.py --diagnostico      # verifica HWiNFO y lista sensores de potencia
  python motor.py --medir            # corre el experimento y escribe mediciones.csv
"""
from __future__ import annotations

import ctypes
import math
import os
import random
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

ALGS = ["Insertion Sort", "Merge Sort"]
J_POR_KWH = 3.6e6


# ===========================================================================
# 1. ALGORITMOS
# ===========================================================================
def insertion_sort(lista):
    a = list(lista)
    for i in range(1, len(a)):
        clave = a[i]
        j = i - 1
        while j >= 0 and a[j] > clave:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = clave
    return a


def merge_sort(lista):
    n = len(lista)
    if n <= 1:
        return list(lista)
    mitad = n // 2
    izq, der = merge_sort(lista[:mitad]), merge_sort(lista[mitad:])
    res, i, j = [], 0, 0
    while i < len(izq) and j < len(der):
        if izq[i] <= der[j]:
            res.append(izq[i]); i += 1
        else:
            res.append(der[j]); j += 1
    res.extend(izq[i:])
    res.extend(der[j:])
    return res


ALGORITMOS = {"Insertion Sort": insertion_sort, "Merge Sort": merge_sort}


def lista_aleatoria(n, semilla=42):
    rng = random.Random(semilla)
    return [rng.random() for _ in range(n)]


# ===========================================================================
# 2. LECTOR DE HWiNFO (memoria compartida, solo Windows)
# ===========================================================================
class HWiNFOError(Exception):
    pass


class MedicionCancelada(Exception):
    pass


class LectorHWiNFO:
    """
    Lee los sensores de HWiNFO en vivo desde la memoria compartida
    "Global\\HWiNFO_SENS_SM2". Estructuras con pack(1), little-endian.
    Los tamaños de elemento se toman de la cabecera, así que tolera versiones nuevas.
    """
    NOMBRES = ("Global\\HWiNFO_SENS_SM2", "HWiNFO_SENS_SM2")
    FIRMA = 0x53695748            # 'HWiS'
    FMT_CAB = "<IIIqIIIIII"       # 44 bytes
    FMT_SENSOR = "<II128s128s"
    FMT_LECT = "<III128s128s16sdddd"
    TIPO_POTENCIA = 5             # SENSOR_TYPE_POWER

    # etiquetas preferidas, en orden (igual que el script original)
    PREFERIDAS = ("cpu package power", "package power", "total cpu power",
                  "cpu ppt", "ppt", "cpu power")

    def __init__(self, etiqueta: Optional[str] = None):
        self.etiqueta_pedida = etiqueta
        self._ptr = None
        self._cab = None
        self._idx = None
        self.etiqueta = None      # etiqueta elegida (para mostrar)

    # -- acceso a memoria ---------------------------------------------------
    def abrir(self):
        if sys.platform != "win32":
            raise HWiNFOError("La lectura de HWiNFO solo funciona en Windows.")
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        k32.OpenFileMappingW.restype = ctypes.c_void_p
        k32.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_uint32, ctypes.c_size_t]
        k32.MapViewOfFile.restype = ctypes.c_void_p
        h = None
        for nombre in self.NOMBRES:
            h = k32.OpenFileMappingW(0x0004, 0, nombre)   # FILE_MAP_READ
            if h:
                break
        if not h:
            raise HWiNFOError(
                "No encontré HWiNFO. Abre HWiNFO en modo Sensors-only y activa "
                "Settings ▸ General ▸ «Shared Memory Support».")
        self._ptr = k32.MapViewOfFile(h, 0x0004, 0, 0, 0)
        if not self._ptr:
            raise HWiNFOError("No pude mapear la memoria compartida de HWiNFO.")
        self._leer_cab()
        self._elegir()

    def cerrar(self):
        self._ptr = None   # la vista se libera al terminar el proceso (es de solo lectura)

    def _bytes(self, offset, n):
        return ctypes.string_at(self._ptr + offset, n)

    def _leer_cab(self):
        c = struct.unpack(self.FMT_CAB, self._bytes(0, struct.calcsize(self.FMT_CAB)))
        if c[0] != self.FIRMA:
            raise HWiNFOError("HWiNFO se cerró o la memoria compartida no es válida.")
        self._cab = {"off_s": c[4], "sz_s": c[5], "n_s": c[6],
                     "off_r": c[7], "sz_r": c[8], "n_r": c[9]}

    @staticmethod
    def _txt(b):
        return b.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()

    def _lectura(self, i):
        c = self._cab
        raw = self._bytes(c["off_r"] + i * c["sz_r"], struct.calcsize(self.FMT_LECT))
        t, s_idx, _id, orig, user, unidad, val, vmin, vmax, vavg = struct.unpack(self.FMT_LECT, raw)
        return {"idx": i, "tipo": t, "sensor_idx": s_idx,
                "label": self._txt(user) or self._txt(orig),
                "unidad": self._txt(unidad), "valor": val}

    def _nombre_sensor(self, s_idx):
        c = self._cab
        raw = self._bytes(c["off_s"] + s_idx * c["sz_s"], struct.calcsize(self.FMT_SENSOR))
        _, _, orig, user = struct.unpack(self.FMT_SENSOR, raw)
        return self._txt(user) or self._txt(orig)

    # -- sensores -----------------------------------------------------------
    def listar_potencias(self):
        """Todas las lecturas en vatios, con el nombre de su sensor."""
        self._leer_cab()
        out = []
        for i in range(self._cab["n_r"]):
            r = self._lectura(i)
            if r["tipo"] == self.TIPO_POTENCIA or r["unidad"].lower() == "w":
                r["sensor"] = self._nombre_sensor(r["sensor_idx"])
                out.append(r)
        return out

    def _elegir(self):
        cands = self.listar_potencias()
        if not cands:
            raise HWiNFOError("HWiNFO no expone ningún sensor de potencia (W).")
        if self.etiqueta_pedida:
            pedida = self.etiqueta_pedida.lower()
            for r in cands:
                if pedida in r["label"].lower():
                    return self._fijar(r)
            raise HWiNFOError(f"No encontré el sensor «{self.etiqueta_pedida}».")
        for clave in self.PREFERIDAS:
            for r in cands:
                if clave in r["label"].lower():
                    return self._fijar(r)
        raise HWiNFOError("No encontré un sensor de potencia de CPU. Elige uno de la lista: "
                          + ", ".join(r["label"] for r in cands[:8]))

    def _fijar(self, r):
        self._idx = r["idx"]
        self.etiqueta = f'{r["sensor"]} · {r["label"]}'

    def leer(self) -> float:
        """Potencia actual (W) del sensor elegido."""
        return self._lectura(self._idx)["valor"]


def diagnostico_hwinfo(etiqueta=None):
    """Para un botón «Verificar HWiNFO». Devuelve (ok, mensaje, [sensores de potencia])."""
    try:
        lec = LectorHWiNFO(etiqueta)
        lec.abrir()
        sensores = [f'{r["sensor"]} · {r["label"]}' for r in lec.listar_potencias()]
        return True, f"HWiNFO conectado. Sensor: {lec.etiqueta} ({lec.leer():.1f} W)", sensores
    except HWiNFOError as e:
        return False, str(e), []


# ===========================================================================
# 3. MEDIDOR
# ===========================================================================
@dataclass
class ConfigMedicion:
    tamanos: tuple = (500, 1000, 2000, 3000, 4000, 6000)
    repeticiones: int = 3
    tiempo_min: float = 6.0        # cada corrida dura al menos esto (repitiendo K veces)
    t_reposo: float = 10.0
    pausa: float = 3.0             # reposo entre corridas (evita arrastrar calor/lectura previa)
    semilla: int = 42
    sensor: Optional[str] = None   # texto de la etiqueta del sensor (None = automático)
    permitir_estimada: bool = True # si no hay HWiNFO, usar potencia supuesta
    p_reposo_supuesto: float = 5.0
    p_alg_supuesto: float = 20.0
    carpeta: str = "."

    @staticmethod
    def preset(nombre):
        if nombre == "rapida":
            return ConfigMedicion(tamanos=(500, 1000, 2000, 3000), repeticiones=2,
                                  tiempo_min=4.0, t_reposo=6.0, pausa=2.0)
        if nombre == "precisa":
            return ConfigMedicion(repeticiones=5, tiempo_min=10.0, t_reposo=15.0, pausa=4.0)
        return ConfigMedicion()


@dataclass
class ResultadoMedicion:
    df: pd.DataFrame
    modo: str                      # "medida" | "estimada"
    p_reposo: float
    sensor: Optional[str]
    intervalo_hwinfo: Optional[float]
    traza: np.ndarray              # columnas: t_s, potencia_W
    ventanas: list
    advertencias: list = field(default_factory=list)
    ruta_csv: str = ""


class _Muestreador(threading.Thread):
    """Lee el sensor ~50 veces/s y guarda solo los instantes en que el valor cambia."""
    def __init__(self, lector, dt=0.02):
        super().__init__(daemon=True)
        self.lector, self.dt = lector, dt
        self.parar = threading.Event()
        self.eventos = []           # [(t_perf, W)]
        self.ultimo = None
        self.error = None

    def run(self):
        try:
            prev = None
            while not self.parar.is_set():
                v = self.lector.leer()
                t = time.perf_counter()
                self.ultimo = v
                if v != prev:
                    self.eventos.append((t, v))
                    prev = v
                time.sleep(self.dt)
        except Exception as e:  # noqa: BLE001
            self.error = e


def energia_ventana(ev_t, ev_p, a, b):
    """
    Energía (J) en [a, b]. Cada lectura de HWiNFO es la potencia promedio del
    intervalo que TERMINA en el instante en que cambia, así que se integra como
    escalón hacia atrás (no trapecio): corrige el desfase de la lectura.
    """
    if len(ev_t) == 0:
        return 0.0
    inicios = np.concatenate([[ev_t[0]], ev_t[:-1]])
    solape = np.clip(np.minimum(ev_t, b) - np.maximum(inicios, a), 0.0, None)
    e = float(np.sum(solape * ev_p))
    if b > ev_t[-1]:                                   # cola sin lectura nueva todavía
        e += (b - max(ev_t[-1], a)) * ev_p[-1]
    return e


class Medidor:
    """
    m = Medidor(cfg, on_progreso=callback, cancelar=threading.Event())
    resultado = m.ejecutar()      # bloqueante: llámalo desde un hilo

    on_progreso recibe un dict: fase, mensaje, fraccion (0-1), potencia_w, restante_s
    """
    def __init__(self, cfg: ConfigMedicion, on_progreso: Optional[Callable] = None,
                 cancelar: Optional[threading.Event] = None, lector=None):
        self.cfg = cfg
        self.cb = on_progreso or (lambda _d: None)
        self.cancelar = cancelar or threading.Event()
        self.lector = lector
        self._t_ini = None
        self._total = 1.0
        self._muestreador = None

    # -- utilidades -----------------------------------------------------------
    def _chk(self):
        if self.cancelar.is_set():
            raise MedicionCancelada()

    def _dormir(self, seg):
        fin = time.perf_counter() + seg
        while True:
            self._chk()
            r = fin - time.perf_counter()
            if r <= 0:
                return
            time.sleep(min(0.1, r))

    def _avisar(self, fase, mensaje):
        el = time.perf_counter() - self._t_ini
        frac = min(0.99, el / self._total) if self._total > 0 else 0.0
        self.cb({"fase": fase, "mensaje": mensaje, "fraccion": frac,
                 "potencia_w": getattr(self._muestreador, "ultimo", None),
                 "restante_s": max(0.0, self._total - el)})

    # -- experimento ----------------------------------------------------------
    def ejecutar(self) -> ResultadoMedicion:
        cfg, advertencias = self.cfg, []

        # 1) conectar con HWiNFO (o caer a modo estimado)
        modo, sensor = "medida", None
        if self.lector is None:
            try:
                self.lector = LectorHWiNFO(cfg.sensor)
                self.lector.abrir()
            except HWiNFOError as e:
                if not cfg.permitir_estimada:
                    raise
                self.lector, modo = None, "estimada"
                advertencias.append(f"Sin HWiNFO ({e}) → energía ESTIMADA con potencia supuesta.")
        if self.lector is not None:
            sensor = getattr(self.lector, "etiqueta", "sensor")

        # 2) datos, calentamiento y calibración de K
        datos = {n: lista_aleatoria(n, cfg.semilla) for n in cfg.tamanos}
        for alg in ALGORITMOS.values():
            alg(lista_aleatoria(500, cfg.semilla))
        K, t_una = {}, {}
        self._t_ini, self._total = time.perf_counter(), 1.0
        self._avisar("calibrar", "Calibrando repeticiones…")
        for nombre, alg in ALGORITMOS.items():
            for n in cfg.tamanos:
                self._chk()
                t0 = time.perf_counter()
                alg(datos[n])
                t_una[(nombre, n)] = max(time.perf_counter() - t0, 1e-6)
                K[(nombre, n)] = max(1, math.ceil(cfg.tiempo_min / t_una[(nombre, n)]))

        corridas_plan = [(r, n, a) for r in range(cfg.repeticiones)
                         for n in cfg.tamanos for a in ALGORITMOS]
        dur_corridas = sum(max(cfg.tiempo_min, t_una[(a, n)]) + cfg.pausa for _, n, a in corridas_plan)
        self._total = 2.0 + cfg.t_reposo * 1.5 + dur_corridas + 2.0

        # 3) arrancar el muestreador (queda corriendo todo el experimento)
        t_ref = time.perf_counter()
        if self.lector is not None:
            self._muestreador = _Muestreador(self.lector)
            self._muestreador.start()
        self._t_ini = t_ref

        ventanas = []
        try:
            self._avisar("reposo", "Midiendo reposo (no uses el computador)…")
            r0 = time.perf_counter(); self._dormir(cfg.t_reposo); r1 = time.perf_counter()
            ventanas.append({"tipo": "reposo", "algoritmo": "", "n": 0, "rep": 0, "K": 0,
                             "t0": r0, "t1": r1, "duracion": r1 - r0})

            hecho = 0
            for rep, n, nombre in corridas_plan:
                hecho += 1
                self._avisar("medir", f"{nombre} · n = {n:,} · repetición {rep + 1}/{cfg.repeticiones}")
                self._dormir(cfg.pausa)
                alg, k, d = ALGORITMOS[nombre], K[(nombre, n)], datos[n]
                t0 = time.perf_counter()
                for _ in range(k):
                    alg(d)
                    self._chk()
                t1 = time.perf_counter()
                ventanas.append({"tipo": "algoritmo", "algoritmo": nombre, "n": n, "rep": rep,
                                 "K": k, "t0": t0, "t1": t1, "duracion": t1 - t0})

            self._avisar("reposo", "Midiendo reposo final…")
            r0 = time.perf_counter(); self._dormir(cfg.t_reposo * 0.5); r1 = time.perf_counter()
            ventanas.append({"tipo": "reposo", "algoritmo": "", "n": 0, "rep": 1, "K": 0,
                             "t0": r0, "t1": r1, "duracion": r1 - r0})
            self._avisar("procesar", "Cerrando mediciones…")
            self._dormir(2.5)             # deja entrar la última lectura de HWiNFO
        finally:
            if self._muestreador is not None:
                self._muestreador.parar.set()
                self._muestreador.join(timeout=2)

        if self._muestreador is not None and self._muestreador.error is not None:
            raise HWiNFOError(f"Falló la lectura de HWiNFO: {self._muestreador.error}")

        # 4) procesar
        res = self._procesar(ventanas, modo, sensor, t_ref, advertencias)
        self.cb({"fase": "listo", "mensaje": "Medición terminada", "fraccion": 1.0,
                 "potencia_w": None, "restante_s": 0.0})
        return res

    def _procesar(self, ventanas, modo, sensor, t_ref, advertencias):
        cfg = self.cfg
        reposos = [v for v in ventanas if v["tipo"] == "reposo"]
        corridas = [v for v in ventanas if v["tipo"] == "algoritmo"]
        intervalo, traza = None, np.zeros((0, 2))
        p_reposo, ruido = cfg.p_reposo_supuesto, 0.0

        if modo == "medida":
            ev = np.array(self._muestreador.eventos, dtype=float)
            if len(ev) < 5:
                raise HWiNFOError("HWiNFO casi no actualizó valores. ¿Está pausado el logging/sensores?")
            ev_t, ev_p = ev[:, 0], ev[:, 1]
            traza = np.column_stack([ev_t - t_ref, ev_p])

            r = reposos[0]
            m = (ev_t >= r["t0"]) & (ev_t <= r["t1"])
            if m.sum() >= 3:
                intervalo = float(np.median(np.diff(ev_t[m])))
                ruido = float(np.std(ev_p[m]))
            e_rep = sum(energia_ventana(ev_t, ev_p, v["t0"], v["t1"]) for v in reposos)
            p_reposo = e_rep / sum(v["duracion"] for v in reposos)

            if intervalo and intervalo > cfg.tiempo_min / 4:
                advertencias.append(
                    f"HWiNFO actualiza cada ≈{intervalo:.1f} s; baja el «Polling period» a 500–1000 ms "
                    f"o sube el tiempo mínimo por corrida para más precisión.")
            for v in corridas:
                v["energia_total"] = energia_ventana(ev_t, ev_p, v["t0"], v["t1"]) - p_reposo * v["duracion"]
        else:
            for v in corridas:
                v["energia_total"] = cfg.p_alg_supuesto * v["duracion"]

        for v in corridas:
            v["tiempo"] = v["duracion"] / v["K"]
            v["energia"] = v["energia_total"] / v["K"]

        filas = []
        for nombre in ALGORITMOS:
            for n in cfg.tamanos:
                g = [v for v in corridas if v["algoritmo"] == nombre and v["n"] == n]
                t = np.array([v["tiempo"] for v in g]); e = np.array([v["energia"] for v in g])
                filas.append({"algoritmo": nombre, "n": n, "K": g[0]["K"],
                              "tiempo_s": t.mean(), "tiempo_std_s": t.std(),
                              "energia_J": e.mean(), "energia_std_J": e.std(),
                              "potencia_alg_W": e.mean() / t.mean(), "modo_energia": modo,
                              "p_reposo_W": p_reposo, "sensor": sensor or "",
                              "intervalo_hwinfo_s": intervalo if intervalo else ""})
        df = pd.DataFrame(filas)

        if modo == "medida":
            neta = df["potencia_alg_W"].median()
            if neta < max(1.0, 3 * ruido):
                advertencias.append("La potencia neta del algoritmo es comparable al ruido del reposo: "
                                    "cierra otras aplicaciones y repite, o usa el preset «precisa».")
            if (df["energia_J"] <= 0).any():
                advertencias.append("Hay energías ≤ 0: el reposo quedó más alto que la ejecución en algún punto.")

        os.makedirs(cfg.carpeta, exist_ok=True)
        ruta = os.path.join(cfg.carpeta, "mediciones.csv")
        df.to_csv(ruta, index=False, encoding="utf-8")
        if len(traza):
            np.savetxt(os.path.join(cfg.carpeta, "traza_potencia.csv"), traza, delimiter=",",
                       header="t_s,potencia_W", comments="")
        pd.DataFrame([{**v, "t0": v["t0"] - t_ref, "t1": v["t1"] - t_ref} for v in ventanas]
                     ).drop(columns=["energia_total", "tiempo", "energia"], errors="ignore"
                            ).to_csv(os.path.join(cfg.carpeta, "ventanas.csv"), index=False)
        return ResultadoMedicion(df, modo, p_reposo, sensor, intervalo, traza,
                                 [{**v, "t0": v["t0"] - t_ref, "t1": v["t1"] - t_ref} for v in ventanas],
                                 advertencias, ruta)


# ===========================================================================
# 4. MODELO PREDICTIVO
# ===========================================================================
def f_ins(n):
    return np.asarray(n, dtype=float) ** 2


def f_merge(n):
    n = np.maximum(np.asarray(n, dtype=float), 2.0)
    return n * np.log2(n)


FORMAS = {"Insertion Sort": f_ins, "Merge Sort": f_merge}


def _ajustar_k(f, y):
    """k de y = k·f minimizando el error RELATIVO (no deja que los n grandes dominen)."""
    f, y = np.asarray(f, float), np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0)
    if not ok.any():
        return None
    r = f[ok] / y[ok]
    k = float(np.sum(r) / np.sum(r * r))
    return k if np.isfinite(k) and k > 0 else None


def pendiente_loglog(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return None
    lx, ly = np.log(x[ok]), np.log(y[ok])
    return float(np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2))


class Modelo:
    """E(n) = k·n² para Insertion y E(n) = k·n·log2(n) para Merge (idem para el tiempo)."""

    def __init__(self, n, E, t):
        self.n = np.asarray(n, float)
        self.E, self.t = E, t
        self.kE = {a: _ajustar_k(FORMAS[a](self.n), E[a]) for a in ALGS}
        self.kT = {a: _ajustar_k(FORMAS[a](self.n), t[a]) for a in ALGS}

    @property
    def valido(self):
        return all(self.kE[a] for a in ALGS)

    def energia(self, alg, n):
        return self.kE[alg] * FORMAS[alg](n)

    def tiempo(self, alg, n):
        return self.kT[alg] * FORMAS[alg](n)

    def razon(self, n):
        return self.energia(ALGS[0], n) / self.energia(ALGS[1], n)

    def error_ajuste(self, alg):
        """Error relativo medio del modelo contra lo medido."""
        y = np.asarray(self.E[alg], float)
        ok = y > 0
        if not ok.any():
            return None
        return float(np.mean(np.abs(self.energia(alg, self.n[ok]) - y[ok]) / y[ok]))

    def pendiente(self, alg):
        """Exponente medido (log-log): ≈2 para Insertion, ≈1.1 para Merge."""
        return pendiente_loglog(self.n, self.E[alg])

    def n_cruce(self, factor=10.0, n_max=1e12):
        """Menor n entero donde Insertion gasta ≥ factor × Merge (None si no ocurre)."""
        if not self.valido:
            return None
        lo, hi = 3.0, float(n_max)
        if self.razon(lo) >= factor:
            return 3
        if self.razon(hi) < factor:
            return None
        while hi - lo > 1:
            mid = math.floor((lo + hi) / 2)
            lo, hi = (lo, mid) if self.razon(mid) >= factor else (mid, hi)
        return int(hi)

    def extrapolando(self, n):
        return n > 10 * self.n.max() or n < self.n.min() / 10

    # -- presupuesto energético ---------------------------------------------
    def max_listas(self, alg, n, presupuesto_j):
        e = float(self.energia(alg, n))
        return int(presupuesto_j // e) if e > 0 else 0

    def max_n(self, alg, listas, presupuesto_j, n_max=1e10):
        """Mayor n tal que ordenar `listas` listas cabe en el presupuesto."""
        def cabe(n):
            return self.energia(alg, n) * listas <= presupuesto_j
        if not cabe(2):
            return 0
        if cabe(n_max):
            return int(n_max)
        lo, hi = 2, int(n_max)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if cabe(mid) else (lo, mid)
        return lo

    def evaluar(self, n, listas, presupuesto_j):
        """¿Cabe la tarea en el presupuesto? Compara ambos algoritmos y recomienda."""
        out = {}
        for a in ALGS:
            E = float(self.energia(a, n)) * listas
            out[a] = {"E": E, "cabe": E <= presupuesto_j,
                      "uso_pct": 100.0 * E / presupuesto_j if presupuesto_j > 0 else math.inf}
        mejor = min(ALGS, key=lambda a: out[a]["E"])
        if not out[mejor]["cabe"]:
            rec = "Ningún algoritmo cabe en el presupuesto: reduce n, la cantidad de listas o cambia de estrategia."
        elif not any(out[a]["cabe"] for a in ALGS if a != mejor):
            rec = f"Solo {mejor} cabe en el presupuesto."
        else:
            rec = f"Ambos caben; {mejor} es el más eficiente."
        return {"por_algoritmo": out, "mejor": mejor, "recomendacion": rec,
                "extrapolando": bool(self.extrapolando(n))}


# ===========================================================================
# 5. DATOS (lee mediciones.csv) — compatible con el app.py actual
# ===========================================================================
class Datos:
    def __init__(self, ruta):
        df = pd.read_csv(ruta)
        faltan = {"algoritmo", "n", "tiempo_s", "energia_J"} - set(df.columns)
        if faltan:
            raise ValueError("Al CSV le faltan las columnas: " + ", ".join(sorted(faltan)))

        self.modo = "desconocido"
        if "modo_energia" in df.columns and df["modo_energia"].notna().any():
            self.modo = str(df["modo_energia"].dropna().iloc[0]).strip().lower()
        self.p_reposo = float(df["p_reposo_W"].iloc[0]) if "p_reposo_W" in df.columns else None
        self.sensor = str(df["sensor"].iloc[0]) if "sensor" in df.columns and pd.notna(df["sensor"].iloc[0]) else ""

        num = [c for c in ("tiempo_s", "tiempo_std_s", "energia_J", "energia_std_J", "potencia_alg_W")
               if c in df.columns]
        df["algoritmo"] = df["algoritmo"].astype(str).str.strip()
        for c in ["n"] + num:
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

        self.ruta, self.tamanos, self.n = ruta, comunes, np.array(comunes, dtype=float)
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

        self.pred = Modelo(self.n, {a: self.S[a]["E"] for a in ALGS}, {a: self.S[a]["t"] for a in ALGS})
        self.modelo = (self.pred.kE[ALGS[0]], self.pred.kE[ALGS[1]]) if self.pred.valido else None
        self.n_10x = self.pred.n_cruce(10) if self.pred.valido else None

        # traza de potencia (si el medidor la guardó junto al CSV)
        carpeta = os.path.dirname(os.path.abspath(ruta))
        self.traza = self.ventanas = None
        try:
            p = os.path.join(carpeta, "traza_potencia.csv")
            if os.path.exists(p):
                self.traza = pd.read_csv(p).to_numpy(float)
            p = os.path.join(carpeta, "ventanas.csv")
            if os.path.exists(p):
                self.ventanas = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            pass

    def razon_modelo(self, n):
        return self.pred.razon(n)

    def razon_medida(self):
        Ei, Em = self.S[ALGS[0]]["E"], self.S[ALGS[1]]["E"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((Ei > 0) & (Em > 0), Ei / Em, np.nan)


# ===========================================================================
# 6. CONVERSIÓN A UNIDADES TANGIBLES
# ===========================================================================
@dataclass
class Parametros:
    costo_kwh: float = 800.0       # COP por kWh
    co2_kg_kwh: float = 0.164      # kg CO2 por kWh (red de Colombia)
    wh_celular: float = 11.1       # batería de un celular (Wh)


def convertir(E_j, p: Parametros):
    kwh = E_j / J_POR_KWH
    return {"E": E_j, "kwh": kwh, "cargas": E_j / (p.wh_celular * 3600.0),
            "cop": kwh * p.costo_kwh, "co2_g": kwh * p.co2_kg_kwh * 1000.0}


def equivalencias(E_j, co2_g):
    """Comparaciones cotidianas para el dashboard."""
    return {"horas_bombillo_led_9W": E_j / (9 * 3600.0),
            "horas_tv_100W": E_j / (100 * 3600.0),
            "km_en_carro": co2_g / 120.0,            # ≈120 g CO2/km
            "arboles_dia": co2_g / 60.0}             # un árbol absorbe ≈22 kg CO2/año ≈ 60 g/día


def presupuesto_a_julios(valor, unidad, p: Parametros):
    """Unidades: 'J', 'kWh', 'COP', 'cargas', 'gCO2'."""
    if unidad == "J":
        return valor
    if unidad == "kWh":
        return valor * J_POR_KWH
    if unidad == "COP":
        return valor / p.costo_kwh * J_POR_KWH
    if unidad == "cargas":
        return valor * p.wh_celular * 3600.0
    if unidad == "gCO2":
        return valor / (p.co2_kg_kwh * 1000.0) * J_POR_KWH
    raise ValueError(f"Unidad desconocida: {unidad}")


def calcular(datos, idx, listas, costo, co2, wh):
    """Misma firma que usa el app.py actual."""
    p = Parametros(costo, co2, wh)
    out = {}
    for a in ALGS:
        c = convertir(datos.S[a]["E"][idx] * listas, p)
        c["T"] = datos.S[a]["t"][idx] * listas
        out[a] = c
    return out


# ===========================================================================
# CLI
# ===========================================================================
def _main():
    if "--diagnostico" in sys.argv:
        ok, msg, sensores = diagnostico_hwinfo()
        print(("✔ " if ok else "✘ ") + msg)
        for s in sensores:
            print("   ·", s)
        return
    if "--medir" in sys.argv:
        preset = sys.argv[sys.argv.index("--medir") + 1] if len(sys.argv) > sys.argv.index("--medir") + 1 else "estandar"
        cfg = ConfigMedicion.preset(preset)

        def mostrar(d):
            w = f'{d["potencia_w"]:.1f} W' if d["potencia_w"] is not None else "—"
            print(f'\r[{d["fraccion"] * 100:5.1f}%] {d["mensaje"][:60]:<60} {w:>8}  faltan {d["restante_s"]:.0f}s',
                  end="", flush=True)
        r = Medidor(cfg, mostrar).ejecutar()
        print(f"\nModo: {r.modo} · reposo {r.p_reposo:.2f} W · archivo: {r.ruta_csv}")
        for a in r.advertencias:
            print("⚠", a)
        return
    print(__doc__)


if __name__ == "__main__":
    _main()
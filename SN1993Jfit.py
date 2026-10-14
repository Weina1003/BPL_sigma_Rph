"""SN1993J: density-following energy; dynesty sigma calibration then fixed-sigma 10D production, <=100 days.
Modified 2026-10-14. SPDX-License-Identifier: GPL-3.0-or-later.
Sampling workflow adapted from the user-supplied TransFit fit.py.
Run: python fit_sn1993j_dynesty_two_stage_100d.py [--resume] [--workers 4].
"""
#Tcolor_ph_K：光球连续谱颜色温度，无有效发光光球时为NaN。
#Tdirect_K：直接辐射连续谱指定温度，等于最佳样本的T_floor；
from pathlib import Path
from dataclasses import dataclass
import json
import hashlib
import os
import multiprocessing as mp
from contextlib import ExitStack
from datetime import datetime, timezone
from uuid import uuid4
# 每个 worker 只用一个数值库线程，避免进程之间争抢 CPU。
for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_thread_var] = '1'
import numpy as np
import numba
# 动态导入检查时不写 Numba 磁盘缓存，避免模块别名影响随后独立运行。
_NUMBA_DISK_CACHE = __name__ in ('__main__','__mp_main__')
try:
    import dynesty
    from dynesty import utils as dyfunc
except ImportError as exc:
    raise ImportError('请先安装依赖：python -m pip install -r requirements.txt') from exc
from scipy.special import logsumexp
import argparse, sys, signal, math
from scipy.stats import truncnorm

# ======================== 只需修改此配置区 ========================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'SN1993j.csv'
OUTPUT_NPZ = DATA_FILE.with_name('SN1993J_density_energy_dynesty_120d.npz')
CALIBRATION_FILE = BASE_DIR / 'calibration' / 'photometry_calibration.npz'
DISTANCE_MPC = 3.63
REDSHIFT = -0.0001
EBV = 0.19
RV = 3.1
MAG_SYSTEM = 'vega'             # 'vega' 或 'ab'，所有观测应采用同一系统
KAPPA0 = 0.1                    # cm2/g，固定，不采样
DELTA = 1.0                     # 内层 BPL 指数，固定，不采样
INITIAL_ENERGY_PROFILE = 'density'  # 第一份默认 p=1 的对应分配：u∝rho
# 可设为 'legacy_inverse_radius' 做原始 u∝1/q 对照；改变后必须使用新输出。
TIME_COLUMN = 'MJD'
TIME_FORMAT = 'JD'              # 此 CSV 列名虽是 MJD，数值实际为 JD
REFERENCE_MJD = None            # None：第一个（时间最早）观测点为零；t_model = phase-t_shift
PHASE_MIN = None
PHASE_MAX = 120.0
# 选择参与拟合的波段：例如改为 ['B','V','R','I']。顺序同时用于输出。
INCLUDE_BANDS = ['V','R']
DATA_COLUMNS = {b:(b,b+'_err') for b in INCLUDE_BANDS}
_TRAPEZOID = getattr(np, 'trapezoid', None)
if _TRAPEZOID is None:
    _TRAPEZOID = np.trapz
if not CALIBRATION_FILE.is_file():
    raise FileNotFoundError(f'缺少通带校准文件：{CALIBRATION_FILE}')
CALIBRATION_SHA256 = hashlib.sha256(CALIBRATION_FILE.read_bytes()).hexdigest()
FILTER_CURVES = {}
with np.load(CALIBRATION_FILE, allow_pickle=False) as _cal:
    CALIBRATION_METADATA = json.loads(str(_cal['metadata_json'].item()))
    ALL_BANDS = CALIBRATION_METADATA['bands']
    for _b,_d in ALL_BANDS.items():
        _wave = np.asarray(_cal[_b+'_wavelength_A'],float)
        _response = np.asarray(_cal[_b+'_response'],float)
        _weight = _response/_wave
        _norm = _TRAPEZOID(_weight,_wave)
        if (_wave.ndim!=1 or _wave.size<2 or _response.shape!=_wave.shape
                or np.any(~np.isfinite(_wave)) or np.any(np.diff(_wave)<=0)
                or np.any(~np.isfinite(_response)) or np.any(_response<0)
                or not np.isfinite(_norm) or _norm<=0):
            raise ValueError(f'{_b} 的通带校准无效。')
        FILTER_CURVES[_b] = (_wave,_response,_weight,_norm)
def select_bands(include, definitions):
    if not include:
        raise ValueError('INCLUDE_BANDS 至少需要一个波段。')
    if len(include) != len(set(include)):
        raise ValueError('INCLUDE_BANDS 不允许重复波段。')
    missing = [b for b in include if b not in definitions]
    if missing:
        raise ValueError(f'请先在 ALL_BANDS 中定义波段：{missing}')
    return {b:dict(definitions[b]) for b in include}

BANDS = select_bands(INCLUDE_BANDS, ALL_BANDS)

NAMES = ['M_ej','E_K','E_Th_in','M_Ni','R_max_in','f_ni',
         'kappa_gamma','n_env','T_floor','t_shift']
UNITS = ['Msun','1e51 erg','1e49 erg','Msun','Rsun','1','cm2/g','1','K','day']
BOUNDS = np.array([
    [0.1,10.0], [0.1,10.0], [0.1,50.0],
    [0.01,1.0], [30.0,1000.0], [0.0,1.0], [0.01,0.5],
    [5.0,15.0], [3000.0,10000.0], [-5.0,0.0],
])
V_MIN, V_MAX = 0.1, 3.0         # 1e9 cm/s，源模型的默认速度先验域
# 默认保留源模型均匀速度先验；在能量坐标中加 Jacobian ∝(M*E)^(-1/2)。
# 改为 False 表示采用均匀能量先验，仍施加速度域约束。
KEEP_VELOCITY_PRIOR = True
NX, NY = 100, 3000
# dynesty 动态嵌套采样；不使用 emcee walkers、burn-in 或链步数。
NLIVE_INIT = 800
NLIVE_BATCH = 400
DLOGZ_INIT = 0.05
N_EFFECTIVE = 5000
BOUND = 'multi'
SAMPLE = 'rwalk'
WALKS = 50
CHECKPOINT_SECONDS = 60.0
MAX_CALLS = None               # None：按 dynesty 停止条件运行
MAX_ITERATIONS = None
MAX_BATCHES = None
SEED = 42
PRODUCTION_SEED = SEED + 100003  # 正式阶段使用独立随机数流和全新先验活点
# 标定阶段临时估计 sigma；正式阶段只采样 NAMES 中的原 10 个参数。
SIGMA_CALIBRATION_BOUNDS = (0.0, 1.0)  # mag，标定阶段的线性均匀先验
SIGMA_ROUNDING_MAG = 0.001
SIGMA_NLIVE_INIT = 400
SIGMA_NLIVE_BATCH = 200
SIGMA_DLOGZ_INIT = 0.1
SIGMA_N_EFFECTIVE = 2000
SIGMA_MAX_CALLS = None
SIGMA_MAX_ITERATIONS = None
SIGMA_MAX_BATCHES = None
CALIBRATION_SIGMA_MAG = 0.0  # 与第一份默认一致；非零时引入同波段共享校准协方差
# Richmond/Woosley 采用 1993-03-27.5 (MJD 49074.0) 为爆炸参考时刻。
# 0.5 d 是保守的外部时刻宽度；若采用其他文献值，只改这两项。
EXPLOSION_MJD_PRIOR = 49074.0
EXPLOSION_MJD_SIGMA_D = 0.5
# 误差结构与第一份一致：测量误差 + 全源共用 sigma_int。
# 不再加入旧的 phase/band 分箱 MODEL_ERROR_MAG；sigma 不参与正式 10 参数采样；只在单独的标定阶段临时估计。
# 1 = 串行；默认最多 8 个进程。建议分别测 4/8 进程。
N_PROCESSES = min(8, max(1, (os.cpu_count() or 1) - 1))
# dynesty 定期保存 checkpoint；--resume 恢复同一模型、数据、先验和固定 sigma；两个阶段分别保存检查点。
# Ctrl+C 后可恢复最近的库检查点；中断时不把未完成的权重当成正式后验导出。
GRID_CHECK_AT_END = True        # 最佳样本处加倍网格检查误差，结果写进NPZ

class _NonPhysicalModelOutput(ValueError):
    pass

# ================= BPL density, moving photosphere and diffusion =================

_BPL_V_MAX_OVER_V_T = 3.0

_BPL_DEFAULT_DELTA = 0.0

_BPL_DEFAULT_N = 10.0

_EXPONENTIAL_V_MAX_OVER_V_E = 12.0

_PROFILE_Q_MIN = 0.0001

_PROFILE_Q_MAX = 1.0

_BPL_Q_T = 1.0 / _BPL_V_MAX_OVER_V_T

_EXPONENTIAL_Q_E = 1.0 / _EXPONENTIAL_V_MAX_OVER_V_E

_PHOTOSPHERE_TAU = 2.0 / 3.0

_TIME_GRID_POWER = 2.0

PI: float = float(np.pi)

C_LIGHT: float = 29979245800.0

H_PLANCK: float = 6.62607015e-27

K_BOLTZ: float = 1.380649e-16

SIGMA_SB: float = 5.670316237574177e-05

SIGMA_T: float = 6.652e-25

M_P: float = 1.673e-24

M_E: float = 9.109e-28

Q_E: float = 4.803e-10

M_SUN: float = 1.9889999e+33

R_SUN: float = 69600000000.0

L_SUN: float = 3.828e+33

PC: float = 3.086e+18

MPC: float = 1000000.0 * PC

KM: float = 100000.0

DAY: float = 86400.0

YR: float = 365.0 * DAY

GHZ: float = 1000000000.0

UJY: float = 1e-29

MJY: float = 1e-26

JY: float = 1e-23

H0: float = 71.0 * KM / MPC

OMEGA_M: float = 0.27

OMEGA_L: float = 0.73

EPSILON_NI: float = 39000000000.0

EPSILON_CO: float = 6800000000.0

TAU_NI: float = 8.8 * DAY

TAU_CO: float = 111.3 * DAY

EPSILON_RATIO: float = EPSILON_CO / (EPSILON_NI - EPSILON_CO)

@dataclass(frozen=True)
class NickelTransportState:
    t_s: np.ndarray
    Lbol: np.ndarray
    Lphotospheric: np.ndarray
    Ldirect: np.ndarray
    q_ph: np.ndarray
    Rph: np.ndarray
    Tph: np.ndarray
    Rhom: np.ndarray
    photosphere_valid: np.ndarray
    density_profile: str
    energy_diagnostics: dict | None = None

def _gamma_deposition_fraction(t_phys, t_gamma):
    t_phys = np.asarray(t_phys, dtype=float)
    deposition = np.ones_like(t_phys)
    positive_time = t_phys > 0.0
    deposition[positive_time] = 1.0 - np.exp(-(float(t_gamma) / t_phys[positive_time]) ** 2)
    return deposition

def _radioactive_heating_shape(t_phys, t_gamma):
    t_phys = np.asarray(t_phys, dtype=float)
    e_co_ratio = EPSILON_CO / (EPSILON_NI - EPSILON_CO)
    radioactive = np.exp(-t_phys / TAU_NI) + e_co_ratio * np.exp(-t_phys / TAU_CO)
    return radioactive * _gamma_deposition_fraction(t_phys, t_gamma)

def _build_time_grid(y_max, Ny):
    y_max = float(y_max)
    Ny = int(Ny)
    if not np.isfinite(y_max) or y_max <= 0.0:
        raise ValueError('Nickel time-grid extent must be finite and positive.')
    if Ny < 1:
        raise ValueError('Nickel time grids require Ny >= 1.')
    fraction = np.linspace(0.0, 1.0, Ny + 1)
    return y_max * fraction ** _TIME_GRID_POWER

def _integral_power_law(x_lo, x_hi, power):
    if not (np.isfinite(x_lo) and np.isfinite(x_hi) and (0.0 < x_lo < x_hi)):
        raise ValueError('Power-law integration bounds must satisfy 0 < x_lo < x_hi.')
    exponent = float(power) + 1.0
    if abs(exponent) < 1e-12:
        return float(np.log(x_hi / x_lo))
    return float((x_hi ** exponent - x_lo ** exponent) / exponent)

@numba.njit(cache=_NUMBA_DISK_CACHE)
def _integral_power_law_scalar_numba(x_lo, x_hi, power):
    exponent = power + 1.0
    if abs(exponent) < 1e-12:
        return np.log(x_hi / x_lo)
    return (x_hi ** exponent - x_lo ** exponent) / exponent

@numba.njit(cache=_NUMBA_DISK_CACHE)
def _q_profile_moment_scalar_numba(q_min, q_max, order, density_profile, delta, n):
    if density_profile == 'uniform':
        return _integral_power_law_scalar_numba(q_min, q_max, order)
    if density_profile == 'exponential':
        q_e = _EXPONENTIAL_Q_E
        x_min = q_min / q_e
        x_max = q_max / q_e
        tail_min = np.exp(-x_min) * (x_min * x_min + 2.0 * x_min + 2.0)
        tail_max = np.exp(-x_max) * (x_max * x_max + 2.0 * x_max + 2.0)
        return q_e ** 3 * (tail_min - tail_max)
    q_t = _BPL_Q_T
    if q_max <= q_t:
        return q_t ** delta * _integral_power_law_scalar_numba(q_min, q_max, order - delta)
    if q_min >= q_t:
        return q_t ** n * _integral_power_law_scalar_numba(q_min, q_max, order - n)
    return q_t ** delta * _integral_power_law_scalar_numba(q_min, q_t, order - delta) + q_t ** n * _integral_power_law_scalar_numba(q_t, q_max, order - n)

def _exponential_tail_moment(x, order):
    x = float(x)
    order = int(order)
    if not np.isfinite(x) or x < 0.0:
        raise ValueError('Exponential-profile coordinates must be finite and non-negative.')
    if order < 0:
        raise ValueError('Exponential-profile moment order must be non-negative.')
    term = 1.0
    series = 1.0
    factorial = 1.0
    for k in range(1, order + 1):
        term *= x / float(k)
        series += term
        factorial *= float(k)
    return float(factorial * np.exp(-x) * series)

def _eta_q(q, density_profile, delta, n):
    q = np.asarray(q, dtype=float)
    if density_profile == 'uniform':
        return np.ones_like(q)
    if density_profile == 'exponential':
        return np.exp(-q / _EXPONENTIAL_Q_E)
    scaled = q / _BPL_Q_T
    return np.where(q < _BPL_Q_T, scaled ** (-float(delta)), scaled ** (-float(n)))

def _q_profile_moment(q_min, q_max, order, density_profile, delta, n):
    q_min = float(q_min)
    q_max = float(q_max)
    order = float(order)
    if not (np.isfinite(q_min) and np.isfinite(q_max) and (0.0 < q_min < q_max <= 1.0)):
        raise ValueError('Common profile bounds must satisfy 0 < q_min < q_max <= 1.')
    if density_profile == 'uniform':
        return _integral_power_law(q_min, q_max, order)
    if density_profile == 'exponential':
        q_e = _EXPONENTIAL_Q_E
        x_min = q_min / q_e
        x_max = q_max / q_e
        return float(q_e ** (order + 1.0) * (_exponential_tail_moment(x_min, int(order)) - _exponential_tail_moment(x_max, int(order))))
    if not (np.isfinite(delta) and 0.0 <= delta < 3.0):
        raise ValueError('delta must be finite and satisfy 0 <= delta < 3.')
    if not (np.isfinite(n) and n > 5.0):
        raise ValueError('n must be finite and > 5.')
    q_t = _BPL_Q_T
    if q_max <= q_t:
        return float(q_t ** delta * _integral_power_law(q_min, q_max, order - delta))
    if q_min >= q_t:
        return float(q_t ** n * _integral_power_law(q_min, q_max, order - n))
    return float(q_t ** delta * _integral_power_law(q_min, q_t, order - delta) + q_t ** n * _integral_power_law(q_t, q_max, order - n))

def _q_enclosed_profile_mass(q, q_min, density_profile, delta, n):
    q = np.asarray(q, dtype=float)
    q_min = float(q_min)
    if density_profile == 'uniform':
        return (q ** 3 - q_min ** 3) / 3.0
    if density_profile == 'exponential':
        q_e = _EXPONENTIAL_Q_E
        x_min = q_min / q_e
        x = q / q_e
        tail_min = np.exp(-x_min) * (x_min * x_min + 2.0 * x_min + 2.0)
        tail_x = np.exp(-x) * (x * x + 2.0 * x + 2.0)
        return q_e ** 3 * (tail_min - tail_x)
    q_t = _BPL_Q_T
    inner_exponent = 3.0 - float(delta)
    outer_exponent = 3.0 - float(n)
    inner_mass = q_t ** delta * (q_t ** inner_exponent - q_min ** inner_exponent) / inner_exponent
    enclosed_inner = q_t ** delta * (q ** inner_exponent - q_min ** inner_exponent) / inner_exponent
    enclosed_outer = inner_mass + q_t ** n * (q ** outer_exponent - q_t ** outer_exponent) / outer_exponent
    return np.where(q <= q_t, enclosed_inner, enclosed_outer)

@numba.njit(cache=_NUMBA_DISK_CACHE)
def _q_enclosed_profile_mass_scalar_numba(q, q_min, density_profile, delta, n):
    if density_profile == 'uniform':
        return (np.power(q, 3.0) - np.power(q_min, 3.0)) / 3.0
    if density_profile == 'exponential':
        q_e = _EXPONENTIAL_Q_E
        x_min = q_min / q_e
        x = q / q_e
        tail_min = np.exp(-x_min) * (x_min * x_min + 2.0 * x_min + 2.0)
        tail_x = np.exp(-x) * (x * x + 2.0 * x + 2.0)
        return q_e ** 3 * (tail_min - tail_x)
    q_t = _BPL_Q_T
    inner_exponent = 3.0 - delta
    outer_exponent = 3.0 - n
    inner_mass = q_t ** delta * (q_t ** inner_exponent - q_min ** inner_exponent) / inner_exponent
    enclosed_inner = q_t ** delta * (q ** inner_exponent - q_min ** inner_exponent) / inner_exponent
    enclosed_outer = inner_mass + q_t ** n * (q ** outer_exponent - q_t ** outer_exponent) / outer_exponent
    if q <= q_t:
        return enclosed_inner
    return enclosed_outer

def _q_mass_fraction_to_radius(f_mass, q_min, density_profile, delta, n):
    f_mass = float(f_mass)
    if not np.isfinite(f_mass) or not 0.0 <= f_mass <= 1.0:
        raise ValueError('Mass fraction must be finite and in [0, 1].')
    if f_mass <= 0.0:
        return float(q_min)
    if f_mass >= 1.0:
        return _PROFILE_Q_MAX
    total_mass = _q_profile_moment(q_min, _PROFILE_Q_MAX, 2.0, density_profile, delta, n)
    target_mass = f_mass * total_mass
    lo = float(q_min)
    hi = _PROFILE_Q_MAX
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        enclosed = float(_q_enclosed_profile_mass(mid, q_min, density_profile, delta, n))
        if enclosed < target_mass:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

def _q_photosphere_radius(expansion_factor, tau_scale, density_profile, delta, n, tau_target=_PHOTOSPHERE_TAU):
    expansion_factor = np.asarray(expansion_factor, dtype=float)
    if np.any(~np.isfinite(expansion_factor)) or np.any(expansion_factor <= 0.0):
        raise ValueError('Expansion factors must be finite and positive.')
    tau_scale = float(tau_scale)
    tau_target = float(tau_target)
    if not np.isfinite(tau_scale) or tau_scale <= 0.0:
        raise ValueError('Optical-depth scale must be finite and positive.')
    if not np.isfinite(tau_target) or tau_target <= 0.0:
        raise ValueError('Photosphere optical depth must be finite and positive.')
    column_target = tau_target * expansion_factor ** 2 / tau_scale
    if density_profile == 'uniform':
        q_ph = _PROFILE_Q_MAX - column_target
    elif density_profile == 'exponential':
        edge_term = np.exp(-_PROFILE_Q_MAX / _EXPONENTIAL_Q_E)
        argument = column_target / _EXPONENTIAL_Q_E + edge_term
        q_ph = -_EXPONENTIAL_Q_E * np.log(argument)
    else:
        q_t = _BPL_Q_T
        delta = float(delta)
        n = float(n)
        outer_column = q_t ** n * (q_t ** (1.0 - n) - _PROFILE_Q_MAX ** (1.0 - n)) / (n - 1.0)
        in_outer = column_target <= outer_column
        outer_power = _PROFILE_Q_MAX ** (1.0 - n) + (n - 1.0) * column_target / q_t ** n
        q_outer = outer_power ** (1.0 / (1.0 - n))
        inner_column = np.maximum(column_target - outer_column, 0.0)
        if abs(1.0 - delta) < 1e-12:
            q_inner = q_t * np.exp(-inner_column / q_t)
        else:
            inner_power = q_t ** (1.0 - delta) - (1.0 - delta) * inner_column / q_t ** delta
            inner_power = np.maximum(inner_power, np.finfo(float).tiny)
            q_inner = inner_power ** (1.0 / (1.0 - delta))
        q_ph = np.where(in_outer, q_outer, q_inner)
    return np.clip(q_ph, _PROFILE_Q_MIN, _PROFILE_Q_MAX)

def _finite_profile_scales(M_ej, E_K, R_max_in, x_max, i_mass, i_kin):
    values = (M_ej, E_K, R_max_in, x_max, i_mass, i_kin)
    if any((not np.isfinite(value) or value <= 0.0 for value in values)):
        raise ValueError('Finite ejecta mass, energy, radius, and profile integrals must be positive.')
    R_scale = float(R_max_in) / float(x_max)
    rho_scale = float(M_ej) / (4.0 * PI * float(i_mass) * R_scale ** 3)
    v_scale = np.sqrt(2.0 * float(i_mass) * float(E_K) / (float(i_kin) * float(M_ej)))
    return (float(rho_scale), float(v_scale), float(R_scale))

@numba.njit(cache=_NUMBA_DISK_CACHE)
def _photospheric_cell_geometry_numba(q_faces, q_heat, xi0, q_photosphere, density_profile, delta, n):
    n_cells = q_faces.size - 1
    n_times = q_photosphere.size
    q_min = q_faces[0]
    thermal_upper = np.zeros((n_times, n_cells), dtype=np.float64)
    thermal_volume = np.zeros((n_times, n_cells), dtype=np.float64)
    density = np.ones((n_times, n_cells), dtype=np.float64)
    centres = np.zeros((n_times, n_cells), dtype=np.float64)
    source_density = np.zeros((n_times, n_cells), dtype=np.float64)
    direct_fraction = np.zeros(n_times, dtype=np.float64)
    active_count = np.zeros(n_times, dtype=np.int64)
    enclosed_lower = np.zeros(n_cells, dtype=np.float64)
    full_centres = np.zeros(n_cells, dtype=np.float64)
    for cell in range(n_cells):
        lower = q_faces[cell]
        upper = q_faces[cell + 1]
        enclosed_lower[cell] = _q_enclosed_profile_mass_scalar_numba(lower, q_min, density_profile, delta, n)
        full_centres[cell] = 0.5 * (lower + upper)
    # 同一次求解中完整单元几何和镍源项不随时间变化。
    full_volume = np.empty(n_cells)
    full_density = np.empty(n_cells)
    full_mass_centres = np.empty(n_cells)
    full_source = np.zeros(n_cells)
    for cell in range(n_cells):
        lo = q_faces[cell]
        hi = q_faces[cell + 1]
        vol = (np.power(hi, 3.0) - np.power(lo, 3.0)) / 3.0
        full_volume[cell] = vol
        full_density[cell] = (_q_enclosed_profile_mass_scalar_numba(
            hi, q_min, density_profile, delta, n) - enclosed_lower[cell]) / vol
        full_mass_centres[cell] = 0.75 * (np.power(hi, 4.0) - np.power(lo, 4.0)) / (np.power(hi, 3.0) - np.power(lo, 3.0))
        source_upper = min(hi, q_heat)
        if source_upper > lo:
            source_mass = _q_enclosed_profile_mass_scalar_numba(source_upper, q_min, density_profile, delta, n) - enclosed_lower[cell]
            full_source[cell] = float(xi0) * source_mass / vol
    total_source = 0.0
    if float(xi0) > 0.0 and float(q_heat) > q_min:
        total_source = float(xi0) * _q_profile_moment_scalar_numba(q_min, float(q_heat), 2.0, density_profile, delta, n)
    for time_index in range(n_times):
        q_ph = q_photosphere[time_index]
        count = 0
        thermal_source = 0.0
        for cell in range(n_cells):
            lower = q_faces[cell]
            upper = q_faces[cell + 1]
            cut_upper = min(upper, q_ph)
            thermal_upper[time_index, cell] = cut_upper
            centres[time_index, cell] = full_centres[cell]
            if cut_upper <= lower:
                continue
            count += 1
            if cut_upper == upper:
                volume = full_volume[cell]
                thermal_volume[time_index, cell] = volume
                density[time_index, cell] = full_density[cell]
                centres[time_index, cell] = full_mass_centres[cell]
                source_density[time_index, cell] = full_source[cell]
                thermal_source += full_source[cell] * volume
            else:
                # 只有光球边界截断的单元需要随时间重新积分。
                volume = (np.power(cut_upper, 3.0) - np.power(lower, 3.0)) / 3.0
                thermal_volume[time_index, cell] = volume
                thermal_mass = _q_enclosed_profile_mass_scalar_numba(cut_upper, q_min, density_profile, delta, n) - enclosed_lower[cell]
                density[time_index, cell] = thermal_mass / volume
                centres[time_index, cell] = 0.75 * (np.power(cut_upper, 4.0) - np.power(lower, 4.0)) / (np.power(cut_upper, 3.0) - np.power(lower, 3.0))
                source_upper = min(cut_upper, q_heat)
                if source_upper > lower:
                    source_mass = _q_enclosed_profile_mass_scalar_numba(source_upper, q_min, density_profile, delta, n) - enclosed_lower[cell]
                    source_density[time_index, cell] = float(xi0) * source_mass / volume
                    thermal_source += source_density[time_index, cell] * volume
        active_count[time_index] = count
        if total_source > 0.0:
            direct_fraction[time_index] = min(max(1.0 - thermal_source / total_source, 0.0), 1.0)
    return (thermal_upper, thermal_volume, density, centres, source_density, direct_fraction, active_count)

def _photospheric_cell_geometry(q_faces, q_heat, xi0, q_photosphere, density_profile, delta, n):
    q_faces = np.ascontiguousarray(np.asarray(q_faces, dtype=float))
    q_photosphere = np.ascontiguousarray(np.asarray(q_photosphere, dtype=float))
    return _photospheric_cell_geometry_numba(q_faces, q_heat, xi0, q_photosphere, density_profile, delta, n)

@numba.njit(cache=_NUMBA_DISK_CACHE)
def _photospheric_transport_coefficients_numba(q_faces, q_photosphere, expansion_factor, tau_scale, thermal_volume, density, centres, boundary_density, active_count, dy_values):
    n_times, n_cells = thermal_volume.shape
    lower = np.zeros_like(thermal_volume)
    upper = np.zeros_like(thermal_volume)
    boundary = np.zeros_like(thermal_volume)
    luminosity_transport = np.zeros(n_times)
    for time_index in range(n_times):
        dy_step = dy_values[time_index]
        count = int(active_count[time_index])
        if count > 1:
            for cell in range(count - 1):
                spacing = centres[time_index, cell + 1] - centres[time_index, cell]
                inv_density_face = 2.0 / (density[time_index, cell] + density[time_index, cell + 1])
                face_transport = q_faces[cell + 1] ** 2 * inv_density_face / spacing
                upper[time_index, cell] = dy_step * face_transport / thermal_volume[time_index, cell]
                lower[time_index, cell + 1] = dy_step * face_transport / thermal_volume[time_index, cell + 1]
        if count > 0:
            boundary_distance = max(q_photosphere[time_index] - centres[time_index, count - 1], 0.0)
            boundary_alpha = 4.0 * expansion_factor[time_index] ** 2 / (3.0 * tau_scale * boundary_density[time_index])
            transport = q_photosphere[time_index] ** 2 / (boundary_density[time_index] * (boundary_distance + boundary_alpha))
            boundary[time_index, count - 1] = dy_step * transport / thermal_volume[time_index, count - 1]
            luminosity_transport[time_index] = transport
    return (lower, upper, boundary, luminosity_transport)

def _photospheric_transport_coefficients(q_faces, q_photosphere, expansion_factor, tau_scale, thermal_volume, density, centres, boundary_density, active_count, dy):
    q_faces = np.asarray(q_faces, dtype=float)
    q_photosphere = np.asarray(q_photosphere, dtype=float)
    expansion_factor = np.asarray(expansion_factor, dtype=float)
    thermal_volume = np.asarray(thermal_volume, dtype=float)
    density = np.asarray(density, dtype=float)
    centres = np.asarray(centres, dtype=float)
    boundary_density = np.asarray(boundary_density, dtype=float)
    active_count = np.asarray(active_count, dtype=np.int64)
    n_times, n_cells = thermal_volume.shape
    dy_values = np.asarray(dy, dtype=float)
    if dy_values.ndim == 0:
        dy_values = np.full(n_times, float(dy_values), dtype=float)
    elif dy_values.shape != (n_times,):
        raise ValueError('dy must be a scalar or contain one step size per time-grid node.')
    if np.any(~np.isfinite(dy_values)) or np.any(dy_values <= 0.0):
        raise ValueError('Nickel time steps must be finite and positive.')
    return _photospheric_transport_coefficients_numba(q_faces, q_photosphere, expansion_factor, float(tau_scale), thermal_volume, density, centres, boundary_density, active_count, np.ascontiguousarray(dy_values))

@numba.njit(fastmath=True, cache=_NUMBA_DISK_CACHE)
def thomas_algorithm(a, b, c_up, d, c_prime, d_prime, x_out):
    n = len(d)
    c_prime[0] = c_up[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c_up[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom
    x_out[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x_out[i] = d_prime[i] - c_prime[i] * x_out[i + 1]

@numba.njit(fastmath=True, cache=_NUMBA_DISK_CACHE)
def _fast_time_loop_photosphere_kernel(Ny, n_cells, dy_steps, fR_vals, heat_vals, source_vals, volume_vals, upper_coeff_vals, lower_coeff_vals, boundary_coeff_vals, active_count_vals, e_initial, luminosity_coeff_vals, L0):
    e_now = e_initial.copy()
    e_next = np.empty_like(e_now)
    L_out = np.zeros(Ny)
    a = np.zeros(n_cells)
    b_diag = np.zeros(n_cells)
    c_up = np.zeros(n_cells)
    rhs = np.zeros(n_cells)
    c_prime = np.zeros(n_cells)
    d_prime = np.zeros(n_cells)
    for time_index in range(Ny):
        dy_step = dy_steps[time_index]
        fR_next = fR_vals[time_index + 1]
        active_next = active_count_vals[time_index + 1]
        swept_energy = 0.0
        for cell in range(n_cells):
            volume_lost = volume_vals[time_index, cell] - volume_vals[time_index + 1, cell]
            if volume_lost > 0.0:
                swept_energy += max(e_now[cell], 0.0) * volume_lost
            if cell < active_next:
                lower = lower_coeff_vals[time_index + 1, cell]
                upper = upper_coeff_vals[time_index + 1, cell]
                boundary = boundary_coeff_vals[time_index + 1, cell]
                a[cell] = -fR_next * lower
                c_up[cell] = -fR_next * upper
                b_diag[cell] = 1.0 + fR_next * (lower + upper + boundary)
                rhs[cell] = e_now[cell] + dy_step * fR_next * heat_vals[time_index + 1] * source_vals[time_index + 1, cell]
            else:
                a[cell] = 0.0
                b_diag[cell] = 1.0
                c_up[cell] = 0.0
                rhs[cell] = 0.0
        thomas_algorithm(a, b_diag, c_up, rhs, c_prime, d_prime, e_next)
        boundary_luminosity = 0.0
        if active_next > 0:
            boundary_luminosity = luminosity_coeff_vals[time_index + 1] * e_next[active_next - 1]
        sweep_luminosity = L0 * swept_energy / (fR_next * dy_step)
        if active_next > 0:
            L_out[time_index] = boundary_luminosity + sweep_luminosity
        else:
            L_out[time_index] = 0.0
            if time_index > 0:
                L_out[time_index - 1] += sweep_luminosity
        e_now, e_next = (e_next, e_now)
    return L_out

def _fast_time_loop_photosphere_numba(Ny, n_cells, dy, fR_vals, heat_vals, source_vals, volume_vals, upper_coeff_vals, lower_coeff_vals, boundary_coeff_vals, active_count_vals, e_initial, luminosity_coeff_vals, L0):
    dy_steps = np.asarray(dy, dtype=float)
    if dy_steps.ndim == 0:
        dy_steps = np.full(int(Ny), float(dy_steps), dtype=float)
    elif dy_steps.shape != (int(Ny),):
        raise ValueError('dy must be a scalar or contain exactly Ny step sizes.')
    if np.any(~np.isfinite(dy_steps)) or np.any(dy_steps <= 0.0):
        raise ValueError('Nickel time steps must be finite and positive.')
    return _fast_time_loop_photosphere_kernel(Ny, n_cells, np.ascontiguousarray(dy_steps), fR_vals, heat_vals, source_vals, volume_vals, upper_coeff_vals, lower_coeff_vals, boundary_coeff_vals, active_count_vals, e_initial, luminosity_coeff_vals, L0)


@numba.njit(cache=_NUMBA_DISK_CACHE)
def _full_domain_kernel(dt_y, expansion, heat, source, volume, face_g, boundary_g, initial, L0, energy_scale, tdiff):
    # Backward Euler on a fixed finite-volume domain; no sweep or direct term.
    nx=initial.size; nt=dt_y.size
    e=initial.copy(); luminosity=np.empty(nt); energy=np.empty(nt+1)
    energy[0]=energy_scale*np.sum(e*volume)/expansion[0]
    cp=np.empty(nx); dp=np.empty(nx); en=np.empty(nx)
    for k in range(nt):
        factor=dt_y[k]*expansion[k+1]
        for i in range(nx):
            left=0.0 if i==0 else face_g[i-1]/volume[i]
            right=0.0 if i==nx-1 else face_g[i]/volume[i]
            edge=boundary_g[k+1]/volume[i] if i==nx-1 else 0.0
            low=-factor*left; up=-factor*right
            diag=1.0+factor*(left+right+edge)
            rhs=e[i]+factor*heat[k+1]*source[i]
            den=diag if i==0 else diag-low*cp[i-1]
            cp[i]=up/den
            dp[i]=(rhs if i==0 else rhs-low*dp[i-1])/den
        en[-1]=dp[-1]
        for i in range(nx-2,-1,-1): en[i]=dp[i]-cp[i]*en[i+1]
        e,en=en,e
        luminosity[k]=L0*boundary_g[k+1]*e[-1]
        energy[k+1]=energy_scale*np.sum(e*volume)/expansion[k+1]
    return luminosity,energy

def _initial_radiation_energy(q_faces, density, volume, total_energy_erg, energy_scale):
    """Normalize shell-average radiation energy density to whole-ejecta energy.

    density mode is the first supplied model's default p=1 analogue on this
    model's existing BPL density. It does not make all other physics identical.
    """
    if INITIAL_ENERGY_PROFILE == 'density':
        shape = np.asarray(density, float)
    elif INITIAL_ENERGY_PROFILE == 'legacy_inverse_radius':
        shape = .5*np.diff(np.asarray(q_faces, float)**2)/volume
    else:
        raise ValueError("INITIAL_ENERGY_PROFILE 必须为 'density' 或 'legacy_inverse_radius'。")
    norm = float(np.sum(shape*volume))
    if not np.isfinite(norm) or norm <= 0. or np.any(~np.isfinite(shape)) or np.any(shape < 0.):
        raise ValueError('初始内能分布必须有限、非负且具有正积分。')
    return (float(total_energy_erg)/float(energy_scale)) * (shape/norm)


def calculate_transport(theta, Nx=100, Ny=1000, t_max_days=150.0, density_profile='broken_power_law') -> NickelTransportState:
    pi, c, day = (PI, C_LIGHT, DAY)
    eNi, eCo = (EPSILON_NI, EPSILON_CO)
    M_ej, v_ej, E_Th_in, M_ni, R_max_in, f_ni, kappa0, kappa_gamma, _T_floor, delta, n = theta
    density_profile = 'broken_power_law'
    M_ej = float(M_ej) * M_SUN
    E_Th_in = float(E_Th_in) * 1e+49
    M_ni = float(M_ni) * M_SUN
    R_max_in = float(R_max_in) * R_SUN
    f_ni = float(f_ni)
    if not np.isfinite(f_ni) or not 0.0 <= f_ni <= 1.0:
        raise ValueError('f_ni must be finite and in [0, 1].')
    mixed_mass_limit = f_ni * float(M_ej)
    tolerance = 1e-12 * max(abs(float(M_ni)), abs(mixed_mass_limit), 1.0)
    if float(M_ni) > mixed_mass_limit + tolerance:
        raise ValueError('Nickel mass-coordinate mixing requires M_ni <= f_ni*M_ej (equivalently f_ni >= M_ni/M_ej).')
    kappa0 = float(kappa0)
    kappa_g = float(kappa_gamma)
    v_ej = float(v_ej) * 1000000000.0
    delta = float(delta)
    n = float(n)
    q_min = _PROFILE_Q_MIN
    q_max = _PROFILE_Q_MAX
    I_M = _q_profile_moment(q_min, q_max, 2.0, density_profile, delta, n)
    I_K = _q_profile_moment(q_min, q_max, 4.0, density_profile, delta, n)
    E_K = 0.5 * M_ej * v_ej * v_ej
    rho_scale, v_scale, R_scale = _finite_profile_scales(M_ej, E_K, R_max_in, q_max, I_M, I_K)
    t_ex = R_scale / v_scale
    t_diff = 3.0 * kappa0 * rho_scale * R_scale ** 2 / c
    t_gamma = np.sqrt(3.0 * kappa_g * M_ej / (4.0 * pi * v_ej * v_ej))
    u0 = rho_scale * (eNi - eCo) * t_diff
    L0 = 4.0 * pi * R_scale * c * u0 / (3.0 * kappa0 * rho_scale)
    tau_scale = kappa0 * rho_scale * R_scale
    q_heat = _q_mass_fraction_to_radius(f_ni, q_min, density_profile, delta, n)
    if q_heat <= q_min + 1e-14:
        xi0 = 0.0
    else:
        denom_heat = _q_profile_moment(q_min, q_heat, 2.0, density_profile, delta, n)
        xi0 = I_M * (M_ni / M_ej) / denom_heat
    xi0 = max(xi0, 0.0)
    Nx, Ny = (int(Nx), int(Ny))
    if Nx < 2 or Ny < 1:
        raise ValueError('NickelModel requires Nx >= 2 and Ny >= 1.')
    q_faces = np.linspace(q_min, q_max, Nx + 1)
    t_max = float(t_max_days) * day
    y_max = t_max / t_diff
    y_vals = _build_time_grid(y_max, Ny)
    dy_steps = np.diff(y_vals)
    dy_by_node = np.empty_like(y_vals)
    dy_by_node[0] = dy_steps[0]
    dy_by_node[1:] = dy_steps
    fR_vals = 1.0 + y_vals * t_diff / t_ex
    t_phys = y_vals * t_diff
    heat = _radioactive_heating_shape(t_phys, t_gamma)
    deposited_heating = M_ni * (eNi - eCo) * heat[1:]
    q_photosphere = _q_photosphere_radius(fR_vals, tau_scale, density_profile, delta, n)
    # Eth is the TOTAL initial radiation energy over q_min <= q <= 1.
    geometry = _photospheric_cell_geometry(q_faces, q_heat, xi0, np.array([q_max]), density_profile, delta, n)
    volume, density, centres, source = (geometry[j][0] for j in (1,2,3,4))
    energy_scale = 4.0*pi*R_scale**3*u0
    e_initial = _initial_radiation_energy(q_faces, density, volume, E_Th_in, energy_scale)
    face_g = q_faces[1:-1]**2 * 2.0/(density[:-1]+density[1:]) / np.diff(centres)
    eta_edge = float(_eta_q(np.array([q_max]), density_profile, delta, n)[0])
    # Eddington vacuum boundary at tau=0: F=c*u/2 (paper outer surface).
    alpha = 2.0*fR_vals**2/(3.0*tau_scale*eta_edge)
    boundary_g = q_max**2/(eta_edge*(q_max-centres[-1]+alpha))
    L_bol, stored = _full_domain_kernel(dy_steps, fR_vals, heat, source, volume,
        face_g, boundary_g, e_initial, L0, energy_scale, t_diff)
    L_photospheric = L_bol.copy()  # Legacy field: escape luminosity used by valid-photosphere SED.
    L_direct = np.zeros(Ny)
    q_ph = q_photosphere[1:]
    photosphere_valid = q_ph > q_min
    R_hom = R_scale*fR_vals[1:]
    R_ph = np.where(photosphere_valid, R_hom*q_ph, np.nan)
    T_ph = np.where(photosphere_valid, (L_bol/(4*pi*SIGMA_SB*R_ph**2))**0.25, np.nan)
    dt = np.diff(t_phys)
    deposited_discrete = L0*heat[1:]*np.sum(source*volume)
    radiated_step = dt*L_bol
    deposited_step = dt*deposited_discrete
    # Exact discrete PdV ledger for the backward-Euler e update, not trapezoidal quadrature.
    pdv_step = stored[:-1]*np.diff(fR_vals)/fR_vals[1:]
    residual_step = np.diff(stored)-deposited_step+radiated_step+pdv_step
    weighted_residual = np.diff(fR_vals*stored)-dt*fR_vals[1:]*(deposited_discrete-L_bol)
    radioactive_power = M_ni*((eNi-eCo)*np.exp(-t_phys[1:]/TAU_NI)+eCo*np.exp(-t_phys[1:]/TAU_CO))
    diagnostics = dict(time_s=t_phys, expansion=fR_vals, stored_energy_erg=stored,
        radioactive_luminosity=radioactive_power, gamma_escape_luminosity=radioactive_power-deposited_heating,
        initial_requested_erg=np.array(E_Th_in), initial_integrated_erg=np.array(stored[0]),
        deposited_luminosity=deposited_discrete, deposited_analytic_luminosity=deposited_heating,
        escape_luminosity=L_bol, deposited_step_erg=deposited_step,
        radiated_step_erg=radiated_step, pdv_step_erg=pdv_step,
        deposited_cumulative_erg=np.r_[0.0,np.cumsum(deposited_step)],
        radiated_cumulative_erg=np.r_[0.0,np.cumsum(radiated_step)],
        pdv_cumulative_erg=np.r_[0.0,np.cumsum(pdv_step)],
        residual_step_erg=residual_step, weighted_residual_step_erg=weighted_residual,
        residual_cumulative_erg=stored-stored[0]-np.r_[0.0,np.cumsum(deposited_step-radiated_step-pdv_step)],
        # Piecewise-constant finite-volume initial state, including the cut cell.
        initial_exterior_energy_erg=np.array(energy_scale*np.sum(e_initial *
            np.maximum(q_faces[1:]**3-np.maximum(q_faces[:-1],q_photosphere[0])**3,0.)/3.)),
        boundary_alpha=alpha, initial_cell_energy_erg=energy_scale*e_initial*volume)
    return NickelTransportState(t_phys[1:], L_bol, L_photospheric, L_direct, q_ph,
        R_ph, T_ph, R_hom, photosphere_valid, density_profile, diagnostics)

def _nickel_two_component_blackbody(transport, T_floor: float):
    floor = float(T_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError('T_floor must be finite and > 0 for photospheric emission.')
    luminosity = np.asarray(transport.Lbol, float)
    photospheric_luminosity = np.asarray(transport.Lphotospheric, float)
    direct_luminosity = np.asarray(transport.Ldirect, float)
    radius_physical = np.asarray(transport.Rph, float)
    photosphere_valid = np.asarray(transport.photosphere_valid, bool)
    if not luminosity.shape == photospheric_luminosity.shape == direct_luminosity.shape == radius_physical.shape == photosphere_valid.shape:
        raise _NonPhysicalModelOutput('Nickel Lbol, Lphotospheric, Ldirect, Rph, and photosphere_valid must have the same shape.')
    if np.any(~np.isfinite(photospheric_luminosity)) or np.any(photospheric_luminosity < 0.0) or np.any(~np.isfinite(direct_luminosity)) or np.any(direct_luminosity < 0.0):
        raise _NonPhysicalModelOutput('Nickel Lphotospheric and Ldirect must be finite and non-negative.')
    if not np.allclose(luminosity, photospheric_luminosity + direct_luminosity, rtol=5e-12, atol=0.0):
        raise _NonPhysicalModelOutput('Nickel transport must satisfy Lbol=Lphotospheric+Ldirect.')
    photo_active = photosphere_valid & (photospheric_luminosity > 0.0)
    invalid_photo_radius = ~np.isfinite(radius_physical) | (radius_physical <= 0.0)
    if np.any(photo_active & invalid_photo_radius):
        raise _NonPhysicalModelOutput('A luminous Nickel photospheric component requires finite positive Rph.')
    temperature_photo = np.full(luminosity.shape, floor, dtype=float)
    radius_photo = np.zeros(luminosity.shape, dtype=float)
    if np.any(photo_active):
        temperature_effective = (photospheric_luminosity[photo_active] / (4.0 * PI * SIGMA_SB * radius_physical[photo_active] ** 2)) ** 0.25
        temperature_photo[photo_active] = np.maximum(temperature_effective, floor)
        radius_photo[photo_active] = np.sqrt(photospheric_luminosity[photo_active] / (4.0 * PI * SIGMA_SB * temperature_photo[photo_active] ** 4))
    temperature_direct = np.full(luminosity.shape, floor, dtype=float)
    radius_direct = np.sqrt(direct_luminosity / (4.0 * PI * SIGMA_SB * floor ** 4))
    temperature_photo[~photosphere_valid] = np.nan
    radius_photo[~photosphere_valid] = np.nan
    return (temperature_photo, radius_photo, temperature_direct, radius_direct)

@dataclass(frozen=True)
class BlackbodySED:
    Tmin: float = 1.0
    Rmin: float = 1.0

    def planck_Bnu(self, nu_hz: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        nu = np.asarray(nu_hz, dtype=float).reshape(-1, 1)
        T = np.asarray(T_K, dtype=float).reshape(1, -1)
        T = np.clip(T, self.Tmin, np.inf)
        x = H_PLANCK * nu / (K_BOLTZ * T)
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            denom = np.expm1(x)
            denom = np.where(denom == 0.0, np.inf, denom)
            Bnu = 2.0 * H_PLANCK * nu ** 3 / C_LIGHT ** 2 / denom
        return np.where(np.isfinite(Bnu), Bnu, np.nan)

    def lnu(self, nu_rest_hz: np.ndarray, Teff_K: np.ndarray, R_cm: np.ndarray) -> np.ndarray:
        R = np.asarray(R_cm, dtype=float).reshape(1, -1)
        R = np.clip(R, self.Rmin, np.inf)
        Bnu = self.planck_Bnu(nu_rest_hz, Teff_K)
        Lnu = 4.0 * PI * PI * R ** 2 * Bnu
        return np.where(np.isfinite(Lnu), Lnu, np.nan)

    def fnu(self, nu_obs_hz: np.ndarray, Teff_K: np.ndarray, R_cm: np.ndarray, DL_cm: float, z: float=0.0) -> np.ndarray:
        zp1 = 1.0 + (z or 0.0)
        nu_rest = np.asarray(nu_obs_hz, dtype=float) * zp1
        Lnu = self.lnu(nu_rest, Teff_K, R_cm)
        Fnu = zp1 * Lnu / (4.0 * PI * DL_cm ** 2)
        return np.where(Fnu > 0.0, Fnu, np.nan)

    def abmag(self, nu_obs_hz: np.ndarray, Teff_K: np.ndarray, R_cm: np.ndarray, DL_cm: float, z: float=0.0) -> np.ndarray:
        Fnu = self.fnu(nu_obs_hz, Teff_K, R_cm, DL_cm=DL_cm, z=z)
        mab = -2.5 * np.log10(Fnu) - 48.6
        return mab

def _as_lambda_um_array(lambda_um: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(lambda_um, float)
    if arr.size == 0:
        raise ValueError('lambda_um must be non-empty.')
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError('lambda_um must contain positive finite values.')
    x = 1.0 / arr
    if np.any((x < 0.3) | (x > 10.0)):
        raise ValueError('Extinction laws currently support wavelengths in the range 0.1-3.33 micron (x = 0.3-10 micron^-1).')
    return arr

def _ccm89_ab(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros_like(x, dtype=float)
    b = np.zeros_like(x, dtype=float)
    ir = (x >= 0.3) & (x < 1.1)
    if np.any(ir):
        xx = x[ir]
        a[ir] = 0.574 * xx ** 1.61
        b[ir] = -0.527 * xx ** 1.61
    opt = (x >= 1.1) & (x < 3.3)
    if np.any(opt):
        y = x[opt] - 1.82
        a[opt] = 1.0 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.7753 * y ** 6 + 0.32999 * y ** 7
        b[opt] = 1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.3026 * y ** 6 - 2.09002 * y ** 7
    uv = (x >= 3.3) & (x <= 8.0)
    if np.any(uv):
        xx = x[uv]
        fa = np.zeros_like(xx, dtype=float)
        fb = np.zeros_like(xx, dtype=float)
        mask = xx >= 5.9
        if np.any(mask):
            y = xx[mask] - 5.9
            fa[mask] = -0.04473 * y ** 2 - 0.009779 * y ** 3
            fb[mask] = 0.213 * y ** 2 + 0.1207 * y ** 3
        a[uv] = 1.752 - 0.316 * xx - 0.104 / ((xx - 4.67) ** 2 + 0.341) + fa
        b[uv] = -3.09 + 1.825 * xx + 1.206 / ((xx - 4.62) ** 2 + 0.263) + fb
    fuv = (x > 8.0) & (x <= 10.0)
    if np.any(fuv):
        y = x[fuv] - 8.0
        a[fuv] = -1.073 - 0.628 * y + 0.137 * y ** 2 - 0.07 * y ** 3
        b[fuv] = 13.67 + 4.257 * y - 0.42 * y ** 2 + 0.374 * y ** 3
    return (a, b)

# ======================== 多波段观测和似然 ========================
def model_vector(theta):
    m,ek,eth,mni,r0,fni,kg,n,tfloor,shift = theta
    velocity = np.sqrt(2*ek*1e51/(m*M_SUN))/1e9
    return (m,velocity,eth,mni,r0,fni,KAPPA0,kg,tfloor,DELTA,n)

def solve_model(theta, tmax_observer, nx=NX, ny=NY):
    return calculate_transport(model_vector(theta), Nx=nx, Ny=ny,
        t_max_days=tmax_observer/(1+REDSHIFT), density_profile='broken_power_law')

def magnitude_grid(state, tfloor):
    tph,rph,tdirect,rdirect = _nickel_two_component_blackbody(state,tfloor)
    sed = BlackbodySED()
    distance = DISTANCE_MPC*MPC
    out = np.full((len(BANDS),state.t_s.size),np.nan)
    for ib,(band,definition) in enumerate(BANDS.items()):
        wave,response,weight,norm = FILTER_CURVES[band]
        nu = C_LIGHT/(wave*1e-8)
        attenuation = _dust_attenuation(wave)
        fph = sed.fnu(nu,tph,rph,DL_cm=distance,z=REDSHIFT)
        fdirect = sed.fnu(nu,tdirect,rdirect,DL_cm=distance,z=REDSHIFT)
        fph[:,rph<=0] = 0.; fdirect[:,rdirect<=0] = 0.
        mean = _TRAPEZOID((fph+fdirect)*attenuation[:,None]*weight[:,None],wave,axis=0)/norm
        zero = definition['zero_jy']*JY if MAG_SYSTEM=='vega' else 3631.*JY
        with np.errstate(divide='ignore',invalid='ignore'):
            out[ib] = -2.5*np.log10(mean/zero)
    return out

def _dust_attenuation(wave_A):
    x = 1.0/(np.asarray(wave_A,float)*1e-4)
    a,b = _ccm89_ab(x)
    return 10.0**(-0.4*EBV*(RV*a+b))

def validate_noise_settings():
    bounds = np.asarray(SIGMA_CALIBRATION_BOUNDS,float)
    if bounds.shape!=(2,) or not np.isfinite(bounds).all() or not 0<=bounds[0]<bounds[1]:
        raise ValueError('SIGMA_CALIBRATION_BOUNDS 必须有限、非负且递增。')
    if not np.isfinite(SIGMA_ROUNDING_MAG) or SIGMA_ROUNDING_MAG<=0:
        raise ValueError('SIGMA_ROUNDING_MAG 必须有限且为正。')
    if not np.isfinite(CALIBRATION_SIGMA_MAG) or CALIBRATION_SIGMA_MAG<0:
        raise ValueError('CALIBRATION_SIGMA_MAG 必须有限且非负。')

def gaussian_calibration_likelihood(observed, predicted, error, groups, calibration_sigma, sigma_int=0.):
    """Exact rank-one shared calibration marginalization (Woodbury identity)."""
    if not np.isfinite([calibration_sigma, sigma_int]).all() or min(calibration_sigma, sigma_int) < 0:
        raise ValueError('Invalid noise scales')
    residual = np.asarray(observed) - np.asarray(predicted)
    variance = np.asarray(error)**2 + sigma_int**2
    if np.any(~np.isfinite(residual)) or np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError('All fixed likelihood rows must be finite')
    inv = 1/variance
    value = float(np.sum(residual**2 * inv + np.log(2*np.pi*variance)))
    s2 = calibration_sigma**2
    for g in np.unique(groups):
        use = groups == g
        weight = float(inv[use].sum())
        projection = float(np.sum(residual[use]*inv[use]))
        value += math.log1p(s2*weight) - s2*projection**2/(1+s2*weight)
    return -.5*value

class FitProblem:
    def __init__(self):
        if MAG_SYSTEM not in ('vega','ab'):
            raise ValueError('MAG_SYSTEM 必须为 vega 或 ab。')
        if not np.isfinite(REDSHIFT) or REDSHIFT<=-1:
            raise ValueError('REDSHIFT 必须有限且大于 -1。')
        if not np.isfinite(DISTANCE_MPC) or DISTANCE_MPC<=0:
            raise ValueError('DISTANCE_MPC 必须有限且为正。')
        if not np.isfinite(EBV) or EBV<0 or not np.isfinite(RV) or RV<=0:
            raise ValueError('要求 EBV>=0、RV>0。')
        validate_noise_settings()
        if BOUNDS.shape!=(10,2) or not np.all(np.isfinite(BOUNDS)) or np.any(BOUNDS[:,1]<=BOUNDS[:,0]):
            raise ValueError('BOUNDS 必须是10行有限递增上下界。')
        if NX<2 or NY<1:
            raise ValueError('要求 NX>=2、NY>=1。')
        self.lower=BOUNDS[:,0]
        self.width=BOUNDS[:,1]-BOUNDS[:,0]
        raw=np.atleast_1d(np.genfromtxt(DATA_FILE,delimiter=',',names=True,dtype=float,encoding='utf-8'))
        mjd=raw[TIME_COLUMN].copy()
        if TIME_FORMAT.upper()=='JD':mjd-=2400000.5
        elif TIME_FORMAT.upper()!='MJD':raise ValueError('TIME_FORMAT 必须为 JD 或 MJD。')
        self.reference_mjd=float(np.nanmin(mjd)) if REFERENCE_MJD is None else float(REFERENCE_MJD)
        phase=mjd-self.reference_mjd
        rows=[]
        for band in BANDS:
            mcol,ecol=DATA_COLUMNS[band]
            m,e=raw[mcol],raw[ecol]
            good=np.isfinite(phase)&np.isfinite(m)&np.isfinite(e)&(m!=0)&(e>0)
            if PHASE_MIN is not None:good&=phase>=PHASE_MIN
            if PHASE_MAX is not None:good&=phase<=PHASE_MAX
            if not good.any():raise ValueError(f'{band} 没有有效观测。')
            rows.extend(zip(phase[good],[band]*int(good.sum()),m[good],e[good]))
        rows.sort(key=lambda row:row[0])
        t,b,m,e=zip(*rows)
        self.t=np.array(t);self.band=np.array(b,dtype=str)
        self.observed=np.array(m);self.err=np.array(e)
        # 消光已经在真实通带内作用于模型SED；观测星等保持原样。
        self.extinction_by_band={b:float(-2.5*np.log10(
            _TRAPEZOID(_dust_attenuation(FILTER_CURVES[b][0])*FILTER_CURVES[b][2],
                       FILTER_CURVES[b][0])/FILTER_CURVES[b][3])) for b in BANDS}
        self.a_mag=np.zeros_like(self.observed)
        self.y=self.observed.copy()
        self.sigma_model=np.zeros_like(self.err)  # 旧 NPZ 字段兼容；分箱模型误差已关闭
        _,self.group_index=np.unique(self.band,return_inverse=True)  # 单 CSV：按波段共享校准项
        self.calibration_sigma=float(CALIBRATION_SIGMA_MAG)
        # sigma 尚未标定。阶段 1 每次显式传入 sigma，阶段 2 前才设置固定值。
        self.sigma_int=None
        self.sigma=None
        self.indices=[np.flatnonzero(self.band==b) for b in BANDS]
        self.tmax=float(self.t.max()-BOUNDS[-1,0]+1.)
        self.input_hash=hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()

    def set_sigma_int(self,value):
        value=float(value)
        if not np.isfinite(value) or value<0:raise ValueError('标定 sigma_int 必须有限且非负。')
        self.sigma_int=value
        # sigma_used 为边际标准差；非零共享校准项的相关性由完整似然处理。
        self.sigma=np.sqrt(self.err**2+value**2+self.calibration_sigma**2)

    def likelihood_from_prediction(self,prediction,sigma_int=None):
        if sigma_int is None and self.sigma_int is None:
            raise RuntimeError('sigma 尚未标定并固定。')
        prediction=np.asarray(prediction,float)
        if prediction.shape!=self.y.shape:raise ValueError('预测和观测形状不一致。')
        return gaussian_calibration_likelihood(self.y,prediction,self.err,self.group_index,
            self.calibration_sigma,self.sigma_int if sigma_int is None else float(sigma_int))

    def physical(self,u):return self.lower+np.asarray(u)*self.width
    def unit(self,theta):return (np.asarray(theta)-self.lower)/self.width

    def predict(self,theta,nx=NX,ny=NY):
        state=solve_model(theta,self.tmax,nx,ny)
        tph,rph,tdirect,rdirect=_nickel_two_component_blackbody(state,theta[8])
        times=state.t_s/DAY*(1+REDSHIFT)
        prediction=np.empty_like(self.t)
        sed=BlackbodySED();distance=DISTANCE_MPC*MPC
        for (band,definition),idx in zip(BANDS.items(),self.indices):
            if not idx.size:continue
            target=self.t[idx]-theta[-1]
            upper=np.clip(np.searchsorted(times,target,side='left'),0,times.size-1)
            nodes=np.unique(np.r_[np.maximum(upper-1,0),upper])
            wave,response,weight,norm=FILTER_CURVES[band]
            nu=C_LIGHT/(wave*1e-8);attenuation=_dust_attenuation(wave)
            fph=sed.fnu(nu,tph[nodes],rph[nodes],DL_cm=distance,z=REDSHIFT)
            fdirect=sed.fnu(nu,tdirect[nodes],rdirect[nodes],DL_cm=distance,z=REDSHIFT)
            fph[:,rph[nodes]<=0]=0.;fdirect[:,rdirect[nodes]<=0]=0.
            mean=_TRAPEZOID((fph+fdirect)*attenuation[:,None]*weight[:,None],wave,axis=0)/norm
            zero=definition['zero_jy']*JY if MAG_SYSTEM=='vega' else 3631.*JY
            mag=-2.5*np.log10(mean/zero)
            prediction[idx]=np.interp(target,times[nodes],mag,left=np.nan,right=np.nan)
        return prediction

    def log_prior(self, theta):
        # 原物理坐标先验（省略与参数无关的常数）。
        m,ek,eth,mni,r0,fni,kg,n,tfloor,shift=theta
        prior=-0.5*np.log(m*ek) if KEEP_VELOCITY_PRIOR else 0.
        if EXPLOSION_MJD_PRIOR is not None and EXPLOSION_MJD_SIGMA_D is not None:
            sigma=EXPLOSION_MJD_SIGMA_D
            if not np.isfinite(sigma) or sigma<=0:
                raise ValueError('EXPLOSION_MJD_SIGMA_D 必须有限且为正。')
            mean=EXPLOSION_MJD_PRIOR-self.reference_mjd
            prior += -0.5*((shift-mean)/sigma)**2-np.log(sigma*np.sqrt(2*PI))
        return float(prior)

    def prior_transform(self, u):
        u=np.asarray(u, dtype=float)
        theta=self.physical(u)
        if KEEP_VELOCITY_PRIOR:
            # p(M,E) ∝ (M E)^(-1/2)，保持原能量坐标中的速度先验 Jacobian。
            # 不能在此简单改成均匀 M、条件均匀 v：那会改变截断后的 M 先验。
            for i in (0,1):
                lo,hi=np.sqrt(BOUNDS[i])
                theta[i]=(lo+u[i]*(hi-lo))**2
        if EXPLOSION_MJD_PRIOR is not None and EXPLOSION_MJD_SIGMA_D is not None:
            sigma=EXPLOSION_MJD_SIGMA_D
            if not np.isfinite(sigma) or sigma<=0:
                raise ValueError('EXPLOSION_MJD_SIGMA_D 必须有限且为正。')
            mean=EXPLOSION_MJD_PRIOR-self.reference_mjd
            a,b=(BOUNDS[-1]-mean)/sigma
            theta[-1]=truncnorm.ppf(u[-1],a,b,loc=mean,scale=sigma)
        return theta

    def log_likelihood(self,theta):
        theta=np.asarray(theta,float)
        if not np.all(np.isfinite(theta)) or np.any(theta<BOUNDS[:,0]) or np.any(theta>BOUNDS[:,1]):
            return -np.inf
        m,ek,eth,mni,r0,fni,kg,n,tfloor,shift=theta
        if min(m,ek,eth,mni,r0,kg,tfloor)<=0 or not (0<fni<=1) or n<=5 or shift>0:
            return -np.inf
        velocity=np.sqrt(2*ek*1e51/(m*M_SUN))/1e9
        # 保留原代码的联合硬约束；新采样先验也检查相同支持域。
        if not (V_MIN<=velocity<=V_MAX) or mni>fni*m or np.any(self.t-shift<=0):
            return -np.inf
        pred=self.predict(theta)
        if not np.all(np.isfinite(pred)):return -np.inf
        return self.likelihood_from_prediction(pred)



# ======================== 文件与原物理支持域 ========================
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.temporary')
    tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def joint_support(problem,theta):
    t=np.asarray(theta,float)
    if t.shape!=(len(NAMES),) or not np.all(np.isfinite(t)):
        return False
    if np.any(t<BOUNDS[:,0]) or np.any(t>BOUNDS[:,1]):return False
    m,ek,eth,mni,r0,fni,kg,n,tfloor,shift=t
    if min(m,ek,eth,mni,r0,kg,tfloor)<=0 or not 0<fni<=1 or n<=5 or shift>0:return False
    v=np.sqrt(2*ek*1e51/(m*M_SUN))/1e9
    return bool(V_MIN<=v<=V_MAX and mni<=fni*m and np.all(problem.t-shift>0))


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def target_identity(problem):
    # 固定前的身份不依赖未知 sigma 值；其标定值单独封存并核对。
    return dict(code_sha256=sha(__file__),data_sha256=problem.input_hash,calibration_sha256=CALIBRATION_SHA256,
        bounds=BOUNDS,names=NAMES,keep_velocity_prior=KEEP_VELOCITY_PRIOR,v_bounds=[V_MIN,V_MAX],
        distance=DISTANCE_MPC,z=REDSHIFT,ebv=EBV,rv=RV,mag_system=MAG_SYSTEM,bands=BANDS,
        data_columns=DATA_COLUMNS,time_column=TIME_COLUMN,time_format=TIME_FORMAT,
        reference_mjd=problem.reference_mjd,phase_min=PHASE_MIN,phase_max=PHASE_MAX,
        sigma_workflow='dynesty calibration then fixed-sigma 10D production',
        sigma_calibration_prior=dict(family='uniform',bounds=SIGMA_CALIBRATION_BOUNDS,unit='mag'),
        sigma_statistic='weighted marginal median',sigma_rounding_mag=SIGMA_ROUNDING_MAG,calibration_sigma_mag=problem.calibration_sigma,
        noise_model='measurement + source-level sigma_int + optional shared single-CSV/band calibration',
        explosion_prior=EXPLOSION_MJD_PRIOR,explosion_sigma=EXPLOSION_MJD_SIGMA_D,
        nx=NX,ny=NY,kappa0=KAPPA0,delta=DELTA,initial_energy_profile=INITIAL_ENERGY_PROFILE)


# ======================== dynesty 正式采样与加权后验导出 ========================
_DYNESTY_PROBLEM = None

def dynesty_prior_transform(unit):
    # 物理先验已经由此映射编码；不能再把 log_prior 加进 loglikelihood。
    return _DYNESTY_PROBLEM.prior_transform(unit)

def dynesty_loglikelihood(theta):
    theta = np.asarray(theta,float)
    if theta.shape!=(len(NAMES),):return -np.inf
    if _DYNESTY_PROBLEM.sigma_int is None:raise RuntimeError('必须先固定 sigma。')
    return float(_DYNESTY_PROBLEM.log_likelihood(theta))

def dynesty_worker_initialize(problem):
    global _DYNESTY_PROBLEM
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _DYNESTY_PROBLEM = problem

def nested_statistics(results, problem):
    raw = np.asarray(results.samples, float)
    logwt = np.asarray(results.logwt, float)
    logl = np.asarray(results.logl, float)
    if raw.ndim != 2 or raw.shape[1] != len(NAMES) or not len(raw):
        raise ValueError('dynesty 尚无可导出的样本。')
    if not np.isfinite(raw).all() or np.isnan(logwt).any() or not np.isfinite(logsumexp(logwt)):
        raise ValueError('dynesty 样本或权重异常。')
    if np.isnan(logl).any() or np.isposinf(logl).any():
        raise ValueError('dynesty 似然异常。')
    weights = np.exp(logwt-logsumexp(logwt))
    weights /= weights.sum()
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError('嵌套采样后验权重异常。')
    quantiles = np.asarray([dyfunc.quantile(raw[:,i], [.16,.5,.84], weights=weights)
                             for i in range(len(NAMES))]).T
    logprior = np.array([problem.log_prior(theta) if joint_support(problem,theta) else -np.inf for theta in raw])
    logposterior = logl+logprior
    valid = np.isfinite(logposterior) & (weights > 0)
    if not valid.any():
        raise ValueError('没有满足联合物理约束的正权重样本。')
    best_index = int(np.argmax(np.where(valid,logposterior,-np.inf)))
    # 兼容旧 corner/光变代码：samples 是等权后验，不是无权重的 raw nested points。
    indices = dyfunc.resample_equal(np.arange(len(raw)), weights,
                                    rstate=np.random.default_rng(PRODUCTION_SEED+1)).astype(int)
    equal = raw[indices]
    ess = float(1./np.sum(weights**2))
    return dict(raw=raw, weights=weights, quantiles=quantiles, logl=logl,
        logposterior=logposterior, best=raw[best_index], best_index=best_index,
        equal=equal, equal_indices=indices, ess=ess)

def nested_diagnostics(results, stats, completed):
    from dynesty.dynamicsampler import stopping_function
    stop, stop_values = stopping_function(results,
        args=dict(pfrac=1.,target_n_effective=N_EFFECTIVE,n_mc=0), return_vals=True)
    qualified = bool(completed and stop and np.isfinite(results.logz[-1]) and
                     np.isfinite(results.logzerr[-1]))
    warnings = [] if qualified else ['Nested stopping target not certified; do not treat output as final.']
    return dict(sampler='dynesty.DynamicNestedSampler',
        qa_grade='NESTED STOP MET' if qualified else 'PRELIMINARY', qualified=qualified,
        run_returned_normally=bool(completed), posterior_stopping_met=bool(stop),
        posterior_weight_ESS=stats['ess'], ESS_min=stats['ess'],
        effective_sample_target=N_EFFECTIVE,
        logz=float(results.logz[-1]), logzerr=float(results.logzerr[-1]),
        niter=int(results.niter), ncall=int(np.sum(results.ncall)),
        stop_values=dict(posterior=float(stop_values[0]), evidence=float(stop_values[1]),combined=float(stop_values[2])),
        warnings=warnings, substantive_problems=[],
        review_scope='Nested weighted ESS/stopping only; no emcee IAT, walker, or global-mode certification.',
        evidence_convention='Normalized separable box prior; joint hard-support violations have zero likelihood. logZ includes allowed-support prior mass; frozen empirical sigma is conditional.')

def export_nested(problem, results, calibration, output, completed=True):
    stats = nested_statistics(results, problem)
    diag = nested_diagnostics(results, stats, completed)
    diag.update(production_dimensions=len(NAMES),fixed_sigma_int_mag=problem.sigma_int,
        sigma_uncertainty_propagated=False)
    best = stats['best']; quantiles = stats['quantiles']
    state = solve_model(best,problem.tmax)
    magnitudes = magnitude_grid(state,best[8])
    prediction = problem.predict(best)
    grid_delta = np.full_like(prediction,np.nan)
    if GRID_CHECK_AT_END:
        grid_delta = problem.predict(best,2*NX,2*NY)-prediction
    tcolor,_,tdirect,_ = _nickel_two_component_blackbody(state,best[8])
    emitting = state.photosphere_valid & (state.Lphotospheric > 0.)
    tcolor = np.where(emitting,tcolor,np.nan)
    age = state.t_s/DAY*(1+REDSHIFT)
    metadata = dict(transport_revision='full-domain-20260917',thermal_energy_definition='whole_ejecta',
        pde_outer_boundary='Rmax_tau0_F=cu/2',luminosity_definition='outer_boundary_escape',
        energy_ledger='backward_Euler_discrete_PdV',
        distance_mpc=DISTANCE_MPC,redshift=REDSHIFT,ebv=EBV,rv=RV,mag_system=MAG_SYSTEM,
        kappa0=KAPPA0,delta=DELTA,bands=BANDS,include_bands=list(BANDS),Nx=NX,Ny=NY,
        phase_min=PHASE_MIN,phase_max=PHASE_MAX,time_column=TIME_COLUMN,time_format=TIME_FORMAT,
        initial_energy_profile=INITIAL_ENERGY_PROFILE,
        initial_exterior_energy_definition='finite-volume piecewise-constant initial radiation',
        keep_velocity_prior=KEEP_VELOCITY_PRIOR,velocity_bounds=[V_MIN,V_MAX],
        sigma_int=problem.sigma_int,sigma_int_requested='dynesty_calibration',
        sigma_inference='calibrate_then_fix',production_dimensions=len(NAMES),
        sigma_uncertainty_propagated=False,calibration_dimensions=len(NAMES)+1,
        calibration_sigma_mag=problem.calibration_sigma,model_error_disabled=True,
        sigma_calibration=calibration,noise_model='TransFit R1 normalized covariance Gaussian',
        explosion_mjd_prior=EXPLOSION_MJD_PRIOR,explosion_mjd_sigma_days=EXPLOSION_MJD_SIGMA_D,
        calibration_file=str(CALIBRATION_FILE),calibration_sha256=CALIBRATION_SHA256,
        extinction_treatment='CCM89 integrated inside each observed bandpass',
        seed=PRODUCTION_SEED,base_seed=SEED,sampler='dynesty.DynamicNestedSampler',dynesty_version=dynesty.__version__,
        nested_settings=nested_settings(),
        samples_semantics='Equal-weight posterior resampled from nested samples with nested_weights',
        nested_samples_semantics='Raw nested points; MUST use nested_weights for posterior summaries',
        quantiles_semantics='Weighted 16/50/84 percentiles of raw nested points',
        best_params_definition='physical-coordinate MAP among positive-weight raw nested samples',
        log_prob_semantics='physical posterior up to a parameter-independent constant',
        incomplete=not diag['qualified'],smoke_only=False,
        time_shift_convention='t_model_observer=phase_observer-t_shift; model_start_mjd=reference_mjd+t_shift',
        input_sha256=problem.input_hash,code_sha256=sha(__file__),
        diagnostic_scope=diag['review_scope'],evidence_convention=diag['evidence_convention'])
    output = Path(output)
    tmp = output.with_name(output.name+'.temporary.npz')
    np.savez_compressed(tmp, **{'energy_'+k:v for k,v in state.energy_diagnostics.items()},
        param_names=np.array(NAMES),param_units=np.array(UNITS),bounds=BOUNDS,
        initial_energy_profile=np.array(INITIAL_ENERGY_PROFILE),sampler_name=np.array('dynesty'),
        sigma_int_mag=np.array(problem.sigma_int),calibration_sigma_mag=np.array(problem.calibration_sigma),
        sigma_calibration_json=np.array(json.dumps(clean(calibration))),
        calibration_group_index=problem.group_index,
        samples=stats['equal'],sample_log_prob=stats['logposterior'][stats['equal_indices']],
        sample_log_likelihood=stats['logl'][stats['equal_indices']],
        nested_samples=stats['raw'],nested_weights=stats['weights'],nested_logwt=results.logwt,
        nested_log_likelihood=stats['logl'],nested_sample_log_prob=stats['logposterior'],
        nested_logz=results.logz,nested_logzerr=results.logzerr,
        nested_ncall=results.ncall,nested_logvol=results.logvol,
        resampled_indices=stats['equal_indices'],
        posterior_effective_sample_size=np.array(stats['ess']),
        convergence_json=np.array(json.dumps(clean(diag))),qa_grade=np.array(diag['qa_grade']),
        diagnostic_criteria_passed=np.array(diag['qualified']),
        percentiles=np.array([16,50,84]),quantiles=quantiles,
        median=quantiles[1],err_minus=quantiles[1]-quantiles[0],err_plus=quantiles[2]-quantiles[1],
        best_params=best,best_mle=stats['raw'][np.argmax(stats['logl'])],
        reference_mjd=problem.reference_mjd,phase_observer_days=problem.t,
        observation_mjd=problem.reference_mjd+problem.t,band=problem.band,
        mag_observed=problem.observed,mag_err=problem.err,extinction_mag=problem.a_mag,
        mag_dereddened=problem.y,mag_likelihood=problem.y,
        sigma_model=problem.sigma_model,sigma_used=problem.sigma,
        best_model_mag_dereddened=prediction,best_model_mag_observed=prediction,best_model_mag_likelihood=prediction,
        model_observer_days=age,model_phase_days=age+best[-1],model_bands=np.array(list(BANDS)),
        model_mag_dereddened=magnitudes,model_mag_observed=magnitudes,model_mag_likelihood=magnitudes,
        model_start_mjd=problem.reference_mjd+best[-1],Rph_cm=state.Rph,Teff_K=state.Tph,
        Lbol_erg_s=state.Lbol,Lphotospheric_erg_s=state.Lphotospheric,Ldirect_erg_s=state.Ldirect,
        photosphere_valid=state.photosphere_valid,Tcolor_ph_K=tcolor,Tdirect_K=tdirect,
        Ldirect_fraction=np.divide(state.Ldirect,state.Lbol,out=np.full_like(state.Lbol,np.nan),where=state.Lbol>0),
        photo_color_floor_active=emitting & (state.Tph < best[8]),
        grid_double_resolution_minus_default_mag=grid_delta,
        metadata_json=np.array(json.dumps(clean(metadata))))
    tmp.replace(output)
    save(output.with_suffix('.run')/'convergence.json',diag)
    print('\nparameter             median       -error       +error',flush=True)
    for name,lo,mid,hi in zip(NAMES,*quantiles):
        print(f'{name:<18} {mid:12.6g} {mid-lo:12.6g} {hi-mid:12.6g}',flush=True)
    print(f"QA: {diag['qa_grade']} | weighted ESS={stats['ess']:.1f}",flush=True)
    print('结果：',output.resolve(),flush=True)
    return diag

def nested_settings():
    return dict(nlive_init=NLIVE_INIT,nlive_batch=NLIVE_BATCH,dlogz_init=DLOGZ_INIT,
        n_effective=N_EFFECTIVE,bound=BOUND,sample=SAMPLE,walks=WALKS,
        maxcall=MAX_CALLS,maxiter=MAX_ITERATIONS,maxbatch=MAX_BATCHES,
        posterior_stopping_pfrac=1.,weight_allocation_pfrac=.8)

def pid_alive(pid):
    if not isinstance(pid,int) or pid<=0:return False
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='SN1993J: dynesty sigma calibration then fixed-sigma 10D production, phase <=100 d')
    parser.add_argument('--output',type=Path,default=OUTPUT_NPZ)
    parser.add_argument('--workers',type=int,default=N_PROCESSES)
    parser.add_argument('--resume',action='store_true',help='restore the most recent compatible dynesty checkpoint')
    parser.add_argument('--no-grid-check',action='store_true')
    return parser.parse_args()

# ======================== 第一阶段：dynesty 标定 sigma ========================
def calibration_prior_transform(unit):
    unit = np.asarray(unit, float)
    if unit.shape != (len(NAMES)+1,) or not np.isfinite(unit).all() or np.any((unit<0)|(unit>1)):
        raise ValueError('sigma 标定阶段的单位立方体坐标异常。')
    lo, hi = SIGMA_CALIBRATION_BOUNDS
    return np.r_[_DYNESTY_PROBLEM.prior_transform(unit[:-1]),lo+(hi-lo)*unit[-1]]

def calibration_loglikelihood(joint):
    joint = np.asarray(joint, float)
    lo, hi = SIGMA_CALIBRATION_BOUNDS
    if (joint.shape != (len(NAMES)+1,) or not np.isfinite(joint).all()
            or not lo <= joint[-1] <= hi or not joint_support(_DYNESTY_PROBLEM,joint[:-1])):
        return -np.inf
    prediction = _DYNESTY_PROBLEM.predict(joint[:-1])
    if not np.isfinite(prediction).all():
        return -np.inf
    # 临时估计 sigma。完整归一化似然，不把物理先验重复乘进似然。
    return float(_DYNESTY_PROBLEM.likelihood_from_prediction(prediction,joint[-1]))

def stage_settings(stage):
    if stage == 'calibration':
        return dict(nlive_init=SIGMA_NLIVE_INIT,nlive_batch=SIGMA_NLIVE_BATCH,
            dlogz_init=SIGMA_DLOGZ_INIT,n_effective=SIGMA_N_EFFECTIVE,
            maxcall=SIGMA_MAX_CALLS,maxiter=SIGMA_MAX_ITERATIONS,maxbatch=SIGMA_MAX_BATCHES,
            seed=SEED,ndim=len(NAMES)+1,bound=BOUND,sample=SAMPLE,walks=WALKS)
    if stage != 'production':
        raise ValueError('未知采样阶段。')
    return dict(nested_settings(),seed=PRODUCTION_SEED,ndim=len(NAMES))

def summarize_calibration(results, identity, posterior_sha256):
    raw = np.asarray(results.samples,float)
    logwt = np.asarray(results.logwt,float)
    if (raw.ndim!=2 or raw.shape[1]!=len(NAMES)+1 or len(raw)==0
            or logwt.shape!=(len(raw),) or not np.isfinite(raw).all()
            or not np.isfinite(logsumexp(logwt))):
        raise ValueError('sigma 标定样本或权重异常。')
    lo, hi = SIGMA_CALIBRATION_BOUNDS
    if np.any((raw[:,-1]<lo)|(raw[:,-1]>hi)):
        raise ValueError('sigma 标定样本超出先验。')
    weights = np.exp(logwt-logsumexp(logwt)); weights /= weights.sum()
    if not np.isfinite(weights).all():
        raise ValueError('sigma 标定权重异常。')
    quantiles = np.asarray(dyfunc.quantile(raw[:,-1],[.16,.5,.84],weights=weights),float)
    # 与 R1 一样保留 0.001 mag 精度；统计量改为 dynesty 加权边缘后验中位数。
    value = float(np.clip(np.floor(quantiles[1]/SIGMA_ROUNDING_MAG+.5)*SIGMA_ROUNDING_MAG,lo,hi))
    result = dict(method='dynesty weighted marginal sigma median, then fixed for independent 10D production',
        calibration_ready=True,preparation_identity=identity,manual_override=False,
        canonical_sigma_int_mag=value,unrounded_sigma_median_mag=quantiles[1],
        sigma_quantiles_mag=quantiles,rounding_mag=SIGMA_ROUNDING_MAG,
        sigma_prior=dict(family='uniform',bounds=SIGMA_CALIBRATION_BOUNDS,unit='mag'),
        calibration_dimensions=len(NAMES)+1,production_dimensions=len(NAMES),
        fitted_bands=list(BANDS),observations_used=len(_DYNESTY_PROBLEM.y),
        posterior_weight_ESS=float(1./np.sum(weights**2)),
        fraction_in_upper_one_percent=float(weights[raw[:,-1]>=hi-.01*(hi-lo)].sum()),
        calibration_posterior='calibration/posterior.npz',posterior_sha256=posterior_sha256,
        created_utc=utc(),warnings=[],
        statistical_interpretation='Same observations reused for plug-in calibration; production is conditional on fixed sigma.',
        uncertainty='Calibration sigma interval is recorded separately, not propagated into production parameter intervals.',
        production_initialization='Fresh 10D sampler and full physical prior; calibration points are never injected.')
    if result['fraction_in_upper_one_percent']>.05:
        result['warnings'].append('Calibration sigma has >5% posterior weight in upper 1% of its prior; inspect the recorded sigma posterior.')
    result['seal_sha256'] = fingerprint(result)
    return clean(result)

def load_calibration(path, expected_identity):
    path = Path(path)
    calibration = json.loads(path.read_text())
    payload = {key:value for key,value in calibration.items() if key!='seal_sha256'}
    if calibration.get('seal_sha256')!=fingerprint(payload):
        raise RuntimeError('sigma 标定文件被修改或不完整；拒绝续接。')
    if calibration.get('preparation_identity')!=expected_identity or not calibration.get('calibration_ready'):
        raise RuntimeError('sigma 标定与当前模型/数据/先验不匹配。')
    posterior = path.parent/calibration['calibration_posterior']
    if not posterior.is_file() or sha(posterior)!=calibration['posterior_sha256']:
        raise RuntimeError('sigma 标定样本文件缺失或改变。')
    value = float(calibration['canonical_sigma_int_mag'])
    lo,hi = SIGMA_CALIBRATION_BOUNDS
    if not np.isfinite(value) or not lo<=value<=hi:
        raise RuntimeError('固定 sigma 不在标定支持域内。')
    return calibration

def stage_diagnostics(results, settings):
    from dynesty.dynamicsampler import stopping_function
    stop, values = stopping_function(results,
        args=dict(pfrac=1.,target_n_effective=settings['n_effective'],n_mc=0),return_vals=True)
    ready = bool(stop and np.isfinite(results.logz[-1]) and np.isfinite(results.logzerr[-1]))
    return dict(qualified=ready,posterior_stopping_met=bool(stop),
        effective_sample_target=settings['n_effective'],sampled_dimensions=settings['ndim'],
        logz=float(results.logz[-1]),logzerr=float(results.logzerr[-1]),
        niter=int(results.niter),ncall=int(np.sum(results.ncall)),
        stopping_ratio=float(values[2]),
        review_scope='Finite nested stopping audit; not a global-mode or physical-model certification.')

def run_nested_stage(problem, stage, folder, workers, resume):
    global _DYNESTY_PROBLEM
    _DYNESTY_PROBLEM = problem
    settings = stage_settings(stage)
    folder = Path(folder);folder.mkdir(parents=True,exist_ok=True)
    checkpoint = folder/'dynesty.checkpoint'
    resume_checkpoint = bool(resume and checkpoint.exists())
    if stage=='production' and (problem.sigma_int is None or not np.isfinite(problem.sigma_int)):
        raise RuntimeError('正式拟合前必须先完成并固定 sigma 标定。')
    likelihood,transform = ((calibration_loglikelihood,calibration_prior_transform)
        if stage=='calibration' else (dynesty_loglikelihood,dynesty_prior_transform))
    with ExitStack() as stack:
        pool = None
        if workers>1:
            pool = stack.enter_context(mp.get_context('spawn').Pool(workers,
                initializer=dynesty_worker_initialize,initargs=(problem,)))
        if resume_checkpoint:
            sampler = dynesty.DynamicNestedSampler.restore(str(checkpoint),pool=pool)
            if sampler.ndim!=settings['ndim']:
                raise RuntimeError('检查点的采样维度与当前阶段不一致。')
            done = getattr(sampler.internal_state,'name','')=='RUN_DONE'
        else:
            # 不传 live_points：两个阶段各自从完整先验生成全新活点。
            sampler = dynesty.DynamicNestedSampler(likelihood,transform,settings['ndim'],
                nlive=settings['nlive_init'],bound=settings['bound'],sample=settings['sample'],walks=settings['walks'],
                pool=pool,queue_size=workers,
                use_pool=dict(prior_transform=False,loglikelihood=True,propose_point=True,update_bound=True),
                rstate=np.random.default_rng(settings['seed']))
            sampler.save(str(checkpoint))
            done = False
        if not done:
            # KeyboardInterrupt 由外层处理；不覆盖库最近的安全检查点。
            sampler.run_nested(nlive_init=settings['nlive_init'],nlive_batch=settings['nlive_batch'],
                dlogz_init=settings['dlogz_init'],n_effective=settings['n_effective'],
                stop_kwargs=dict(pfrac=1.,n_mc=0),wt_kwargs=dict(pfrac=.8),
                maxcall=settings['maxcall'],maxiter=settings['maxiter'],maxbatch=settings['maxbatch'],
                checkpoint_file=str(checkpoint),checkpoint_every=CHECKPOINT_SECONDS,
                resume=resume_checkpoint,print_progress=True)
        results = sampler.results
        diagnostics = stage_diagnostics(results,settings)
        save(folder/'convergence.json',diagnostics)
        return results,diagnostics

def export_calibration(results, identity, folder):
    folder = Path(folder)
    posterior = folder/'calibration'/'posterior.npz'
    posterior.parent.mkdir(parents=True,exist_ok=True)
    tmp = posterior.with_name(posterior.name+'.temporary.npz')
    logwt = np.asarray(results.logwt,float)
    weights = np.exp(logwt-logsumexp(logwt));weights /= weights.sum()
    np.savez_compressed(tmp,param_names=np.array(NAMES+['sigma_int_mag']),
        param_units=np.array(UNITS+['mag']),nested_samples=results.samples,nested_weights=weights,
        nested_logwt=results.logwt,nested_log_likelihood=results.logl,
        nested_logz=results.logz,nested_logzerr=results.logzerr,
        role=np.array('Calibration only; never merge with production posterior'),
        sigma_prior_bounds=np.asarray(SIGMA_CALIBRATION_BOUNDS))
    tmp.replace(posterior)
    calibration = summarize_calibration(results,identity,sha(posterior))
    save(folder/'sigma_calibration.json',calibration)
    return calibration

def main():
    global _DYNESTY_PROBLEM,GRID_CHECK_AT_END
    args = parse_args()
    if args.workers<1:raise ValueError('--workers 必须至少为 1。')
    if args.no_grid_check:GRID_CHECK_AT_END=False
    if tuple(map(int,dynesty.__version__.split('.')[:2]))!=(3,1):
        raise RuntimeError('请安装 requirements.txt 中的 dynesty==3.1.0。')
    for stage in ('calibration','production'):
        settings = stage_settings(stage)
        if min(settings['nlive_init'],settings['nlive_batch'])<2*settings['ndim'] or settings['n_effective']<1 or settings['dlogz_init']<=0:
            raise ValueError(stage+' 的活点/ESS/dlogz 设置无效。')
    output = Path(args.output).resolve();folder = output.with_suffix('.run')
    folder.mkdir(parents=True,exist_ok=True)
    manifest_path = folder/'sampler.json';calibration_path = folder/'sigma_calibration.json'
    lock = folder/'writer.lock'
    if lock.exists():
        owner = json.loads(lock.read_text())
        if pid_alive(owner.get('pid')):raise RuntimeError('此输出目录有存活的写入进程。')
        lock.rename(folder/('writer.stale.'+uuid4().hex+'.json'))
    with lock.open('x',encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(),created_utc=utc()),handle)
    manifest = None
    try:
        existing = (output.exists() or manifest_path.exists() or calibration_path.exists()
            or (folder/'calibration'/'dynesty.checkpoint').exists()
            or (folder/'production'/'dynesty.checkpoint').exists())
        if not args.resume and existing:
            raise FileExistsError('结果目录已有内容；请使用 --resume 或新的 --output。')
        problem = FitProblem()
        identity = clean(dict(target=target_identity(problem),
            stages={name:stage_settings(name) for name in ('calibration','production')},
            dynesty_version=dynesty.__version__,numpy_version=np.__version__,python=sys.version.split()[0]))
        preparation_identity = fingerprint(identity)
        if args.resume:
            if not manifest_path.exists():raise RuntimeError('--resume 缺少 sampler.json。')
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('identity')!=identity:
                raise RuntimeError('代码、模型、数据、先验或依赖改变；拒绝续接。')
            if manifest.get('status')=='completed':
                if not output.is_file() or sha(output)!=manifest.get('output_sha256'):
                    raise RuntimeError('已完成的正式结果缺失或改变。')
                print('此两阶段运行已完成。',flush=True)
                return
        else:
            manifest = dict(identity=identity,status='ready',created_utc=utc(),output_npz=str(output),
                calibration_dimensions=len(NAMES)+1,production_dimensions=len(NAMES))
            save(manifest_path,manifest)
        _DYNESTY_PROBLEM = problem
        if calibration_path.exists():
            if (manifest.get('sigma_calibration_sha256') is not None
                    and manifest['sigma_calibration_sha256']!=sha(calibration_path)):
                raise RuntimeError('已封存的 sigma 标定文件改变；拒绝续接正式拟合。')
            calibration = load_calibration(calibration_path,preparation_identity)
            print('读取已完成的 dynesty sigma 标定，不重新标定。',flush=True)
        else:
            if (folder/'production'/'dynesty.checkpoint').exists():
                raise RuntimeError('已有正式检查点但 sigma 标定文件缺失；不能重新标定后续接旧正式链。')
            manifest.update(status='calibrating_sigma');save(manifest_path,manifest)
            save(folder/'status.json',dict(status='calibrating_sigma',updated_utc=utc()))
            print('阶段 1/2：dynesty 临时估计物理参数与 sigma，仅用于标定。',flush=True)
            results,diag = run_nested_stage(problem,'calibration',folder/'calibration',args.workers,args.resume)
            if not diag['qualified']:
                raise RuntimeError('sigma 标定尚未满足停止目标；不会启动正式拟合。')
            calibration = export_calibration(results,preparation_identity,folder)
            del results
        problem.set_sigma_int(calibration['canonical_sigma_int_mag'])
        manifest.update(status='sigma_fixed',fixed_sigma_int_mag=problem.sigma_int,
            sigma_calibration_sha256=sha(calibration_path));save(manifest_path,manifest)
        print(f"sigma 加权中位数={calibration['unrounded_sigma_median_mag']:.6g} mag；固定值={problem.sigma_int:.3f} mag。",flush=True)
        print(f'阶段 2/2：全新 10 参数正式 dynesty；波段 {list(BANDS)}；相位 <= {PHASE_MAX:g} 天。',flush=True)
        manifest.update(status='production_sampling');save(manifest_path,manifest)
        save(folder/'status.json',dict(status='production_sampling',fixed_sigma_int_mag=problem.sigma_int,updated_utc=utc()))
        results,_ = run_nested_stage(problem,'production',folder/'production',args.workers,args.resume)
        diag = export_nested(problem,results,calibration,output,completed=True)
        manifest.update(status='completed',diagnostics=diag,output_sha256=sha(output),updated_utc=utc())
        save(manifest_path,manifest)
        save(folder/'status.json',dict(status='completed',updated_utc=utc(),**diag))
    except KeyboardInterrupt:
        if manifest is not None:
            manifest.update(interrupted_stage=manifest.get('status'),status='interrupted',updated_utc=utc())
            save(manifest_path,manifest)
        save(folder/'status.json',dict(status='interrupted',updated_utc=utc()))
        print('\n已中断；下次加 --resume，按阶段恢复最近安全检查点。未合并或导出中断阶段的样本。',flush=True)
    finally:
        if lock.exists():lock.unlink()


GPL_LICENSE_TEXT = '                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n\n Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>\n Everyone is permitted to copy and distribute verbatim copies\n of this license document, but changing it is not allowed.\n\n                            Preamble\n\n  The GNU General Public License is a free, copyleft license for\nsoftware and other kinds of works.\n\n  The licenses for most software and other practical works are designed\nto take away your freedom to share and change the works.  By contrast,\nthe GNU General Public License is intended to guarantee your freedom to\nshare and change all versions of a program--to make sure it remains free\nsoftware for all its users.  We, the Free Software Foundation, use the\nGNU General Public License for most of our software; it applies also to\nany other work released this way by its authors.  You can apply it to\nyour programs, too.\n\n  When we speak of free software, we are referring to freedom, not\nprice.  Our General Public Licenses are designed to make sure that you\nhave the freedom to distribute copies of free software (and charge for\nthem if you wish), that you receive source code or can get it if you\nwant it, that you can change the software or use pieces of it in new\nfree programs, and that you know you can do these things.\n\n  To protect your rights, we need to prevent others from denying you\nthese rights or asking you to surrender the rights.  Therefore, you have\ncertain responsibilities if you distribute copies of the software, or if\nyou modify it: responsibilities to respect the freedom of others.\n\n  For example, if you distribute copies of such a program, whether\ngratis or for a fee, you must pass on to the recipients the same\nfreedoms that you received.  You must make sure that they, too, receive\nor can get the source code.  And you must show them these terms so they\nknow their rights.\n\n  Developers that use the GNU GPL protect your rights with two steps:\n(1) assert copyright on the software, and (2) offer you this License\ngiving you legal permission to copy, distribute and/or modify it.\n\n  For the developers\' and authors\' protection, the GPL clearly explains\nthat there is no warranty for this free software.  For both users\' and\nauthors\' sake, the GPL requires that modified versions be marked as\nchanged, so that their problems will not be attributed erroneously to\nauthors of previous versions.\n\n  Some devices are designed to deny users access to install or run\nmodified versions of the software inside them, although the manufacturer\ncan do so.  This is fundamentally incompatible with the aim of\nprotecting users\' freedom to change the software.  The systematic\npattern of such abuse occurs in the area of products for individuals to\nuse, which is precisely where it is most unacceptable.  Therefore, we\nhave designed this version of the GPL to prohibit the practice for those\nproducts.  If such problems arise substantially in other domains, we\nstand ready to extend this provision to those domains in future versions\nof the GPL, as needed to protect the freedom of users.\n\n  Finally, every program is threatened constantly by software patents.\nStates should not allow patents to restrict development and use of\nsoftware on general-purpose computers, but in those that do, we wish to\navoid the special danger that patents applied to a free program could\nmake it effectively proprietary.  To prevent this, the GPL assures that\npatents cannot be used to render the program non-free.\n\n  The precise terms and conditions for copying, distribution and\nmodification follow.\n\n                       TERMS AND CONDITIONS\n\n  0. Definitions.\n\n  "This License" refers to version 3 of the GNU General Public License.\n\n  "Copyright" also means copyright-like laws that apply to other kinds of\nworks, such as semiconductor masks.\n\n  "The Program" refers to any copyrightable work licensed under this\nLicense.  Each licensee is addressed as "you".  "Licensees" and\n"recipients" may be individuals or organizations.\n\n  To "modify" a work means to copy from or adapt all or part of the work\nin a fashion requiring copyright permission, other than the making of an\nexact copy.  The resulting work is called a "modified version" of the\nearlier work or a work "based on" the earlier work.\n\n  A "covered work" means either the unmodified Program or a work based\non the Program.\n\n  To "propagate" a work means to do anything with it that, without\npermission, would make you directly or secondarily liable for\ninfringement under applicable copyright law, except executing it on a\ncomputer or modifying a private copy.  Propagation includes copying,\ndistribution (with or without modification), making available to the\npublic, and in some countries other activities as well.\n\n  To "convey" a work means any kind of propagation that enables other\nparties to make or receive copies.  Mere interaction with a user through\na computer network, with no transfer of a copy, is not conveying.\n\n  An interactive user interface displays "Appropriate Legal Notices"\nto the extent that it includes a convenient and prominently visible\nfeature that (1) displays an appropriate copyright notice, and (2)\ntells the user that there is no warranty for the work (except to the\nextent that warranties are provided), that licensees may convey the\nwork under this License, and how to view a copy of this License.  If\nthe interface presents a list of user commands or options, such as a\nmenu, a prominent item in the list meets this criterion.\n\n  1. Source Code.\n\n  The "source code" for a work means the preferred form of the work\nfor making modifications to it.  "Object code" means any non-source\nform of a work.\n\n  A "Standard Interface" means an interface that either is an official\nstandard defined by a recognized standards body, or, in the case of\ninterfaces specified for a particular programming language, one that\nis widely used among developers working in that language.\n\n  The "System Libraries" of an executable work include anything, other\nthan the work as a whole, that (a) is included in the normal form of\npackaging a Major Component, but which is not part of that Major\nComponent, and (b) serves only to enable use of the work with that\nMajor Component, or to implement a Standard Interface for which an\nimplementation is available to the public in source code form.  A\n"Major Component", in this context, means a major essential component\n(kernel, window system, and so on) of the specific operating system\n(if any) on which the executable work runs, or a compiler used to\nproduce the work, or an object code interpreter used to run it.\n\n  The "Corresponding Source" for a work in object code form means all\nthe source code needed to generate, install, and (for an executable\nwork) run the object code and to modify the work, including scripts to\ncontrol those activities.  However, it does not include the work\'s\nSystem Libraries, or general-purpose tools or generally available free\nprograms which are used unmodified in performing those activities but\nwhich are not part of the work.  For example, Corresponding Source\nincludes interface definition files associated with source files for\nthe work, and the source code for shared libraries and dynamically\nlinked subprograms that the work is specifically designed to require,\nsuch as by intimate data communication or control flow between those\nsubprograms and other parts of the work.\n\n  The Corresponding Source need not include anything that users\ncan regenerate automatically from other parts of the Corresponding\nSource.\n\n  The Corresponding Source for a work in source code form is that\nsame work.\n\n  2. Basic Permissions.\n\n  All rights granted under this License are granted for the term of\ncopyright on the Program, and are irrevocable provided the stated\nconditions are met.  This License explicitly affirms your unlimited\npermission to run the unmodified Program.  The output from running a\ncovered work is covered by this License only if the output, given its\ncontent, constitutes a covered work.  This License acknowledges your\nrights of fair use or other equivalent, as provided by copyright law.\n\n  You may make, run and propagate covered works that you do not\nconvey, without conditions so long as your license otherwise remains\nin force.  You may convey covered works to others for the sole purpose\nof having them make modifications exclusively for you, or provide you\nwith facilities for running those works, provided that you comply with\nthe terms of this License in conveying all material for which you do\nnot control copyright.  Those thus making or running the covered works\nfor you must do so exclusively on your behalf, under your direction\nand control, on terms that prohibit them from making any copies of\nyour copyrighted material outside their relationship with you.\n\n  Conveying under any other circumstances is permitted solely under\nthe conditions stated below.  Sublicensing is not allowed; section 10\nmakes it unnecessary.\n\n  3. Protecting Users\' Legal Rights From Anti-Circumvention Law.\n\n  No covered work shall be deemed part of an effective technological\nmeasure under any applicable law fulfilling obligations under article\n11 of the WIPO copyright treaty adopted on 20 December 1996, or\nsimilar laws prohibiting or restricting circumvention of such\nmeasures.\n\n  When you convey a covered work, you waive any legal power to forbid\ncircumvention of technological measures to the extent such circumvention\nis effected by exercising rights under this License with respect to\nthe covered work, and you disclaim any intention to limit operation or\nmodification of the work as a means of enforcing, against the work\'s\nusers, your or third parties\' legal rights to forbid circumvention of\ntechnological measures.\n\n  4. Conveying Verbatim Copies.\n\n  You may convey verbatim copies of the Program\'s source code as you\nreceive it, in any medium, provided that you conspicuously and\nappropriately publish on each copy an appropriate copyright notice;\nkeep intact all notices stating that this License and any\nnon-permissive terms added in accord with section 7 apply to the code;\nkeep intact all notices of the absence of any warranty; and give all\nrecipients a copy of this License along with the Program.\n\n  You may charge any price or no price for each copy that you convey,\nand you may offer support or warranty protection for a fee.\n\n  5. Conveying Modified Source Versions.\n\n  You may convey a work based on the Program, or the modifications to\nproduce it from the Program, in the form of source code under the\nterms of section 4, provided that you also meet all of these conditions:\n\n    a) The work must carry prominent notices stating that you modified\n    it, and giving a relevant date.\n\n    b) The work must carry prominent notices stating that it is\n    released under this License and any conditions added under section\n    7.  This requirement modifies the requirement in section 4 to\n    "keep intact all notices".\n\n    c) You must license the entire work, as a whole, under this\n    License to anyone who comes into possession of a copy.  This\n    License will therefore apply, along with any applicable section 7\n    additional terms, to the whole of the work, and all its parts,\n    regardless of how they are packaged.  This License gives no\n    permission to license the work in any other way, but it does not\n    invalidate such permission if you have separately received it.\n\n    d) If the work has interactive user interfaces, each must display\n    Appropriate Legal Notices; however, if the Program has interactive\n    interfaces that do not display Appropriate Legal Notices, your\n    work need not make them do so.\n\n  A compilation of a covered work with other separate and independent\nworks, which are not by their nature extensions of the covered work,\nand which are not combined with it such as to form a larger program,\nin or on a volume of a storage or distribution medium, is called an\n"aggregate" if the compilation and its resulting copyright are not\nused to limit the access or legal rights of the compilation\'s users\nbeyond what the individual works permit.  Inclusion of a covered work\nin an aggregate does not cause this License to apply to the other\nparts of the aggregate.\n\n  6. Conveying Non-Source Forms.\n\n  You may convey a covered work in object code form under the terms\nof sections 4 and 5, provided that you also convey the\nmachine-readable Corresponding Source under the terms of this License,\nin one of these ways:\n\n    a) Convey the object code in, or embodied in, a physical product\n    (including a physical distribution medium), accompanied by the\n    Corresponding Source fixed on a durable physical medium\n    customarily used for software interchange.\n\n    b) Convey the object code in, or embodied in, a physical product\n    (including a physical distribution medium), accompanied by a\n    written offer, valid for at least three years and valid for as\n    long as you offer spare parts or customer support for that product\n    model, to give anyone who possesses the object code either (1) a\n    copy of the Corresponding Source for all the software in the\n    product that is covered by this License, on a durable physical\n    medium customarily used for software interchange, for a price no\n    more than your reasonable cost of physically performing this\n    conveying of source, or (2) access to copy the\n    Corresponding Source from a network server at no charge.\n\n    c) Convey individual copies of the object code with a copy of the\n    written offer to provide the Corresponding Source.  This\n    alternative is allowed only occasionally and noncommercially, and\n    only if you received the object code with such an offer, in accord\n    with subsection 6b.\n\n    d) Convey the object code by offering access from a designated\n    place (gratis or for a charge), and offer equivalent access to the\n    Corresponding Source in the same way through the same place at no\n    further charge.  You need not require recipients to copy the\n    Corresponding Source along with the object code.  If the place to\n    copy the object code is a network server, the Corresponding Source\n    may be on a different server (operated by you or a third party)\n    that supports equivalent copying facilities, provided you maintain\n    clear directions next to the object code saying where to find the\n    Corresponding Source.  Regardless of what server hosts the\n    Corresponding Source, you remain obligated to ensure that it is\n    available for as long as needed to satisfy these requirements.\n\n    e) Convey the object code using peer-to-peer transmission, provided\n    you inform other peers where the object code and Corresponding\n    Source of the work are being offered to the general public at no\n    charge under subsection 6d.\n\n  A separable portion of the object code, whose source code is excluded\nfrom the Corresponding Source as a System Library, need not be\nincluded in conveying the object code work.\n\n  A "User Product" is either (1) a "consumer product", which means any\ntangible personal property which is normally used for personal, family,\nor household purposes, or (2) anything designed or sold for incorporation\ninto a dwelling.  In determining whether a product is a consumer product,\ndoubtful cases shall be resolved in favor of coverage.  For a particular\nproduct received by a particular user, "normally used" refers to a\ntypical or common use of that class of product, regardless of the status\nof the particular user or of the way in which the particular user\nactually uses, or expects or is expected to use, the product.  A product\nis a consumer product regardless of whether the product has substantial\ncommercial, industrial or non-consumer uses, unless such uses represent\nthe only significant mode of use of the product.\n\n  "Installation Information" for a User Product means any methods,\nprocedures, authorization keys, or other information required to install\nand execute modified versions of a covered work in that User Product from\na modified version of its Corresponding Source.  The information must\nsuffice to ensure that the continued functioning of the modified object\ncode is in no case prevented or interfered with solely because\nmodification has been made.\n\n  If you convey an object code work under this section in, or with, or\nspecifically for use in, a User Product, and the conveying occurs as\npart of a transaction in which the right of possession and use of the\nUser Product is transferred to the recipient in perpetuity or for a\nfixed term (regardless of how the transaction is characterized), the\nCorresponding Source conveyed under this section must be accompanied\nby the Installation Information.  But this requirement does not apply\nif neither you nor any third party retains the ability to install\nmodified object code on the User Product (for example, the work has\nbeen installed in ROM).\n\n  The requirement to provide Installation Information does not include a\nrequirement to continue to provide support service, warranty, or updates\nfor a work that has been modified or installed by the recipient, or for\nthe User Product in which it has been modified or installed.  Access to a\nnetwork may be denied when the modification itself materially and\nadversely affects the operation of the network or violates the rules and\nprotocols for communication across the network.\n\n  Corresponding Source conveyed, and Installation Information provided,\nin accord with this section must be in a format that is publicly\ndocumented (and with an implementation available to the public in\nsource code form), and must require no special password or key for\nunpacking, reading or copying.\n\n  7. Additional Terms.\n\n  "Additional permissions" are terms that supplement the terms of this\nLicense by making exceptions from one or more of its conditions.\nAdditional permissions that are applicable to the entire Program shall\nbe treated as though they were included in this License, to the extent\nthat they are valid under applicable law.  If additional permissions\napply only to part of the Program, that part may be used separately\nunder those permissions, but the entire Program remains governed by\nthis License without regard to the additional permissions.\n\n  When you convey a copy of a covered work, you may at your option\nremove any additional permissions from that copy, or from any part of\nit.  (Additional permissions may be written to require their own\nremoval in certain cases when you modify the work.)  You may place\nadditional permissions on material, added by you to a covered work,\nfor which you have or can give appropriate copyright permission.\n\n  Notwithstanding any other provision of this License, for material you\nadd to a covered work, you may (if authorized by the copyright holders of\nthat material) supplement the terms of this License with terms:\n\n    a) Disclaiming warranty or limiting liability differently from the\n    terms of sections 15 and 16 of this License; or\n\n    b) Requiring preservation of specified reasonable legal notices or\n    author attributions in that material or in the Appropriate Legal\n    Notices displayed by works containing it; or\n\n    c) Prohibiting misrepresentation of the origin of that material, or\n    requiring that modified versions of such material be marked in\n    reasonable ways as different from the original version; or\n\n    d) Limiting the use for publicity purposes of names of licensors or\n    authors of the material; or\n\n    e) Declining to grant rights under trademark law for use of some\n    trade names, trademarks, or service marks; or\n\n    f) Requiring indemnification of licensors and authors of that\n    material by anyone who conveys the material (or modified versions of\n    it) with contractual assumptions of liability to the recipient, for\n    any liability that these contractual assumptions directly impose on\n    those licensors and authors.\n\n  All other non-permissive additional terms are considered "further\nrestrictions" within the meaning of section 10.  If the Program as you\nreceived it, or any part of it, contains a notice stating that it is\ngoverned by this License along with a term that is a further\nrestriction, you may remove that term.  If a license document contains\na further restriction but permits relicensing or conveying under this\nLicense, you may add to a covered work material governed by the terms\nof that license document, provided that the further restriction does\nnot survive such relicensing or conveying.\n\n  If you add terms to a covered work in accord with this section, you\nmust place, in the relevant source files, a statement of the\nadditional terms that apply to those files, or a notice indicating\nwhere to find the applicable terms.\n\n  Additional terms, permissive or non-permissive, may be stated in the\nform of a separately written license, or stated as exceptions;\nthe above requirements apply either way.\n\n  8. Termination.\n\n  You may not propagate or modify a covered work except as expressly\nprovided under this License.  Any attempt otherwise to propagate or\nmodify it is void, and will automatically terminate your rights under\nthis License (including any patent licenses granted under the third\nparagraph of section 11).\n\n  However, if you cease all violation of this License, then your\nlicense from a particular copyright holder is reinstated (a)\nprovisionally, unless and until the copyright holder explicitly and\nfinally terminates your license, and (b) permanently, if the copyright\nholder fails to notify you of the violation by some reasonable means\nprior to 60 days after the cessation.\n\n  Moreover, your license from a particular copyright holder is\nreinstated permanently if the copyright holder notifies you of the\nviolation by some reasonable means, this is the first time you have\nreceived notice of violation of this License (for any work) from that\ncopyright holder, and you cure the violation prior to 30 days after\nyour receipt of the notice.\n\n  Termination of your rights under this section does not terminate the\nlicenses of parties who have received copies or rights from you under\nthis License.  If your rights have been terminated and not permanently\nreinstated, you do not qualify to receive new licenses for the same\nmaterial under section 10.\n\n  9. Acceptance Not Required for Having Copies.\n\n  You are not required to accept this License in order to receive or\nrun a copy of the Program.  Ancillary propagation of a covered work\noccurring solely as a consequence of using peer-to-peer transmission\nto receive a copy likewise does not require acceptance.  However,\nnothing other than this License grants you permission to propagate or\nmodify any covered work.  These actions infringe copyright if you do\nnot accept this License.  Therefore, by modifying or propagating a\ncovered work, you indicate your acceptance of this License to do so.\n\n  10. Automatic Licensing of Downstream Recipients.\n\n  Each time you convey a covered work, the recipient automatically\nreceives a license from the original licensors, to run, modify and\npropagate that work, subject to this License.  You are not responsible\nfor enforcing compliance by third parties with this License.\n\n  An "entity transaction" is a transaction transferring control of an\norganization, or substantially all assets of one, or subdividing an\norganization, or merging organizations.  If propagation of a covered\nwork results from an entity transaction, each party to that\ntransaction who receives a copy of the work also receives whatever\nlicenses to the work the party\'s predecessor in interest had or could\ngive under the previous paragraph, plus a right to possession of the\nCorresponding Source of the work from the predecessor in interest, if\nthe predecessor has it or can get it with reasonable efforts.\n\n  You may not impose any further restrictions on the exercise of the\nrights granted or affirmed under this License.  For example, you may\nnot impose a license fee, royalty, or other charge for exercise of\nrights granted under this License, and you may not initiate litigation\n(including a cross-claim or counterclaim in a lawsuit) alleging that\nany patent claim is infringed by making, using, selling, offering for\nsale, or importing the Program or any portion of it.\n\n  11. Patents.\n\n  A "contributor" is a copyright holder who authorizes use under this\nLicense of the Program or a work on which the Program is based.  The\nwork thus licensed is called the contributor\'s "contributor version".\n\n  A contributor\'s "essential patent claims" are all patent claims\nowned or controlled by the contributor, whether already acquired or\nhereafter acquired, that would be infringed by some manner, permitted\nby this License, of making, using, or selling its contributor version,\nbut do not include claims that would be infringed only as a\nconsequence of further modification of the contributor version.  For\npurposes of this definition, "control" includes the right to grant\npatent sublicenses in a manner consistent with the requirements of\nthis License.\n\n  Each contributor grants you a non-exclusive, worldwide, royalty-free\npatent license under the contributor\'s essential patent claims, to\nmake, use, sell, offer for sale, import and otherwise run, modify and\npropagate the contents of its contributor version.\n\n  In the following three paragraphs, a "patent license" is any express\nagreement or commitment, however denominated, not to enforce a patent\n(such as an express permission to practice a patent or covenant not to\nsue for patent infringement).  To "grant" such a patent license to a\nparty means to make such an agreement or commitment not to enforce a\npatent against the party.\n\n  If you convey a covered work, knowingly relying on a patent license,\nand the Corresponding Source of the work is not available for anyone\nto copy, free of charge and under the terms of this License, through a\npublicly available network server or other readily accessible means,\nthen you must either (1) cause the Corresponding Source to be so\navailable, or (2) arrange to deprive yourself of the benefit of the\npatent license for this particular work, or (3) arrange, in a manner\nconsistent with the requirements of this License, to extend the patent\nlicense to downstream recipients.  "Knowingly relying" means you have\nactual knowledge that, but for the patent license, your conveying the\ncovered work in a country, or your recipient\'s use of the covered work\nin a country, would infringe one or more identifiable patents in that\ncountry that you have reason to believe are valid.\n\n  If, pursuant to or in connection with a single transaction or\narrangement, you convey, or propagate by procuring conveyance of, a\ncovered work, and grant a patent license to some of the parties\nreceiving the covered work authorizing them to use, propagate, modify\nor convey a specific copy of the covered work, then the patent license\nyou grant is automatically extended to all recipients of the covered\nwork and works based on it.\n\n  A patent license is "discriminatory" if it does not include within\nthe scope of its coverage, prohibits the exercise of, or is\nconditioned on the non-exercise of one or more of the rights that are\nspecifically granted under this License.  You may not convey a covered\nwork if you are a party to an arrangement with a third party that is\nin the business of distributing software, under which you make payment\nto the third party based on the extent of your activity of conveying\nthe work, and under which the third party grants, to any of the\nparties who would receive the covered work from you, a discriminatory\npatent license (a) in connection with copies of the covered work\nconveyed by you (or copies made from those copies), or (b) primarily\nfor and in connection with specific products or compilations that\ncontain the covered work, unless you entered into that arrangement,\nor that patent license was granted, prior to 28 March 2007.\n\n  Nothing in this License shall be construed as excluding or limiting\nany implied license or other defenses to infringement that may\notherwise be available to you under applicable patent law.\n\n  12. No Surrender of Others\' Freedom.\n\n  If conditions are imposed on you (whether by court order, agreement or\notherwise) that contradict the conditions of this License, they do not\nexcuse you from the conditions of this License.  If you cannot convey a\ncovered work so as to satisfy simultaneously your obligations under this\nLicense and any other pertinent obligations, then as a consequence you may\nnot convey it at all.  For example, if you agree to terms that obligate you\nto collect a royalty for further conveying from those to whom you convey\nthe Program, the only way you could satisfy both those terms and this\nLicense would be to refrain entirely from conveying the Program.\n\n  13. Use with the GNU Affero General Public License.\n\n  Notwithstanding any other provision of this License, you have\npermission to link or combine any covered work with a work licensed\nunder version 3 of the GNU Affero General Public License into a single\ncombined work, and to convey the resulting work.  The terms of this\nLicense will continue to apply to the part which is the covered work,\nbut the special requirements of the GNU Affero General Public License,\nsection 13, concerning interaction through a network will apply to the\ncombination as such.\n\n  14. Revised Versions of this License.\n\n  The Free Software Foundation may publish revised and/or new versions of\nthe GNU General Public License from time to time.  Such new versions will\nbe similar in spirit to the present version, but may differ in detail to\naddress new problems or concerns.\n\n  Each version is given a distinguishing version number.  If the\nProgram specifies that a certain numbered version of the GNU General\nPublic License "or any later version" applies to it, you have the\noption of following the terms and conditions either of that numbered\nversion or of any later version published by the Free Software\nFoundation.  If the Program does not specify a version number of the\nGNU General Public License, you may choose any version ever published\nby the Free Software Foundation.\n\n  If the Program specifies that a proxy can decide which future\nversions of the GNU General Public License can be used, that proxy\'s\npublic statement of acceptance of a version permanently authorizes you\nto choose that version for the Program.\n\n  Later license versions may give you additional or different\npermissions.  However, no additional obligations are imposed on any\nauthor or copyright holder as a result of your choosing to follow a\nlater version.\n\n  15. Disclaimer of Warranty.\n\n  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY\nAPPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT\nHOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY\nOF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,\nTHE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR\nPURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM\nIS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF\nALL NECESSARY SERVICING, REPAIR OR CORRECTION.\n\n  16. Limitation of Liability.\n\n  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING\nWILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS\nTHE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY\nGENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE\nUSE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF\nDATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD\nPARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),\nEVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF\nSUCH DAMAGES.\n\n  17. Interpretation of Sections 15 and 16.\n\n  If the disclaimer of warranty and limitation of liability provided\nabove cannot be given local legal effect according to their terms,\nreviewing courts shall apply local law that most closely approximates\nan absolute waiver of all civil liability in connection with the\nProgram, unless a warranty or assumption of liability accompanies a\ncopy of the Program in return for a fee.\n\n                     END OF TERMS AND CONDITIONS\n\n            How to Apply These Terms to Your New Programs\n\n  If you develop a new program, and you want it to be of the greatest\npossible use to the public, the best way to achieve this is to make it\nfree software which everyone can redistribute and change under these terms.\n\n  To do so, attach the following notices to the program.  It is safest\nto attach them to the start of each source file to most effectively\nstate the exclusion of warranty; and each file should have at least\nthe "copyright" line and a pointer to where the full notice is found.\n\n    <one line to give the program\'s name and a brief idea of what it does.>\n    Copyright (C) <year>  <name of author>\n\n    This program is free software: you can redistribute it and/or modify\n    it under the terms of the GNU General Public License as published by\n    the Free Software Foundation, either version 3 of the License, or\n    (at your option) any later version.\n\n    This program is distributed in the hope that it will be useful,\n    but WITHOUT ANY WARRANTY; without even the implied warranty of\n    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n    GNU General Public License for more details.\n\n    You should have received a copy of the GNU General Public License\n    along with this program.  If not, see <https://www.gnu.org/licenses/>.\n\nAlso add information on how to contact you by electronic and paper mail.\n\n  If the program does terminal interaction, make it output a short\nnotice like this when it starts in an interactive mode:\n\n    <program>  Copyright (C) <year>  <name of author>\n    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w\'.\n    This is free software, and you are welcome to redistribute it\n    under certain conditions; type `show c\' for details.\n\nThe hypothetical commands `show w\' and `show c\' should show the appropriate\nparts of the General Public License.  Of course, your program\'s commands\nmight be different; for a GUI interface, you would use an "about box".\n\n  You should also get your employer (if you work as a programmer) or school,\nif any, to sign a "copyright disclaimer" for the program, if necessary.\nFor more information on this, and how to apply and follow the GNU GPL, see\n<https://www.gnu.org/licenses/>.\n\n  The GNU General Public License does not permit incorporating your program\ninto proprietary programs.  If your program is a subroutine library, you\nmay consider it more useful to permit linking proprietary applications with\nthe library.  If this is what you want to do, use the GNU Lesser General\nPublic License instead of this License.  But first, please read\n<https://www.gnu.org/licenses/why-not-lgpl.html>.\n\n'


if __name__ == "__main__":
    mp.freeze_support()
    main()

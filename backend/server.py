from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import numpy as np
import json
import io
import csv

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class MaterialProperties(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name_en: str
    name_ua: str
    grade: str
    E_GPa: float
    uts_MPa: float
    yield_MPa: float
    poisson: float
    density_kg_m3: float
    elongation_pct: float
    category: str = "both"
    is_default: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class MaterialCreate(BaseModel):
    name_en: str
    name_ua: str
    grade: str
    E_GPa: float
    uts_MPa: float
    yield_MPa: float
    poisson: float
    density_kg_m3: float
    elongation_pct: float
    category: str = "both"

class MaterialUpdate(BaseModel):
    name_en: Optional[str] = None
    name_ua: Optional[str] = None
    grade: Optional[str] = None
    E_GPa: Optional[float] = None
    uts_MPa: Optional[float] = None
    yield_MPa: Optional[float] = None
    poisson: Optional[float] = None
    density_kg_m3: Optional[float] = None
    elongation_pct: Optional[float] = None
    category: Optional[str] = None

class InterfaceParams(BaseModel):
    diffusion_layer_thickness_um: float = 5.0
    quality_coefficient: float = 0.85
    kirkendall_porosity: float = 0.05
    intermetallic_thickness_um: float = 2.0
    intermetallic_E_GPa: float = 150.0
    intermetallic_strength_MPa: float = 200.0
    intermetallic_elongation_pct: float = 0.5
    # Oxide layer model (non-vacuum welding)
    welding_atmosphere: str = "vacuum"
    oxide_thickness_nm: float = 0.0
    oxide_E_GPa: float = 250.0
    oxide_strength_MPa: float = 150.0
    oxide_elongation_pct: float = 0.05
    oxide_continuity: float = 0.0

class CalculationRequest(BaseModel):
    plate_material_id: str
    interlayer_material_id: str
    plate_thickness_mm: float = 2.0
    plate_width_mm: float = 10.0
    plate_length_mm: float = 50.0
    interlayer_thickness_um: float = 20.0
    interface_params: InterfaceParams = InterfaceParams()
    name: str = ""

class ExportRequest(BaseModel):
    calculation_id: str

class ParametricRequest(BaseModel):
    plate_material_id: str
    interlayer_material_id: str
    plate_thickness_mm: float = 2.0
    plate_width_mm: float = 10.0
    plate_length_mm: float = 50.0
    interlayer_thickness_um: float = 20.0
    interface_params: InterfaceParams = InterfaceParams()
    vary_parameter: str
    vary_min: float
    vary_max: float
    vary_steps: int = 20

# =============================================================================
# DEFAULT MATERIALS
# =============================================================================

DEFAULT_MATERIALS = [
    {
        "name_en": "MA2-1 (Mg-Al-Zn alloy)", "name_ua": "МА2-1 (Mg-Al-Zn сплав)",
        "grade": "MA2-1", "E_GPa": 45.0, "uts_MPa": 145.0, "yield_MPa": 85.0,
        "poisson": 0.35, "density_kg_m3": 1780, "elongation_pct": 8.0, "category": "both"
    },
    {
        "name_en": "VT1-0 (CP Titanium Grade 1)", "name_ua": "ВТ1-0 (Чистий титан)",
        "grade": "VT1-0", "E_GPa": 103.0, "uts_MPa": 295.0, "yield_MPa": 175.0,
        "poisson": 0.34, "density_kg_m3": 4505, "elongation_pct": 25.0, "category": "interlayer"
    },
    {
        "name_en": "AMg6 (Al-Mg alloy)", "name_ua": "АМг6 (Al-Mg сплав)",
        "grade": "AMg6", "E_GPa": 71.0, "uts_MPa": 340.0, "yield_MPa": 170.0,
        "poisson": 0.33, "density_kg_m3": 2640, "elongation_pct": 15.0, "category": "both"
    },
    {
        "name_en": "VT6 (Ti-6Al-4V)", "name_ua": "ВТ6 (Ti-6Al-4V)",
        "grade": "VT6", "E_GPa": 114.0, "uts_MPa": 950.0, "yield_MPa": 880.0,
        "poisson": 0.342, "density_kg_m3": 4430, "elongation_pct": 10.0, "category": "both"
    },
    {
        "name_en": "D16 (Al-Cu alloy, 2024-T4)", "name_ua": "Д16 (Al-Cu сплав, 2024-T4)",
        "grade": "D16", "E_GPa": 72.0, "uts_MPa": 440.0, "yield_MPa": 290.0,
        "poisson": 0.33, "density_kg_m3": 2780, "elongation_pct": 17.0, "category": "both"
    },
    {
        "name_en": "Steel 12Kh18N10T (AISI 321)", "name_ua": "Сталь 12Х18Н10Т (AISI 321)",
        "grade": "12Kh18N10T", "E_GPa": 200.0, "uts_MPa": 510.0, "yield_MPa": 200.0,
        "poisson": 0.3, "density_kg_m3": 7900, "elongation_pct": 40.0, "category": "both"
    },
    {
        "name_en": "Pure Copper M1", "name_ua": "Чиста мідь М1",
        "grade": "M1", "E_GPa": 130.0, "uts_MPa": 210.0, "yield_MPa": 60.0,
        "poisson": 0.34, "density_kg_m3": 8940, "elongation_pct": 45.0, "category": "interlayer"
    },
    {
        "name_en": "Nickel N2", "name_ua": "Нікель Н2",
        "grade": "N2", "E_GPa": 200.0, "uts_MPa": 400.0, "yield_MPa": 100.0,
        "poisson": 0.31, "density_kg_m3": 8900, "elongation_pct": 40.0, "category": "interlayer"
    },
    {
        "name_en": "Pure Annealed Titanium (Grade 2)", "name_ua": "Чистий відпалений титан (Grade 2)",
        "grade": "CP-Ti-2", "E_GPa": 105.0, "uts_MPa": 345.0, "yield_MPa": 275.0,
        "poisson": 0.34, "density_kg_m3": 4510, "elongation_pct": 20.0, "category": "both"
    },
    {
        "name_en": "Pure Zinc Foil (Zn99.95)", "name_ua": "Чиста цинкова фольга (Zn99.95)",
        "grade": "Zn99", "E_GPa": 108.0, "uts_MPa": 37.0, "yield_MPa": 30.0,
        "poisson": 0.25, "density_kg_m3": 7140, "elongation_pct": 60.0, "category": "interlayer"
    },
]


# Oxide presets by base metal (typical native oxide properties)
OXIDE_PRESETS = {
    "Mg": {"oxide_name": "MgO", "E_GPa": 250.0, "strength_MPa": 150.0, "elongation_pct": 0.05},
    "Ti": {"oxide_name": "TiO2", "E_GPa": 230.0, "strength_MPa": 250.0, "elongation_pct": 0.1},
    "Al": {"oxide_name": "Al2O3", "E_GPa": 370.0, "strength_MPa": 350.0, "elongation_pct": 0.02},
    "Fe": {"oxide_name": "Cr2O3/FeO", "E_GPa": 250.0, "strength_MPa": 200.0, "elongation_pct": 0.03},
    "Cu": {"oxide_name": "CuO/Cu2O", "E_GPa": 80.0, "strength_MPa": 100.0, "elongation_pct": 0.1},
    "Ni": {"oxide_name": "NiO", "E_GPa": 200.0, "strength_MPa": 180.0, "elongation_pct": 0.05},
    "Zn": {"oxide_name": "ZnO", "E_GPa": 140.0, "strength_MPa": 120.0, "elongation_pct": 0.08}
}

GRADE_TO_BASE_METAL = {
    "MA2-1": "Mg", "VT1-0": "Ti", "VT6": "Ti", "CP-Ti-2": "Ti",
    "AMg6": "Al", "D16": "Al", "12Kh18N10T": "Fe", "M1": "Cu", "N2": "Ni", "Zn99": "Zn",
}

ATMOSPHERE_DEFAULTS = {
    "vacuum":  {"oxide_thickness_nm": 0, "oxide_continuity": 0.0},
    "argon":   {"oxide_thickness_nm": 15, "oxide_continuity": 0.3},
    "air":     {"oxide_thickness_nm": 80, "oxide_continuity": 0.85},
}

# =============================================================================
# CALCULATION ENGINE
# =============================================================================

def calculate_sandwich_models(plate_mat, inter_mat, config, iface):
    # Detect same-material sandwich — no intermetallics or Kirkendall possible
    same_material = plate_mat.get('grade') == inter_mat.get('grade')
    if same_material:
        iface = dict(iface)
        iface['intermetallic_thickness_um'] = 0
        iface['intermetallic_E_GPa'] = plate_mat['E_GPa']
        iface['intermetallic_strength_MPa'] = plate_mat['uts_MPa']
        iface['intermetallic_elongation_pct'] = plate_mat['elongation_pct']
        iface['kirkendall_porosity'] = 0

    t_plate = config['plate_thickness_mm']
    t_inter = config['interlayer_thickness_um'] / 1000.0
    t_diff = iface['diffusion_layer_thickness_um'] / 1000.0
    t_imc = iface['intermetallic_thickness_um'] / 1000.0
    eta = iface['quality_coefficient']
    porosity = iface['kirkendall_porosity']
    L = config['plate_length_mm']

    E_p = plate_mat['E_GPa']
    E_i = inter_mat['E_GPa']
    uts_p = plate_mat['uts_MPa']
    uts_i = inter_mat['uts_MPa']
    y_p = plate_mat['yield_MPa']
    y_i = inter_mat['yield_MPa']
    nu_p = plate_mat['poisson']
    nu_i = inter_mat['poisson']

    if same_material:
        E_diff = E_p
        uts_diff = uts_p
        y_diff = y_p
    else:
        E_diff = (E_p + E_i) / 2.0 * 0.85
        uts_diff = min(uts_p, uts_i) * 0.75
        y_diff = min(y_p, y_i) * 0.75

    E_imc = iface['intermetallic_E_GPa']
    uts_imc = iface['intermetallic_strength_MPa']
    y_imc = uts_imc * 0.95
    elong_imc = iface['intermetallic_elongation_pct']

    # Oxide layer
    oxide_cont = iface.get('oxide_continuity', 0.0)
    oxide_t_nm = iface.get('oxide_thickness_nm', 0.0)
    t_oxide = oxide_t_nm / 1e6 * oxide_cont  # effective thickness in mm
    E_oxide = iface.get('oxide_E_GPa', 250.0)
    uts_oxide = iface.get('oxide_strength_MPa', 150.0)
    y_oxide = uts_oxide * 0.95
    elong_oxide = iface.get('oxide_elongation_pct', 0.05)

    # Structure: plate | diff | imc | oxide | interlayer | oxide | imc | diff | plate
    t_total = 2 * t_plate + 2 * t_diff + 2 * t_imc + 2 * t_oxide + t_inter
    if t_total <= 0:
        t_total = 1e-6

    f_plate = 2 * t_plate / t_total
    f_inter = t_inter / t_total
    f_diff = 2 * t_diff / t_total
    f_imc = 2 * t_imc / t_total
    f_oxide = 2 * t_oxide / t_total

    # --- MODEL 1: Modified Rule of Mixtures with nonlinear interaction term ---
    # Nonlinear interaction coefficient based on elastic mismatch
    E_ratio = max(E_p, E_i) / max(min(E_p, E_i), 1.0)
    mismatch_factor = 1.0 - 0.08 * np.log(E_ratio)  # logarithmic penalty for mismatch
    
    # Nonlinear porosity degradation (exponential decay)
    porosity_factor = np.exp(-2.5 * porosity) if porosity > 0 else 1.0
    
    # Nonlinear quality degradation (sigmoid-like)
    quality_factor = 1.0 / (1.0 + np.exp(-12.0 * (eta - 0.5)))
    quality_factor = 0.3 + 0.7 * quality_factor  #

    uts_mrom = uts_rom_base * mismatch_factor * quality_factor * porosity_factor
    y_mrom = y_rom_base * mismatch_factor * quality_factor * porosity_factor

    model_mrom = {
        'name': 'modified_rom',
        'uts_mpa': round(float(uts_mrom), 2),
        'yield_mpa': round(float(y_mrom), 2),
        'description_en': 'Modified Rule of Mixtures with nonlinear interface quality and porosity correction',
        'description_ua': 'Модифіковане правило сумішей з нелінійною корекцією якості з\'єднання та пористості'
    }

   # --- MODEL 2: Shear-Lag with nonlinear stress transfer ---
    G_p = E_p / (2 * (1 + nu_p))
    G_i = E_i / (2 * (1 + nu_i))
    G_interface = min(G_p, G_i) * eta * (1.0 - oxide_cont * 0.5)
    t_eff_inter = t_inter + 2 * t_diff + 2 * t_imc + 2 * t_oxide

    if t_plate > 0 and t_eff_inter > 0 and G_interface > 0 and E_i > 0:
        # Enhanced shear-lag with nonlinear aspect ratio effect
        aspect_ratio = L / (2 * t_plate) if t_plate > 0 else 100
        beta = float(np.sqrt(G_interface / (E_i * t_plate * t_eff_inter)))
        arg = beta * L / 2.0
        
        if 0 < arg < 500:
            sl_efficiency = float(np.tanh(arg) / arg)
        elif arg >= 500:
            sl_efficiency = 0.0
        else:
            sl_efficiency = 1.0
        
        # Nonlinear aspect ratio correction (short specimens have stress concentration at ends)
        ar_correction = 1.0 - 0.15 * np.exp(-aspect_ratio / 10.0)
        sl_efficiency *= ar_correction
    else:
        sl_efficiency = 1.0

    # Nonlinear porosity effect with percolation threshold
    porosity_sl = 1.0 - porosity ** 0.7 if porosity > 0 else 1.0
    
    uts_sl = (f_plate * uts_p + (f_inter * uts_i + f_diff * uts_diff + f_imc * uts_imc + f_oxide * uts_oxide) * sl_efficiency) * quality_factor * porosity_sl
    y_sl = (f_plate * y_p + (f_inter * y_i + f_diff * y_diff + f_imc * y_imc + f_oxide * y_oxide) * sl_efficiency) * quality_factor * porosity_sl

    model_sl = {
        'name': 'shear_lag',
        'uts_mpa': round(float(uts_sl), 2),
        'yield_mpa': round(float(y_sl), 2),
        'efficiency': round(float(sl_efficiency), 4),
        'description_en': 'Shear-Lag model with nonlinear aspect ratio and stress transfer correction',
        'description_ua': 'Модель зсувного відставання з нелінійною корекцією передачі навантаження'
    }

    # --- MODEL 3: First Ply Failure ---
    layers = [
        {'name': 'plate', 'E': E_p, 'uts': uts_p, 'yield': y_p, 't': 2 * t_plate},
        {'name': 'interlayer', 'E': E_i, 'uts': uts_i, 'yield': y_i, 't': t_inter},
        {'name': 'diffusion', 'E': E_diff, 'uts': uts_diff, 'yield': y_diff, 't': 2 * t_diff},
        {'name': 'intermetallic', 'E': E_imc, 'uts': uts_imc, 'yield': y_imc, 't': 2 * t_imc},
    ]
    if t_oxide > 0 and oxide_cont > 0:
        # Oxide effective properties blended with base by continuity
        E_ox_eff = E_oxide * oxide_cont + ((E_p + E_i) / 2.0) * (1.0 - oxide_cont)
        uts_ox_eff = uts_oxide * oxide_cont + min(uts_p, uts_i) * (1.0 - oxide_cont)
        y_ox_eff = y_oxide * oxide_cont + min(y_p, y_i) * (1.0 - oxide_cont)
        layers.append({'name': 'oxide', 'E': E_ox_eff, 'uts': uts_ox_eff, 'yield': y_ox_eff, 't': 2 * t_oxide})

    Et_total = sum(l['E'] * l['t'] for l in layers if l['t'] > 0)

    min_uts_fpf = float('inf')
    min_y_fpf = float('inf')
    fpf_layer = 'none'

    for l in layers:
        if l['E'] > 0 and l['t'] > 0:
            eps_uts = l['uts'] / (l['E'] * 1000.0)
            eps_yield = l['yield'] / (l['E'] * 1000.0)
            sigma_uts = eps_uts * Et_total * 1000.0 / t_total if t_total > 0 else 0
            sigma_yield = eps_yield * Et_total * 1000.0 / t_total if t_total > 0 else 0
if sigma_uts < min_uts_fpf:
                min_uts_fpf = sigma_uts
                fpf_layer = l['name']
            min_y_fpf = min(min_y_fpf, sigma_yield)

if sigma_uts < min_uts_fpf:
                min_uts_fpf = sigma_uts
                fpf_layer = l['name']
            min_y_fpf = min(min_y_fpf, sigma_yield)

    uts_fpf = min_uts_fpf * eta * (1.0 - porosity) if min_uts_fpf < float('inf') else 0
    y_fpf = min_y_fpf * eta * (1.0 - porosity) if min_y_fpf < float('inf') else 0

    model_fpf = {
        'name': 'fpf',
        'uts_mpa': round(float(uts_fpf), 2),
        'yield_mpa': round(float(y_fpf), 2),
        'critical_layer': fpf_layer,
        'description_en': f'First Ply Failure criterion - critical layer: {fpf_layer}',
        'description_ua': f'Критерій першого руйнування шару - критичний шар: {fpf_layer}'
    }

    # --- MODEL 4: Kirkendall-Corrected ---
    K_t = 1.0 + 2.0 * float(np.sqrt(porosity / np.pi)) if porosity > 0 else 1.0
    A_eff = 1.0 - porosity

    uts_kirk = uts_rom_base * eta * A_eff / K_t
    y_kirk = y_rom_base * eta * A_eff / K_t

    model_kirk = {
        'name': 'kirkendall',
        'uts_mpa': round(float(uts_kirk), 2),
        'yield_mpa': round(float(y_kirk), 2),
        'stress_concentration_factor': round(float(K_t), 4),
        'description_en': 'Kirkendall porosity model with stress concentration at voids',
        'description_ua': 'Модель пористості Кіркендала з концентрацією напружень біля пор'
    }

    # --- MODEL 5: Comprehensive ---
    uts_comp = min(uts_mrom, uts_sl, uts_fpf, uts_kirk)
    y_comp = min(y_mrom, y_sl, y_fpf, y_kirk)

    if fpf_layer == 'intermetallic' and Et_total > 0:
        remaining_Et = Et_total - E_imc * 2 * t_imc
        t_remaining = t_total - 2 * t_imc
        if t_remaining > 0 and remaining_Et > 0:
            eps_imc_fail = uts_imc / (E_imc * 1000.0)
            sigma_remaining = eps_imc_fail * remaining_Et * 1000.0 / t_remaining
            uts_post_fpf = sigma_remaining * t_remaining / t_total * eta * (1.0 - porosity) * 0.9
            uts_comp = max(uts_comp, float(uts_post_fpf))

    # Oxide FPF: if oxide is the first to fail, check post-oxide residual
    if fpf_layer == 'oxide' and Et_total > 0 and t_oxide > 0:
        remaining_Et = Et_total - E_ox_eff * 2 * t_oxide if oxide_cont > 0 else Et_total
        t_remaining = t_total - 2 * t_oxide
        if t_remaining > 0 and remaining_Et > 0:
            eps_ox_fail = uts_ox_eff / (E_ox_eff * 1000.0) if oxide_cont > 0 else 1.0
            sigma_remaining = eps_ox_fail * remaining_Et * 1000.0 / t_remaining
            uts_post_oxide = sigma_remaining * t_remaining / t_total * eta * (1.0 - porosity) * 0.85
            uts_comp = max(uts_comp, float(uts_post_oxide))
}

    # --- MODEL 5: Comprehensive ---
    uts_comp = min(uts_mrom, uts_sl, uts_fpf, uts_kirk)
    y_comp = min(y_mrom, y_sl, y_fpf, y_kirk)

    if fpf_layer == 'intermetallic' and Et_total > 0:
        remaining_Et = Et_total - E_imc * 2 * t_imc
        t_remaining = t_total - 2 * t_imc
        if t_remaining > 0 and remaining_Et > 0:
            eps_imc_fail = uts_imc / (E_imc * 1000.0)
            sigma_remaining = eps_imc_fail * remaining_Et * 1000.0 / t_remaining
            uts_post_fpf = sigma_remaining * t_remaining / t_total * eta * (1.0 - porosity) * 0.9
            uts_comp = max(uts_comp, float(uts_post_fpf))

    # Oxide FPF: if oxide is the first to fail, check post-oxide residual
    if fpf_layer == 'oxide' and Et_total > 0 and t_oxide > 0:
        remaining_Et = Et_total - E_ox_eff * 2 * t_oxide if oxide_cont > 0 else Et_total
        t_remaining = t_total - 2 * t_oxide
        if t_remaining > 0 and remaining_Et > 0:
            eps_ox_fail = uts_ox_eff / (E_ox_eff * 1000.0) if oxide_cont > 0 else 1.0
            sigma_remaining = eps_ox_fail * remaining_Et * 1000.0 / t_remaining
            uts_post_oxide = sigma_remaining * t_remaining / t_total * eta * (1.0 - porosity) * 0.85
            uts_comp = max(uts_comp, float(uts_post_oxide))

    model_comp = {
        'name': 'comprehensive',
        'uts_mpa': round(float(uts_comp), 2),
        'yield_mpa': round(float(y_comp), 2),
        'description_en': 'Comprehensive model - conservative envelope with progressive damage assessment',
        'description_ua': 'Комплексна модель - консервативна оболонка з оцінкою прогресивного пошкодження'
    }

    return [model_mrom, model_sl, model_fpf, model_kirk, model_comp]


def calculate_stress_distribution(plate_mat, inter_mat, iface):
    # Same-material: no intermetallic, no Kirkendall, no composition gradient
    same_material = plate_mat.get('grade') == inter_mat.get('grade')
    if same_material:
        iface = dict(iface)
        iface['intermetallic_thickness_um'] = 0
        iface['kirkendall_porosity'] = 0

    t_diff = iface['diffusion_layer_thickness_um']
    t_imc = iface['intermetallic_thickness_um']
    eta = iface['quality_coefficient']
    porosity = iface['kirkendall_porosity']

    # Oxide
    oxide_cont = iface.get('oxide_continuity', 0.0)
    oxide_t_nm = iface.get('oxide_thickness_nm', 0.0)
    t_oxide_um = oxide_t_nm / 1000.0 * oxide_cont  # effective thickness in μm
    E_oxide = iface.get('oxide_E_GPa', 250.0)

    E_p = plate_mat['E_GPa']
    E_i = inter_mat['E_GPa']
    E_imc = iface['intermetallic_E_GPa']
    E_diff = E_p if same_material else (E_p + E_i) / 2.0 * 0.85

    eps_ref = plate_mat['yield_MPa'] * 0.6 / (E_p * 1000.0)

    positions = np.linspace(-30.0, 30.0, 241)
    data = []

    exp_norm = 1.0 - np.exp(-3.0)
    if abs(exp_norm) < 1e-12:
        exp_norm = 1.0

    # Zone boundaries from center outward:
    # |oxide| imc | diffusion | plate/interlayer
    half_oxide = t_oxide_um / 2.0
    half_imc = t_imc / 2.0
    bnd_oxide = half_oxide
    bnd_imc = half_oxide + half_imc
    bnd_diff_plate = half_oxide + half_imc + t_diff

    for x in positions:
        xv = float(x)
        ax = abs(xv)

        if ax < bnd_oxide and t_oxide_um > 0:
            # Oxide zone
            E_base_ox = E_imc if t_imc > 0 else (E_p if xv < 0 else E_i)
            E_local = E_oxide * oxide_cont + E_base_ox * (1.0 - oxide_cont)
            zone = "oxide"
        elif ax < bnd_imc and half_imc > 0:
            # Intermetallic zone
            E_local = E_imc
            zone = "intermetallic"
        elif ax < bnd_diff_plate and t_diff > 0:
            # Diffusion zone (exponential transition)
            frac = (ax - bnd_imc) / t_diff if t_diff > 0 else 1.0
            frac = max(0.0, min(1.0, frac))
            if xv < 0:
                E_local = E_diff + (E_p - E_diff) * (1.0 - np.exp(-3.0 * frac)) / exp_norm
            else:
                E_local = E_diff + (E_i - E_diff) * (1.0 - np.exp(-3.0 * frac)) / exp_norm
            zone = "diffusion"
        else:
            # Bulk material
            if xv < 0:
                E_local = E_p
                zone = "plate"
            else:
                E_local = E_i
                zone = "interlayer"

        sigma = float(E_local) * 1000.0 * eps_ref * eta

        if porosity > 0:
            kirk_plane = -t_diff / 3.0
            pore_width = max(t_diff / 2.0, 1.0)
            pore_factor = porosity * np.exp(-((xv - kirk_plane) ** 2) / (2.0 * pore_width ** 2))
            K_local = 1.0 + 2.0 * float(pore_factor)
            sigma *= K_local

        data.append({
            "position_um": round(xv, 2),
            "stress_mpa": round(float(sigma), 3),
            "zone": zone,
            "E_local_GPa": round(float(E_local), 2)
        })

    return data


def estimate_additional_properties(plate_mat, inter_mat, config, iface, models):
    """Estimate grain size, hardness, elongation, and bend angle for the sandwich."""
    same_material = plate_mat.get('grade') == inter_mat.get('grade')
    if same_material:
        iface = dict(iface)
        iface['intermetallic_thickness_um'] = 0
        iface['intermetallic_elongation_pct'] = plate_mat['elongation_pct']
        iface['kirkendall_porosity'] = 0

    t_plate = config['plate_thickness_mm']
    t_inter = config['interlayer_thickness_um'] / 1000.0
    t_diff = iface['diffusion_layer_thickness_um'] / 1000.0
    t_imc = iface['intermetallic_thickness_um'] / 1000.0
    t_total = 2 * t_plate + 2 * t_diff + 2 * t_imc + t_inter
    eta = iface['quality_coefficient']
    porosity = iface['kirkendall_porosity']

    # Oxide in additional properties
    oxide_cont = iface.get('oxide_continuity', 0.0)
    oxide_t_nm = iface.get('oxide_thickness_nm', 0.0)
    t_oxide = oxide_t_nm / 1e6 * oxide_cont
    t_total += 2 * t_oxide

    if t_total <= 0:
        t_total = 1e-6

    # Effective elongation (plasticity) — limited by FPF of least ductile layer
    elong_imc = iface['intermetallic_elongation_pct']
    elong_diff = min(plate_mat['elongation_pct'], inter_mat['elongation_pct']) * 0.5
    min_elong = min(
        plate_mat['elongation_pct'],
        inter_mat['elongation_pct'],
        elong_diff if t_diff > 0 else 999.0,
        elong_imc if t_imc > 0 else 999.0,
        iface.get('oxide_elongation_pct', 999.0) if oxide_cont > 0 and t_oxide > 0 else 999.0,
    )
    elong_eff = min_elong * eta * (1.0 - porosity)

    # Grain size in interface zone (inverse Hall-Petch approximation + diffusion growth)
    d0_plate = 200.0 / max(plate_mat['uts_MPa'], 1) * 30.0
    d0_inter = 200.0 / max(inter_mat['uts_MPa'], 1) * 30.0
    t_diff_um = iface['diffusion_layer_thickness_um']
    d_interface = (d0_plate + d0_inter) / 2.0 * (1.0 + 0.04 * np.sqrt(max(t_diff_um, 0.1)))

    # Hardness (Vickers) via Tabor relation: HV ~ sigma_y / 2.75
    comp_yield = min(m['yield_mpa'] for m in models)
    hv_composite = comp_yield / 2.75
    hv_imc = iface['intermetallic_strength_MPa'] * 0.95 / 2.75
    f_imc = 2.0 * t_imc / t_total
    hv_avg = hv_composite * (1.0 - f_imc) + hv_imc * f_imc

    # Bend angle estimation from elongation and geometry
    eps_f = max(elong_eff / 100.0, 0.001)
    R_min = t_total / (2.0 * eps_f)
    L = config['plate_length_mm']
    bend_angle = min(180.0, float(2.0 * np.degrees(np.arctan2(L, 2.0 * R_min))))

    return {
        'elongation_pct': round(float(elong_eff), 2),
        'grain_size_um': round(float(d_interface), 2),
        'hardness_hv': round(float(hv_avg), 1),
        'bend_angle_deg': round(float(bend_angle), 1),
    }


# =============================================================================
# MATERIAL ROUTES
# =============================================================================

@api_router.get("/materials", response_model=List[dict])
async def get_materials():
    docs = await db.materials.find({}, {"_id": 0}).to_list(1000)
    return docs

@api_router.get("/materials/{material_id}")
async def get_material(material_id: str):
    doc = await db.materials.find_one({"id": material_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Material not found")
    return doc

@api_router.post("/materials", status_code=201)
async def create_material(data: MaterialCreate):
    mat = MaterialProperties(**data.model_dump(), is_default=False)
    doc = mat.model_dump()
    await db.materials.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.put("/materials/{material_id}")
async def update_material(material_id: str, data: MaterialUpdate):
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = await db.materials.update_one({"id": material_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Material not found")
    doc = await db.materials.find_one({"id": material_id}, {"_id": 0})
    return doc

@api_router.delete("/materials/{material_id}")
async def delete_material(material_id: str):
    result = await db.materials.delete_one({"id": material_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Material not found")
    return {"status": "deleted"}

# =============================================================================
# CALCULATION ROUTES
# =============================================================================

@api_router.post("/calculate")
async def run_calculation(req: CalculationRequest):
    plate_mat = await db.materials.find_one({"id": req.plate_material_id}, {"_id": 0})
    inter_mat = await db.materials.find_one({"id": req.interlayer_material_id}, {"_id": 0})

    if not plate_mat:
        raise HTTPException(status_code=404, detail="Plate material not found")
    if not inter_mat:
        raise HTTPException(status_code=404, detail="Interlayer material not found")

    config = {
        "plate_thickness_mm": req.plate_thickness_mm,
        "plate_width_mm": req.plate_width_mm,
        "plate_length_mm": req.plate_length_mm,
        "interlayer_thickness_um": req.interlayer_thickness_um,
    }
    iface = req.interface_params.model_dump()

    models = calculate_sandwich_models(plate_mat, inter_mat, config, iface)
    stress_dist = calculate_stress_distribution(plate_mat, inter_mat, iface)

    calc_id = str(uuid.uuid4())
    calc_name = req.name or f"Calc-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    additional = estimate_additional_properties(plate_mat, inter_mat, config, iface, models)

    same_mat = plate_mat.get('grade') == inter_mat.get('grade')
    notes = []
    if same_mat:
        notes.append({
            "type": "same_material",
            "en": f"Same material detected ({plate_mat.get('grade')}): intermetallic formation and Kirkendall porosity are physically impossible and have been suppressed in all models.",
            "ua": f"Виявлено однаковий матеріал ({plate_mat.get('grade')}): утворення інтерметалідів та пористість Кіркендала фізично неможливі і були виключені з усіх моделей."
        })

    oxide_cont = iface.get('oxide_continuity', 0)
    oxide_t = iface.get('oxide_thickness_nm', 0)
    atm = iface.get('welding_atmosphere', 'vacuum')
    if oxide_cont > 0 and oxide_t > 0:
        notes.append({
            "type": "oxide_active",
            "en": f"Oxide layer model active ({atm} atmosphere): {oxide_t:.0f} nm oxide film, continuity = {oxide_cont:.0%}. Oxide reduces bond strength and may cause early FPF.",
            "ua": f"Модель оксидного шару активна (атмосфера: {atm}): оксидна плівка {oxide_t:.0f} нм, суцільність = {oxide_cont:.0%}. Оксид знижує міцність та може викликати раннє руйнування."
        })

    result = {
        "id": calc_id,
        "name": calc_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "interface_params": iface,
        "plate_material": {k: v for k, v in plate_mat.items() if k != '_id'},
        "interlayer_material": {k: v for k, v in inter_mat.items() if k != '_id'},
        "models": models,
        "additional_properties": additional,
        "notes": notes,
        "stress_distribution": stress_dist,
    }

    await db.calculations.insert_one({**result, "_mongo_id": True})
    result.pop("_mongo_id", None)
    return result

@api_router.get("/calculations")
async def get_calculations():
    docs = await db.calculations.find({}, {"_id": 0, "stress_distribution": 0, "_mongo_id": 0}).sort("timestamp", -1).to_list(100)
    return docs

@api_router.get("/calculations/{calc_id}")
async def get_calculation(calc_id: str):
    doc = await db.calculations.find_one({"id": calc_id}, {"_id": 0, "_mongo_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Calculation not found")
    return doc

@api_router.delete("/calculations/{calc_id}")
async def delete_calculation(calc_id: str):
    result = await db.calculations.delete_one({"id": calc_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Calculation not found")
    return {"status": "deleted"}

# =============================================================================
# EXPORT ROUTES
# =============================================================================

@api_router.post("/export/pdf")
async def export_pdf(req: ExportRequest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    doc = await db.calculations.find_one({"id": req.calculation_id}, {"_id": 0, "_mongo_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Calculation not found")

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: Summary
        fig1, ax1 = plt.subplots(figsize=(8.5, 11))
        ax1.axis('off')
        fig1.patch.set_facecolor('#09090b')

        summary_lines = [
            f"DiffBond Analyzer Report",
            f"",
            f"Calculation: {doc['name']}",
            f"Date: {doc['timestamp'][:19]}",
            f"",
            f"Plate Material: {doc['plate_material']['grade']}",
            f"Interlayer Material: {doc['interlayer_material']['grade']}",
            f"Plate Thickness: {doc['config']['plate_thickness_mm']} mm",
            f"Interlayer Thickness: {doc['config']['interlayer_thickness_um']} um",
            f"",
            f"Interface Parameters:",
            f"  Diffusion Layer: {doc['interface_params']['diffusion_layer_thickness_um']} um",
            f"  Quality Coefficient: {doc['interface_params']['quality_coefficient']}",
            f"  Kirkendall Porosity: {doc['interface_params']['kirkendall_porosity']}",
            f"  Intermetallic Thickness: {doc['interface_params']['intermetallic_thickness_um']} um",
            f"  Intermetallic E: {doc['interface_params']['intermetallic_E_GPa']} GPa",
            f"  Intermetallic Strength: {doc['interface_params']['intermetallic_strength_MPa']} MPa",
        ]
        for i, line in enumerate(summary_lines):
            fontsize = 16 if i == 0 else 11
            weight = 'bold' if i == 0 else 'normal'
            ax1.text(0.1, 0.95 - i * 0.045, line, transform=ax1.transAxes,
                    fontsize=fontsize, color='white', fontweight=weight,
                    fontfamily='monospace', verticalalignment='top')
        pdf.savefig(fig1, facecolor='#09090b')
        plt.close(fig1)

        # Page 2: Model comparison bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig2.patch.set_facecolor('#09090b')
        ax2.set_facecolor('#18181b')

        models = doc['models']
        names = [m['name'] for m in models]
        uts_vals = [m['uts_mpa'] for m in models]
        yield_vals = [m['yield_mpa'] for m in models]

        x = np.arange(len(names))
        width = 0.35
        ax2.bar(x - width/2, uts_vals, width, label='UTS (MPa)', color='#0ea5e9')
        ax2.bar(x + width/2, yield_vals, width, label='Yield (MPa)', color='#f43f5e')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=30, ha='right', color='#a1a1aa', fontsize=10)
        ax2.set_ylabel('Stress (MPa)', color='#a1a1aa')
        ax2.set_title('Model Comparison', color='white', fontsize=14, fontweight='bold')
        ax2.legend(facecolor='#18181b', edgecolor='#27272a', labelcolor='white')
        ax2.tick_params(colors='#71717a')
        ax2.spines['bottom'].set_color('#27272a')
        ax2.spines['left'].set_color('#27272a')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='y', color='#27272a', linestyle='--', alpha=0.5)
        plt.tight_layout()
        pdf.savefig(fig2, facecolor='#09090b')
        plt.close(fig2)

        # Page 3: Stress distribution
        if doc.get('stress_distribution'):
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            fig3.patch.set_facecolor('#09090b')
            ax3.set_facecolor('#18181b')

            sd = doc['stress_distribution']
            pos = [d['position_um'] for d in sd]
            stress = [d['stress_mpa'] for d in sd]

            ax3.plot(pos, stress, color='#0ea5e9', linewidth=2)
            ax3.set_xlabel('Position (um)', color='#a1a1aa')
            ax3.set_ylabel('Stress (MPa)', color='#a1a1aa')
            ax3.set_title('Stress Distribution in Interface Zone', color='white', fontsize=14, fontweight='bold')
            ax3.tick_params(colors='#71717a')
            ax3.spines['bottom'].set_color('#27272a')
            ax3.spines['left'].set_color('#27272a')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.grid(color='#27272a', linestyle='--', alpha=0.5)
            plt.tight_layout()
            pdf.savefig(fig3, facecolor='#09090b')
            plt.close(fig3)

    buf.seek(0)
    return StreamingResponse(buf, media_type='application/pdf',
                           headers={'Content-Disposition': f'attachment; filename=diffbond_report_{doc["id"][:8]}.pdf'})


@api_router.post("/export/csv")
async def export_csv_endpoint(req: ExportRequest):
    doc = await db.calculations.find_one({"id": req.calculation_id}, {"_id": 0, "_mongo_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Calculation not found")

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(['DiffBond Analyzer - Calculation Results'])
    writer.writerow([])
    writer.writerow(['Calculation Name', doc['name']])
    writer.writerow(['Date', doc['timestamp'][:19]])
    writer.writerow(['Plate Material', doc['plate_material']['grade']])
    writer.writerow(['Interlayer Material', doc['interlayer_material']['grade']])
    writer.writerow(['Plate Thickness (mm)', doc['config']['plate_thickness_mm']])
    writer.writerow(['Interlayer Thickness (um)', doc['config']['interlayer_thickness_um']])
    writer.writerow([])

    writer.writerow(['Interface Parameters'])
    for k, v in doc['interface_params'].items():
        writer.writerow([k, v])
    writer.writerow([])

    writer.writerow(['Model Comparison'])
    writer.writerow(['Model', 'UTS (MPa)', 'Yield (MPa)'])
    for m in doc['models']:
        writer.writerow([m['name'], m['uts_mpa'], m['yield_mpa']])
    writer.writerow([])

    if doc.get('stress_distribution'):
        writer.writerow(['Stress Distribution'])
        writer.writerow(['Position (um)', 'Stress (MPa)', 'Zone', 'E_local (GPa)'])
        for d in doc['stress_distribution']:
            writer.writerow([d['position_um'], d['stress_mpa'], d['zone'], d['E_local_GPa']])

    content = output.getvalue()
    return StreamingResponse(
        io.BytesIO(content.encode('utf-8-sig')),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename=diffbond_results_{doc["id"][:8]}.csv'}
    )


@api_router.post("/export/chart")
async def export_chart(req: ExportRequest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    doc = await db.calculations.find_one({"id": req.calculation_id}, {"_id": 0, "_mongo_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Calculation not found")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#09090b')

    # Left: Model comparison
    ax1.set_facecolor('#18181b')
    models = doc['models']
    names = [m['name'] for m in models]
    uts_vals = [m['uts_mpa'] for m in models]
    yield_vals = [m['yield_mpa'] for m in models]
    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x - width/2, uts_vals, width, label='UTS', color='#0ea5e9')
    ax1.bar(x + width/2, yield_vals, width, label='Yield', color='#f43f5e')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right', color='#a1a1aa', fontsize=9)
    ax1.set_ylabel('MPa', color='#a1a1aa')
    ax1.set_title('Model Comparison', color='white', fontsize=12)
    ax1.legend(facecolor='#18181b', edgecolor='#27272a', labelcolor='white')
    ax1.tick_params(colors='#71717a')
    for sp in ['top', 'right']:
        ax1.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax1.spines[sp].set_color('#27272a')
    ax1.grid(axis='y', color='#27272a', linestyle='--', alpha=0.5)

    # Right: Stress distribution
    ax2.set_facecolor('#18181b')
    if doc.get('stress_distribution'):
        sd = doc['stress_distribution']
        pos = [d['position_um'] for d in sd]
        stress = [d['stress_mpa'] for d in sd]
        ax2.plot(pos, stress, color='#0ea5e9', linewidth=2)
    ax2.set_xlabel('Position (um)', color='#a1a1aa')
    ax2.set_ylabel('Stress (MPa)', color='#a1a1aa')
    ax2.set_title('Stress Distribution', color='white', fontsize=12)
    ax2.tick_params(colors='#71717a')
    for sp in ['top', 'right']:
        ax2.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax2.spines[sp].set_color('#27272a')
    ax2.grid(color='#27272a', linestyle='--', alpha=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#09090b')
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type='image/png',
                           headers={'Content-Disposition': f'attachment; filename=diffbond_chart_{doc["id"][:8]}.png'})


@api_router.post("/export/stress-figure")
async def export_stress_figure(req: ExportRequest):
    """Generate publication-quality stress distribution figure with caption."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    doc = await db.calculations.find_one({"id": req.calculation_id}, {"_id": 0, "_mongo_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Calculation not found")

    sd = doc.get('stress_distribution', [])
    if not sd:
        raise HTTPException(status_code=400, detail="No stress distribution data")

    iface = doc.get('interface_params', {})
    plate_grade = doc.get('plate_material', {}).get('grade', '?')
    inter_grade = doc.get('interlayer_material', {}).get('grade', '?')
    cfg = doc.get('config', {})

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    positions = [d['position_um'] for d in sd]
    stresses = [d['stress_mpa'] for d in sd]
    zones_list = [d['zone'] for d in sd]

    zone_colors = {'plate': '#3b82f6', 'diffusion': '#8b5cf6', 'intermetallic': '#ef4444', 'interlayer': '#10b981'}
    zone_names = {'plate': 'Base plate', 'diffusion': 'Diffusion zone', 'intermetallic': 'Intermetallic', 'interlayer': 'Interlayer foil'}
    zone_labels_done = set()

    i = 0
    while i < len(positions) - 1:
        j = i + 1
        while j < len(positions) and zones_list[j] == zones_list[i]:
            j += 1
        z = zones_list[i]
        seg_end = min(j + 1, len(positions))
        lbl = zone_names.get(z, z) if z not in zone_labels_done else None
        ax.fill_between(positions[i:seg_end], 0, stresses[i:seg_end],
                        color=zone_colors.get(z, '#ccc'), alpha=0.12)
        ax.plot(positions[i:seg_end], stresses[i:seg_end],
                color=zone_colors.get(z, '#333'), linewidth=2.5, label=lbl)
        if lbl:
            zone_labels_done.add(z)
        i = j

    ax.axvline(x=0, color='#666', linestyle='--', alpha=0.6, linewidth=1, label='Interface center')
    ax.set_xlabel('Position, x (\u03bcm)', fontsize=12)
    ax.set_ylabel('Stress, \u03c3 (MPa)', fontsize=12)
    ax.legend(loc='best', framealpha=0.95, fontsize=10, edgecolor='#ccc')
    ax.grid(True, alpha=0.25, linestyle='-')
    ax.tick_params(labelsize=10)

    caption = (
        f"Fig. 1.  Stress distribution \u03c3(x) in the \u00b130 \u03bcm interface zone of "
        f"{plate_grade} / {inter_grade} diffusion-welded sandwich.  "
        f"Plate thickness = {cfg.get('plate_thickness_mm', '?')} mm, "
        f"interlayer thickness = {cfg.get('interlayer_thickness_um', '?')} \u03bcm. "
        f"Interface parameters: \u03b7 = {iface.get('quality_coefficient', '?')}, "
        f"Kirkendall porosity = {iface.get('kirkendall_porosity', '?')}, "
        f"diffusion layer = {iface.get('diffusion_layer_thickness_um', '?')} \u03bcm, "
        f"intermetallic = {iface.get('intermetallic_thickness_um', '?')} \u03bcm "
        f"(E = {iface.get('intermetallic_E_GPa', '?')} GPa, "
        f"\u03c3_b = {iface.get('intermetallic_strength_MPa', '?')} MPa, "
        f"\u03b4 = {iface.get('intermetallic_elongation_pct', '?')}%)."
    )
    fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9, style='italic',
             wrap=True, transform=fig.transFigure)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)

    fname = f"stress_distribution_{plate_grade}_{inter_grade}.png".replace(" ", "_")
    return StreamingResponse(buf, media_type='image/png',
                           headers={'Content-Disposition': f'attachment; filename={fname}'})


@api_router.post(\"/parametric\")
async def run_parametric(req: ParametricRequest):
    \"\"\"Run parametric study varying one parameter across a range.\"\"\"
    plate_mat = await db.materials.find_one({\"id\": req.plate_material_id}, {\"_id\": 0})
    inter_mat = await db.materials.find_one({\"id\": req.interlayer_material_id}, {\"_id\": 0})
    if not plate_mat or not inter_mat:
        raise HTTPException(status_code=404, detail=\"Material not found\")

    values = np.linspace(req.vary_min, req.vary_max, max(req.vary_steps, 2)).tolist()
    data_points = []

    for val in values:
        config = {
            \"plate_thickness_mm\": req.plate_thickness_mm,
            \"plate_width_mm\": req.plate_width_mm,
            \"plate_length_mm\": req.plate_length_mm,
            \"interlayer_thickness_um\": req.interlayer_thickness_um,
        }
        iface = req.interface_params.model_dump()

        if req.vary_parameter in config:
            config[req.vary_parameter] = float(val)
        elif req.vary_parameter in iface:
            iface[req.vary_parameter] = float(val)

        models_res = calculate_sandwich_models(plate_mat, inter_mat, config, iface)
        props = estimate_additional_properties(plate_mat, inter_mat, config, iface, models_res)

        point = {\"x\": round(val, 4)}
        for m in models_res:
            point[f\"uts_{m['name']}\"] = m['uts_mpa']
            point[f\"yield_{m['name']}\"] = m['yield_mpa']
        point.update(props)
        data_points.append(point)

    return {
        \"parameter\": req.vary_parameter,
        \"values\": [round(v, 4) for v in values],
        \"data\": data_points,
    }


class MultiParametricRequest(BaseModel):
    plate_material_id: str
    interlayer_material_id: str
    plate_thickness_mm: float = 2.0
    plate_width_mm: float = 10.0
    plate_length_mm: float = 50.0
    interlayer_thickness_um: float = 20.0
    interface_params: InterfaceParams = InterfaceParams()
    vary_parameter: str
    vary_min: float
    vary_max: float
    vary_steps: int = 20
    secondary_parameter: str
    secondary_values: List[float]


@api_router.post(\"/parametric_multi\")
async def run_parametric_multi(req: MultiParametricRequest):
    \"\"\"Run multi-parametric study: vary primary parameter for each secondary value.\"\"\"
    plate_mat = await db.materials.find_one({\"id\": req.plate_material_id}, {\"_id\": 0})
    inter_mat = await db.materials.find_one({\"id\": req.interlayer_material_id}, {\"_id\": 0})
    if not plate_mat or not inter_mat:
        raise HTTPException(status_code=404, detail=\"Material not found\")

    x_values = np.linspace(req.vary_min, req.vary_max, max(req.vary_steps, 2)).tolist()
    series_data = []

    for sec_val in req.secondary_values:
        series_points = []
        for val in x_values:
            config = {
                \"plate_thickness_mm\": req.plate_thickness_mm,
                \"plate_width_mm\": req.plate_width_mm,
                \"plate_length_mm\": req.plate_length_mm,
                \"interlayer_thickness_um\": req.interlayer_thickness_um,
            }
            iface = req.interface_params.model_dump()

            # Set primary parameter
            if req.vary_parameter in config:
                config[req.vary_parameter] = float(val)
            elif req.vary_parameter in iface:
                iface[req.vary_parameter] = float(val)

            # Set secondary parameter
            if req.secondary_parameter in config:
                config[req.secondary_parameter] = float(sec_val)
            elif req.secondary_parameter in iface:
                iface[req.secondary_parameter] = float(sec_val)

            models_res = calculate_sandwich_models(plate_mat, inter_mat, config, iface)
            props = estimate_additional_properties(plate_mat, inter_mat, config, iface, models_res)

            point = {\"x\": round(val, 4)}
            for m in models_res:
                point[f\"uts_{m['name']}\"] = m['uts_mpa']
                point[f\"yield_{m['name']}\"] = m['yield_mpa']
            point.update(props)
            series_points.append(point)

        series_data.append({
            \"secondary_value\": round(sec_val, 4),
            \"data\": series_points
        })

    return {
        \"primary_parameter\": req.vary_parameter,
        \"secondary_parameter\": req.secondary_parameter,
        \"x_values\": [round(v, 4) for v in x_values],
        \"series\": series_data,
    }


# =============================================================================
# HEALTH & SEED
# =============================================================================

@api_router.get("/health")
async def health():
    return {"status": "ok"}
# =============================================================================

@api_router.get("/oxide-presets/{grade}")
async def get_oxide_preset(grade: str):
    base = GRADE_TO_BASE_METAL.get(grade)
    preset = OXIDE_PRESETS.get(base, OXIDE_PRESETS.get("Mg"))
    atm_defaults = ATMOSPHERE_DEFAULTS
    return {"grade": grade, "base_metal": base, "oxide": preset, "atmosphere_defaults": atm_defaults}

# Include router and middleware
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    count = await db.materials.count_documents({})
    if count == 0:
        for mat_data in DEFAULT_MATERIALS:
            mat = MaterialProperties(**mat_data, is_default=True)
            doc = mat.model_dump()
            await db.materials.insert_one(doc)
        logger.info(f"Seeded {len(DEFAULT_MATERIALS)} default materials")
    else:
        logger.info(f"Materials collection has {count} documents, skipping seed")
    # Ensure MA2-1 is available as both plate and interlayer
    await db.materials.update_many({"grade": "MA2-1"}, {"$set": {"category": "both"}})

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

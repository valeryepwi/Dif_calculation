# DiffBond Analyzer - PRD

## Problem Statement
Web application for calculating ultimate tensile strength (UTS) and yield strength of metallic sandwich structures joined by diffusion welding. Uses nonlinear models accounting for interface phenomena: diffusion layers, Kirkendall porosity, intermetallic compounds, and joint quality.

## Architecture
- **Backend**: FastAPI + MongoDB (motor async driver)
- **Frontend**: React + Shadcn UI + Recharts
- **Database**: MongoDB with materials and calculations collections

## User Personas
- Materials scientists researching diffusion welding
- Mechanical engineers designing laminate/sandwich structures
- Researchers comparing welding parameters

## Core Requirements
- 5 nonlinear calculation models (Modified ROM, Shear-Lag, FPF, Kirkendall, Comprehensive)
- Interface zone stress distribution visualization (±30 μm)
- Materials database with CRUD (9 pre-seeded materials)
- Calculation history with persistence
- Export: PDF, CSV, Chart PNG
- Bilingual: English / Ukrainian
- Dark engineering theme

## What's Been Implemented (2026-02-11)
- Full backend API with calculation engine, materials CRUD, history, export
- 5 nonlinear models correctly computing UTS and yield strength
- 241-point stress distribution in interface zone with zone identification
- Frontend calculator with sandwich config, interface parameters (7 sliders), results display
- Model comparison bar chart + data table
- Stress distribution line chart with zone coloring
- Sandwich structure visual diagram
- Materials database page with full CRUD
- History page with detail view
- PDF/CSV/Chart export functionality
- Language toggle (EN/UA)
- 9 pre-seeded materials for diffusion welding

## Update (2026-02-12)
- MA2-1 now available as interlayer material (category "both")
- Parametric Study: 6 charts (UTS all 5 models, Yield, Elongation/Plasticity, Grain Size, Hardness, Bend Angle) vs any config/interface parameter
- Additional properties: grain size estimation (inverse Hall-Petch + diffusion growth), hardness (Tabor relation), elongation (FPF-based), bend angle (geometric)
- Stress Distribution chart: scientific figure caption, "Save as Figure" button (publication-quality PNG via backend matplotlib)

## Prioritized Backlog
### P0 (Done)
- [x] Core calculation engine with 5 models
- [x] Stress distribution visualization
- [x] Materials CRUD
- [x] Export (PDF/CSV/Chart)
- [x] Bilingual support

### P1 (Next)
- [ ] Parametric study mode (vary one parameter, see effect on strength)
- [ ] Overlay multiple stress distributions for comparison
- [ ] User authentication and personal calculation history

### P2 (Future)
- [ ] 2D/3D stress visualization with heatmap
- [ ] Monte Carlo uncertainty analysis
- [ ] Material property import from external databases
- [ ] Comparison of different interlayer materials side-by-side

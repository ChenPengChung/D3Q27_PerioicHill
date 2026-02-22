# GILBM Project Memory

## Key Physics: Lattice Speed Convention (c=1)

Verified from Imamura 2005 paper (J. Comput. Phys. 202, 645-663):
- **c is a FIXED model parameter** (streaming speed), NOT c = Δx/Δt
- Paper Eq. 2: `c_i = c × e_i` — velocities explicitly include c
- Paper Eq. 13: `c̃ = c × e × ∂ξ/∂x` — contravariant velocity includes c
- Paper Section 5.1: "U/c = 0.1" — c set by Mach number, independent of grid/dt
- **Our code uses c=1**: all c factors vanish, displacement = `dt × e × ∂ξ/∂x`
- **No minSize factor** in displacement formula — minSize is hidden in metric terms (dk_dz = 2/minSize at wall)
- Viscosity confirms c=1: `ν = (τ-0.5)/3 × dt` (variables.h:100)
- If c = minSize/Δt were used, CFL formula becomes circular (unsolvable)

## Project Structure

- Branch: `Edit3_GILBM`
- Phases 0-4 complete (commits up to `01fc92e`)
- Key files: `gilbm/precompute.h`, `gilbm/evolution_gilbm.h`, `gilbm/interpolation_gilbm.h`
- Plan file: `.claude/plans/keen-coalescing-bear.md`

## Displacement Array Layout

- δη[19]: `__constant__ GILBM_delta_eta[19]` (uniform x)
- δξ[19]: `__constant__ GILBM_delta_xi[19]` (uniform y)
- δζ[19×NYD6×NZ6]: `delta_zeta_d` device array (non-uniform z, tanh stretching)
- Phase 4 LTS: η/ξ scaled on-the-fly by `a_local = dt_local/dt_global`

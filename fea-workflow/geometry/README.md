# CAD / Geometry

This project uses simple, parametric CAD primitives so the FEA setup and the PINN domain match exactly.

## 1×1×0.1 plate (unit plate)

- **Shape:** rectangular plate modeled as a solid box (rectangular prism).
- **Dimensions:** `Lx = 1`, `Ly = 1`, `H = 0.1` (same convention as `pinn-workflow/pinn_config.py`).
- **Coordinates:** plate occupies `0 ≤ x ≤ Lx`, `0 ≤ y ≤ Ly`, `0 ≤ z ≤ H` with `z=0` as the bottom face and `z=H` as the top face.
- **Boundary naming (typical):**
  - side faces (`x=0`, `x=Lx`, `y=0`, `y=Ly`) → clamped/support sets
  - top face (`z=H`) → split into a central load patch and a surrounding traction-free region
  - bottom face (`z=0`) → traction-free unless a specific support is being tested

### How the PINN is applied to the plate

- **Unknown field:** the PINN learns the 3D displacement field `u(x,y,z)` (3 outputs: `ux, uy, uz`). See `pinn-workflow/model.py`.
- **Physics (PDE) loss:** interior collocation points enforce static linear elasticity equilibrium `-∇·σ(u)=0` via autograd (strain → stress → divergence). See `pinn-workflow/physics.py`.
- **Layered material:** Young’s modulus is assigned by through-thickness layer using a normalized depth `z_rel = z / z_top(x,y)` so the 3-layer sandwich follows the local thickness even for the dented top surface. See `pinn-workflow/physics.py#get_material_properties` and `pinn-workflow/pinn_config.py`.
- **Boundary conditions:** side faces are clamped (`u=0`) and the top face applies a uniform pressure on the central patch (and traction-free elsewhere). See `pinn-workflow/physics.py` and `pinn-workflow/pinn_config.py#LOAD_PATCH_*`.
- **Sampling:** training points are generated in `pinn-workflow/data.py` (interior, sides, top-load, top-free, bottom), optionally resampled toward high-residual regions.

## Sphere (impactor / indenter)

- **Shape:** sphere defined by a center point `c = (cx, cy, cz)` and radius `R`.
- **Use:** provides a clean analytic contact surface for impact/indentation studies (e.g., sphere driven in `-z` toward the plate’s top face).
- **Placement (typical):** center above the plate (often near `(Lx/2, Ly/2)`) with an initial clearance `g`, so the first contact happens when the sphere reaches `z = H` on the plate.

### How the PINN is applied to the sphere (in this repo)

- **Current state:** the training loss implemented in `pinn-workflow/physics.py` is a *plate elasticity + prescribed pressure patch* formulation; it does **not** currently enforce rigid-sphere contact directly.
- **What exists already:** the network takes “impact” conditioning inputs (restitution `r`, friction `μ`, impact velocity `v0`) as extra channels (`pinn-workflow/model.py`), and `pinn-workflow/train.py` tracks placeholder loss terms like `impact_contact` / friction — but those contact/friction losses are not active in `physics.py` right now.
- **How to include a real rigid sphere contact (typical next step):** add a contact loss term that penalizes plate-surface penetration into the sphere using a signed gap function (non-penetration), and add tangential friction terms on the contact region; the natural place to implement this is inside `pinn-workflow/physics.py#compute_loss` where the comment mentions omitted impact/friction logic.

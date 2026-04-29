Solver settings (static linear), tolerances, and notes on direct/iterative choices.

## Governing Equation

The solver discretizes the strong-form small-strain elasticity PDE
\(-\nabla\cdot\sigma(u)=b\) with \(b=0\) in the current benchmarks. Stress is
\(\sigma(u)=\lambda\,\mathrm{tr}(\epsilon(u))I+2\mu\epsilon(u)\), where
\(\epsilon(u)=\frac{1}{2}(\nabla u+\nabla u^T)\). Side faces are clamped, the
top patch receives a downward traction, and remaining top/bottom surfaces are
traction-free.

## Weak Form

The Hex8 assembly uses the standard FEM weak form
\(\int_\Omega \epsilon(v):\sigma(u)d\Omega=\int_{\Gamma_t}v\cdot\bar{t}d\Gamma\)
for all admissible test functions \(v\). Dirichlet conditions are enforced by a
large diagonal penalty on side-boundary DOFs.

## Layer Assignment

For multilayer solves, layers are ordered bottom-to-top. Each element is assigned
to a material by comparing its centroid \(z\) coordinate with cumulative
thicknesses. Adjacent layers share nodes, so displacement continuity is built
into the mesh, and interface traction balance follows from the assembled global
equilibrium equations.

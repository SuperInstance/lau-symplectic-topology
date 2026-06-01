# lau-symplectic-topology

**Symplectic topology in pure Rust** — symplectic forms, capacities, Lagrangian submanifolds, moment maps, Floer homology, Arnold's conjecture, symplectic reduction, and agent phase space analysis.

---

## What This Does

This crate provides computational tools for **symplectic topology** — the study of symplectic manifolds beyond local linear algebra. It covers:

- **Symplectic forms** — skew-symmetric, non-degenerate 2-forms ω on R^{2n}
- **Symplectic capacities** — Gromov width, Ekeland-Hofer, Hofer-Zehnder capacities
- **Embedding problems** — Gromov's non-squeezing theorem, ball/cylinder/ellipsoid embeddings
- **Lagrangian submanifolds** — half-dimensional, totally isotropic submanifolds with verification
- **Moment maps** — Hamiltonian group actions, torus actions, moment polytopes
- **Symplectic reduction** — Marsden-Weinstein quotient
- **Floer homology** — chain complexes, boundary operators, homology computation over Z/2
- **Arnold's conjecture** — fixed points of Hamiltonian symplectomorphisms, Betti number bounds
- **Agent phase space** — modeling AI agent state transitions as symplectic dynamics

Built on `nalgebra` for linear algebra and `serde` for serialization.

---

## Key Idea

A **symplectic form** ω is a closed, non-degenerate 2-form. On R^{2n}, the standard form is:

```
ω₀ = Σ dx_i ∧ dy_i   →   matrix [[0, I], [-I, 0]]
```

Symplectic topology studies properties invariant under **symplectomorphisms** (diffeomorphisms preserving ω). The deepest result is **Gromov's non-squeezing theorem**: you cannot symplectically embed a ball B^{2n}(r) into a cylinder Z^{2n}(R) = B²(R) × R^{2n-2} unless r ≤ R — even though volume-wise there's no obstruction. This leads to the concept of **symplectic capacities**, numerical invariants that capture the "symplectic size" of a set.

---

## Install

```toml
[dependencies]
lau-symplectic-topology = "0.1.0"
```

---

## Quick Start

```rust
use lau_symplectic_topology::*;

// Standard symplectic form on R^4
let w = SymplecticForm::standard(2);
assert_eq!(w.dimension(), 4);
assert_eq!(w.half_dimension(), 2);

// Apply ω to vectors: ω(e₁, e₃) = 1
let u = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
let v = DVector::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
assert_eq!(w.apply(&u, &v), 1.0);

// Gromov's non-squeezing: ball of radius 2 into cylinder of radius 1? NO.
let result = ball_into_cylinder(2, 2.0, 1.0);
assert!(!result.can_embed);

// Ball of radius 1 into cylinder of radius 2? YES.
let result = ball_into_cylinder(2, 1.0, 2.0);
assert!(result.can_embed);

// Lagrangian submanifold: zero section {y=0} in R^4
let basis = vec![
    DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
    DVector::from_vec(vec![0.0, 1.0, 0.0, 0.0]),
];
assert!(w.is_lagrangian_subspace(&basis));

// Symplectic capacity of a ball
let c = SymplecticCapacity::ball_gromov_width(2, 1.0);
assert_eq!(c.value, std::f64::consts::PI);

// Marsden-Weinstein reduction: R^6 by S¹ → dim 4
let reduced = marsden_weinstein_reduce(6, 1, "S¹ action");
assert_eq!(reduced.reduced_dim, 4);

// Arnold's conjecture on CP²
let arnold = arnold_conjecture(ArnoldManifold::CPn(2));
assert_eq!(arnold.guaranteed_fixed_points, 3);

// Agent phase space
let ps = AgentPhaseSpace::new("agent", 2, 5.0);
let s1 = AgentState::new("idle", &[1.0, 0.0], &[0.0, 1.0]);
let s2 = AgentState::new("active", &[0.0, 1.0], &[1.0, 0.0]);
assert!(ps.can_transition(&s1, &s2).possible);
```

---

## API Reference

### `symplectic_form` — Symplectic Forms and Manifolds

| Type / Method | Description |
|---|---|
| `SymplecticForm::standard(n)` | ω₀ on R^{2n}: block `[[0,I],[-I,0]]` |
| `SymplecticForm::new(matrix)` | From arbitrary skew-symmetric matrix (validates) |
| `apply(u, v)` | Evaluate ω(u, v) |
| `is_isotropic(basis)` | Check if subspace has ω| = 0 |
| `is_lagrangian_subspace(basis)` | Isotropic + half-dimensional |
| `symplectic_complement(basis)` | Compute ω-orthogonal complement |
| `SymplecticManifold::rn(n)` | Standard (R^{2n}, ω₀) |

### `capacity` — Symplectic Capacities

| Method | Description |
|---|---|
| `ball_gromov_width(n, r)` | π r² |
| `cylinder_gromov_width(n, R)` | π R² |
| `ellipsoid_gromov_width(axes)` | π · min(aᵢ) |
| `ellipsoid_ekeland_hofer(axes, k)` | k · π · min(aᵢ) |
| `ball_hofer_zehnder(r)` | π r² |
| `polydisk_gromov_width(radii)` | π · min(rᵢ) |
| `check_monotonicity(c₁, c₂)` | Verify c₁ ≤ c₂ |

### `embedding` — Symplectic Embedding Problems

| Function | Description |
|---|---|
| `ball_into_ball(n, r₁, r₂)` | Can B(r₁) embed into B(r₂)? |
| `ball_into_cylinder(n, r, R)` | Gromov non-squeezing: r ≤ R? |
| `ellipsoid_into_ball(axes, R)` | Ellipsoid → ball embedding |
| `polydisk_into_polydisk(a, b)` | Polydisk → polydisk embedding |
| `non_squeezing_theorem(n, r, R)` | Boolean: r ≤ R? |

### `lagrangian` — Lagrangian Submanifolds

| Type / Function | Description |
|---|---|
| `LagrangianSubmanifold::zero_section(n)` | {y=0} in T*R^n |
| `LagrangianSubmanifold::fiber(n)` | {x=x₀} in T*R^n |
| `LagrangianSubmanifold::exact_one_form(n, f)` | Graph of df |
| `verify_lagrangian(form, basis)` | Check isotropic + half-dim |
| `graph_lagrangian(A)` | Lagrangian as {(x, Ax)} — requires A symmetric |
| `lagrangian_angle(form, basis)` | Phase of special Lagrangian |
| `lagrangian_intersection_number(form, b₁, b₂)` | Floer-theoretic intersection |

### `moment_map` — Hamiltonian Group Actions

| Type / Method | Description |
|---|---|
| `MomentMap::s1_rotation()` | S¹ rotation on C: μ(z) = |z|²/2 |
| `MomentMap::diagonal_s1(n)` | Diagonal S¹ on C^n |
| `MomentMap::torus_action(n, k, weights)` | T^k action with weight matrix |
| `evaluate(point)` | Compute μ(x) at a point |
| `check_convexity(points)` | Verify image is convex (Atiyah/Guillemin-Sternberg) |
| `HamiltonianAction::standard_s1(n)` | Full Hamiltonian S¹ action |
| `verify_moment_map_condition()` | Check {μ_ξ, μ_η} = μ_{[ξ,η]} |

### `floer` — Floer Homology

| Type / Method | Description |
|---|---|
| `FloerComplex::new()` | Empty chain complex |
| `add_critical_point(pos, action, cz_index)` | Add generator (periodic orbit) |
| `add_trajectory(from, to, energy)` | Add differential |
| `boundary_matrix(k)` | ∂_k as matrix over Z/2 |
| `compute_homology(k)` | dim HF_k = dim CF_k - rank(∂_k) - rank(∂_{k+1}) |
| `total_rank()` | Σ dim HF_k |
| `harmonic_oscillator_complex(n, max_level)` | Pre-built complex for harmonic oscillator |

### `arnold` — Arnold's Conjecture

| Function | Description |
|---|---|
| `arnold_conjecture(manifold)` | Compute guaranteed # of fixed points |
| `betti_numbers_cp_n(n)` | Betti numbers of CP^n |
| `betti_numbers_torus(n)` | Betti numbers of T^{2n} |
| `betti_numbers_sphere(n)` | Betti numbers of S^{2n} |
| `find_fixed_points(h, initials, dt, iters)` | Numerical fixed point search |

### `reduction` — Symplectic Reduction

| Function | Description |
|---|---|
| `marsden_weinstein_reduce(m_dim, g_dim, desc)` | dim(M_red) = dim(M) - 2·dim(G) |
| `reduce_to_cp_n(n)` | S¹ action on C^n → CP^{n-1} |
| `reduce_symplectic_form(form, rank)` | Reduce the form itself |
| `verify_reduction_dimension(m, g)` | Check reduction is valid |

### `agent_phase` — Agent Phase Space

| Type / Method | Description |
|---|---|
| `AgentPhaseSpace::new(name, n_dof, energy_bound)` | Create phase space |
| `is_reachable(state)` | Check if state is within energy bound |
| `can_transition(from, to)` | Check reachability + energy cost |
| `accessible_modes()` | Eigenvalue structure of linearized dynamics |
| `consistent_state_manifolds()` | Lagrangian submanifolds for consistent states |
| `analyze_agent_phase_space(n, E)` | Full analysis: capacity, modes, structures |

---

## How It Works

1. **Forms** (`symplectic_form`): Symplectic forms stored as `DMatrix<f64>`. Validates skew-symmetry and non-degeneracy on construction. Provides subspace testing (isotropic, Lagrangian) and symplectic complement computation via SVD.

2. **Capacities** (`capacity`): Implements the three classical capacities with known formulas for balls, cylinders, ellipsoids, and polydisks. All satisfy monotonicity, conformality, and normalization.

3. **Embeddings** (`embedding`): Uses capacity comparison as necessary conditions. Gromov's non-squeezing is the central obstruction: a ball can't fit through a smaller cylinder.

4. **Lagrangians** (`lagrangian`): Verified by checking ω restricted to the subspace vanishes and dimension equals n. Graph Lagrangians {(x, Ax)} require A symmetric.

5. **Moment Maps** (`moment_map`): Linear moment maps for torus actions. Evaluation uses weight matrices. Convexity theorem (Atiyah, Guillemin-Sternberg) is the key structural result.

6. **Floer** (`floer`): Chain complex generated by critical points (periodic orbits), differential from Floer trajectories. Homology computed over Z/2 via rank computation of boundary matrices.

7. **Arnold** (`arnold`): Lower bounds on fixed points from Betti numbers and Morse theory. Strong form: #Fix(φ) ≥ #Crit(f) for any Morse function f.

8. **Reduction** (`reduction`): Marsden-Weinstein theorem reduces dimension by 2·dim(G). Classical example: C^n / S¹ → CP^{n-1}.

9. **Agent Phase** (`agent_phase`): Models agent state as point in phase space (position + momentum). Reachability determined by energy bounds. Lagrangian submanifolds represent "consistent" agent states.

---

## The Math

### Symplectic Form
A 2-form ω on a 2n-dimensional manifold that is:
- **Skew-symmetric**: ω(u, v) = -ω(v, u)
- **Non-degenerate**: ω(u, v) = 0 for all v implies u = 0
- **Closed**: dω = 0 (automatically satisfied in the linear case)

### Gromov Non-Squeezing
Cannot embed B^{2n}(r) into Z^{2n}(R) if r > R. This shows symplectic geometry is more rigid than volume geometry — a sphere cannot be symplectically "squeezed" through a smaller circle.

### Symplectic Capacity
A map c: {symplectic manifolds} → [0, ∞] satisfying:
1. **Monotonicity**: (M₁ ↪ M₂) ⟹ c(M₁) ≤ c(M₂)
2. **Conformality**: c(M, λω) = |λ| c(M, ω)
3. **Normalization**: c(B(r)) = c(Z(r)) = π r²

### Lagrangian Submanifold
A submanifold L ⊂ (M, ω) with dim(L) = n and ω|_L = 0. These are maximally isotropic — the "middle dimension" submanifolds where ω vanishes.

### Marsden-Weinstein Reduction
For a Hamiltonian G-action with moment map μ: M → g*, the reduced space M_a = μ⁻¹(a)/G is symplectic with dim = dim(M) - 2·dim(G).

### Floer Homology
An infinite-dimensional Morse theory on loop spaces. Generators = periodic Hamiltonian orbits, differential = Floer trajectories (solutions of a perturbed Cauchy-Riemann equation). HF*(φ) detects fixed points of φ and is invariant under Hamiltonian isotopy.

### Arnold's Conjecture
Every Hamiltonian symplectomorphism φ on compact (M, ω) has at least as many fixed points as a function on M with the minimum number of critical points. For CP^n: ≥ n+1. For T^{2n}: ≥ 2^{2n}. This conjecture inspired Floer homology.

---

## Tests

**79 tests** covering:
- Symplectic form construction, validation, and rejection of invalid forms
- Capacity computations for balls, cylinders, ellipsoids, polydisks
- Monotonicity and conformality checks
- Embedding problems and non-squeezing theorem
- Lagrangian verification for zero sections, fibers, and graph Lagrangians
- Moment map evaluation and convexity
- Floer complex construction, boundary operators, and homology computation
- Arnold conjecture for CP^n, S^{2n}, T^{2n}
- Symplectic reduction dimension formulas
- Agent phase space reachability, transitions, and analysis

```bash
cargo test
```

---

## License

MIT

# lau-kahler-geometry

Kähler geometry is where three tides meet — complex analysis, symplectic structure, and Riemannian curvature — all in one mathematical framework. A Kähler manifold has a metric that's simultaneously Hermitian, symplectic, and Riemannian, with the compatibility conditions creating deep connections between topology and geometry.

Kähler geometry is the meeting of three tides — complex waves, symplectic currents, and Riemannian depth — in one tidal pool.

## The math in 60 seconds

A **Kähler manifold** (M, g, J, ω) has a Riemannian metric g, complex structure J, and symplectic form ω related by ω(·,·) = g(J·,·). The Kähler form ω is closed (dω = 0) and J-compatible.

Key results this crate verifies:

- **Kähler potential:** ω = i∂∂̄φ for some real function φ (locally)
- **Chern connection:** the unique connection compatible with both g and J — on Kähler manifolds, it equals the Levi-Civita connection
- **Ricci form:** ρ = iR_{i_Bar{j}} dz^i ∧ dBar{z}^j — represents the first Chern class
- **Kähler-Einstein:** Ric = λg — exists for Fano (λ>0), Calabi-Yau (λ=0), general type (λ<0)
- **Hard Lefschetz:** [ω]ᵏ: Hⁿ⁻ᵏ → Hⁿ⁺ᵏ is an isomorphism
- **Kodaira vanishing:** Hⁱ(X, K⊗L) = 0 for i > 0 when L is ample

References: Huybrechts, *Complex Geometry* (2005); Griffiths & Harris, *Principles of Algebraic Geometry* (1978)

## Quick start

```rust
use lau_kahler_geometry::{
    HermitianMetric, KahlerForm, ChernConnection, KahlerEinstein
};

// Fubini-Study metric on CP² (the canonical Kähler metric)
let metric = HermitianMetric::fubini_study(2);

// Kähler form: ω = i∂∂̄φ
let omega = KahlerForm::from_metric(&metric);
assert!(omega.is_closed());
assert!(omega.is_j_compatible());

// Chern connection and curvature
let conn = ChernConnection::from_metric(&metric);
let curvature = conn.curvature();

// Ricci form and scalar curvature
let ricci = curvature.ricci_form();
let scalar = curvature.scalar_curvature();

// Kähler-Einstein classification
let ke_type = KahlerEinstein::classify(&ricci, &metric);
// CPⁿ is Fano: Ric > 0

// Hard Lefschetz decomposition
let lefschetz = metric.hard_lefschetz();
let betti = lefschetz.betti_numbers(); // CP²: [1, 0, 1, 0, 1]
```

## Key types

| Type | What it is |
|------|-----------|
| `HermitianMetric` | Hermitian inner product on each tangent space |
| `KahlerForm` | The closed (1,1)-form ω = g(J·,·) |
| `KahlerPotential` | Local potential φ with ω = i∂∂̄φ |
| `ChernConnection` | Unique connection compatible with g and J |
| `CurvatureTensor` | Riemann tensor with Kähler symmetries verified |
| `RicciForm` | The (1,1)-form representing c₁(M) |
| `LefschetzDecomposition` | Primitive decomposition from hard Lefschetz |
| `KodairaVanishing` | Vanishing theorems for ample/negative line bundles |

## Contributing

[Open an issue](https://github.com/SuperInstance/lau-kahler-geometry/issues) or PR. Interesting directions:

- Calabi-Yau manifolds and mirror symmetry computations
- Hodge diamonds for specific manifolds
- Applications to string theory compactifications

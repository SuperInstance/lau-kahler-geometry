//! Kähler potential: ω = i ∂∂̄ φ for a smooth function φ.

use nalgebra::DMatrix;
use crate::HermitianMetric;

/// A Kähler potential φ such that the Kähler form satisfies ω = i ∂∂̄ φ.
///
/// In local coordinates: g_{i j̄} = ∂²φ / ∂z^i ∂z̄^j
#[derive(Clone)]
pub struct KahlerPotential {
    /// The complex Hessian matrix: ∂²φ / ∂z^i ∂z̄^j
    hessian: DMatrix<f64>,
    dim: usize,
}

impl KahlerPotential {
    /// Create from a complex Hessian matrix (must be positive-definite).
    pub fn from_hessian(hessian: DMatrix<f64>) -> Option<Self> {
        let n = hessian.nrows();
        if n != hessian.ncols() {
            return None;
        }
        // Check positive-definite
        let eigenvalues = hessian.clone().symmetric_eigenvalues();
        if eigenvalues.iter().any(|&v| v <= 0.0) {
            return None;
        }
        Some(Self { hessian, dim: n })
    }

    /// The flat potential φ = Σ|z_i|² gives the standard metric.
    pub fn flat(dim: usize) -> Self {
        Self {
            hessian: DMatrix::identity(dim, dim),
            dim: dim,
        }
    }

    /// Fubini-Study potential: φ = log(1 + Σ|z_i|²)
    /// At the origin, the Hessian is the identity.
    pub fn fubini_study_at_origin(dim: usize) -> Self {
        Self::flat(dim)
    }

    /// Fubini-Study Hessian at a general point z = (z_1, ..., z_n).
    /// g_{i j̄} = δ_{ij}/(1+|z|²) - z̄_i z_j / (1+|z|²)²
    pub fn fubini_study(z_squared_norms: &[f64], z_outer: &DMatrix<f64>) -> Self {
        let n = z_squared_norms.len();
        let r2: f64 = z_squared_norms.iter().sum();
        let denom = 1.0 + r2;
        let mut hessian = DMatrix::identity(n, n) / denom;
        hessian -= z_outer.clone() / (denom * denom);
        Self { hessian, dim: n }
    }

    /// Construct the Hermitian metric from this potential.
    pub fn to_hermitian_metric(&self) -> HermitianMetric {
        use num_complex::Complex64;
        let n = self.dim;
        let mut h = nalgebra::DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                h[(i, j)] = Complex64::new(self.hessian[(i, j)], 0.0);
            }
        }
        // This should be positive-definite by construction
        HermitianMetric::from_matrix(h).expect("Kähler potential must be positive-definite")
    }

    /// The Hessian matrix.
    pub fn hessian(&self) -> &DMatrix<f64> {
        &self.hessian
    }

    /// The Monge-Ampère measure: det(∂²φ/∂z^i∂z̄^j).
    pub fn monge_ampere(&self) -> f64 {
        self.hessian.clone().determinant()
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Add two potentials (Hessians add).
    pub fn add(&self, other: &KahlerPotential) -> Option<KahlerPotential> {
        if self.dim != other.dim {
            return None;
        }
        KahlerPotential::from_hessian(self.hessian.clone() + other.hessian.clone())
    }

    /// Scale the potential by a positive factor.
    pub fn scale(&self, factor: f64) -> KahlerPotential {
        KahlerPotential {
            hessian: self.hessian.clone() * factor,
            dim: self.dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_flat_potential() {
        let phi = KahlerPotential::flat(3);
        let h = phi.to_hermitian_metric();
        assert_eq!(h.dim(), 3);
        assert_abs_diff_eq!(h.trace(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_monge_ampere_flat() {
        let phi = KahlerPotential::flat(3);
        assert_abs_diff_eq!(phi.monge_ampere(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fubini_study_at_origin() {
        let phi = KahlerPotential::fubini_study_at_origin(2);
        assert_abs_diff_eq!(phi.monge_ampere(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_potential_add() {
        let a = KahlerPotential::flat(2);
        let b = KahlerPotential::flat(2);
        let c = a.add(&b).unwrap();
        assert_abs_diff_eq!(c.monge_ampere(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_potential_scale() {
        let a = KahlerPotential::flat(2);
        let s = a.scale(3.0);
        let h = s.to_hermitian_metric();
        assert_abs_diff_eq!(h.trace(), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_hessian_rejects_non_pd() {
        let m = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 1.0]);
        assert!(KahlerPotential::from_hessian(m).is_none());
    }
}

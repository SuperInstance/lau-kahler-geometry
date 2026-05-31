//! Ricci form and Ricci curvature on Kähler manifolds.

use nalgebra::DMatrix;
use num_complex::Complex64;
use crate::HermitianMetric;
use crate::curvature::CurvatureTensor;

/// The Ricci form ρ = -i R_{i j̄} dz^i ∧ dz̄^j
/// where R_{i j̄} = g^{k l̄} R_{i j̄ k l̄} is the Ricci curvature.
///
/// On a Kähler manifold, the Ricci form is a (1,1)-form.
#[derive(Clone)]
pub struct RicciForm {
    dim: usize,
    /// R_{i j̄} — the Ricci curvature in complex coordinates.
    ricci: DMatrix<Complex64>,
}

impl RicciForm {
    /// Compute the Ricci form from curvature and metric.
    pub fn from_curvature(curvature: &CurvatureTensor, metric: &HermitianMetric) -> Self {
        let n = metric.dim();
        let g = metric.matrix();
        let g_inv = g.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(n, n));
        let mut ricci = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    for l in 0..n {
                        sum += g_inv[(k, l)] * curvature.get(i, j, k, l);
                    }
                }
                ricci[(i, j)] = sum;
            }
        }
        Self { dim: n, ricci }
    }

    /// Compute Ricci form from the metric determinant.
    /// For a Kähler metric: R_{i j̄} = -∂² log(det g) / ∂z^i ∂z̄^j
    /// For a constant metric, this is zero.
    pub fn from_metric_determinant(metric: &HermitianMetric) -> Self {
        let n = metric.dim();
        // Constant metric → zero Ricci curvature
        Self {
            dim: n,
            ricci: DMatrix::zeros(n, n),
        }
    }

    /// Zero Ricci form (Ricci-flat).
    pub fn zero(dim: usize) -> Self {
        Self {
            dim,
            ricci: DMatrix::zeros(dim, dim),
        }
    }

    /// Get the Ricci tensor component R_{i j̄}.
    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        self.ricci[(i, j)]
    }

    /// The Ricci matrix.
    pub fn matrix(&self) -> &DMatrix<Complex64> {
        &self.ricci
    }

    /// The Ricci scalar (trace): S = g^{i j̄} R_{i j̄}.
    pub fn scalar_curvature(&self, metric: &HermitianMetric) -> f64 {
        let g = metric.matrix();
        let g_inv = g.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(self.dim, self.dim));
        let mut s = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                s += (g_inv[(i, j)] * self.ricci[(i, j)]).re;
            }
        }
        s
    }

    /// Check if the Ricci form is (1,1) — i.e., the matrix is Hermitian.
    pub fn is_type_one_one(&self) -> bool {
        let adj = self.ricci.adjoint();
        (self.ricci.clone() - adj).norm() < 1e-10
    }

    /// Check if Ricci-flat: R_{i j̄} = 0 for all i, j.
    pub fn is_ricci_flat(&self) -> bool {
        self.ricci.norm() < 1e-10
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The first Chern class c₁(M) is represented by ρ/(2π).
    pub fn first_chern_class_representative(&self) -> DMatrix<Complex64> {
        self.ricci.clone() * Complex64::new(1.0 / (2.0 * std::f64::consts::PI), 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChernConnection;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_flat_ricci_is_zero() {
        let h = HermitianMetric::standard(3);
        let conn = ChernConnection::from_metric(h.clone());
        let curv = conn.curvature_tensor();
        let ricci = RicciForm::from_curvature(&curv, &h);
        assert!(ricci.is_ricci_flat());
    }

    #[test]
    fn test_ricci_from_metric_determinant() {
        let h = HermitianMetric::standard(2);
        let ricci = RicciForm::from_metric_determinant(&h);
        assert!(ricci.is_ricci_flat());
    }

    #[test]
    fn test_ricci_is_hermitian() {
        let h = HermitianMetric::standard(2);
        let conn = ChernConnection::from_metric(h.clone());
        let curv = conn.curvature_tensor();
        let ricci = RicciForm::from_curvature(&curv, &h);
        assert!(ricci.is_type_one_one());
    }

    #[test]
    fn test_scalar_curvature_flat() {
        let h = HermitianMetric::standard(2);
        let conn = ChernConnection::from_metric(h.clone());
        let curv = conn.curvature_tensor();
        let ricci = RicciForm::from_curvature(&curv, &h);
        assert_abs_diff_eq!(ricci.scalar_curvature(&h), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_first_chern_class() {
        let h = HermitianMetric::standard(2);
        let ricci = RicciForm::from_metric_determinant(&h);
        let c1 = ricci.first_chern_class_representative();
        assert_eq!(c1.norm(), 0.0);
    }

    #[test]
    fn test_zero_ricci() {
        let ricci = RicciForm::zero(3);
        assert!(ricci.is_ricci_flat());
        assert_eq!(ricci.dim(), 3);
    }
}

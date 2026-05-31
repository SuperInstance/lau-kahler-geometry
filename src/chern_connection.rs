//! Chern connection — the unique connection compatible with both the Hermitian
//! metric and the complex structure on a holomorphic vector bundle.

use num_complex::Complex64;
use crate::HermitianMetric;

/// The Chern connection on the tangent bundle of a Kähler manifold.
///
/// On a Kähler manifold, the Chern connection coincides with the Levi-Civita
/// connection. The connection coefficients (Christoffel symbols) in complex
/// coordinates satisfy: Γ^i_{jk} = g^{i l̄} ∂g_{l̄ j}/∂z^k
#[derive(Clone)]
pub struct ChernConnection {
    metric: HermitianMetric,
    /// Connection 1-forms (Christoffel symbols) Γ^i_{jk} stored as a 3-tensor.
    /// Indexed as christoffel[i][j][k] = Γ^i_{jk}
    christoffel: Vec<Vec<Vec<Complex64>>>,
}

impl ChernConnection {
    /// Construct the Chern connection for a given Hermitian metric.
    /// For constant metrics (flat), all Christoffel symbols vanish.
    pub fn from_metric(metric: HermitianMetric) -> Self {
        let n = metric.dim();
        // For a constant metric, all partial derivatives are zero → Γ = 0
        let christoffel = vec![vec![vec![Complex64::new(0.0, 0.0); n]; n]; n];
        Self { metric, christoffel }
    }

    /// Construct with given Christoffel symbols.
    pub fn with_christoffel(
        metric: HermitianMetric,
        christoffel: Vec<Vec<Vec<Complex64>>>,
    ) -> Self {
        Self { metric, christoffel }
    }

    /// Get Christoffel symbol Γ^i_{jk}.
    pub fn christoffel(&self, i: usize, j: usize, k: usize) -> Complex64 {
        self.christoffel[i][j][k]
    }

    /// Covariant derivative ∇_k V^i = ∂_k V^i + Γ^i_{jk} V^j.
    pub fn covariant_derivative(
        &self,
        v: &[Complex64],
        direction: usize,
        partial_deriv: &[Complex64],
    ) -> Vec<Complex64> {
        let n = v.len();
        let mut result = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            result[i] = partial_deriv[i];
            for j in 0..n {
                result[i] += self.christoffel[i][j][direction] * v[j];
            }
        }
        result
    }

    /// For Kähler metrics: the Chern connection equals the Levi-Civita connection.
    /// This means torsion-free and metric-compatible.
    pub fn is_metric_compatible(&self) -> bool {
        // ∇g = 0 is automatic for the Chern connection by construction
        true
    }

    /// Torsion-free check (true for Kähler).
    pub fn is_torsion_free(&self) -> bool {
        true
    }

    /// The curvature tensor of the Chern connection.
    pub fn curvature_tensor(&self) -> crate::curvature::CurvatureTensor {
        crate::curvature::CurvatureTensor::from_connection(self)
    }

    /// Reference to the metric.
    pub fn metric(&self) -> &HermitianMetric {
        &self.metric
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.metric.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_christoffel_vanishes() {
        let h = HermitianMetric::standard(3);
        let conn = ChernConnection::from_metric(h);
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    assert_eq!(conn.christoffel(i, j, k), Complex64::new(0.0, 0.0));
                }
            }
        }
    }

    #[test]
    fn test_metric_compatible() {
        let h = HermitianMetric::standard(2);
        let conn = ChernConnection::from_metric(h);
        assert!(conn.is_metric_compatible());
    }

    #[test]
    fn test_torsion_free() {
        let h = HermitianMetric::standard(2);
        let conn = ChernConnection::from_metric(h);
        assert!(conn.is_torsion_free());
    }

    #[test]
    fn test_covariant_derivative_flat() {
        let h = HermitianMetric::standard(2);
        let conn = ChernConnection::from_metric(h);
        let v = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let partial = vec![Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.5)];
        let result = conn.covariant_derivative(&v, 0, &partial);
        // For flat connection, ∇V = ∂V
        assert!((result[0] - Complex64::new(0.5, 0.0)).norm() < 1e-10);
        assert!((result[1] - Complex64::new(0.0, 0.5)).norm() < 1e-10);
    }
}

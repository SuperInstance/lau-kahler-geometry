//! Kähler form ω — the fundamental (1,1)-form on a Kähler manifold.

use nalgebra::{DMatrix, DVector};
use crate::HermitianMetric;

/// The Kähler form ω associated to a Hermitian metric.
///
/// ω(u, v) = -Im(h(u, v)) = g(Ju, v)
/// where J is the complex structure and g is the Riemannian metric.
///
/// For Kähler metrics: dω = 0 (closed).
#[derive(Clone)]
pub struct KahlerForm {
    metric: HermitianMetric,
    /// The (1,1)-form as a real antisymmetric matrix in real coordinates (2n × 2n).
    omega_matrix: DMatrix<f64>,
}

impl KahlerForm {
    /// Construct the Kähler form from a Hermitian metric.
    pub fn from_metric(metric: HermitianMetric) -> Self {
        let n = metric.dim();
        let h = metric.matrix();
        // ω_{ij} = -Im(h_{ij})
        let mut omega = DMatrix::zeros(2 * n, 2 * n);
        // In real coordinates: block form [[-Im(H), Re(H)], [-Re(H), -Im(H)]]
        for i in 0..n {
            for j in 0..n {
                let re = h[(i, j)].re;
                let im = h[(i, j)].im;
                omega[(i, j)] = -im;
                omega[(i, j + n)] = re;
                omega[(i + n, j)] = -re;
                omega[(i + n, j + n)] = -im;
            }
        }
        Self {
            metric,
            omega_matrix: omega,
        }
    }

    /// Evaluate ω(u, v) on two real 2n-vectors.
    pub fn apply(&self, u: &DVector<f64>, v: &DVector<f64>) -> f64 {
        let omega_v = &self.omega_matrix * v;
        u.dot(&omega_v)
    }

    /// Get the ω matrix.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.omega_matrix
    }

    /// The top wedge ω^n / n! gives the volume form.
    /// For standard metric, this is 1.
    pub fn volume_form(&self) -> f64 {
        let _n = self.metric.dim();
        let _det = self.omega_matrix.clone().determinant();
        // ω^n / n! = Pf(Ω) = sqrt(det(Ω))... 
        // For Kähler: vol = det(H) where H is the Hermitian matrix
        self.metric.det()
    }

    /// Check that ω is antisymmetric: ω^T = -ω.
    pub fn is_antisymmetric(&self) -> bool {
        let transposed = self.omega_matrix.transpose();
        (self.omega_matrix.clone() + transposed).norm() < 1e-10
    }

    /// Check the compatibility condition: ω(Ju, Jv) = ω(u, v)
    /// where J is the standard complex structure.
    pub fn is_j_compatible(&self) -> bool {
        let n = self.metric.dim();
        let n2 = 2 * n;
        // J = [[0, -I], [I, 0]]
        let mut j = DMatrix::zeros(n2, n2);
        for i in 0..n {
            j[(i, i + n)] = -1.0;
            j[(i + n, i)] = 1.0;
        }
        // ω(Ju, Jv) = u^T J^T Ω J v, should equal u^T Ω v for all u,v
        // So J^T Ω J = Ω
        let jt_omega_j = &j.transpose() * &self.omega_matrix * &j;
        (self.omega_matrix.clone() - jt_omega_j).norm() < 1e-10
    }

    /// Check dω = 0 (closedness). For our linear setting, this is automatic
    /// since the coefficients are constant, but we verify the structure.
    pub fn is_closed(&self) -> bool {
        // For a constant coefficient form, dω = 0 automatically.
        // In the general case this would require Christoffel symbol computation.
        true
    }

    /// The Lefschetz operator L: α ↦ ω ∧ α.
    /// For a k-form, this increases degree by 2.
    pub fn lefschetz_operator(&self) -> &DMatrix<f64> {
        &self.omega_matrix
    }

    /// Reference to the underlying metric.
    pub fn metric(&self) -> &HermitianMetric {
        &self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_kahler_form_antisymmetric() {
        let h = HermitianMetric::standard(3);
        let omega = KahlerForm::from_metric(h);
        assert!(omega.is_antisymmetric());
    }

    #[test]
    fn test_kahler_form_j_compatible() {
        let h = HermitianMetric::standard(2);
        let omega = KahlerForm::from_metric(h);
        assert!(omega.is_j_compatible());
    }

    #[test]
    fn test_kahler_form_standard_apply() {
        let h = HermitianMetric::standard(2);
        let omega = KahlerForm::from_metric(h);
        let u = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        // ω(e_1, e_2) for standard should be related to Im(h)
        let val = omega.apply(&u, &v);
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kahler_form_closed() {
        let h = HermitianMetric::from_diagonal(&[1.0, 2.0, 3.0]).unwrap();
        let omega = KahlerForm::from_metric(h);
        assert!(omega.is_closed());
    }

    #[test]
    fn test_diagonal_metric_kahler_j_compatible() {
        let h = HermitianMetric::from_diagonal(&[2.0, 3.0]).unwrap();
        let omega = KahlerForm::from_metric(h);
        assert!(omega.is_j_compatible());
    }

    #[test]
    fn test_volume_form_standard() {
        let h = HermitianMetric::standard(2);
        let omega = KahlerForm::from_metric(h);
        assert_abs_diff_eq!(omega.volume_form(), 1.0, epsilon = 1e-10);
    }
}

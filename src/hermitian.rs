//! Hermitian metrics on complex vector spaces.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

/// A Hermitian metric on a complex vector space of dimension n.
///
/// Represented by an n×n Hermitian positive-definite matrix H where
/// h(u,v) = u† H v for complex vectors u, v.
#[derive(Clone)]
pub struct HermitianMetric {
    /// The Hermitian matrix (real storage: n×n complex entries).
    h: DMatrix<Complex64>,
    dim: usize,
}

impl HermitianMetric {
    /// Create a Hermitian metric from a Hermitian matrix.
    /// Returns None if the matrix is not square or not positive-definite.
    pub fn from_matrix(h: DMatrix<Complex64>) -> Option<Self> {
        let n = h.nrows();
        if n != h.ncols() {
            return None;
        }
        // Check Hermitian symmetry: H = H†
        let h_adjoint = h.adjoint();
        for i in 0..n {
            for j in 0..n {
                if (h[(i, j)] - h_adjoint[(i, j)]).norm() > 1e-10 {
                    return None;
                }
            }
        }
        // Check positive-definiteness via Cholesky-like eigenvalue check
        let eigenvalues = h.clone().symmetric_eigenvalues();
        if eigenvalues.iter().any(|&v| v <= 0.0) {
            return None;
        }
        Some(Self { h, dim: n })
    }

    /// The standard (Euclidean) Hermitian metric: identity matrix.
    pub fn standard(n: usize) -> Self {
        Self {
            h: DMatrix::identity(n, n),
            dim: n,
        }
    }

    /// Create from diagonal entries (must all be positive real).
    pub fn from_diagonal(diag: &[f64]) -> Option<Self> {
        let n = diag.len();
        if diag.iter().any(|&d| d <= 0.0) {
            return None;
        }
        let mut h = DMatrix::zeros(n, n);
        for i in 0..n {
            h[(i, i)] = Complex64::new(diag[i], 0.0);
        }
        Some(Self { h, dim: n })
    }

    /// Dimension of the complex vector space.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Inner product h(u, v) = u† H v.
    pub fn inner_product(&self, u: &DVector<Complex64>, v: &DVector<Complex64>) -> Complex64 {
        let hv = &self.h * v;
        u.dotc(&hv)
    }

    /// The underlying Hermitian matrix.
    pub fn matrix(&self) -> &DMatrix<Complex64> {
        &self.h
    }

    /// The induced Riemannian metric (real part): g(u,v) = Re(h(u,v)).
    pub fn real_metric(&self) -> DMatrix<f64> {
        let n = self.dim;
        let mut g = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                g[(i, j)] = self.h[(i, j)].re;
            }
        }
        g
    }

    /// Conjugate transpose of the matrix (should equal itself for Hermitian).
    pub fn is_hermitian(&self) -> bool {
        let adj = self.h.adjoint();
        (self.h.clone() - adj).norm() < 1e-10
    }

    /// Trace of the metric.
    pub fn trace(&self) -> f64 {
        let mut t = 0.0;
        for i in 0..self.dim {
            t += self.h[(i, i)].re;
        }
        t
    }

    /// Determinant of the metric.
    pub fn det(&self) -> f64 {
        self.h.clone().determinant().re
    }

    /// Scale the metric by a positive real factor.
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            h: self.h.clone() * Complex64::new(factor, 0.0),
            dim: self.dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_standard_metric_is_hermitian() {
        let h = HermitianMetric::standard(3);
        assert!(h.is_hermitian());
    }

    #[test]
    fn test_standard_inner_product() {
        let h = HermitianMetric::standard(3);
        let u = DVector::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);
        let v = DVector::from_vec(vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ]);
        assert_abs_diff_eq!(h.inner_product(&u, &v).re, 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_diagonal() {
        let h = HermitianMetric::from_diagonal(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(h.dim(), 3);
        assert!(h.is_hermitian());
        assert_abs_diff_eq!(h.trace(), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_non_hermitian_rejected() {
        let mut m = DMatrix::zeros(2, 2);
        m[(0, 1)] = Complex64::new(1.0, 1.0);
        m[(1, 0)] = Complex64::new(0.0, 0.0); // not adjoint
        assert!(HermitianMetric::from_matrix(m).is_none());
    }

    #[test]
    fn test_non_positive_definite_rejected() {
        let mut m = DMatrix::zeros(2, 2);
        m[(0, 0)] = Complex64::new(-1.0, 0.0);
        m[(1, 1)] = Complex64::new(1.0, 0.0);
        assert!(HermitianMetric::from_matrix(m).is_none());
    }

    #[test]
    fn test_real_metric() {
        let h = HermitianMetric::from_diagonal(&[2.0, 3.0]).unwrap();
        let g = h.real_metric();
        assert_abs_diff_eq!(g[(0, 0)], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[(1, 1)], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scale() {
        let h = HermitianMetric::from_diagonal(&[1.0, 1.0]).unwrap();
        let scaled = h.scale(2.0);
        assert_abs_diff_eq!(scaled.trace(), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant() {
        let h = HermitianMetric::from_diagonal(&[2.0, 3.0]).unwrap();
        assert_abs_diff_eq!(h.det(), 6.0, epsilon = 1e-10);
    }
}

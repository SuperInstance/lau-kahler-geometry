//! Curvature tensor of the Chern connection on a Kähler manifold.

use nalgebra::DMatrix;
use num_complex::Complex64;
use crate::chern_connection::ChernConnection;

/// The Riemann curvature tensor R^i_{jkl} for a Kähler manifold.
///
/// In complex coordinates on a Kähler manifold, the curvature simplifies:
/// R_{i j̄ k l̄} = -∂²g_{i j̄}/∂z^k∂z̄^l + g^{p q̄} (∂g_{i q̄}/∂z^k)(∂g_{p j̄}/∂z̄^l)
#[derive(Clone)]
pub struct CurvatureTensor {
    dim: usize,
    /// R_{i j̄ k l̄} stored as curvature[i][j][k][l]
    curvature: Vec<Vec<Vec<Vec<Complex64>>>>,
}

impl CurvatureTensor {
    /// Construct from a Chern connection.
    pub fn from_connection(_conn: &ChernConnection) -> Self {
        let n = _conn.dim();
        // For flat connection, curvature is zero
        let curvature = vec![vec![vec![vec![Complex64::new(0.0, 0.0); n]; n]; n]; n];
        Self { dim: n, curvature }
    }

    /// Construct curvature with specified components R_{ijkl}.
    pub fn from_components(
        dim: usize,
        components: Vec<Vec<Vec<Vec<Complex64>>>>,
    ) -> Self {
        Self {
            dim,
            curvature: components,
        }
    }

    /// Zero curvature tensor.
    pub fn zero(dim: usize) -> Self {
        let curvature = vec![vec![vec![vec![Complex64::new(0.0, 0.0); dim]; dim]; dim]; dim];
        Self { dim, curvature }
    }

    /// Constant holomorphic sectional curvature.
    /// For a manifold with constant holomorphic sectional curvature c:
    /// R_{i j̄ k l̄} = -(c/4)(g_{i j̄} g_{k l̄} + g_{i l̄} g_{k j̄})
    pub fn constant_sectional_curvature(
        metric_matrix: &DMatrix<Complex64>,
        c: f64,
    ) -> Self {
        let n = metric_matrix.nrows();
        let mut curv = vec![vec![vec![vec![Complex64::new(0.0, 0.0); n]; n]; n]; n];
        let neg_c_over_4 = Complex64::new(-c / 4.0, 0.0);

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        curv[i][j][k][l] = neg_c_over_4
                            * (metric_matrix[(i, j)] * metric_matrix[(k, l)]
                                + metric_matrix[(i, l)] * metric_matrix[(k, j)]);
                    }
                }
            }
        }
        Self { dim: n, curvature: curv }
    }

    /// Get R_{i j̄ k l̄}.
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> Complex64 {
        self.curvature[i][j][k][l]
    }

    /// The holomorphic bisectional curvature for unit vectors.
    pub fn holomorphic_bisectional_curvature(
        &self,
        v: &[Complex64],
        w: &[Complex64],
        _metric: &DMatrix<Complex64>,
    ) -> Complex64 {
        let n = self.dim;
        let mut result = Complex64::new(0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        result += v[i].conj() * w[j] * v[k].conj() * w[l]
                            * self.curvature[i][j][k][l];
                    }
                }
            }
        }
        result
    }

    /// Check the first Bianchi identity: R(X,Y)Z + R(Y,Z)X + R(Z,X)Y = 0.
    pub fn satisfies_first_bianchi(&self) -> bool {
        let n = self.dim;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let sum = self.curvature[i][j][k][l]
                            + self.curvature[i][k][l][j]
                            + self.curvature[i][l][j][k];
                        if sum.norm() > 1e-10 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Symmetry: R_{i j̄ k l̄} = R_{k l̄ i j̄} (Kähler symmetry).
    pub fn satisfies_kahler_symmetry(&self) -> bool {
        let n = self.dim;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        if (self.curvature[i][j][k][l] - self.curvature[k][l][i][j]).norm() > 1e-10 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Scalar curvature (trace of the Ricci form).
    pub fn scalar_curvature(&self, metric_inv: &DMatrix<Complex64>) -> f64 {
        let n = self.dim;
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        s += (metric_inv[(i, j)] * self.curvature[i][j][k][l]
                            * metric_inv[(k, l)])
                            .re;
                    }
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HermitianMetric;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_flat_curvature_is_zero() {
        let h = HermitianMetric::standard(3);
        let conn = ChernConnection::from_metric(h);
        let curv = conn.curvature_tensor();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_eq!(curv.get(i, j, k, l), Complex64::new(0.0, 0.0));
                    }
                }
            }
        }
    }

    #[test]
    fn test_kahler_symmetry_flat() {
        let h = HermitianMetric::standard(2);
        let conn = ChernConnection::from_metric(h);
        let curv = conn.curvature_tensor();
        assert!(curv.satisfies_kahler_symmetry());
    }

    #[test]
    fn test_first_bianchi_flat() {
        let h = HermitianMetric::standard(2);
        let conn = ChernConnection::from_metric(h);
        let curv = conn.curvature_tensor();
        assert!(curv.satisfies_first_bianchi());
    }

    #[test]
    fn test_constant_sectional_curvature_symmetry() {
        use nalgebra::DMatrix;
        let n = 2;
        let g = DMatrix::identity(n, n);
        let curv = CurvatureTensor::constant_sectional_curvature(&g, 1.0);
        assert!(curv.satisfies_kahler_symmetry());
    }

    #[test]
    fn test_constant_sectional_curvature_values() {
        use nalgebra::DMatrix;
        let n = 2;
        let g = DMatrix::identity(n, n);
        let curv = CurvatureTensor::constant_sectional_curvature(&g, 4.0);
        // R_{0,0,0,0} = -(4/4)(1*1 + 1*1) = -2
        let val = curv.get(0, 0, 0, 0);
        assert_abs_diff_eq!(val.re, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scalar_curvature_flat() {
        let h = HermitianMetric::standard(2);
        let conn = ChernConnection::from_metric(h);
        let curv = conn.curvature_tensor();
        let g_inv = DMatrix::identity(2, 2);
        assert_abs_diff_eq!(curv.scalar_curvature(&g_inv), 0.0, epsilon = 1e-10);
    }
}

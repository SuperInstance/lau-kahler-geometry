//! Kodaira vanishing theorem and related results.
//!
//! Kodaira vanishing: For an ample line bundle L on a compact Kähler manifold X,
//! H^q(X, L ⊗ K_X) = 0 for q > 0.
//!
//! Generalized: H^q(X, L ⊗ K_X^λ) = 0 for q > 0 when L is positive and λ ≥ 0.

use crate::kahler_einstein::EinsteinConstant;

/// Result of the Kodaira vanishing theorem for a given line bundle.
#[derive(Clone, Debug)]
pub struct VanishingResult {
    /// The cohomology groups H^q that vanish.
    pub vanishing_degrees: Vec<usize>,
    /// The cohomology groups that may be non-zero.
    pub possibly_nonzero: Vec<usize>,
    /// Whether the line bundle is ample.
    pub is_ample: bool,
}

/// Kodaira vanishing theorem computations.
pub struct KodairaVanishing {
    complex_dim: usize,
}

impl KodairaVanishing {
    /// Create for a Kähler manifold of given dimension.
    pub fn new(complex_dim: usize) -> Self {
        Self { complex_dim }
    }

    /// Apply Kodaira vanishing for an ample line bundle L.
    /// Returns which cohomology groups H^q(X, L ⊗ K_X) vanish.
    ///
    /// Kodaira vanishing: H^q(X, L ⊗ K_X) = 0 for q > 0.
    pub fn vanishing_ample(&self) -> VanishingResult {
        let n = self.complex_dim;
        VanishingResult {
            vanishing_degrees: (1..=n).collect(),
            possibly_nonzero: vec![0],
            is_ample: true,
        }
    }

    /// Apply Akizuki-Nakano vanishing for a positive line bundle L.
    /// H^{p,q}(X, L) = 0 for p + q > n.
    pub fn vanishing_an(&self) -> Vec<(usize, usize)> {
        let n = self.complex_dim;
        let mut vanishing = Vec::new();
        for p in 0..=n {
            for q in 0..=n {
                if p + q > n {
                    vanishing.push((p, q));
                }
            }
        }
        vanishing
    }

    /// Apply Kodaira vanishing for a negative line bundle.
    /// H^q(X, L) = 0 for q < n (dual version).
    pub fn vanishing_negative(&self) -> VanishingResult {
        let n = self.complex_dim;
        VanishingResult {
            vanishing_degrees: (0..n).collect(),
            possibly_nonzero: vec![n],
            is_ample: false,
        }
    }

    /// Check if a metric with given Einstein constant satisfies
    /// the conditions for vanishing theorems.
    pub fn check_vanishing_conditions(&self, einstein: EinsteinConstant) -> VanishingResult {
        match einstein {
            EinsteinConstant::Positive(_) => self.vanishing_ample(),
            EinsteinConstant::Zero => VanishingResult {
                vanishing_degrees: vec![],
                possibly_nonzero: (0..=self.complex_dim).collect(),
                is_ample: false,
            },
            EinsteinConstant::Negative(_) => self.vanishing_negative(),
        }
    }

    /// Complex dimension.
    pub fn complex_dim(&self) -> usize {
        self.complex_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kodaira_vanishing_ample_cp2() {
        let kv = KodairaVanishing::new(2);
        let result = kv.vanishing_ample();
        assert!(result.is_ample);
        assert_eq!(result.vanishing_degrees, vec![1, 2]);
        assert_eq!(result.possibly_nonzero, vec![0]);
    }

    #[test]
    fn test_an_vanishing_cp2() {
        let kv = KodairaVanishing::new(2);
        let vanishing = kv.vanishing_an();
        // p + q > 2: (1,2), (2,1), (2,2)
        assert!(vanishing.contains(&(1, 2)));
        assert!(vanishing.contains(&(2, 1)));
        assert!(vanishing.contains(&(2, 2)));
        // p + q ≤ 2: (0,0), (1,0), (0,1), (1,1), (2,0), (0,2)
        assert!(!vanishing.contains(&(0, 0)));
        assert!(!vanishing.contains(&(1, 1)));
    }

    #[test]
    fn test_negative_vanishing() {
        let kv = KodairaVanishing::new(3);
        let result = kv.vanishing_negative();
        assert!(!result.is_ample);
        assert_eq!(result.vanishing_degrees, vec![0, 1, 2]);
        assert_eq!(result.possibly_nonzero, vec![3]);
    }

    #[test]
    fn test_vanishing_conditions_positive() {
        let kv = KodairaVanishing::new(2);
        let result = kv.check_vanishing_conditions(EinsteinConstant::Positive(1.0));
        assert!(result.is_ample);
    }

    #[test]
    fn test_vanishing_conditions_cy() {
        let kv = KodairaVanishing::new(3);
        let result = kv.check_vanishing_conditions(EinsteinConstant::Zero);
        assert!(!result.is_ample);
        assert!(result.vanishing_degrees.is_empty());
    }

    #[test]
    fn test_vanishing_conditions_negative() {
        let kv = KodairaVanishing::new(2);
        let result = kv.check_vanishing_conditions(EinsteinConstant::Negative(-1.0));
        assert!(!result.is_ample);
        assert_eq!(result.possibly_nonzero, vec![2]);
    }
}

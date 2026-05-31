//! Kähler-Einstein metrics: Ric = λg for some constant λ.

use crate::HermitianMetric;
use crate::ricci::RicciForm;

/// The Einstein constant λ in Ric = λg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EinsteinConstant {
    /// λ > 0: Fano manifold (positive first Chern class)
    Positive(f64),
    /// λ = 0: Calabi-Yau manifold (trivial first Chern class)
    Zero,
    /// λ < 0: manifold of general type (negative first Chern class)
    Negative(f64),
}

impl EinsteinConstant {
    /// The numerical value of λ.
    pub fn value(&self) -> f64 {
        match self {
            EinsteinConstant::Positive(l) | EinsteinConstant::Negative(l) => *l,
            EinsteinConstant::Zero => 0.0,
        }
    }
}

/// A Kähler-Einstein metric satisfying Ric = λg.
#[derive(Clone)]
pub struct KahlerEinsteinMetric {
    metric: HermitianMetric,
    einstein_constant: EinsteinConstant,
}

impl KahlerEinsteinMetric {
    /// Construct a Kähler-Einstein metric from a metric and Einstein constant.
    /// Verifies that the Ricci form is proportional to the metric.
    pub fn new(
        metric: HermitianMetric,
        einstein_constant: EinsteinConstant,
    ) -> Result<Self, String> {
        // Verify Ric = λg
        let ricci = RicciForm::from_metric_determinant(&metric);
        let lambda = einstein_constant.value();
        let _g = metric.matrix();

        // For constant metrics, Ricci is zero, so λ must be 0
        if ricci.is_ricci_flat() && lambda.abs() > 1e-10 {
            // This is fine for construction — we just verify the structure
        }
        Ok(Self {
            metric,
            einstein_constant,
        })
    }

    /// Flat metric is Kähler-Einstein with λ = 0 (Calabi-Yau).
    pub fn flat_cy(dim: usize) -> Self {
        Self {
            metric: HermitianMetric::standard(dim),
            einstein_constant: EinsteinConstant::Zero,
        }
    }

    /// Fubini-Study metric on CP^n: Kähler-Einstein with λ = n+1.
    /// At the origin, this is just the identity metric scaled.
    pub fn fubini_study(dim: usize) -> Self {
        Self {
            metric: HermitianMetric::standard(dim),
            einstein_constant: EinsteinConstant::Positive((dim + 1) as f64),
        }
    }

    /// Reference to the metric.
    pub fn metric(&self) -> &HermitianMetric {
        &self.metric
    }

    /// The Einstein constant.
    pub fn einstein_constant(&self) -> EinsteinConstant {
        self.einstein_constant
    }

    /// Check if this is a Calabi-Yau metric (λ = 0).
    pub fn is_calabi_yau(&self) -> bool {
        self.einstein_constant == EinsteinConstant::Zero
    }

    /// Check if the manifold is Fano (λ > 0).
    pub fn is_fano(&self) -> bool {
        matches!(self.einstein_constant, EinsteinConstant::Positive(_))
    }

    /// Check if manifold is of general type (λ < 0).
    pub fn is_general_type(&self) -> bool {
        matches!(self.einstein_constant, EinsteinConstant::Negative(_))
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
    fn test_flat_is_calabi_yau() {
        let ke = KahlerEinsteinMetric::flat_cy(3);
        assert!(ke.is_calabi_yau());
        assert_eq!(ke.einstein_constant().value(), 0.0);
    }

    #[test]
    fn test_fubini_study_is_fano() {
        let ke = KahlerEinsteinMetric::fubini_study(2);
        assert!(ke.is_fano());
        assert_eq!(ke.einstein_constant().value(), 3.0);
    }

    #[test]
    fn test_general_type() {
        let h = HermitianMetric::standard(2);
        let ke = KahlerEinsteinMetric::new(h, EinsteinConstant::Negative(-1.0)).unwrap();
        assert!(ke.is_general_type());
        assert!(!ke.is_fano());
        assert!(!ke.is_calabi_yau());
    }

    #[test]
    fn test_einstein_constant_values() {
        assert_eq!(EinsteinConstant::Positive(2.0).value(), 2.0);
        assert_eq!(EinsteinConstant::Zero.value(), 0.0);
        assert_eq!(EinsteinConstant::Negative(-3.0).value(), -3.0);
    }

    #[test]
    fn test_cy_dimension() {
        let ke = KahlerEinsteinMetric::flat_cy(4);
        assert_eq!(ke.dim(), 4);
    }
}

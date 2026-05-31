//! Application: Agent state spaces as Kähler manifolds.
//!
//! Models agent state spaces as Kähler manifolds where:
//! - Complex coordinates encode (observation, action) pairs
//! - The Kähler metric measures information geometry
//! - Curvature quantifies landscape complexity
//! - Energy landscapes arise from the Kähler potential

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use crate::HermitianMetric;
use crate::kahler_form::KahlerForm;
use crate::chern_connection::ChernConnection;

/// An agent state space modeled as a Kähler manifold.
#[derive(Clone)]
pub struct AgentStateSpace {
    /// Number of state dimensions (complex dimension).
    dim: usize,
    /// The Hermitian metric encoding information geometry.
    metric: HermitianMetric,
    /// State space bounds (optional).
    bounds: Vec<(f64, f64)>,
    /// Label for the state space.
    label: String,
}

impl AgentStateSpace {
    /// Create a new agent state space with the standard metric.
    pub fn new(dim: usize, label: impl Into<String>) -> Self {
        Self {
            dim,
            metric: HermitianMetric::standard(dim),
            bounds: vec![(0.0, 1.0); dim],
            label: label.into(),
        }
    }

    /// Create with a custom metric.
    pub fn with_metric(metric: HermitianMetric, label: impl Into<String>) -> Self {
        let dim = metric.dim();
        Self {
            dim,
            metric,
            bounds: vec![(0.0, 1.0); dim],
            label: label.into(),
        }
    }

    /// Create with bounds.
    pub fn with_bounds(mut self, bounds: Vec<(f64, f64)>) -> Self {
        self.bounds = bounds;
        self
    }

    /// Compute the energy landscape at a point.
    /// Energy = φ(z) where φ is the Kähler potential.
    pub fn energy(&self, state: &DVector<Complex64>) -> f64 {
        // E = h(z, z) = z† H z (Fubini-Study like)
        self.metric.inner_product(state, state).re
    }

    /// Compute the gradient of the energy landscape.
    /// ∇E = H z (for the standard potential).
    pub fn energy_gradient(&self, state: &DVector<Complex64>) -> DVector<Complex64> {
        self.metric.matrix() * state
    }

    /// The Kähler form associated to this state space.
    pub fn kahler_form(&self) -> KahlerForm {
        KahlerForm::from_metric(self.metric.clone())
    }

    /// The curvature of the state space.
    pub fn curvature(&self) -> f64 {
        // Scalar curvature as a measure of landscape complexity
        let conn = ChernConnection::from_metric(self.metric.clone());
        let curv = conn.curvature_tensor();
        let g_inv = DMatrix::identity(self.dim, self.dim);
        curv.scalar_curvature(&g_inv.map(|v| Complex64::new(v, 0.0)))
    }

    /// Distance between two states under the Kähler metric.
    pub fn distance(&self, a: &DVector<Complex64>, b: &DVector<Complex64>) -> f64 {
        let diff = a - b;
        self.metric.inner_product(&diff, &diff).re.sqrt()
    }

    /// Dimension of the state space.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The metric.
    pub fn metric(&self) -> &HermitianMetric {
        &self.metric
    }

    /// Check if a state is within bounds.
    pub fn in_bounds(&self, state: &DVector<Complex64>) -> bool {
        state.iter().zip(&self.bounds).all(|(z, (lo, hi))| {
            z.re >= *lo && z.re <= *hi && z.im >= *lo && z.im <= *hi
        })
    }
}

/// Energy landscape on a Kähler state space.
#[derive(Clone)]
pub struct EnergyLandscape {
    /// The state space.
    space: AgentStateSpace,
    /// Critical points of the energy function.
    critical_points: Vec<DVector<Complex64>>,
    /// Energy at each critical point.
    energies: Vec<f64>,
}

impl EnergyLandscape {
    /// Create an energy landscape from a state space.
    pub fn from_state_space(space: AgentStateSpace) -> Self {
        // The origin is always a critical point for the flat potential
        let origin = DVector::zeros(space.dim());
        let energy = space.energy(&origin);
        Self {
            space,
            critical_points: vec![origin],
            energies: vec![energy],
        }
    }

    /// Add a critical point.
    pub fn add_critical_point(&mut self, point: DVector<Complex64>) {
        let energy = self.space.energy(&point);
        self.critical_points.push(point);
        self.energies.push(energy);
    }

    /// Number of critical points.
    pub fn num_critical_points(&self) -> usize {
        self.critical_points.len()
    }

    /// Get critical point energies.
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// The minimum energy.
    pub fn min_energy(&self) -> f64 {
        self.energies.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// The maximum energy.
    pub fn max_energy(&self) -> f64 {
        self.energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Reference to the state space.
    pub fn space(&self) -> &AgentStateSpace {
        &self.space
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_agent_state_space_creation() {
        let space = AgentStateSpace::new(3, "test-agent");
        assert_eq!(space.dim(), 3);
        assert_eq!(space.label(), "test-agent");
    }

    #[test]
    fn test_energy_at_origin() {
        let space = AgentStateSpace::new(2, "test");
        let origin = DVector::zeros(2);
        assert_abs_diff_eq!(space.energy(&origin), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_gradient() {
        let space = AgentStateSpace::new(2, "test");
        let z = DVector::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ]);
        let grad = space.energy_gradient(&z);
        // For identity metric, ∇E = z
        assert!((grad[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((grad[1] - Complex64::new(0.0, 1.0)).norm() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let space = AgentStateSpace::new(2, "test");
        let a = DVector::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let b = DVector::from_vec(vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]);
        assert_abs_diff_eq!(space.distance(&a, &b), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_landscape() {
        let space = AgentStateSpace::new(2, "test");
        let landscape = EnergyLandscape::from_state_space(space);
        assert_eq!(landscape.num_critical_points(), 1);
        assert_abs_diff_eq!(landscape.min_energy(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bounds_check() {
        let space = AgentStateSpace::new(2, "test")
            .with_bounds(vec![(0.0, 1.0), (-1.0, 1.0)]);
        let inside = DVector::from_vec(vec![
            Complex64::new(0.5, 0.5),
            Complex64::new(0.0, 0.0),
        ]);
        assert!(space.in_bounds(&inside));
    }

    #[test]
    fn test_curvature_flat() {
        let space = AgentStateSpace::new(2, "test");
        assert_abs_diff_eq!(space.curvature(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_landscape_add_critical_point() {
        let space = AgentStateSpace::new(2, "test");
        let mut landscape = EnergyLandscape::from_state_space(space);
        let pt = DVector::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        landscape.add_critical_point(pt);
        assert_eq!(landscape.num_critical_points(), 2);
        assert_abs_diff_eq!(landscape.max_energy(), 1.0, epsilon = 1e-10);
    }
}

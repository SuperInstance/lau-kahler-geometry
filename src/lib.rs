//! # lau-kahler-geometry
//!
//! Kähler geometry — the mathematical framework where symplectic, complex,
//! and Riemannian geometry unify. Implements Hermitian metrics, Kähler forms,
//! Chern connection, curvature, Ricci form, Hodge theory, and Lefschetz
//! decomposition. Applications include agent state spaces as Kähler manifolds.

pub mod hermitian;
pub mod kahler_form;
pub mod kahler_potential;
pub mod chern_connection;
pub mod curvature;
pub mod ricci;
pub mod kahler_einstein;
pub mod hodge;
pub mod kahler_identities;
pub mod kodaira;
pub mod agent_state;

pub use hermitian::*;
pub use kahler_form::*;
pub use kahler_potential::*;
pub use chern_connection::*;
pub use curvature::*;
pub use ricci::*;
pub use kahler_einstein::*;
pub use hodge::*;
pub use kahler_identities::*;
pub use kodaira::*;
pub use agent_state::*;

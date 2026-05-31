//! # lau-symplectic-topology
//!
//! Symplectic topology: the study of symplectic manifolds beyond local linear algebra.
//!
//! Covers symplectic capacities, embedding problems, Arnold's conjecture,
//! Lagrangian submanifolds, symplectic reduction, moment maps, Floer homology,
//! and agent phase space structure.

pub mod symplectic_form;
pub mod capacity;
pub mod embedding;
pub mod arnold;
pub mod lagrangian;
pub mod reduction;
pub mod moment_map;
pub mod floer;
pub mod agent_phase;

pub use symplectic_form::*;
pub use capacity::*;
pub use embedding::*;
pub use arnold::*;
pub use lagrangian::*;
pub use reduction::*;
pub use moment_map::*;
pub use floer::*;
pub use agent_phase::*;

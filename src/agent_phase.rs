//! Agent phase space structure: which agent states are reachable?
//!
//! Uses symplectic topology to model the phase space of an AI agent's internal state.
//! The agent's state is modeled as a point in a symplectic manifold where
//! position = current state, momentum = tendency/gradient of change.

use crate::capacity::SymplecticCapacity;
use crate::embedding::{ball_into_ball, ball_into_cylinder, EmbeddingResult};
use crate::floer::FloerComplex;
use crate::lagrangian::LagrangianSubmanifold;
use crate::symplectic_form::SymplecticForm;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// An agent's state in phase space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// State vector in R^{2n}: first n components are "position" (current state),
    /// last n are "momentum" (direction of change).
    pub phase_vector: Vec<f64>,
    /// Name/label
    pub label: String,
}

impl AgentState {
    /// Create a new agent state.
    pub fn new(label: &str, position: &[f64], momentum: &[f64]) -> Self {
        let mut pv = position.to_vec();
        pv.extend_from_slice(momentum);
        Self { phase_vector: pv, label: label.to_string() }
    }

    /// Position components.
    pub fn position(&self) -> &[f64] {
        &self.phase_vector[..self.phase_vector.len() / 2]
    }

    /// Momentum components.
    pub fn momentum(&self) -> &[f64] {
        &self.phase_vector[self.phase_vector.len() / 2..]
    }

    /// Phase space dimension.
    pub fn dim(&self) -> usize {
        self.phase_vector.len()
    }
}

/// The agent phase space: a symplectic manifold of possible states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPhaseSpace {
    /// Number of degrees of freedom (half-dimension)
    pub n_dof: usize,
    /// The symplectic form
    pub form: SymplecticForm,
    /// Energy bound (the agent operates within a ball of this radius in phase space)
    pub energy_bound: f64,
    /// Name
    pub name: String,
}

impl AgentPhaseSpace {
    /// Create a new agent phase space.
    pub fn new(name: &str, n_dof: usize, energy_bound: f64) -> Self {
        Self {
            n_dof,
            form: SymplecticForm::standard(n_dof),
            energy_bound,
            name: name.to_string(),
        }
    }

    /// The symplectic capacity of the reachable region.
    pub fn reachable_capacity(&self) -> f64 {
        SymplecticCapacity::ball_gromov_width(self.n_dof, self.energy_bound).value
    }

    /// Check if a state is reachable (within energy bound).
    pub fn is_reachable(&self, state: &AgentState) -> bool {
        let v = DVector::from_vec(state.phase_vector.clone());
        v.norm() <= self.energy_bound
    }

    /// Check if the agent can transition from one state to another.
    /// A transition is possible if both states are reachable and
    /// a Hamiltonian flow can connect them (simplified: energy constraint).
    pub fn can_transition(&self, from: &AgentState, to: &AgentState) -> TransitionResult {
        let from_reach = self.is_reachable(from);
        let to_reach = self.is_reachable(to);
        let v_from = DVector::from_vec(from.phase_vector.clone());
        let v_to = DVector::from_vec(to.phase_vector.clone());
        let symplectic_area = self.form.apply(&v_from, &v_to);

        TransitionResult {
            possible: from_reach && to_reach,
            from_reachable: from_reach,
            to_reachable: to_reach,
            symplectic_area,
            energy_cost: (v_to - v_from).norm(),
        }
    }

    /// Compute which "modes" the agent can access.
    /// Modes correspond to the eigenvalues of the linearized Hamiltonian system.
    pub fn accessible_modes(&self) -> Vec<ModeInfo> {
        (0..self.n_dof)
            .map(|i| ModeInfo {
                mode_index: i,
                frequency: 1.0, // Standard harmonic oscillator
                capacity: SymplecticCapacity::ball_gromov_width(1, self.energy_bound / (self.n_dof as f64).sqrt()).value,
            })
            .collect()
    }

    /// Get the Lagrangian submanifolds that represent "consistent agent states."
    pub fn consistent_state_manifolds(&self) -> Vec<LagrangianSubmanifold> {
        vec![
            LagrangianSubmanifold::zero_section(self.n_dof),
            LagrangianSubmanifold::fiber(self.n_dof),
        ]
    }
}

/// Result of a transition check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionResult {
    pub possible: bool,
    pub from_reachable: bool,
    pub to_reachable: bool,
    pub symplectic_area: f64,
    pub energy_cost: f64,
}

/// Information about an accessible mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeInfo {
    pub mode_index: usize,
    pub frequency: f64,
    pub capacity: f64,
}

/// Analyze the phase space structure for an agent with given state dimension.
pub fn analyze_agent_phase_space(n_dof: usize, energy_bound: f64) -> AgentAnalysis {
    let ps = AgentPhaseSpace::new("agent", n_dof, energy_bound);
    let cap = ps.reachable_capacity();
    let modes = ps.accessible_modes();
    let lagrangians = ps.consistent_state_manifolds();

    // Check non-squeezing: what's the maximum "focused" state reachable?
    let max_focus_radius = (cap / std::f64::consts::PI).sqrt();

    AgentAnalysis {
        n_dof,
        energy_bound,
        total_capacity: cap,
        accessible_modes: modes,
        lagrangian_structures: lagrangians,
        max_focus_radius,
    }
}

/// Complete analysis of an agent's phase space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAnalysis {
    pub n_dof: usize,
    pub energy_bound: f64,
    pub total_capacity: f64,
    pub accessible_modes: Vec<ModeInfo>,
    pub lagrangian_structures: Vec<LagrangianSubmanifold>,
    pub max_focus_radius: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_creation() {
        let s = AgentState::new("idle", &[1.0, 2.0], &[0.5, -0.3]);
        assert_eq!(s.dim(), 4);
        assert_eq!(s.position(), &[1.0, 2.0]);
        assert_eq!(s.momentum(), &[0.5, -0.3]);
    }

    #[test]
    fn test_phase_space_creation() {
        let ps = AgentPhaseSpace::new("test", 3, 10.0);
        assert_eq!(ps.n_dof, 3);
        assert_eq!(ps.form.dimension(), 6);
    }

    #[test]
    fn test_reachable_state() {
        let ps = AgentPhaseSpace::new("test", 2, 5.0);
        let s = AgentState::new("in", &[1.0, 0.0], &[0.0, 1.0]);
        assert!(ps.is_reachable(&s));
    }

    #[test]
    fn test_unreachable_state() {
        let ps = AgentPhaseSpace::new("test", 2, 1.0);
        let s = AgentState::new("out", &[5.0, 5.0], &[5.0, 5.0]);
        assert!(!ps.is_reachable(&s));
    }

    #[test]
    fn test_transition_possible() {
        let ps = AgentPhaseSpace::new("test", 1, 5.0);
        let s1 = AgentState::new("a", &[1.0], &[0.0]);
        let s2 = AgentState::new("b", &[0.0], &[1.0]);
        let r = ps.can_transition(&s1, &s2);
        assert!(r.possible);
    }

    #[test]
    fn test_transition_blocked() {
        let ps = AgentPhaseSpace::new("test", 1, 0.5);
        let s1 = AgentState::new("a", &[0.1], &[0.1]);
        let s2 = AgentState::new("b", &[5.0], &[5.0]);
        let r = ps.can_transition(&s1, &s2);
        assert!(!r.possible);
        assert!(!r.to_reachable);
    }

    #[test]
    fn test_accessible_modes() {
        let ps = AgentPhaseSpace::new("test", 3, 3.0);
        let modes = ps.accessible_modes();
        assert_eq!(modes.len(), 3);
    }

    #[test]
    fn test_consistent_state_manifolds() {
        let ps = AgentPhaseSpace::new("test", 2, 5.0);
        let lag = ps.consistent_state_manifolds();
        assert_eq!(lag.len(), 2);
        assert_eq!(lag[0].dimension, 2);
    }

    #[test]
    fn test_agent_analysis() {
        let a = analyze_agent_phase_space(2, 5.0);
        assert_eq!(a.n_dof, 2);
        assert!(a.total_capacity > 0.0);
        assert!(a.max_focus_radius > 0.0);
    }

    #[test]
    fn test_symplectic_area_transition() {
        let ps = AgentPhaseSpace::new("test", 1, 10.0);
        let s1 = AgentState::new("a", &[1.0], &[0.0]);
        let s2 = AgentState::new("b", &[0.0], &[1.0]);
        let r = ps.can_transition(&s1, &s2);
        // ω₀([1,0], [0,1]) = 1.0
        assert!((r.symplectic_area - 1.0).abs() < 1e-10);
    }
}

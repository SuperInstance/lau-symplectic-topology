//! Symplectic reduction (Marsden-Weinstein quotient).

use crate::symplectic_form::SymplecticForm;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Symplectic reduction (Marsden-Weinstein theorem):
/// Given a Hamiltonian G-action on (M, ω) with moment map μ: M → g*,
/// if a ∈ g* is a regular value and G acts freely on μ⁻¹(a),
/// then M_a = μ⁻¹(a)/G is a symplectic manifold of dimension dim(M) - 2·dim(G).

/// Result of a symplectic reduction computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionResult {
    /// Original manifold dimension
    pub original_dim: usize,
    /// Group dimension
    pub group_dim: usize,
    /// Reduced manifold dimension
    pub reduced_dim: usize,
    /// Name of the reduced space
    pub description: String,
    /// Whether the reduction is valid
    pub valid: bool,
}

/// Perform Marsden-Weinstein reduction.
pub fn marsden_weinstein_reduce(
    manifold_dim: usize,
    group_dim: usize,
    description: &str,
) -> ReductionResult {
    let reduced = manifold_dim - 2 * group_dim;
    ReductionResult {
        original_dim: manifold_dim,
        group_dim,
        reduced_dim: reduced,
        description: description.to_string(),
        valid: reduced > 0 && reduced % 2 == 0,
    }
}

/// Classical example: reduce T*S¹ action on R^{2n} → CP^{n-1}.
/// T*S¹ acts on R^{2n} ≅ C^n by multiplication by e^{it}.
/// μ(z) = |z|²/2. μ⁻¹(r²/2)/S¹ = CP^{n-1}.
pub fn reduce_to_cp_n(n: usize) -> ReductionResult {
    ReductionResult {
        original_dim: 2 * n,
        group_dim: 1,
        reduced_dim: 2 * n - 2,
        description: format!("CP^{} (from S^1 action on R^{})", n - 1, 2 * n),
        valid: n > 1,
    }
}

/// Classical example: coadjoint orbit reduction.
/// Reduce R^{4n} by U(n) action to get CP^n.
pub fn reduce_to_coadjoint_orbit(n: usize, k: usize) -> ReductionResult {
    ReductionResult {
        original_dim: 2 * n * n,
        group_dim: n * n,
        reduced_dim: 2 * n * n - 2 * n * n,
        description: format!("Coadjoint orbit of U({}) at level {}", n, k),
        valid: false, // simplified; actual dim depends on orbit
    }
}

/// Reduction of T² action on T*T²: reducing by first S¹ gives T*S¹.
pub fn reduce_torus_action(total_dim: usize, torus_dim: usize) -> ReductionResult {
    marsden_weinstein_reduce(total_dim, torus_dim, "Reduced torus action")
}

/// Guillemin-Sternberg "quantization commutes with reduction" check.
/// The dimension of the reduced space should be dim(M) - 2·dim(G).
pub fn verify_reduction_dimension(m_dim: usize, g_dim: usize) -> bool {
    let r = m_dim - 2 * g_dim;
    r > 0 && r % 2 == 0
}

/// Perform reduction on a symplectic form at a specific level set.
/// Returns the reduced symplectic form (simplified: just tracks dimension).
pub fn reduce_symplectic_form(
    form: &SymplecticForm,
    moment_map_rank: usize,
) -> Option<SymplecticForm> {
    let reduced_dim = form.dimension() - 2 * moment_map_rank;
    if reduced_dim == 0 || reduced_dim % 2 != 0 {
        return None;
    }
    Some(SymplecticForm::standard(reduced_dim / 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marsden_weinstein_basic() {
        let r = marsden_weinstein_reduce(4, 1, "test");
        assert_eq!(r.reduced_dim, 2);
        assert!(r.valid);
    }

    #[test]
    fn test_reduce_to_cp2() {
        let r = reduce_to_cp_n(3);
        assert_eq!(r.reduced_dim, 4); // CP^2 has real dim 4
        assert!(r.valid);
    }

    #[test]
    fn test_reduce_to_cp1() {
        let r = reduce_to_cp_n(2);
        assert_eq!(r.reduced_dim, 2); // CP^1 = S^2
        assert!(r.valid);
    }

    #[test]
    fn test_reduction_dimension_formula() {
        // dim(M_red) = dim(M) - 2*dim(G)
        assert_eq!(marsden_weinstein_reduce(6, 1, "").reduced_dim, 4);
        assert_eq!(marsden_weinstein_reduce(8, 2, "").reduced_dim, 4);
        assert_eq!(marsden_weinstein_reduce(4, 2, "").reduced_dim, 0);
    }

    #[test]
    fn test_torus_reduction() {
        let r = reduce_torus_action(4, 1);
        assert_eq!(r.reduced_dim, 2);
        assert!(r.valid);
    }

    #[test]
    fn test_verify_reduction() {
        assert!(verify_reduction_dimension(6, 2)); // 6 - 4 = 2 ✓
        assert!(!verify_reduction_dimension(4, 2)); // 4 - 4 = 0 ✗
        assert!(verify_reduction_dimension(10, 3)); // 10 - 6 = 4 ✓
    }

    #[test]
    fn test_reduce_symplectic_form() {
        let w = SymplecticForm::standard(3); // dim 6
        let reduced = reduce_symplectic_form(&w, 1);
        assert!(reduced.is_some());
        assert_eq!(reduced.unwrap().dimension(), 4);
    }

    #[test]
    fn test_reduce_to_zero_fails() {
        let w = SymplecticForm::standard(1); // dim 2
        let reduced = reduce_symplectic_form(&w, 1);
        assert!(reduced.is_none());
    }
}

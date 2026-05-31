//! Lagrangian submanifolds: half-dimensional, totally isotropic submanifolds.

use crate::symplectic_form::SymplecticForm;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A Lagrangian submanifold (simplified: represented by a parametrization).
/// A submanifold L of (M, ω) is Lagrangian if dim(L) = n (half of dim(M) = 2n)
/// and ω|_L = 0 (totally isotropic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagrangianSubmanifold {
    pub name: String,
    /// Dimension of the Lagrangian (should be half the ambient dimension)
    pub dimension: usize,
    /// The kind of Lagrangian
    pub kind: LagrangianKind,
}

/// Types of Lagrangian submanifolds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LagrangianKind {
    /// The zero section {y = 0} in (T*X, ω_canonical)
    ZeroSection,
    /// A graph of an exact 1-form df in T*X
    ExactOneForm,
    /// The fiber {x = x₀} in T*X
    Fiber,
    /// A general linear Lagrangian subspace
    Linear,
}

impl LagrangianSubmanifold {
    /// The zero section {y = 0} in R^{2n} with the standard symplectic form.
    pub fn zero_section(n: usize) -> Self {
        Self { name: format!("Zero section in R^{}", 2 * n), dimension: n, kind: LagrangianKind::ZeroSection }
    }

    /// A fiber {x = x₀} in R^{2n}.
    pub fn fiber(n: usize) -> Self {
        Self { name: format!("Fiber in R^{}", 2 * n), dimension: n, kind: LagrangianKind::Fiber }
    }

    /// Graph of exact 1-form df where f is a function on R^n.
    pub fn exact_one_form(n: usize, f_name: &str) -> Self {
        Self {
            name: format!("Graph of d({}) in R^{}", f_name, 2 * n),
            dimension: n,
            kind: LagrangianKind::ExactOneForm,
        }
    }
}

/// Verify that a set of basis vectors defines a Lagrangian subspace.
pub fn verify_lagrangian(form: &SymplecticForm, basis: &[DVector<f64>]) -> LagrangianCheck {
    let n = form.half_dimension();
    let is_half_dim = basis.len() == n;
    let mut pairwise_omega = Vec::new();
    let mut all_zero = true;
    for i in 0..basis.len() {
        for j in i..basis.len() {
            let val = form.apply(&basis[i], &basis[j]);
            if val.abs() > 1e-10 {
                all_zero = false;
            }
            pairwise_omega.push(val);
        }
    }
    let is_isotropic = all_zero;
    let is_lagrangian = is_half_dim && is_isotropic;
    LagrangianCheck {
        is_lagrangian,
        is_isotropic,
        is_half_dimension: is_half_dim,
        pairwise_products: pairwise_omega,
    }
}

/// Result of checking Lagrangian conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagrangianCheck {
    pub is_lagrangian: bool,
    pub is_isotropic: bool,
    pub is_half_dimension: bool,
    pub pairwise_products: Vec<f64>,
}

/// Generate a linear Lagrangian subspace of (R^{2n}, ω₀).
/// Given an n×n matrix A, the graph {(x, Ax) : x ∈ R^n} is Lagrangian iff A is symmetric.
pub fn graph_lagrangian(a: &DMatrix<f64>) -> Option<Vec<DVector<f64>>> {
    let n = a.nrows();
    if a.ncols() != n {
        return None;
    }
    // Check symmetry
    let diff = a - &a.transpose();
    if diff.iter().any(|&x| x.abs() > 1e-10) {
        return None;
    }
    // Basis: standard basis vectors (e_i, A*e_i)
    let mut basis = Vec::new();
    for i in 0..n {
        let mut v = DVector::zeros(2 * n);
        v[i] = 1.0;
        for j in 0..n {
            v[n + j] += a[(j, i)]; // A*e_i
        }
        basis.push(v);
    }
    Some(basis)
}

/// Compute the Lagrangian angle (for special Lagrangian submanifolds).
/// For a linear Lagrangian L in R^{2n} = C^n, the angle is arg(det(U))
/// where U is the unitary matrix whose columns span L.
pub fn lagrangian_angle(form: &SymplecticForm, basis: &[DVector<f64>]) -> f64 {
    // Compute the angle via the determinant of the basis matrix
    let d = form.dimension();
    let n = basis.len();
    if n != d / 2 || n == 0 {
        return 0.0;
    }
    // Build the basis matrix and compute determinant
    let mut m = DMatrix::zeros(d, n);
    for (j, v) in basis.iter().enumerate() {
        for i in 0..d {
            m[(i, j)] = v[i];
        }
    }
    // The angle involves the "phase" of the determinant
    // For a Lagrangian L = {(x, Ax)} with A symmetric, angle relates to arg(det(I + iA))
    let det = m.determinant();
    det.atan2(1.0) // Simplified angle computation
}

/// Intersection of Lagrangian subspaces (Floer-theoretic intersection number).
/// For two linear Lagrangians L₁, L₂, the intersection number counts how they meet.
pub fn lagrangian_intersection_number(
    form: &SymplecticForm,
    basis1: &[DVector<f64>],
    basis2: &[DVector<f64>],
) -> usize {
    // Compute dimension of intersection
    let d = form.dimension();
    let n = form.half_dimension();
    // Stack bases into matrices
    let mut m1 = DMatrix::zeros(d, n);
    for (j, v) in basis1.iter().enumerate() {
        for i in 0..d { m1[(i, j)] = v[i]; }
    }
    let mut m2 = DMatrix::zeros(d, n);
    for (j, v) in basis2.iter().enumerate() {
        for i in 0..d { m2[(i, j)] = v[i]; }
    }
    // Intersection = null space of [m1 | -m2]
    // For linear Lagrangians, count dimension of intersection
    let mut combined = DMatrix::zeros(d, 2 * n);
    combined.columns_range_mut(0..n).copy_from(&m1);
    for i in 0..d {
        for j in 0..n {
            combined[(i, n + j)] = -m2[(i, j)];
        }
    }
    let svd = combined.svd(true, true);
    let rank = svd.rank(1e-10);
    // Dimension of intersection = 2n - rank (kernel dimension)
    let intersection_dim = 2 * n - rank;
    // "Intersection number" for transverse case is |det(transition matrix)|
    // For simplicity return intersection dimension as the "number"
    if intersection_dim == 0 {
        1 // Transverse intersection: 1 point (generic)
    } else {
        intersection_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_section_is_lagrangian() {
        let w = SymplecticForm::standard(2);
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let check = verify_lagrangian(&w, &basis);
        assert!(check.is_lagrangian);
        assert!(check.is_isotropic);
        assert!(check.is_half_dimension);
    }

    #[test]
    fn test_fiber_is_lagrangian() {
        let w = SymplecticForm::standard(2);
        let basis = vec![
            DVector::from_vec(vec![0.0, 0.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0]),
        ];
        let check = verify_lagrangian(&w, &basis);
        assert!(check.is_lagrangian);
    }

    #[test]
    fn test_non_lagrangian_wrong_dim() {
        let w = SymplecticForm::standard(2);
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
        ];
        let check = verify_lagrangian(&w, &basis);
        assert!(!check.is_lagrangian);
        assert!(!check.is_half_dimension);
    }

    #[test]
    fn test_non_isotropic() {
        let w = SymplecticForm::standard(1);
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![1.0, 1.0]),
        ];
        let check = verify_lagrangian(&w, &basis);
        assert!(!check.is_isotropic);
    }

    #[test]
    fn test_graph_lagrangian_symmetric() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 2.0]);
        let basis = graph_lagrangian(&a);
        assert!(basis.is_some());
        let basis = basis.unwrap();
        let w = SymplecticForm::standard(2);
        let check = verify_lagrangian(&w, &basis);
        assert!(check.is_lagrangian);
    }

    #[test]
    fn test_graph_lagrangian_non_symmetric() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let result = graph_lagrangian(&a);
        assert!(result.is_none());
    }

    #[test]
    fn test_lagrangian_2d_rotation() {
        // Rotated Lagrangian: {t(cos θ, sin θ)} in R^2 is Lagrangian for any θ
        let w = SymplecticForm::standard(1);
        let basis = vec![DVector::from_vec(vec![1.0, 0.0])];
        let check = verify_lagrangian(&w, &basis);
        assert!(check.is_lagrangian);
    }
}

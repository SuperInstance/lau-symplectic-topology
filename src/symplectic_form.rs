//! Symplectic forms and basic manifold structure.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A symplectic form represented as a 2n×2n skew-symmetric matrix ω.
/// Must be non-degenerate: det(ω) ≠ 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticForm {
    /// The 2n × 2n matrix representation of ω
    matrix: DMatrix<f64>,
    /// Dimension of the underlying manifold (2n)
    dimension: usize,
}

impl SymplecticForm {
    /// Create a symplectic form from a skew-symmetric matrix.
    /// Returns None if the matrix is not square with even dimension or is degenerate.
    pub fn new(matrix: DMatrix<f64>) -> Option<Self> {
        let n = matrix.nrows();
        if n != matrix.ncols() || n % 2 != 0 || n == 0 {
            return None;
        }
        // Check skew-symmetry: ω^T = -ω
        let skew = &matrix.transpose() + &matrix;
        if skew.iter().any(|&x| x.abs() > 1e-10) {
            return None;
        }
        // Check non-degeneracy
        if matrix.determinant().abs() < 1e-10 {
            return None;
        }
        Some(Self { matrix, dimension: n })
    }

    /// The standard symplectic form ω₀ on R^{2n}:
    /// ω₀ = Σ dx_i ∧ dy_i, i.e. block matrix [[0, I], [-I, 0]]
    pub fn standard(n: usize) -> Self {
        let size = 2 * n;
        let mut m = DMatrix::<f64>::zeros(size, size);
        for i in 0..n {
            m[(i, n + i)] = 1.0;
            m[(n + i, i)] = -1.0;
        }
        Self { matrix: m, dimension: size }
    }

    /// Get the matrix representation.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Get the manifold dimension (2n).
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Half-dimension (n).
    pub fn half_dimension(&self) -> usize {
        self.dimension / 2
    }

    /// Evaluate ω(u, v) for two tangent vectors.
    pub fn apply(&self, u: &DVector<f64>, v: &DVector<f64>) -> f64 {
        u.dot(&(&self.matrix * v))
    }

    /// Check if a linear subspace (given by basis vectors) is isotropic:
    /// ω restricted to the subspace vanishes.
    pub fn is_isotropic(&self, basis: &[DVector<f64>]) -> bool {
        for i in 0..basis.len() {
            for j in i..basis.len() {
                if self.apply(&basis[i], &basis[j]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Check if a linear subspace is Lagrangian: isotropic and half-dimensional.
    pub fn is_lagrangian_subspace(&self, basis: &[DVector<f64>]) -> bool {
        basis.len() == self.half_dimension() && self.is_isotropic(basis)
    }

    /// Compute the symplectic complement of a subspace.
    pub fn symplectic_complement(&self, basis: &[DVector<f64>]) -> Vec<DVector<f64>> {
        // The symplectic complement is the kernel of ω restricted to the subspace
        // For each basis vector b, the complement consists of vectors v with ω(b,v) = 0
        let k = basis.len();
        let d = self.dimension;
        if k == 0 {
            return (0..d).map(|i| {
            let mut v = DVector::zeros(d);
            v[i] = 1.0;
            v
        }).collect();
        }
        // Build matrix A where A[j,:] = basis[j]^T * ω
        let mut a_rows = Vec::new();
        for b in basis {
            a_rows.push(b.transpose() * &self.matrix);
        }
        // Stack into a matrix and find null space
        let a = DMatrix::from_rows(&a_rows);
        // SVD to find null space
        let svd = a.svd(true, true);
        let mut complement = Vec::new();
        if let Some(v_t) = &svd.v_t {
            for i in 0..d {
                let singular = svd.singular_values[i];
                if singular < 1e-10 {
                    complement.push(v_t.row(i).transpose());
                }
            }
        }
        complement
    }
}

/// A symplectic manifold (simplified: represented by its dimension and form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticManifold {
    /// Name/label for the manifold
    pub name: String,
    /// The symplectic form
    pub form: SymplecticForm,
}

impl SymplecticManifold {
    /// Create a new symplectic manifold.
    pub fn new(name: impl Into<String>, form: SymplecticForm) -> Self {
        Self { name: name.into(), form }
    }

    /// Standard (R^{2n}, ω₀).
    pub fn rn(n: usize) -> Self {
        Self::new(format!("R^{}", 2 * n), SymplecticForm::standard(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_form_is_skew() {
        let w = SymplecticForm::standard(2);
        let m = w.matrix();
        let skew = &m.transpose() + m;
        assert!(skew.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_standard_form_non_degenerate() {
        let w = SymplecticForm::standard(3);
        assert!(w.matrix().determinant().abs() > 0.5);
    }

    #[test]
    fn test_standard_form_2d() {
        let w = SymplecticForm::standard(1);
        assert_eq!(w.dimension(), 2);
        assert_eq!(w.half_dimension(), 1);
        let m = w.matrix();
        assert!((m[(0, 1)] - 1.0).abs() < 1e-10);
        assert!((m[(1, 0)] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symplectic_apply() {
        let w = SymplecticForm::standard(1);
        let u = DVector::from_vec(vec![1.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 1.0]);
        assert!((w.apply(&u, &v) - 1.0).abs() < 1e-10);
        assert!((w.apply(&v, &u) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reject_non_skew() {
        let m = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        assert!(SymplecticForm::new(m).is_none());
    }

    #[test]
    fn test_reject_odd_dimension() {
        let m = DMatrix::from_vec(3, 3, vec![0.0; 9]);
        assert!(SymplecticForm::new(m).is_none());
    }

    #[test]
    fn test_isotropic_x_axis_2d() {
        let w = SymplecticForm::standard(1);
        let basis = vec![DVector::from_vec(vec![1.0, 0.0])];
        assert!(w.is_isotropic(&basis));
    }

    #[test]
    fn test_not_isotropic_xy_2d() {
        let w = SymplecticForm::standard(1);
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        assert!(!w.is_isotropic(&basis));
    }
}

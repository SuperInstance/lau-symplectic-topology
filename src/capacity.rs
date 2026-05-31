//! Symplectic capacities: Ekeland-Hofer, Gromov width.

use crate::SymplecticForm;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A symplectic capacity is a map c from symplectic manifolds to [0, ∞] satisfying:
/// 1. Monotonicity: if (M₁,ω₁) symplectically embeds into (M₂,ω₂), then c(M₁) ≤ c(M₂)
/// 2. Conformality: c(M, λω) = |λ| c(M, ω)
/// 3. Normalization: c(B²ⁿ(r)) = c(Z²ⁿ(r)) = π r²
/// where B²ⁿ(r) is the ball and Z²ⁿ(r) is the cylinder of radius r.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SymplecticCapacity {
    /// The capacity value
    pub value: f64,
    /// Which capacity was computed
    pub kind: CapacityKind,
}

/// Types of symplectic capacities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacityKind {
    /// Gromov width: sup { π r² | B(r) symplectically embeds into M }
    GromovWidth,
    /// Ekeland-Hofer capacity (via spectral methods)
    EkelandHofer,
    /// Hofer-Zehnder capacity
    HoferZehnder,
}

impl SymplecticCapacity {
    /// Gromov width of a 2n-dimensional ball of radius r: π r².
    pub fn ball_gromov_width(n: usize, r: f64) -> Self {
        let _ = n;
        Self { value: std::f64::consts::PI * r * r, kind: CapacityKind::GromovWidth }
    }

    /// Gromov width of a symplectic cylinder Z(r) = B²(r) × R^{2n-2}: π r².
    pub fn cylinder_gromov_width(n: usize, r: f64) -> Self {
        let _ = n;
        Self { value: std::f64::consts::PI * r * r, kind: CapacityKind::GromovWidth }
    }

    /// Ekeland-Hofer capacity for a ball: π r².
    pub fn ball_ekeland_hofer(r: f64) -> Self {
        Self { value: std::f64::consts::PI * r * r, kind: CapacityKind::EkelandHofer }
    }

    /// Ekeland-Hofer capacity for a cylinder: π r².
    pub fn cylinder_ekeland_hofer(r: f64) -> Self {
        Self { value: std::f64::consts::PI * r * r, kind: CapacityKind:: EkelandHofer }
    }

    /// Hofer-Zehnder capacity for a ball: π r².
    pub fn ball_hofer_zehnder(r: f64) -> Self {
        Self { value: std::f64::consts::PI * r * r, kind: CapacityKind::HoferZehnder }
    }

    /// Capacity of an ellipsoid E(a₁,...,aₙ) = { Σ x_i²/aᵢ ≤ 1 }
    /// The Ekeland-Hofer capacity c_k(E) = k · min(a₁,...,aₙ) for k-th capacity.
    pub fn ellipsoid_ekeland_hofer(semi_axes: &[f64], k: usize) -> Self {
        let min_a = semi_axes.iter().cloned().fold(f64::INFINITY, f64::min);
        Self { value: (k as f64) * min_a * std::f64::consts::PI, kind: CapacityKind::EkelandHofer }
    }

    /// Gromov width of an ellipsoid: π · min(a₁,...,aₙ).
    pub fn ellipsoid_gromov_width(semi_axes: &[f64]) -> Self {
        let min_a = semi_axes.iter().cloned().fold(f64::INFINITY, f64::min);
        Self { value: std::f64::consts::PI * min_a, kind: CapacityKind::GromovWidth }
    }

    /// Check monotonicity property: if c₁ embeds into c₂, capacity must not decrease.
    pub fn check_monotonicity(c_source: &SymplecticCapacity, c_target: &SymplecticCapacity) -> bool {
        c_source.value <= c_target.value + 1e-10
    }

    /// Check conformality: c(M, λω) = |λ| c(M, ω).
    pub fn check_conformality(capacity: &SymplecticCapacity, lambda: f64) -> f64 {
        lambda.abs() * capacity.value
    }

    /// Compute capacity for a polydisk P(a₁,...,aₙ) = B²(a₁) × ... × B²(aₙ).
    /// Gromov width = π · min(a₁,...,aₙ).
    pub fn polydisk_gromov_width(radii: &[f64]) -> Self {
        let min_r = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        Self { value: std::f64::consts::PI * min_r, kind: CapacityKind::GromovWidth }
    }
}

/// Capacity computation for a general domain via Monte Carlo / grid methods.
pub fn compute_gromov_width_grid<F>(form: &SymplecticForm, contains_point: &F, grid_res: usize) -> f64
where
    F: Fn(&DVector<f64>) -> bool,
{
    let n = form.half_dimension();
    let dim = form.dimension();
    let mut max_r_sq = 0.0;

    // Try embedding balls of increasing radius
    for k in 1..=grid_res {
        let r_sq = (k as f64) / (grid_res as f64);
        let r = r_sq.sqrt();
        // Check if a ball of radius r centered at origin fits
        let mut fits = true;
        let samples = grid_res * 2;
        for i in 0..samples {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (samples as f64);
            let mut point = DVector::zeros(dim);
            point[0] = r * theta.cos();
            point[1] = r * theta.sin();
            if !contains_point(&point) {
                fits = false;
                break;
            }
        }
        if fits {
            max_r_sq = std::f64::consts::PI * r_sq;
        }
    }
    max_r_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ball_gromov_width_2d() {
        let c = SymplecticCapacity::ball_gromov_width(1, 1.0);
        assert!((c.value - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_ball_gromov_width_r2() {
        let c = SymplecticCapacity::ball_gromov_width(2, 2.0);
        assert!((c.value - 4.0 * std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_cylinder_capacity() {
        let c = SymplecticCapacity::cylinder_gromov_width(2, 3.0);
        assert!((c.value - 9.0 * std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_normalization_ball_eq_cylinder() {
        let cb = SymplecticCapacity::ball_gromov_width(3, 1.0);
        let cc = SymplecticCapacity::cylinder_gromov_width(3, 1.0);
        assert!((cb.value - cc.value).abs() < 1e-10);
    }

    #[test]
    fn test_monotonicity() {
        let c1 = SymplecticCapacity::ball_gromov_width(2, 1.0);
        let c2 = SymplecticCapacity::ball_gromov_width(2, 2.0);
        assert!(SymplecticCapacity::check_monotonicity(&c1, &c2));
        assert!(!SymplecticCapacity::check_monotonicity(&c2, &c1));
    }

    #[test]
    fn test_conformality() {
        let c = SymplecticCapacity::ball_gromov_width(1, 1.0);
        let scaled = SymplecticCapacity::check_conformality(&c, 2.0);
        assert!((scaled - 2.0 * std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_ellipsoid_capacity() {
        let c = SymplecticCapacity::ellipsoid_ekeland_hofer(&[1.0, 2.0, 3.0], 1);
        assert!((c.value - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_ellipsoid_gromov_width() {
        let c = SymplecticCapacity::ellipsoid_gromov_width(&[2.0, 3.0]);
        assert!((c.value - 2.0 * std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_polydisk_gromov_width() {
        let c = SymplecticCapacity::polydisk_gromov_width(&[1.0, 2.0, 3.0]);
        assert!((c.value - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_ekeland_hofer_kth() {
        let c1 = SymplecticCapacity::ellipsoid_ekeland_hofer(&[1.0, 3.0], 1);
        let c2 = SymplecticCapacity::ellipsoid_ekeland_hofer(&[1.0, 3.0], 2);
        assert!((c2.value - 2.0 * c1.value).abs() < 1e-10);
    }

    #[test]
    fn test_hofer_zehnder_ball() {
        let c = SymplecticCapacity::ball_hofer_zehnder(1.0);
        assert!((c.value - std::f64::consts::PI).abs() < 1e-10);
    }
}

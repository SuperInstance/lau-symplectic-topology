//! Symplectic embedding problems: when can one symplectic manifold fit inside another?

use crate::capacity::{CapacityKind, SymplecticCapacity};
use serde::{Deserialize, Serialize};

/// Result of a symplectic embedding check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    /// Whether the embedding exists
    pub can_embed: bool,
    /// The Gromov width of the source
    pub source_capacity: f64,
    /// The capacity of the target
    pub target_capacity: f64,
    /// Reason for the result
    pub reason: String,
}

/// Check if a ball B^{2n}(r₁) symplectically embeds into B^{2n}(r₂).
pub fn ball_into_ball(n: usize, r1: f64, r2: f64) -> EmbeddingResult {
    let c_source = SymplecticCapacity::ball_gromov_width(n, r1);
    let c_target = SymplecticCapacity::ball_gromov_width(n, r2);
    let can = r1 <= r2;
    EmbeddingResult {
        can_embed: can,
        source_capacity: c_source.value,
        target_capacity: c_target.value,
        reason: if can {
            format!("B^{}({}) fits into B^{}({}) by volume", 2 * n, r1, 2 * n, r2)
        } else {
            format!("Gromov width {} > target capacity {}", c_source.value, c_target.value)
        },
    }
}

/// Check if a ball B^{2n}(r) symplectically embeds into cylinder Z^{2n}(R) = B²(R) × R^{2n-2}.
/// By Gromov's non-squeezing theorem: embedding exists iff r ≤ R.
pub fn ball_into_cylinder(n: usize, r: f64, R: f64) -> EmbeddingResult {
    let c_ball = SymplecticCapacity::ball_gromov_width(n, r);
    let c_cyl = SymplecticCapacity::cylinder_gromov_width(n, R);
    let can = r <= R;
    EmbeddingResult {
        can_embed: can,
        source_capacity: c_ball.value,
        target_capacity: c_cyl.value,
        reason: if can {
            "Ball fits through cylinder (Gromov non-squeezing)".into()
        } else {
            format!("Gromov non-squeezing: r={} > R={}, impossible", r, R)
        },
    }
}

/// Check if an ellipsoid E(a₁,...,aₙ) symplectically embeds into a ball B(R).
pub fn ellipsoid_into_ball(semi_axes: &[f64], R: f64) -> EmbeddingResult {
    let n = semi_axes.len();
    let c_ell = SymplecticCapacity::ellipsoid_gromov_width(semi_axes);
    let c_ball = SymplecticCapacity::ball_gromov_width(n, R);
    let max_axis = semi_axes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Necessary: Gromov width ≤ target capacity. Sufficient: max semi-axis ≤ R.
    let can = max_axis <= R;
    EmbeddingResult {
        can_embed: can,
        source_capacity: c_ell.value,
        target_capacity: c_ball.value,
        reason: if can {
            "Ellipsoid fits inside ball".into()
        } else {
            format!("Largest semi-axis {} > ball radius {}", max_axis, R)
        },
    }
}

/// Check if a polydisk P(a₁,...,aₙ) symplectically embeds into another P(b₁,...,bₙ).
/// By monotonicity of Gromov width, need min(aᵢ) ≤ min(bᵢ) (necessary, not sufficient in general).
pub fn polydisk_into_polydisk(a: &[f64], b: &[f64]) -> EmbeddingResult {
    let c_src = SymplecticCapacity::polydisk_gromov_width(a);
    let c_tgt = SymplecticCapacity::polydisk_gromov_width(b);
    let min_a = a.iter().cloned().fold(f64::INFINITY, f64::min);
    let min_b = b.iter().cloned().fold(f64::INFINITY, f64::min);
    let can = min_a <= min_b;
    EmbeddingResult {
        can_embed: can,
        source_capacity: c_src.value,
        target_capacity: c_tgt.value,
        reason: if can {
            "Necessary condition (Gromov width) satisfied".into()
        } else {
            "Gromov width obstruction".into()
        },
    }
}

/// Gromov's non-squeezing theorem: you cannot symplectically embed B^{2n}(r) into Z^{2n}(R) if r > R,
/// even though volume-wise there's no obstruction.
pub fn non_squeezing_theorem(n: usize, ball_radius: f64, cylinder_radius: f64) -> bool {
    ball_radius <= cylinder_radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ball_into_larger_ball() {
        let r = ball_into_ball(2, 1.0, 2.0);
        assert!(r.can_embed);
    }

    #[test]
    fn test_ball_into_smaller_ball() {
        let r = ball_into_ball(2, 2.0, 1.0);
        assert!(!r.can_embed);
    }

    #[test]
    fn test_ball_into_same_ball() {
        let r = ball_into_ball(1, 1.0, 1.0);
        assert!(r.can_embed);
    }

    #[test]
    fn test_non_squeezing_blocked() {
        let r = ball_into_cylinder(2, 2.0, 1.0);
        assert!(!r.can_embed);
    }

    #[test]
    fn test_non_squeezing_allowed() {
        let r = ball_into_cylinder(2, 1.0, 2.0);
        assert!(r.can_embed);
    }

    #[test]
    fn test_gromov_non_squeezing() {
        assert!(!non_squeezing_theorem(2, 1.5, 1.0));
        assert!(non_squeezing_theorem(2, 0.5, 1.0));
    }

    #[test]
    fn test_ellipsoid_into_ball_yes() {
        let r = ellipsoid_into_ball(&[1.0, 2.0], 2.0);
        assert!(r.can_embed);
    }

    #[test]
    fn test_ellipsoid_into_ball_no() {
        let r = ellipsoid_into_ball(&[1.0, 2.0], 1.5);
        assert!(!r.can_embed);
    }

    #[test]
    fn test_polydisk_embedding() {
        let r = polydisk_into_polydisk(&[1.0, 2.0], &[2.0, 3.0]);
        assert!(r.can_embed);
    }

    #[test]
    fn test_polydisk_no_embedding() {
        let r = polydisk_into_polydisk(&[1.0, 2.0], &[0.5, 3.0]);
        assert!(!r.can_embed);
    }
}

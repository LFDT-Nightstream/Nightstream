//! Stable diagnostic paths for the NIFS tail after Pi_CCS and Pi_RLC.
//!
//! Owns: incoming and outgoing Pi_DEC stage names, point-binding stage names,
//! and immediate-child ownership.
//!
//! Does not own: Pi_CCS/Pi_RLC nodes, constraint emission, or cost totals.
//!
//! Emits constraints: no.
//!
//! Authority boundary: labels are diagnostic metadata; the checked parent,
//! children, and point-equality rows remain the semantic authority.
//!
//! | Child path | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | `pi_dec.verify` | Strictly recompose the claimed parent and check every shared/canonical family | yes | `pi_dec_circuit` | PiDEC bridge partial; leaf ownership lives in `pi_dec_circuit::stage` |
//! | `point_binding` | Equate the Pi_DEC parent point with Pi_CCS `r_prime` | yes | `circuit::mod` | NIFS bridge open |

use crate::paper::reductions::pi_dec_circuit::stage as pi_dec_stage;

pub const PI_DEC: &str = pi_dec_stage::ROOT;
pub const PI_DEC_VERIFY: &str = pi_dec_stage::VERIFY;
pub const RUNNING_PARENT_PI_DEC: &str = "nifs.running_parent_pi_dec";
pub const POINT_BINDING: &str = "nifs.point_binding";

pub const ALL: &[&str] = &[RUNNING_PARENT_PI_DEC, PI_DEC, PI_DEC_VERIFY, POINT_BINDING];

pub const HIERARCHY: &[(&str, &[&str])] = &[(PI_DEC, &[PI_DEC_VERIFY])];

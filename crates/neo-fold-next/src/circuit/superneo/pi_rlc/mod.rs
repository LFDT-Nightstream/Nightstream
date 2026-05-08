//! Owns public Π_RLC arithmetic checks over CE claims for the RV32IM main-relation circuit.
//!
//! This module owns rho-driven claim folding and the last-chunk shortcut that still lives on
//! the Π_RLC side of the bridge theorem boundary. Pure b-ary Π_DEC checks live in `pi_dec.rs`.

mod basis;
mod constraints;
#[path = "../rlc_dec/diagnostics.rs"]
mod diagnostics;
mod public;
mod rho_action;
mod ring_action;
mod y_rows;

use crate::spartan_backend::SpartanF;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{KExtensions, D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::sync::LazyLock;

use super::claim::CircuitCeClaim;
use super::k_field::{enforce_k_eq, enforce_k_eq_constant_f_linear_combination, k_base_mul_var, KNum, KNumVar};
use super::rho_sampling::{RotRhoMatrixVar, RotRhoVar};

static GOLDILOCKS_ROT_BASIS_MATS: LazyLock<Vec<Mat<F>>> = LazyLock::new(basis::build_goldilocks_rot_basis_mats);

pub(crate) use diagnostics::{
    debug_locate_rlc_public_with_split_rho_views_stage,
    debug_measure_rlc_public_with_rho_coeffs_for_constant_children_stage_ranges,
    debug_measure_rlc_public_with_split_rho_views_stage_ranges, RlcPublicStageCheckpoints,
};
pub use public::*;

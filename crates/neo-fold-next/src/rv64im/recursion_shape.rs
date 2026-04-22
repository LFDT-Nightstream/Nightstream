//! Owns the fixed RV64IM recursion-shape surface used to key deterministic setup.
//!
//! This module does not own the live root kernel params, but the fixed-shape
//! family compiled into `F'` / `vk_fs` must mirror the live `b` / `k_rho`
//! pin or the verifier-key digest stops describing the real recursive family.

use core::hash::{Hash, Hasher};
use std::collections::{BTreeMap, BTreeSet};

use neo_math::D;
use neo_reductions::engines::utils::build_dims_and_policy;
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::rv64im::ccs::RV64IM_ROOT_PUBLIC_INPUTS;
use crate::rv64im::kernel::{
    phase0_family_order, rv64im_cached_root_main_lane_context, rv64im_simple_root_params_for_step_cap,
    FamilyEvalSchemaId, SimpleKernelError,
};

pub const RV64IM_RECURSION_SHAPE_SOUNDNESS_BIG_K: u32 = 1;
pub const RV64IM_RECURSION_SOUNDNESS_T: u32 = 216;
pub const RV64IM_RECURSION_DEFAULT_STEP_CAP: u32 = 1;

const RV64IM_STAGE1_PHASE0_SLOT_COUNT: u32 = 4;
const RV64IM_SINGLETON_PHASE0_SLOT_COUNT: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl Hash for ProtocolVersion {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.major.hash(state);
        self.minor.hash(state);
    }
}

pub const RV64IM_RECURSION_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion { major: 1, minor: 0 };

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct RecursionShape {
    pub step_cap: u32,
    pub soundness_k: u32,
    pub soundness_big_k: u32,
    pub t_matrices: u32,
    pub log_m: u32,
    pub d_sc: u32,
    pub n_R: u32,
    pub n_R_in: u32,
    pub b: u8,
    pub decomposition_k: u8,
    pub side_families_active: BTreeSet<FamilyEvalSchemaId>,
    pub side_slots_per_family: BTreeMap<FamilyEvalSchemaId, u32>,
    pub version: ProtocolVersion,
}

impl Hash for RecursionShape {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.step_cap.hash(state);
        self.soundness_k.hash(state);
        self.soundness_big_k.hash(state);
        self.t_matrices.hash(state);
        self.log_m.hash(state);
        self.d_sc.hash(state);
        self.n_R.hash(state);
        self.n_R_in.hash(state);
        self.b.hash(state);
        self.decomposition_k.hash(state);
        for schema in &self.side_families_active {
            schema.tag().hash(state);
        }
        for (schema, slots) in &self.side_slots_per_family {
            schema.tag().hash(state);
            slots.hash(state);
        }
        self.version.hash(state);
    }
}

impl RecursionShape {
    pub fn canonical_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/recursion_shape");
        tr.append_u64s(
            b"neo.fold.next/rv64im/recursion_shape/meta",
            &[
                self.version.major as u64,
                self.version.minor as u64,
                self.step_cap as u64,
                self.soundness_k as u64,
                self.soundness_big_k as u64,
                self.t_matrices as u64,
                self.log_m as u64,
                self.d_sc as u64,
                self.n_R as u64,
                self.n_R_in as u64,
                self.b as u64,
                self.decomposition_k as u64,
                self.side_families_active.len() as u64,
                self.side_slots_per_family.len() as u64,
            ],
        );
        for schema in &self.side_families_active {
            tr.append_u64s(b"neo.fold.next/rv64im/recursion_shape/side_family", &[schema.tag()]);
        }
        for (schema, slots) in &self.side_slots_per_family {
            tr.append_u64s(
                b"neo.fold.next/rv64im/recursion_shape/side_slots",
                &[schema.tag(), *slots as u64],
            );
        }
        tr.digest32()
    }

    pub fn validate_soundness(&self) -> Result<(), ShapeError> {
        if self.version != RV64IM_RECURSION_PROTOCOL_VERSION {
            return Err(ShapeError::UnsupportedVersion {
                major: self.version.major,
                minor: self.version.minor,
            });
        }

        let big_b = (self.b as u64)
            .checked_pow(self.decomposition_k as u32)
            .unwrap_or(u64::MAX);
        let k_plus_big_k = self.soundness_k.saturating_add(self.soundness_big_k);
        let lhs =
            (k_plus_big_k as u128) * (RV64IM_RECURSION_SOUNDNESS_T as u128) * ((self.b as u128).saturating_sub(1));
        if lhs >= big_b as u128 {
            return Err(ShapeError::SoundnessViolation {
                k_plus_big_k,
                t: RV64IM_RECURSION_SOUNDNESS_T,
                b: self.b,
                big_b,
            });
        }

        Ok(())
    }

    pub fn side_slot_count(&self, schema: FamilyEvalSchemaId) -> Option<u32> {
        self.side_slots_per_family.get(&schema).copied()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ShapeError {
    #[error("RV64IM recursion shape violates Def 14: (K+k)={k_plus_big_k}, T={t}, b={b}, B={big_b}")]
    SoundnessViolation {
        k_plus_big_k: u32,
        t: u32,
        b: u8,
        big_b: u64,
    },
    #[error("unsupported RV64IM recursion-shape version {major}.{minor}")]
    UnsupportedVersion { major: u16, minor: u16 },
}

pub fn build_rv64im_recursion_shape() -> Result<RecursionShape, SimpleKernelError> {
    build_rv64im_recursion_shape_for_step_cap(RV64IM_RECURSION_DEFAULT_STEP_CAP as usize)
}

pub fn build_rv64im_recursion_shape_for_step_cap(step_cap: usize) -> Result<RecursionShape, SimpleKernelError> {
    let step_cap = u32::try_from(step_cap)
        .map_err(|_| SimpleKernelError::Build("RV64IM recursion shape step_cap overflowed u32".into()))?;
    if step_cap == 0 {
        return Err(SimpleKernelError::Build(
            "RV64IM recursion shape step_cap must be at least one public step".into(),
        ));
    }
    let (_, _, structure) = rv64im_cached_root_main_lane_context()?;
    let params = rv64im_simple_root_params_for_step_cap(step_cap as usize);
    let dims = build_dims_and_policy(&params, structure)
        .map_err(|err| SimpleKernelError::Build(format!("RV64IM recursion shape dims failed: {err}")))?;
    let side_families_active = phase0_family_order().into_iter().collect::<BTreeSet<_>>();
    let side_slots_per_family = phase0_family_order()
        .into_iter()
        .map(|schema| (schema, rv64im_phase0_slot_count(schema)))
        .collect::<BTreeMap<_, _>>();

    let shape = RecursionShape {
        step_cap,
        soundness_k: params.k_rho,
        soundness_big_k: RV64IM_RECURSION_SHAPE_SOUNDNESS_BIG_K,
        t_matrices: structure.t() as u32,
        log_m: dims.ell_m as u32,
        d_sc: dims.d_sc as u32,
        n_R: ceil_div_usize(structure.n, D) as u32,
        n_R_in: ceil_div_usize(RV64IM_ROOT_PUBLIC_INPUTS, D) as u32,
        b: params.b as u8,
        decomposition_k: params.k_rho as u8,
        side_families_active,
        side_slots_per_family,
        version: RV64IM_RECURSION_PROTOCOL_VERSION,
    };
    shape
        .validate_soundness()
        .map_err(|err| SimpleKernelError::Build(format!("RV64IM recursion shape invalid: {err}")))?;
    Ok(shape)
}

fn ceil_div_usize(value: usize, divisor: usize) -> usize {
    value.saturating_add(divisor.saturating_sub(1)) / divisor
}

fn rv64im_phase0_slot_count(schema: FamilyEvalSchemaId) -> u32 {
    match schema {
        FamilyEvalSchemaId::Stage1Rows => RV64IM_STAGE1_PHASE0_SLOT_COUNT,
        FamilyEvalSchemaId::Stage2RegisterReads
        | FamilyEvalSchemaId::Stage2RegisterWrites
        | FamilyEvalSchemaId::Stage2RamEvents
        | FamilyEvalSchemaId::Stage2TwistLinks
        | FamilyEvalSchemaId::Stage3Continuity => RV64IM_SINGLETON_PHASE0_SLOT_COUNT,
    }
}

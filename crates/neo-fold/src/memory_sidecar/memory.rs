use crate::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use crate::memory_sidecar::sumcheck_ds::{run_batched_sumcheck_prover_ds, verify_batched_sumcheck_rounds_ds};
use crate::memory_sidecar::transcript::{bind_batched_claim_sums, bind_twist_val_eval_claim_sums, digest_fields};
use crate::memory_sidecar::utils::{bitness_weights, RoundOraclePrefix};
use crate::shard_proof_types::{
    MemOrLutProof, MemSidecarProof, ShoutAddrPreGroupProof, ShoutAddrPreProof, ShoutProofK, TwistProofK,
};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{KExtensions, F, K};
use neo_memory::bit_ops::{eq_bit_affine, eq_bits_prod};
use neo_memory::cpu::{
    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, BusLayout, ShoutInstanceShape,
};
use neo_memory::identity::shout_oracle::IdentityAddressLookupOracleSparse;
use neo_memory::mle::{eq_points, lt_eval};
use neo_memory::riscv::shout_oracle::RiscvAddressLookupOracleSparse;
use neo_memory::riscv::trace::{
    riscv_decode_lookup_backed_row_from_instr_word, riscv_decode_lookup_table_id_for_col,
    riscv_decode_lookup_transport_cols, riscv_decode_lookup_val_slot_for_col, riscv_is_decode_lookup_table_id,
    riscv_trace_is_width_lookup_table_id, riscv_trace_shared_width_lookup_backed_cols,
    riscv_trace_shared_width_lookup_table_id_for_col, riscv_trace_shared_width_lookup_val_slot_for_col,
    riscv_trace_uses_shared_width_lookup_table_id, Rv32DecodeSidecarLayout, Rv32TraceLayout, Rv32WidthSidecarLayout,
};
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::ts_common as ts;
use neo_memory::twist_oracle::{
    AddressLookupOracle, IndexAdapterOracleSparseTime, LazyWeightedBitnessOracleSparseTime,
    Rv32NonVirtualArchDomainOracleSparseTime, Rv32PackedAddOracleSparseTime, Rv32PackedAndOracleSparseTime,
    Rv32PackedAndnOracleSparseTime, Rv32PackedBitwiseAdapterOracleSparseTime, Rv32PackedDivOracleSparseTime,
    Rv32PackedDivRemAdapterOracleSparseTime, Rv32PackedDivRemuAdapterOracleSparseTime, Rv32PackedDivuOracleSparseTime,
    Rv32PackedEqAdapterOracleSparseTime, Rv32PackedEqOracleSparseTime, Rv32PackedMulHiOracleSparseTime,
    Rv32PackedMulOracleSparseTime, Rv32PackedMulhAdapterOracleSparseTime, Rv32PackedMulhsuAdapterOracleSparseTime,
    Rv32PackedMulhuOracleSparseTime, Rv32PackedNeqAdapterOracleSparseTime, Rv32PackedNeqOracleSparseTime,
    Rv32PackedOrOracleSparseTime, Rv32PackedRemOracleSparseTime, Rv32PackedRemuOracleSparseTime,
    Rv32PackedSllOracleSparseTime, Rv32PackedSltOracleSparseTime, Rv32PackedSltuOracleSparseTime,
    Rv32PackedSraAdapterOracleSparseTime, Rv32PackedSraOracleSparseTime, Rv32PackedSrlAdapterOracleSparseTime,
    Rv32PackedSrlOracleSparseTime, Rv32PackedSubOracleSparseTime, Rv32PackedXorOracleSparseTime,
    Rv32VirtualWriteDomainOracleSparseTime, ShoutValueOracleSparse, TwistLaneSparseCols,
    TwistReadCheckAddrOracleSparseTimeMultiLane, TwistReadCheckOracleSparseTime, TwistTotalIncOracleSparseTime,
    TwistValEvalOracleSparseTime, TwistWriteCheckAddrOracleSparseTimeMultiLane, TwistWriteCheckOracleSparseTime,
    U32DecompOracleSparseTime, ZeroOracleSparseTime,
};
use neo_memory::witness::{LutInstance, LutTableSpec, MemInstance, StepInstanceBundle, StepWitnessBundle};
use neo_memory::{eval_init_at_r_addr, twist, BatchedAddrProof, MemInit};
use neo_params::NeoParams;
use neo_reductions::sumcheck::{BatchedClaim, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField64;
use std::collections::{BTreeMap, BTreeSet};

#[path = "memory/addr_pre_proofs.rs"]
mod addr_pre_proofs;
#[path = "memory/event_table_context.rs"]
mod event_table_context;
#[path = "memory/precompiles/poseidon2/claim_builders.rs"]
mod poseidon_claim_builders;
#[path = "memory/precompiles/poseidon2/link_claim_builders.rs"]
mod poseidon_link_claim_builders;
#[path = "memory/precompiles/poseidon2/local_commit.rs"]
mod poseidon_local_commit;
#[path = "memory/precompiles/poseidon2/terminal_checks.rs"]
mod poseidon_terminal_checks;
#[path = "memory/route_a_claim_builders.rs"]
mod route_a_claim_builders;
#[path = "memory/route_a_claims.rs"]
mod route_a_claims;
#[path = "memory/route_a_finalize.rs"]
mod route_a_finalize;
#[path = "memory/route_a_oracles.rs"]
mod route_a_oracles;
#[path = "memory/route_a_rv64_fullword.rs"]
mod route_a_rv64_fullword;
#[path = "memory/route_a_rv64_reg_output.rs"]
mod route_a_rv64_reg_output;
#[path = "memory/route_a_terminal_checks.rs"]
mod route_a_terminal_checks;
#[path = "memory/route_a_verify.rs"]
mod route_a_verify;
#[path = "memory/sparse_oracles_and_twist_pre.rs"]
mod sparse_oracles_and_twist_pre;
#[path = "memory/transcript_and_common.rs"]
mod transcript_and_common;
#[path = "memory/w2_virtual_constraints.rs"]
mod w2_virtual_constraints;

pub use addr_pre_proofs::{verify_shout_addr_pre_time, verify_twist_addr_pre_time};
pub use route_a_verify::verify_route_a_memory_step;
pub use transcript_and_common::{absorb_step_memory, TwistTimeLaneOpenings};

pub(crate) use addr_pre_proofs::*;
pub(crate) use event_table_context::*;
pub(crate) use poseidon_claim_builders::*;
pub(crate) use poseidon_link_claim_builders::*;
pub(crate) use poseidon_local_commit::*;
pub(crate) use poseidon_terminal_checks::*;
pub(crate) use route_a_claim_builders::*;
pub(crate) use route_a_claims::*;
pub(crate) use route_a_finalize::*;
pub(crate) use route_a_oracles::*;
pub(crate) use route_a_rv64_fullword::*;
pub(crate) use route_a_rv64_reg_output::*;
pub(crate) use route_a_terminal_checks::*;
pub(crate) use sparse_oracles_and_twist_pre::*;
pub(crate) use transcript_and_common::*;
pub(crate) use w2_virtual_constraints::*;

//! Canonical serial SHA-256 packed-state lifecycle benchmark workload.
//!
//! Owns: the bellpepper serial-SHA circuit, its 56-bit public state-lane
//! packing, and the R1CS-F' plan/derived-structure construction for it.
//! Kept byte-for-byte compatible with the CUDA e2e workload and the
//! `neo-fold-clean` CPU reference.
//! Owns no protocol semantics: everything delegates to `neo-fold-clean`.

use bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper::gadgets::{multipack, sha256};
use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::bellpepper::{synthesize_to_ccs, BellpepperCcs, BellpepperGoldilocks};
use neo_fold_clean::frontends::f_prime::image::{FPrimeImageLayout, NifsCeClaimShape, NifsPayloadShape};
use neo_fold_clean::frontends::f_prime::recursive_plan::{
    build_recursive_step_image_config, build_semantic_state_preimage_fields, AccumulatorPlanOptions,
    RecursiveStepImagePlan, StateXOutPlanOptions,
};
use neo_fold_clean::frontends::r1cs_f_prime::{self, SparseR1cs};
use neo_fold_clean::paper::digest::digest_fields_as_digest32;
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

/// Ajtai seed for the serial packed-state preprocessing (matches the
/// CPU-reference test's `SHA256_AJTAI_SEED ^ 0x5154_4154_455f_0056`).
pub const SHA256_SERIAL_AJTAI_SEED: u64 = 0x5348_4132_3556_5345 ^ 0x5154_4154_455f_0056;

const STATE_LANES56: usize = 5;
const STATE_LIMB_BITS: usize = 56;

/// One serial-SHA chunk: `transitions` SHA-256 application(s) to a 32-byte
/// state, with `state_in`/`state_out` exposed as 56-bit public lanes.
struct Sha256SerialPackedStateCircuit {
    state_in: Vec<u8>,
    transitions: usize,
}

impl Circuit<BellpepperGoldilocks> for Sha256SerialPackedStateCircuit {
    fn synthesize<CS: ConstraintSystem<BellpepperGoldilocks>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        assert_eq!(self.state_in.len(), 32, "serial SHA state is a 32-byte digest");
        assert!(self.transitions > 0, "serial SHA circuit needs at least one transition");
        let state_in_bits = multipack::bytes_to_bits(&self.state_in);
        let mut current = state_in_bits
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, bit)| {
                AllocatedBit::alloc(cs.namespace(|| format!("packed_state_in_private_bit_{idx}")), Some(bit))
            })
            .map(|bit| bit.map(Boolean::from))
            .collect::<Result<Vec<_>, _>>()?;

        enforce_public_lanes_from_bits(
            cs.namespace(|| "state_in_public_lanes"),
            "state_in_lane",
            &current,
            &state_lanes56_fields(&self.state_in),
        )?;

        for step in 0..self.transitions {
            current = sha256::sha256(cs.namespace(|| format!("packed_sha256_step_{step}")), &current)?;
        }
        let state_out_bytes = sha_state_trace(&self.state_in, self.transitions)
            .pop()
            .expect("state trace includes final state");
        enforce_public_lanes_from_bits(
            cs.namespace(|| "state_out_public_lanes"),
            "state_out_lane",
            &current,
            &state_lanes56_fields(&state_out_bytes),
        )?;
        Ok(())
    }
}

/// Synthesize one serial chunk starting at `state_in`.
pub fn serial_chunk(state_in: Vec<u8>, transitions: usize) -> BellpepperCcs {
    synthesize_to_ccs(Sha256SerialPackedStateCircuit { state_in, transitions })
        .expect("synthesize packed-state serial SHA chunk")
}

/// The fixed initial 32-byte state the reference workload starts from.
pub fn initial_sha_state() -> Vec<u8> {
    (0..32).map(|idx| idx as u8).collect()
}

/// `[initial, SHA(initial), SHA^2(initial), ...]` with `transitions` steps.
pub fn sha_state_trace(initial: &[u8], transitions: usize) -> Vec<Vec<u8>> {
    let mut states = Vec::with_capacity(transitions + 1);
    states.push(initial.to_vec());
    for _ in 0..transitions {
        let next = Sha256::digest(states.last().expect("non-empty state trace")).to_vec();
        states.push(next);
    }
    states
}

/// The semantic-state digest the chain reports for a given 32-byte state.
pub fn serial_state_lanes56_semantic_digest(state: &[u8]) -> [u8; 32] {
    digest_fields_as_digest32(
        encode_poseidon_trace(&build_semantic_state_preimage_fields(&state_lanes56_fields(state))).digest_native,
    )
}

/// Derive the packed-state serial R1CS-F' structure, iterating the CE shape
/// until the challenge lengths converge. Returns the derived structure and
/// the number of plan iterations it took.
pub fn packed_state_derived_structure(
    r1cs: &SparseR1cs,
    params: &Params,
    initial_state: &[u8],
) -> (r1cs_f_prime::R1csFPrimeDerivedStructure, usize) {
    let shape = r1cs_f_prime::R1csShape::from(r1cs);
    let widths = shape.conservative_app_private_var_widths();
    let typed_bits: usize = widths.iter().sum();
    let c_data_entries = params.kappa() as usize * params.d() as usize;
    let child_count = params.k_rho() as u64;
    let initial_anchor = serial_state_lanes56_semantic_digest(initial_state);
    let mut r_len = challenge_len_for_domain(shape.n());
    let mut s_col_len = challenge_len_for_domain(typed_bits + 1);

    for iteration in 1..=8 {
        let mut plan =
            lifecycle_plan_with_ce_shape(shape.m(), shape.m_in(), c_data_entries, child_count, r_len, s_col_len);
        let state_x_out = plan
            .state_x_out
            .as_mut()
            .expect("SHA lifecycle plan installs state_x_out");
        state_x_out.app_public_input_var_indices = (0..shape.m_in()).collect();
        state_x_out.app_public_input_bit_var_indices = Vec::new();
        state_x_out.semantic_state_in_var_indices = (1..=STATE_LANES56).collect();
        state_x_out.semantic_state_out_var_indices = ((1 + STATE_LANES56)..=(2 * STATE_LANES56)).collect();
        state_x_out.initial_semantic_state_digest_anchor = Some(initial_anchor);
        plan.limbs = typed_bits + 1;
        plan.app_private_var_widths = widths.clone();
        let derived = r1cs_f_prime::derive_sparse_preprocessing_structure(r1cs, &plan)
            .expect("derive packed-state SHA R1CS-F' structure");
        let next_r_len = challenge_len_for_domain(derived.structure().ccs.n);
        let next_s_col_len = challenge_len_for_domain(derived.structure().ccs.m);
        if next_r_len == r_len && next_s_col_len == s_col_len {
            return (derived, iteration);
        }
        r_len = next_r_len;
        s_col_len = next_s_col_len;
    }

    panic!("SHA packed-state serial R1CS-F' CE shape did not converge")
}

fn lifecycle_plan_with_ce_shape(
    m: usize,
    m_in: usize,
    c_data_entries: usize,
    child_count: u64,
    r_len: usize,
    s_col_len: usize,
) -> RecursiveStepImagePlan {
    let limbs = m * POSEIDON2_GOLDILOCKS_BITS + 1;
    let ce_shape = NifsCeClaimShape {
        c_data_entries,
        x_rows: 54,
        x_active_cols: 5,
        r_len,
        y_ring_inner_lens: vec![64; 8],
        y_zcol_len: 64,
        s_col_len,
    };
    let probe_plan = RecursiveStepImagePlan {
        limbs,
        app_private_var_widths: Vec::new(),
        boundary_bits: 4 * POSEIDON2_GOLDILOCKS_BITS,
        kmul_count: 0,
        ring_action_pair_count: 0,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        sponge_transcript_permutes: 0,
        nifs_payload_shapes: vec![NifsPayloadShape::CeClaim(ce_shape)],
        projection_batches: Vec::new(),
        accumulator: Some(AccumulatorPlanOptions {
            ce_claim_payload_index: 0,
            c_data_entries,
            child_count,
            unified: true,
        }),
        state_x_out: None,
    };
    let probe_layout = FPrimeImageLayout::new(build_recursive_step_image_config(&probe_plan));
    let public_x_out_lane_bit_starts: [usize; 4] =
        std::array::from_fn(|i| probe_layout.boundary.offset + i * POSEIDON2_GOLDILOCKS_BITS);
    let mut plan = probe_plan;
    plan.state_x_out = Some(StateXOutPlanOptions {
        pc: 1,
        public_x_out_lane_bit_starts,
        app_public_input_var_indices: Vec::new(),
        app_public_input_bit_var_indices: (0..m_in).collect(),
        semantic_state_in_var_indices: Vec::new(),
        semantic_state_out_var_indices: Vec::new(),
        initial_semantic_state_digest_anchor: None,
    });
    plan
}

fn challenge_len_for_domain(size: usize) -> usize {
    size.next_power_of_two().max(2).trailing_zeros() as usize
}

fn state_lanes56_fields(state: &[u8]) -> Vec<F> {
    assert_eq!(state.len(), 32, "SHA state must be 32 bytes");
    multipack::bytes_to_bits(state)
        .chunks(STATE_LIMB_BITS)
        .map(|chunk| {
            let mut value = 0u64;
            for (idx, bit) in chunk.iter().enumerate() {
                if *bit {
                    value |= 1u64 << idx;
                }
            }
            F::from_u64(value)
        })
        .collect()
}

fn enforce_public_lanes_from_bits<CS: ConstraintSystem<BellpepperGoldilocks>>(
    mut cs: CS,
    label: &str,
    bits: &[Boolean],
    lane_values: &[F],
) -> Result<(), SynthesisError> {
    assert_eq!(lane_values.len(), STATE_LANES56);
    for (lane_idx, value) in lane_values.iter().enumerate() {
        let input = cs.alloc_input(
            || format!("{label}_{lane_idx}"),
            || Ok(BellpepperGoldilocks::from(value.as_canonical_u64())),
        )?;
        let start = lane_idx * STATE_LIMB_BITS;
        let end = usize::min(start + STATE_LIMB_BITS, bits.len());
        let lane_bits = &bits[start..end];
        cs.enforce(
            || format!("{label}_{lane_idx}_matches_bits"),
            |lc| {
                let mut out = lc;
                let mut coeff = BellpepperGoldilocks::ONE;
                for bit in lane_bits {
                    out = out + &bit.lc(CS::one(), coeff);
                    coeff += coeff;
                }
                out
            },
            |lc| lc + CS::one(),
            |lc| lc + input,
        );
    }
    Ok(())
}

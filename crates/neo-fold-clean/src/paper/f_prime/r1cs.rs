//! F' R1CS step — the in-circuit augmented function from
//! Hypernova §6.3 / SuperNeo Construction 2.
//!
//! Two internal branch emitters mirror the paper's case split. They are not
//! independently foldable public relations; the production caller combines
//! them into one selector-controlled implementation language in
//! `frontends::r1cs_f_prime::full_relation`.
//!
//! Owns: the base and recursive Construction-2 branch emitters and their
//! state/accumulator/public-link outputs.
//!
//! Does not own: fixed branch selection, application constraints, final public
//! pinning, or low-norm lowering.
//!
//! Emits constraints: yes; callers must compose these internal branches into
//! one fixed selector-controlled relation. That relation is an implementation
//! artifact, not the independent semantic authority for F'/NIFS.
//!
//! Authority boundary: recursive state is accepted only through the public
//! prior-state link, exact paper-level child accumulator, and checked NIFS
//! parent cache; digests compress but never replace those constrained inputs.
//! The optimized `y_zcol` source-binding gap is not claimed closed here.
//!
//! | Branch/phase | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | Base | Initialize state and the canonical empty accumulator | yes | this file | FPrime base semantics |
//! | Recursive transcript | Bind the prior public state before NIFS | yes | this file | transcript refinement open |
//! | NIFS transition | Fold fresh/running claims into checked parent plus exact children | yes | `paper/nifs/circuit/` | NIFS/FPrime bridge |
//! | Accumulator continuity | Link incoming authority and recompute outgoing authority | yes | this file | authority refinement open |
//! | Counters and `x_out` | Advance counters and derive the public next-state digest | yes | this file | FPrime step semantics |
//!
//!   - [`enforce_f_prime_base_step_circuit`] (i = 0). No NIFS.V; enforces
//!     `z_i = z_0`, `chunk_count_in = 0`, and `acc_digest_in = empty_acc`.
//!     `acc_digest_out` is the same empty-acc constant. Strict mode also
//!     requires `rows_in_chunk >= 1`, matching lifecycle's no-empty-batch
//!     boundary.
//!   - [`enforce_f_prime_recursive_step_circuit`] (i ≥ 1). Runs NIFS.V to
//!     fold `u_i` into `U_i` under a transcript bound to the full F' state
//!     input, enforces the HyperNova recursive link
//!     `u_i.public == bits(prior_x_out)`, and binds `acc_digest_in` to
//!     the exact ordered-child handle of the actual `running` accumulator and
//!     `acc_digest_out` to the ordered-child handle of NIFS.V's output.
//!
//! ## Bindings enforced (recursive case)
//!
//! 1. Every fresh `u_i.x` is `[1 || enc_inst(prior_x_out)]`.
//! 2. `acc_digest_in` hashes the exact ordered running children.
//! 3. `acc_digest_out` hashes the exact ordered NIFS output children. The
//!    checked Π_RLC parent is only a cache because Π_DEC recomposition is not
//!    injective in its child vector.
//! 4. **`pc == TRIVIAL_PC`** (ℓ = 1 in this build). `pc` is pinned,
//!    linked as state, and absorbed into `state_x_out` so the local
//!    recursive link retains HyperNova's `pc_i` binding even before
//!    multi-program support exists.
//! 5. **Strict-mode shape**: non-empty fresh batch, every fresh `u_i`
//!    has the verifier-owned physical carrier width. Plain F′ therefore uses
//!    `[1 | 256 enc_inst bits | 13 fixed zeros]`. Shape is checked before
//!    allocation and the padding is constrained to zero in-circuit.
//! 6. **Current chunk-shape digest**: `chunk_digest` is recomputed with
//!    Poseidon2 from the in-circuit `step_count`, fixed claim geometry, and
//!    `rows_in_chunk`. Native witness generation is not trusted to supply this
//!    state boundary honestly.

use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::poseidon2::DIGEST_LEN;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::u64_arith::{
    alloc_u64_bits, decompose_var_to_u64_bits, enforce_u64_add, enforce_u64_constant, enforce_u64_increment,
};
use crate::paper::construction2::{NebulaConfig, NebulaLane, TRIVIAL_PC};
use crate::paper::digest::AccumulatorHandle;
use crate::paper::digest::StateXOutDigestMode;
use crate::paper::f_prime::digest_circuit::{
    enforce_f_prime_chunk_public_digest_circuit, enforce_state_x_out_digest_circuit,
    enforce_state_x_out_digest_with_nebula_circuit, StateXOutDigestInputs,
};
use crate::paper::f_prime::nebula_lane_circuit::{
    alloc_nebula_lane_wires, decode_delayed_nebula_public_suffix_circuit, delayed_nebula_public_suffix_len,
    enforce_delayed_nebula_claim_circuit, enforce_nebula_lane_constant_circuit,
    enforce_nebula_lane_digest_selected_circuit, NebulaLaneWires, NebulaOpenContextWires,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use crate::paper::f_prime::source_image_circuit::{enforce_goldilocks_word_canonical, SourceImageWires};
use crate::paper::f_prime::stage;
use crate::paper::nifs::circuit::{
    enforce_nifs_v_circuit_with_transcript_and_header_bundle,
    enforce_nifs_v_circuit_with_transcript_and_header_bundle_wires, NifsVCircuitConfig, NifsVCircuitMessages,
};
use crate::paper::params::Params;

mod accumulator;
pub(crate) use accumulator::enforce_terminal_output_acc_digest;

/// Canonical bits per `x_out` digest lane. Goldilocks canonical form fits
/// in 64 bits.
pub const X_OUT_BITS_PER_LANE: usize = 64;

/// Number of `enc_inst(x_out)` bits — the bit-decomposed digest body.
pub const F_PRIME_ENC_INST_BITS: usize = DIGEST_LEN * X_OUT_BITS_PER_LANE;

/// Index of the constant-one slot in the F' CCS public input.
pub const F_PRIME_PUBLIC_ONE_OFFSET: usize = 0;

/// First index of the `enc_inst(x_out)` body inside the F' public input.
pub const F_PRIME_ENC_INST_OFFSET: usize = 1;

/// Logical F' public-input length: `[1, enc_inst(x_out)…]`.
///
/// `enc_inst` is the **public-instance encoding boundary**. It does *not*
/// mean every internal F' field value is bit-backed — Poseidon2 outputs,
/// transcript challenges, sumcheck values, etc. all remain ordinary
/// field values during F' execution. `enc_inst` only ensures that when
/// the hash output is carried as the *next* fresh CCS public input, it
/// is low-norm under `b = 2` (Definition 12 requires `‖z‖_∞ < b` on the
/// full assignment, public input included).
///
/// Standard R1CS-as-CCS layout requires a fixed constant-one slot so
/// affine terms have a column to ride on; without it the CCS instance
/// is not committable as a real `u_i`.
///
/// HyperNova §6.3 has `u_{i+1}.public == enc_inst(F'.x_out)` where
/// `enc_inst` is the protocol's mapping from the raw augmented-step
/// output into the CCS instance public-input shape. For SuperNeo under
/// `b = 2`, the body is "canonical 64-bit decomposition of each of the
/// four Goldilocks digest lanes". See `encoding.md`.
pub const F_PRIME_PUBLIC_INPUT_LEN: usize = 1 + F_PRIME_ENC_INST_BITS;

/// Authoritative SuperNeo carrier width for plain F'.
///
/// SuperNeo acts on complete `D`-coefficient ring columns. The logical
/// 257-field F' link therefore occupies a 270-coordinate public carrier:
/// columns `0..257` hold `[1, enc_inst(x_out)]`, while columns `257..270`
/// are verifier-fixed zero padding in every fresh claim. Running CE claims
/// retain all 270 coordinates because ring-linear folding may populate the
/// final thirteen lanes.
pub const F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN: usize = (F_PRIME_PUBLIC_INPUT_LEN + D - 1) / D * D;

/// Verifier-owned public-input shape for one F' relation.
///
/// Plain F' uses no suffix. A composed application may append public step
/// data after `enc_inst(x_out)`; the next recursive step receives those
/// coordinates through NIFS.V without treating them as part of the hash
/// link itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FPrimePublicInputLayout {
    suffix_len: usize,
}

impl FPrimePublicInputLayout {
    pub const fn plain() -> Self {
        Self { suffix_len: 0 }
    }

    pub const fn with_suffix(suffix_len: usize) -> Self {
        Self { suffix_len }
    }

    /// Logical fields carried before the verifier-fixed ring padding.
    pub const fn logical_len(self) -> usize {
        F_PRIME_PUBLIC_INPUT_LEN + self.suffix_len
    }

    /// Complete public carrier consumed by SuperNeo.
    pub const fn total_len(self) -> usize {
        let logical = self.logical_len();
        (logical + D - 1) / D * D
    }

    pub const fn carrier_padding_len(self) -> usize {
        self.total_len() - self.logical_len()
    }

    pub const fn suffix_len(self) -> usize {
        self.suffix_len
    }

    pub const fn suffix_offset(self) -> usize {
        F_PRIME_PUBLIC_INPUT_LEN
    }

    pub const fn suffix_end(self) -> usize {
        self.suffix_offset() + self.suffix_len
    }

    pub const fn carrier_padding_offset(self) -> usize {
        self.logical_len()
    }
}

/// `enc_inst(x_out)` body: encode `x_out`'s four Goldilocks lanes as
/// `F_PRIME_ENC_INST_BITS` canonical bits (little-endian), so the body is
/// low-norm under `b = 2`. Does **not** prepend the constant-one slot —
/// see [`encode_f_prime_public_input`] for the full CCS public input.
/// Mirrors [`decompose_var_to_u64_bits`]'s in-circuit layout.
pub fn encode_x_out_public_bits(x_out: [F; DIGEST_LEN]) -> Vec<F> {
    let mut out = Vec::with_capacity(F_PRIME_ENC_INST_BITS);
    for lane in x_out {
        let v = lane.as_canonical_u64();
        for bit in 0..X_OUT_BITS_PER_LANE {
            out.push(F::from_u64((v >> bit) & 1));
        }
    }
    out
}

/// Build the full F' CCS instance public input: `[1, enc_inst(x_out)…]`.
///
/// This is the logical HyperNova link. SuperNeo fresh claims must use
/// [`encode_f_prime_superneo_public_input`] so the active ring carrier is
/// completed by verifier-fixed zeros.
pub fn encode_f_prime_public_input(x_out: [F; DIGEST_LEN]) -> Vec<F> {
    let mut out = Vec::with_capacity(F_PRIME_PUBLIC_INPUT_LEN);
    out.push(F::ONE);
    out.extend(encode_x_out_public_bits(x_out));
    out
}

/// Build the plain F' public carrier consumed by SuperNeo:
/// `[1, enc_inst(x_out), 0^13]`.
pub fn encode_f_prime_superneo_public_input(x_out: [F; DIGEST_LEN]) -> Vec<F> {
    let mut out = encode_f_prime_public_input(x_out);
    out.resize(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN, F::ZERO);
    out
}

/// In-circuit inverse of [`encode_x_out_public_bits`]: assert that the
/// length-`F_PRIME_ENC_INST_BITS` `public_bits` wires are the canonical
/// little-endian 64-bit decomposition of `digest[lane]` for each lane.
/// Each lane is canonicity-checked inside [`decompose_var_to_u64_bits`].
///
/// This is public only as an audit/export surface for the Rust-to-Lean
/// artifact pipeline. Both F' branches call this exact helper through their
/// public-output and prior-link bindings.
#[doc(hidden)]
pub fn enforce_public_bits_encode_digest(
    builder: &mut R1csBuilder,
    public_bits: &[Var],
    digest: &[Var; DIGEST_LEN],
) -> Result<(), Error> {
    if public_bits.len() != F_PRIME_ENC_INST_BITS {
        return Err(Error::Inner(format!(
            "F' enc_inst body length {} != {F_PRIME_ENC_INST_BITS}",
            public_bits.len(),
        )));
    }
    for lane in 0..DIGEST_LEN {
        let canonical_bits = decompose_var_to_u64_bits(builder, digest[lane]);
        let offset = lane * X_OUT_BITS_PER_LANE;
        for bit in 0..X_OUT_BITS_PER_LANE {
            builder.enforce_eq(
                &Lc::from_var(public_bits[offset + bit]),
                &Lc::from_var(canonical_bits[bit]),
            );
        }
    }
    Ok(())
}

/// Bind an in-circuit `F`-valued state var to a source-image
/// [`Word64Image`]: enforces the source-image word is canonical Goldilocks
/// (`< p`) AND that `var == Σ 2^i · bit_i` of those source-image bits.
///
/// Used by Phase 7-pre Step 4 to route F' u64 counters (`chunk_count_in`,
/// `step_count_in`, `pc`) through the source image, so the authoritative
/// low-norm witness for each counter is a 64-bit slice of bits — not a
/// freely-allocated field var.
fn enforce_var_matches_source_word64(
    builder: &mut R1csBuilder,
    source_wires: &SourceImageWires,
    word: Word64Image,
    var: Var,
) {
    enforce_goldilocks_word_canonical(builder, source_wires, word);
    let decoded = source_wires.word64_lc(word);
    builder.enforce_eq(&Lc::from_var(var), &decoded);
}

fn source_word_bits(source_wires: &SourceImageWires, word: Word64Image) -> [Var; 64] {
    let bits = source_wires.range(word.bits());
    assert_eq!(bits.len(), 64, "F' counter source word must be 64 bits");
    std::array::from_fn(|i| bits[i])
}

fn enforce_counter_increment_no_wrap(builder: &mut R1csBuilder, old_bits: &[Var; 64], new_counter: Var) -> [Var; 64] {
    let new_bits = decompose_var_to_u64_bits(builder, new_counter);
    enforce_u64_increment(builder, old_bits, &new_bits);
    new_bits
}

fn enforce_counter_add_no_wrap(
    builder: &mut R1csBuilder,
    old_bits: &[Var; 64],
    increment: u64,
    new_counter: Var,
) -> ([Var; 64], [Var; 64]) {
    let increment_bits = alloc_u64_bits(builder, increment);
    enforce_u64_constant(builder, &increment_bits, increment);
    let new_bits = decompose_var_to_u64_bits(builder, new_counter);
    enforce_u64_add(builder, old_bits, &increment_bits, &new_bits);
    (increment_bits, new_bits)
}

/// Wires produced while binding the two authoritative F' input counters to
/// their low-norm source-image words.
///
/// This is an audit surface for the Rust-to-Lean artifact pipeline. The same
/// helper is used by both production F' branches, so the exported subcircuit
/// cannot silently diverge from the rows used in recursive proofs.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct FPrimeCounterInputWires {
    pub chunk_count_bits: [Var; 64],
    pub step_count_bits: [Var; 64],
}

/// Bind the field-valued F' input counters to canonical 64-bit words from the
/// source image, returning the exact bit wires used by the no-wrap arithmetic.
#[doc(hidden)]
pub fn enforce_f_prime_counter_input_binding(
    builder: &mut R1csBuilder,
    source_wires: &SourceImageWires,
    chunk_count_word: Word64Image,
    step_count_word: Word64Image,
    chunk_count: Var,
    step_count: Var,
) -> FPrimeCounterInputWires {
    enforce_var_matches_source_word64(builder, source_wires, chunk_count_word, chunk_count);
    enforce_var_matches_source_word64(builder, source_wires, step_count_word, step_count);
    FPrimeCounterInputWires {
        chunk_count_bits: source_word_bits(source_wires, chunk_count_word),
        step_count_bits: source_word_bits(source_wires, step_count_word),
    }
}

/// Complete wire layout of the production recursive F' counter transition.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct FPrimeCounterTransitionWires {
    pub chunk_count_out: Var,
    pub step_count_out: Var,
    pub chunk_count_out_bits: [Var; 64],
    pub rows_in_chunk_bits: [Var; 64],
    pub step_count_out_bits: [Var; 64],
}

/// Enforce the recursive F' counter equations over both the field and the
/// canonical no-wrap u64 representation:
///
/// `chunk_count_out = chunk_count_in + 1`
/// `step_count_out = step_count_in + rows_in_chunk`.
///
/// The production recursive branch and the formal-artifact exporter call this
/// exact helper. `*_out_value` are witness values only; the emitted equations
/// determine whether those claims are valid.
#[doc(hidden)]
pub fn enforce_f_prime_recursive_counter_transition(
    builder: &mut R1csBuilder,
    chunk_count_in: Var,
    step_count_in: Var,
    input_bits: &FPrimeCounterInputWires,
    rows_in_chunk: u64,
    chunk_count_out_value: u64,
    step_count_out_value: u64,
) -> FPrimeCounterTransitionWires {
    let chunk_count_out = builder.alloc(F::from_u64(chunk_count_out_value));
    let step_count_out = builder.alloc(F::from_u64(step_count_out_value));

    let mut chunk_sum = Lc::from_var(chunk_count_in);
    chunk_sum.add_constant(F::ONE);
    builder.enforce_eq(&Lc::from_var(chunk_count_out), &chunk_sum);
    let mut step_sum = Lc::from_var(step_count_in);
    step_sum.add_constant(F::from_u64(rows_in_chunk));
    builder.enforce_eq(&Lc::from_var(step_count_out), &step_sum);

    let chunk_count_out_bits =
        enforce_counter_increment_no_wrap(builder, &input_bits.chunk_count_bits, chunk_count_out);
    let (rows_in_chunk_bits, step_count_out_bits) =
        enforce_counter_add_no_wrap(builder, &input_bits.step_count_bits, rows_in_chunk, step_count_out);

    FPrimeCounterTransitionWires {
        chunk_count_out,
        step_count_out,
        chunk_count_out_bits,
        rows_in_chunk_bits,
        step_count_out_bits,
    }
}

/// Given pre-allocated source-image bit wires `expected_bits`, constrain
/// them to be `enc_inst(x_out)` — the canonical 64-bit decomposition of
/// each digest lane of `x_out`.
///
/// This is the **output** half of the recursive link. F' produces:
///   - `x_out`: the raw 4-lane Goldilocks `state_x_out_digest` hash.
///   - `x_out_bits`: `enc_inst(x_out)`, the encoding that gets committed
///     as this step's `CcsInstance.x` and read by the next step as
///     `fresh.x`.
///
/// The bits are *not* allocated here. They come from `SourceImageWires`,
/// which already allocated them as bit-constrained witnesses. This keeps
/// every committed F' coordinate bit-valued (low-norm-native discipline).
fn enforce_x_out_public_bit_wires(
    builder: &mut R1csBuilder,
    expected_bits: &[Var],
    x_out: &[Var; DIGEST_LEN],
) -> Result<Vec<Var>, Error> {
    if expected_bits.len() != F_PRIME_ENC_INST_BITS {
        return Err(Error::Inner(format!(
            "F' output enc_inst body length {} != {F_PRIME_ENC_INST_BITS}",
            expected_bits.len()
        )));
    }
    enforce_public_bits_encode_digest(builder, expected_bits, x_out)?;
    Ok(expected_bits.to_vec())
}

/// Configuration for one F' R1CS step.
///
/// The NIFS.V config carries references into the caller's `Preprocessing`
/// (`&Params`, `&Structure`, etc.); the lifetime `'a` is tied to those
/// borrows.
pub struct FPrimeStepConfig<'a> {
    /// NIFS.V composition config.
    pub nifs: NifsVCircuitConfig<'a>,
    /// Norm bound `b` (used to derive the empty-acc constant).
    pub b: u32,
    /// Optional initialization label for the F' transcript. Static so the
    /// in-circuit `TranscriptGadget` can fast-forward its init.
    pub transcript_label: &'static [u8],
    /// Public-input shape of the foldable F' relation. The first 257
    /// coordinates always remain `[1 || enc_inst(x_out)]`; any suffix is
    /// application data surfaced from the previous fresh claims.
    pub public_input_layout: FPrimePublicInputLayout,
    /// Nebula CC-IVC constants. Presence selects the paper's delayed
    /// commitment-carrying relation; plain F' keeps the original relation
    /// and hash/transcript preimages byte-for-byte.
    pub nebula: Option<&'a NebulaConfig>,
    /// Native/circuit state-x_out preimage mode. Stateless mode omits the
    /// duplicate semantic digest and this circuit enforces semantic == acc.
    pub state_x_out_digest_mode: StateXOutDigestMode,
}

/// Common state-in fields shared between base and recursive F' steps.
#[derive(Clone)]
pub struct FPrimeStateIn {
    pub vk_fs_digest: [F; DIGEST_LEN],
    /// SplitNc header of the relation being folded. Carried as verifier-key
    /// advice so folded F' never embeds a digest of its own matrices.
    pub pi_ccs_header_bundle: [F; DIGEST_LEN],
    pub chunk_count_in: u64,
    pub step_count_in: u64,
    pub z_0: [F; DIGEST_LEN],
    pub z_i_in: [F; DIGEST_LEN],
    pub pc: u64,
    pub semantic_state_digest_in: [F; DIGEST_LEN],
    pub acc_digest_in: [F; DIGEST_LEN],
    pub public_trace_in: [F; DIGEST_LEN],
    /// Carried memory state. Presence must match [`FPrimeStepConfig::nebula`].
    pub nebula: Option<NebulaLane>,
}

/// Inputs to the F' **base** step (i = 0).
///
/// Base is a real first F' step: it consumes a `chunk_digest` and advances
/// both `chunk_count` and `step_count`. The `rows_in_chunk` field carries
/// the K-value the native `advance_state` increments `step_count` by.
pub struct FPrimeBaseInputs<'a> {
    pub state: FPrimeStateIn,
    pub chunk_digest: [F; DIGEST_LEN],
    pub semantic_state_digest_out: [F; DIGEST_LEN],
    /// Number of fresh-instance rows this chunk encodes. Native
    /// `advance_state(prev, _, fresh_count, _)` adds `fresh_count` to
    /// `step_count`; we mirror that here.
    pub rows_in_chunk: u64,
    /// Source image for F' public-boundary encodings. At this stage it
    /// backs `enc_inst` bits and selected u64 boundary counters only; it
    /// is not the full private `enc(F')` witness — internal F' computation
    /// slots remain ordinary field values. See `encoding.md` for the generic
    /// reference encoder and the production feasibility gate.
    pub source_image: &'a FPrimeSourceImage,
    /// Source-image word for `state.chunk_count_in` (Step 4).
    pub chunk_count_in_word: Word64Image,
    /// Source-image word for `state.step_count_in` (Step 4).
    pub step_count_in_word: Word64Image,
    /// Source-image word for `state.pc` (Step 4).
    pub pc_word: Word64Image,
    /// Slice inside `source_image` that holds the `enc_inst(x_out)` body
    /// for this step (`F_PRIME_ENC_INST_BITS` bits).
    pub public_x_out_bits: BitRange,
}

/// Inputs to the F' **recursive** step (i ≥ 1). Includes NIFS proof.
pub struct FPrimeRecursiveInputs<'a> {
    pub state: FPrimeStateIn,
    pub chunk_digest: [F; DIGEST_LEN],
    pub semantic_state_digest_out: [F; DIGEST_LEN],
    /// Outgoing accumulator handle carried in `state_out` and absorbed by
    /// `state_x_out`. The recursive F' step recomputes this value from
    /// NIFS.V's output `(children, parent)` and rejects if the supplied
    /// witness value disagrees.
    pub acc_digest_out: [F; DIGEST_LEN],
    pub nifs_msg: NifsVCircuitMessages<'a>,
    /// Size of the **current** chunk being deposited as the new latest —
    /// i.e., `next_latest.len()` in [`crate::paper::f_prime::native::prove`].
    /// Drives the `step_count` advance in this step
    /// (`step_count_out = step_count_in + rows_in_chunk`), matching native
    /// `advance_state(..., fresh_count, ...)` semantics.
    ///
    /// This is **not** `nifs_msg.fresh.len()`: that is the *previous*
    /// latest batch being folded by NIFS.V at this step. When batch
    /// sizes vary across steps (e.g. `K=61` then `K=62`), the two
    /// differ and using `nifs_msg.fresh.len()` would advance
    /// `step_count` by the wrong amount.
    pub rows_in_chunk: u64,
    /// Low-norm-native witness image (Phase 7-pre); see
    /// [`FPrimeBaseInputs::source_image`].
    pub source_image: &'a FPrimeSourceImage,
    /// Source-image word for `state.chunk_count_in` (Step 4).
    pub chunk_count_in_word: Word64Image,
    /// Source-image word for `state.step_count_in` (Step 4).
    pub step_count_in_word: Word64Image,
    /// Source-image word for `state.pc` (Step 4).
    pub pc_word: Word64Image,
    /// Slice inside `source_image` for `enc_inst(prior_x_out)` — the body
    /// of the **input** recursive link. F' constrains both
    /// `source_image[prior_x_out_bits] == enc_inst(prior_x_out)` AND
    /// `fresh[i].x[1..] == source_image[prior_x_out_bits]` for **every**
    /// fresh `u_i` in the batch, so the authoritative low-norm witness
    /// for the input link lives in the source image rather than inside
    /// the NIFS proof message. Same `prior_x_out` for the whole chunk:
    /// every fresh in the batch is rooted at the same prior
    /// Construction-2 state.
    pub prior_x_out_bits: BitRange,
    /// Slice inside `source_image` that holds the `enc_inst(x_out)` body
    /// for this step's **output** link.
    pub public_x_out_bits: BitRange,
}

/// Output wires of one F' R1CS step.
///
/// - `x_out`: the four-lane Goldilocks `state_x_out_digest` (HyperNova's
///   raw `F'.x_out`).
/// - `x_out_bits`: `enc_inst(x_out)` — the canonical bit encoding committed
///   as this step's `CcsInstance.x` and read by the next step as `fresh.x`.
/// - `state_in` / `state_out`: full state-in/out wire bundles. The decider
///   chains these across steps (enforces `prev.state_out == next.state_in`)
///   and pins `state_out` of the last step to `decider::PublicImage`
///   (with the post-fold accumulator digest substituted in).
/// - `nifs_running` / `nifs_children`: per-claim wire bundles from the
///   embedded NIFS.V (recursive step only; both `None` for base). The
///   decider uses these to enforce CE-claim continuity: every recursive
///   step k+1's `nifs_running` must equal step k's `nifs_children`
///   across all currently bound fields (not just by accumulator-digest
///   equality). Π_DEC's child y_zcol sidecar is excluded by the current
///   encoding; the delayed-projection authority bridge must close that gap.
pub struct FPrimeStepOutput {
    pub x_out: [Var; DIGEST_LEN],
    pub x_out_bits: Vec<Var>,
    /// Recursive consumer-side link wires. `None` for the base branch.
    /// This is a read-only formal-artifact surface; the enforcing rows remain
    /// authoritative.
    #[doc(hidden)]
    pub prior_link: Option<FPrimePriorLinkWires>,
    pub state_in: FPrimeStateWires,
    pub state_out: FPrimeStateWires,
    pub nifs_running: Option<Vec<crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires>>,
    pub nifs_running_parent_authority:
        Option<crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires>,
    pub nifs_parent: Option<crate::paper::reductions::pi_dec_circuit::CeClaimWires>,
    pub nifs_children: Option<Vec<crate::paper::reductions::pi_dec_circuit::CeClaimWires>>,
    /// Application public suffixes carried by the fresh claims consumed in
    /// this recursive step. Empty for the base step and for plain F'.
    pub fresh_public_suffixes: Vec<Vec<Var>>,
    /// Product-commitment coordinates paired index-for-index with
    /// [`Self::fresh_public_suffixes`].
    pub fresh_adv: Vec<Option<crate::paper::relations::product_commitment_circuit::AdvCommitmentWires>>,
}

/// Exact wires participating in the delayed HyperNova public-input link.
#[doc(hidden)]
#[derive(Clone)]
pub struct FPrimePriorLinkWires {
    pub row_start: usize,
    pub row_end: usize,
    pub first_allocated_column: usize,
    pub digest: [Var; DIGEST_LEN],
    pub encoded_bits: Vec<Var>,
    pub fresh_public_inputs: Vec<Vec<Var>>,
}

/// Full state wire bundle — used both for pre-step (`state_in`) and
/// post-step (`state_out`) views. Fields that don't change across a step
/// (vk_fs_digest, pi_ccs_header_bundle, z_0, pc) share wires between `state_in`
/// and `state_out`.
#[derive(Clone, Copy)]
pub struct FPrimeStateWires {
    pub vk_fs_digest: [Var; DIGEST_LEN],
    pub pi_ccs_header_bundle: [Var; DIGEST_LEN],
    pub chunk_count: Var,
    pub step_count: Var,
    pub z_0: [Var; DIGEST_LEN],
    pub z_i: [Var; DIGEST_LEN],
    pub pc: Var,
    pub semantic_state_digest: [Var; DIGEST_LEN],
    pub acc_digest: [Var; DIGEST_LEN],
    pub public_trace: [Var; DIGEST_LEN],
    pub nebula: Option<NebulaLaneWires>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("F' R1CS: {0}")]
    Inner(String),
}

impl From<crate::paper::nifs::circuit::Error> for Error {
    fn from(e: crate::paper::nifs::circuit::Error) -> Self {
        Self::Inner(format!("NIFS.V: {e}"))
    }
}

fn validate_nebula_configuration(
    cfg: &FPrimeStepConfig<'_>,
    state: &FPrimeStateIn,
    fresh_count: Option<usize>,
) -> Result<(), Error> {
    match (cfg.nebula, state.nebula.as_ref()) {
        (None, None) => Ok(()),
        (Some(nebula), Some(_)) => {
            if nebula.steps_per_segment == 0 {
                return Err(Error::Inner("Nebula F' requires steps_per_segment >= 1".into()));
            }
            let expected_suffix = delayed_nebula_public_suffix_len(nebula.stacks);
            if cfg.public_input_layout.suffix_len() != expected_suffix {
                return Err(Error::Inner(format!(
                    "Nebula F' public suffix length {} != delayed contract {expected_suffix}",
                    cfg.public_input_layout.suffix_len()
                )));
            }
            if let Some(fresh_count) = fresh_count {
                if fresh_count != 1 {
                    return Err(Error::Inner(format!(
                        "Nebula F' currently requires exactly one delayed fresh claim, got {fresh_count}"
                    )));
                }
            }
            Ok(())
        }
        _ => Err(Error::Inner("Nebula F' config/state presence mismatch".into())),
    }
}

/// Allocate state-in wires + the `pc == TRIVIAL_PC` guard. Shared between
/// base and recursive F' entry points.
struct StateInWires {
    vk_fs: [Var; DIGEST_LEN],
    pi_ccs_header_bundle: [Var; DIGEST_LEN],
    chunk_count_in: Var,
    step_count_in: Var,
    z_0: [Var; DIGEST_LEN],
    z_i_in: [Var; DIGEST_LEN],
    pc: Var,
    semantic_state_digest_in: [Var; DIGEST_LEN],
    acc_digest_in: [Var; DIGEST_LEN],
    public_trace_in: [Var; DIGEST_LEN],
    nebula: Option<NebulaLaneWires>,
}

fn alloc_state_in(builder: &mut R1csBuilder, state: &FPrimeStateIn) -> StateInWires {
    let pc = builder.alloc(F::from_u64(state.pc));
    builder.enforce_eq(&Lc::from_var(pc), &Lc::from_const(F::from_u64(TRIVIAL_PC)));
    StateInWires {
        vk_fs: alloc_4(builder, state.vk_fs_digest),
        pi_ccs_header_bundle: alloc_4(builder, state.pi_ccs_header_bundle),
        chunk_count_in: builder.alloc(F::from_u64(state.chunk_count_in)),
        step_count_in: builder.alloc(F::from_u64(state.step_count_in)),
        z_0: alloc_4(builder, state.z_0),
        z_i_in: alloc_4(builder, state.z_i_in),
        pc,
        semantic_state_digest_in: alloc_4(builder, state.semantic_state_digest_in),
        acc_digest_in: alloc_4(builder, state.acc_digest_in),
        public_trace_in: alloc_4(builder, state.public_trace_in),
        nebula: state
            .nebula
            .as_ref()
            .map(|lane| alloc_nebula_lane_wires(builder, lane)),
    }
}

/// Repackage [`StateInWires`] (private, pre-step view) as the public
/// [`FPrimeStateWires`] returned to callers.
fn state_in_to_wires(sw: &StateInWires) -> FPrimeStateWires {
    FPrimeStateWires {
        vk_fs_digest: sw.vk_fs,
        pi_ccs_header_bundle: sw.pi_ccs_header_bundle,
        chunk_count: sw.chunk_count_in,
        step_count: sw.step_count_in,
        z_0: sw.z_0,
        z_i: sw.z_i_in,
        pc: sw.pc,
        semantic_state_digest: sw.semantic_state_digest_in,
        acc_digest: sw.acc_digest_in,
        public_trace: sw.public_trace_in,
        nebula: sw.nebula,
    }
}

/// Build the public state-out view from the unchanged-across-step wires
/// (taken from `sw`) plus the updated wires (counters, z_i, public_trace,
/// acc_digest) emitted by the step body.
fn state_out_wires(
    sw: &StateInWires,
    chunk_count: Var,
    step_count: Var,
    z_i: [Var; DIGEST_LEN],
    public_trace: [Var; DIGEST_LEN],
    semantic_state_digest: [Var; DIGEST_LEN],
    acc_digest: [Var; DIGEST_LEN],
    nebula: Option<NebulaLaneWires>,
) -> FPrimeStateWires {
    FPrimeStateWires {
        vk_fs_digest: sw.vk_fs,
        pi_ccs_header_bundle: sw.pi_ccs_header_bundle,
        chunk_count,
        step_count,
        z_0: sw.z_0,
        z_i,
        pc: sw.pc,
        semantic_state_digest,
        acc_digest,
        public_trace,
        nebula,
    }
}

/// Build the `x_out` wires for the post-step state. Common tail between
/// base and recursive entry points. Returns both the `x_out` digest and
/// the carried `new_z_i` / mirrored `new_public_trace` wires so callers can
/// thread the post-step state into the next step (decider chain) or
/// into the terminal-fold gate.
fn build_x_out(
    builder: &mut R1csBuilder,
    mode: StateXOutDigestMode,
    sw: &StateInWires,
    chunk_digest: [Var; DIGEST_LEN],
    new_semantic_state_digest: [Var; DIGEST_LEN],
    new_acc_digest: [Var; DIGEST_LEN],
    new_chunk_count: Var,
    new_step_count: Var,
    nebula: Option<&NebulaLaneWires>,
) -> ([Var; DIGEST_LEN], [Var; DIGEST_LEN], [Var; DIGEST_LEN]) {
    if matches!(mode, StateXOutDigestMode::Stateless) {
        enforce_digest_eq(builder, &new_semantic_state_digest, &new_acc_digest);
    }
    let new_z_i = chunk_digest;
    let new_public_trace = new_z_i;
    let x_out_inputs = StateXOutDigestInputs {
        mode,
        vk_fs_digest: sw.vk_fs,
        pi_ccs_header_bundle: sw.pi_ccs_header_bundle,
        structure_digest: sw.pi_ccs_header_bundle,
        chunk_count: new_chunk_count,
        step_count: new_step_count,
        initial_boundary: sw.z_0,
        current_boundary: new_z_i,
        pc: sw.pc,
        semantic_acc: new_semantic_state_digest,
        construction2_acc: new_acc_digest,
        public_trace: new_public_trace,
    };
    let x_out = match nebula {
        None => enforce_state_x_out_digest_circuit(builder, &x_out_inputs),
        Some(lane) => {
            let lane_digest = enforce_nebula_lane_digest_selected_circuit(builder, lane);
            enforce_state_x_out_digest_with_nebula_circuit(builder, &x_out_inputs, lane_digest)
        }
    };
    (x_out, new_z_i, new_public_trace)
}

fn enforce_digest_eq(builder: &mut R1csBuilder, a: &[Var; DIGEST_LEN], b: &[Var; DIGEST_LEN]) {
    for lane in 0..DIGEST_LEN {
        builder.enforce_eq(&Lc::from_var(a[lane]), &Lc::from_var(b[lane]));
    }
}

/// F' **base** step (i = 0). No NIFS.V. Enforces `chunk_count_in == 0`,
/// `z_i_in == z_0`, and `acc_digest_in == AccumulatorHandle::empty()`.
/// Sets `acc_digest_out = empty_acc_digest`. Returns the `x_out` wires.
pub fn enforce_f_prime_base_step_circuit(
    builder: &mut R1csBuilder,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeBaseInputs<'_>,
) -> Result<FPrimeStepOutput, Error> {
    enforce_f_prime_base_step_with_output_acc(builder, cfg, inputs, AccumulatorHandle::empty().digest_fields())
}

/// Authoritative Construction-2 base branch.
///
/// Unlike the legacy shell entrypoint, this emits the formal SuperNeo
/// `u_perp in CE(b,L)^k`: exactly `k` zero CE children plus their derived
/// decomposition parent. The incoming pre-chain sentinel remains empty, but
/// the first `x_out` commits to the fixed-shape accumulator that the next
/// recursive NIFS call consumes.
pub fn enforce_construction2_f_prime_base_step_circuit(
    builder: &mut R1csBuilder,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeBaseInputs<'_>,
) -> Result<FPrimeStepOutput, Error> {
    let relation = &cfg.nifs.pi_ccs.structure;
    let zero = crate::paper::construction2::RunningInstance::canonical_zero_for_shape(
        cfg.nifs.pi_ccs.params,
        relation.n(),
        relation.m(),
        relation.t(),
        cfg.public_input_layout.total_len(),
    )
    .map_err(|error| Error::Inner(format!("canonical Construction-2 accumulator: {error}")))?;
    let zero_digest = crate::paper::digest::digest32_as_fields(
        zero.accumulator_digest_for_relation_columns(relation.m())
            .map_err(|error| Error::Inner(format!("canonical Construction-2 accumulator digest: {error}")))?,
    );
    enforce_f_prime_base_step_with_output_acc(builder, cfg, inputs, zero_digest)
}

fn enforce_f_prime_base_step_with_output_acc(
    builder: &mut R1csBuilder,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeBaseInputs<'_>,
    output_acc_digest: [F; DIGEST_LEN],
) -> Result<FPrimeStepOutput, Error> {
    let base_start = builder.rows();
    validate_nebula_configuration(cfg, &inputs.state, None)?;
    if inputs.rows_in_chunk == 0 {
        return Err(Error::Inner(
            "strict F' base: rows_in_chunk must be \u{2265} 1 (first chunk must be non-empty)".into(),
        ));
    }
    let fresh_len = usize::try_from(inputs.rows_in_chunk)
        .map_err(|_| Error::Inner("strict F' base: rows_in_chunk exceeds platform usize".into()))?;
    if fresh_len > cfg.nifs.pi_ccs.params.max_fresh_count() {
        return Err(Error::Inner(format!(
            "strict F' base: rows_in_chunk {} exceeds max_fresh_count {}",
            inputs.rows_in_chunk,
            cfg.nifs.pi_ccs.params.max_fresh_count()
        )));
    }

    builder.begin_encoding_stage(stage::BASE_STEP);
    builder.begin_encoding_stage(stage::BASE_PRELUDE);
    let sw = alloc_state_in(builder, &inputs.state);
    let chunk_digest = alloc_4(builder, inputs.chunk_digest);
    let expected_chunk_digest = enforce_f_prime_chunk_public_digest_circuit(
        builder,
        sw.step_count_in,
        fresh_len,
        D,
        cfg.nifs.pi_ccs.params.kappa() as usize,
        cfg.public_input_layout.total_len(),
    );
    enforce_digest_eq(builder, &chunk_digest, &expected_chunk_digest);
    if let (Some(nebula_cfg), Some(lane)) = (cfg.nebula, sw.nebula.as_ref()) {
        enforce_nebula_lane_constant_circuit(builder, lane, &NebulaLane::base(nebula_cfg));
    }
    builder.record_row_family("fprime.base.prelude", base_start);

    // Allocate source-image bits once for the whole F' step. Each
    // coordinate becomes a bit-constrained witness wire; the public
    // x_out bits are sliced from this image below.
    let source_start = builder.rows();
    builder.begin_encoding_stage(stage::BASE_SOURCE);
    let source_wires = SourceImageWires::alloc(builder, inputs.source_image);

    // Bind u64 counters to source-image words (Step 4). The `Var`s
    // remain in use downstream; this just pins them to the canonical
    // low-norm bit decomposition stored in the source image.
    let counter_inputs = enforce_f_prime_counter_input_binding(
        builder,
        &source_wires,
        inputs.chunk_count_in_word,
        inputs.step_count_in_word,
        sw.chunk_count_in,
        sw.step_count_in,
    );
    enforce_var_matches_source_word64(builder, &source_wires, inputs.pc_word, sw.pc);
    builder.record_row_family("fprime.base.source", source_start);

    // Base pre-state: chunk_count_in == 0, step_count_in == 0, z_i_in == z_0.
    let initial_start = builder.rows();
    builder.begin_encoding_stage(stage::BASE_INITIAL);
    builder.enforce_eq(&Lc::from_var(sw.chunk_count_in), &Lc::zero());
    builder.enforce_eq(&Lc::from_var(sw.step_count_in), &Lc::zero());
    for k in 0..DIGEST_LEN {
        builder.enforce_eq(&Lc::from_var(sw.z_i_in[k]), &Lc::from_var(sw.z_0[k]));
    }

    // acc_digest_in must equal the empty-accumulator digest constant.
    let empty_acc = AccumulatorHandle::empty().digest_fields();
    for k in 0..DIGEST_LEN {
        builder.enforce_eq(&Lc::from_var(sw.acc_digest_in[k]), &Lc::from_const(empty_acc[k]));
    }
    builder.record_row_family("fprime.base.initial", initial_start);

    let advance_start = builder.rows();
    builder.begin_encoding_stage(stage::BASE_ADVANCE);
    let new_acc_digest = alloc_4_const(builder, output_acc_digest);
    let new_semantic_state_digest = alloc_4(builder, inputs.semantic_state_digest_out);

    // Counter advance: chunk_count' = 1, step_count' = rows_in_chunk.
    // Base IS a real F' step (consumes a chunk_digest, updates z_i and
    // public_trace), so it must also advance step_count consistently with
    // native `advance_state(prev, _, fresh_count, _)`.
    let new_chunk_count = builder.alloc(F::ONE);
    builder.enforce_eq(&Lc::from_var(new_chunk_count), &Lc::from_const(F::ONE));
    let new_step_count = builder.alloc(F::from_u64(inputs.rows_in_chunk));
    builder.enforce_eq(
        &Lc::from_var(new_step_count),
        &Lc::from_const(F::from_u64(inputs.rows_in_chunk)),
    );
    enforce_counter_increment_no_wrap(builder, &counter_inputs.chunk_count_bits, new_chunk_count);
    enforce_counter_add_no_wrap(
        builder,
        &counter_inputs.step_count_bits,
        inputs.rows_in_chunk,
        new_step_count,
    );
    builder.record_row_family("fprime.base.advance", advance_start);

    let output_start = builder.rows();
    builder.begin_encoding_stage(stage::BASE_OUTPUT);
    let (x_out, new_z_i, new_public_trace) = build_x_out(
        builder,
        cfg.state_x_out_digest_mode,
        &sw,
        chunk_digest,
        new_semantic_state_digest,
        new_acc_digest,
        new_chunk_count,
        new_step_count,
        sw.nebula.as_ref(),
    );
    let expected_bits = source_wires.range(inputs.public_x_out_bits);
    let x_out_bits = enforce_x_out_public_bit_wires(builder, expected_bits, &x_out)?;
    let state_in = state_in_to_wires(&sw);
    let state_out = state_out_wires(
        &sw,
        new_chunk_count,
        new_step_count,
        new_z_i,
        new_public_trace,
        new_semantic_state_digest,
        new_acc_digest,
        sw.nebula,
    );
    builder.record_row_family("fprime.base.output", output_start);
    builder.record_row_family("fprime.base.total", base_start);
    Ok(FPrimeStepOutput {
        x_out,
        x_out_bits,
        prior_link: None,
        state_in,
        state_out,
        // Base step does not run NIFS.V; no children/running wires.
        nifs_running: None,
        nifs_running_parent_authority: None,
        nifs_parent: None,
        nifs_children: None,
        fresh_public_suffixes: Vec::new(),
        fresh_adv: Vec::new(),
    })
}

/// F' **recursive** step (i ≥ 1). Runs NIFS.V to fold the K fresh
/// `u_i,j` into `U_i`, enforces the HyperNova recursive link
/// `u_i,j.public == prior x_out` for every j in the batch, and binds
/// `acc_digest_in` to the digest of the actual running accumulator.
///
/// Strict mode preconditions (rejected at message-shape check):
///   - non-empty fresh batch (`inputs.nifs_msg.fresh.len() >= 1`).
///   - `inputs.rows_in_chunk >= 1` — the new chunk being deposited
///     must be non-empty.
///   - every fresh `u_i,j` has the verifier-configured physical carrier
///     length. Its fixed prefix encodes the prior `x_out` digest as
///     `DIGEST_LEN * 64` low-norm bits; an optional application suffix follows
///     it; the remaining ring-completion coordinates are fixed to zero.
///   - all K fresh public inputs are bound to the **same** prior
///     `x_out` source-image bits; the whole batch is treated as one
///     SuperNeo chunk rooted at one prior Construction-2 state.
pub fn enforce_f_prime_recursive_step_circuit(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeRecursiveInputs<'_>,
) -> Result<FPrimeStepOutput, Error> {
    enforce_f_prime_recursive_step_circuit_impl(builder, pp, cfg, None, inputs)
}

/// Fixed-relation recursive branch. The Π_CCS header is verifier-key witness
/// data shared with the surrounding `vk_fs` digest constraint.
pub(crate) fn enforce_f_prime_recursive_step_circuit_with_header_bundle_wires(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &FPrimeStepConfig<'_>,
    header_bundle_wires: [Var; DIGEST_LEN],
    inputs: &FPrimeRecursiveInputs<'_>,
) -> Result<FPrimeStepOutput, Error> {
    enforce_f_prime_recursive_step_circuit_impl(builder, pp, cfg, Some(header_bundle_wires), inputs)
}

fn enforce_f_prime_recursive_step_circuit_impl(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &FPrimeStepConfig<'_>,
    header_bundle_wires: Option<[Var; DIGEST_LEN]>,
    inputs: &FPrimeRecursiveInputs<'_>,
) -> Result<FPrimeStepOutput, Error> {
    let recursive_start = builder.rows();
    // ── Strict-mode shape ──────────────────────────────────────────────
    if inputs.nifs_msg.fresh.is_empty() {
        return Err(Error::Inner(
            "strict F' recursive: fresh batch must be non-empty".into(),
        ));
    }
    validate_nebula_configuration(cfg, &inputs.state, Some(inputs.nifs_msg.fresh.len()))?;
    if inputs.rows_in_chunk == 0 {
        return Err(Error::Inner(
            "strict F' recursive: rows_in_chunk must be \u{2265} 1 (new chunk must be non-empty)".into(),
        ));
    }
    let next_fresh_len = usize::try_from(inputs.rows_in_chunk)
        .map_err(|_| Error::Inner("strict F' recursive: rows_in_chunk exceeds platform usize".into()))?;
    if next_fresh_len > cfg.nifs.pi_ccs.params.max_fresh_count() {
        return Err(Error::Inner(format!(
            "strict F' recursive: rows_in_chunk {} exceeds max_fresh_count {}",
            inputs.rows_in_chunk,
            cfg.nifs.pi_ccs.params.max_fresh_count()
        )));
    }
    let expected_public_input_len = cfg.public_input_layout.total_len();
    for (idx, fresh) in inputs.nifs_msg.fresh.iter().enumerate() {
        if fresh.m_in != expected_public_input_len || fresh.x.len() != expected_public_input_len {
            return Err(Error::Inner(format!(
                "strict F' recursive: fresh[{idx}] public input must be [1 | {F_PRIME_ENC_INST_BITS} enc_inst bits | {} suffix fields | {} fixed-zero padding fields] (total {}), \
                 got m_in={}, x.len={}",
                cfg.public_input_layout.suffix_len(),
                cfg.public_input_layout.carrier_padding_len(),
                expected_public_input_len,
                fresh.m_in,
                fresh.x.len()
            )));
        }
    }

    // Recursive means i ≥ 1; both counters must have advanced past the
    // true base state. `chunk_count_in != 0` separates the recursive
    // branch from NoFold; `step_count_in != 0` prevents a forged Active
    // state from representing "some prior chunk, but zero folded rows."
    if inputs.state.chunk_count_in == 0 {
        return Err(Error::Inner(
            "strict F' recursive: chunk_count_in must be nonzero".into(),
        ));
    }

    builder.begin_encoding_stage(stage::RECURSIVE_STEP);
    builder.begin_encoding_stage(stage::RECURSIVE_PRELUDE);
    let sw = alloc_state_in(builder, &inputs.state);
    let chunk_digest = alloc_4(builder, inputs.chunk_digest);
    let expected_chunk_digest = enforce_f_prime_chunk_public_digest_circuit(
        builder,
        sw.step_count_in,
        next_fresh_len,
        D,
        cfg.nifs.pi_ccs.params.kappa() as usize,
        cfg.public_input_layout.total_len(),
    );
    enforce_digest_eq(builder, &chunk_digest, &expected_chunk_digest);
    let nebula_lane_in_digest = sw
        .nebula
        .as_ref()
        .map(|lane| enforce_nebula_lane_digest_selected_circuit(builder, lane));

    // Allocate source-image bits once for the whole F' step. Each
    // coordinate becomes a bit-constrained witness wire; the public
    // x_out bits are sliced from this image at the end of the step.
    let source_wires = SourceImageWires::alloc(builder, inputs.source_image);

    // Bind u64 counters to source-image words (Step 4).
    let counter_inputs = enforce_f_prime_counter_input_binding(
        builder,
        &source_wires,
        inputs.chunk_count_in_word,
        inputs.step_count_in_word,
        sw.chunk_count_in,
        sw.step_count_in,
    );
    enforce_var_matches_source_word64(builder, &source_wires, inputs.pc_word, sw.pc);

    // In-circuit nonzero gadgets: prove counter · inv == 1. If a malicious
    // prover sets a counter wire to zero, no inverse can satisfy the row.
    // The native chunk-count check above keeps honest witness construction
    // from taking the zero inverse path; the step-count row is intentionally
    // still emitted for forged witnesses so the circuit itself owns the
    // branch invariant.
    {
        use p3_field::Field;
        let cc_in_f = F::from_u64(inputs.state.chunk_count_in);
        let inv = builder.alloc(cc_in_f.inverse());
        builder.enforce(
            &Lc::from_var(sw.chunk_count_in),
            &Lc::from_var(inv),
            &Lc::from_const(F::ONE),
        );

        let sc_in_f = F::from_u64(inputs.state.step_count_in);
        let sc_inv_val = if sc_in_f == F::ZERO { F::ZERO } else { sc_in_f.inverse() };
        let sc_inv = builder.alloc(sc_inv_val);
        builder.enforce(
            &Lc::from_var(sw.step_count_in),
            &Lc::from_var(sc_inv),
            &Lc::from_const(F::ONE),
        );
    }

    // ── NIFS.V composition ──────────────────────────────────────────────
    builder.record_row_family("fprime.recursive.prelude", recursive_start);
    let transcript_start = builder.rows();
    builder.begin_encoding_stage(stage::RECURSIVE_TRANSCRIPT);
    let mut transcript = TranscriptGadget::new(builder, cfg.transcript_label);
    transcript.append_fields(builder, b"f_prime/vk_fs", &sw.vk_fs);
    transcript.append_fields(builder, b"f_prime/pi_ccs_header", &sw.pi_ccs_header_bundle);
    transcript.append_fields(builder, b"f_prime/chunk_count_in", &[sw.chunk_count_in]);
    transcript.append_fields(builder, b"f_prime/step_count_in", &[sw.step_count_in]);
    transcript.append_fields(builder, b"f_prime/z_0", &sw.z_0);
    transcript.append_fields(builder, b"f_prime/z_i_in", &sw.z_i_in);
    transcript.append_fields(builder, b"f_prime/pc", &[sw.pc]);
    transcript.append_fields(builder, b"f_prime/semantic_state_in", &sw.semantic_state_digest_in);
    transcript.append_fields(builder, b"f_prime/acc_digest_in", &sw.acc_digest_in);
    transcript.append_fields(builder, b"f_prime/public_trace_in", &sw.public_trace_in);
    if let Some(lane_digest) = nebula_lane_in_digest.as_ref() {
        transcript.append_fields(builder, b"f_prime/nebula_lane_in", lane_digest);
    }
    transcript.append_fields(builder, b"f_prime/chunk_digest", &chunk_digest);

    builder.record_row_family("fprime.recursive.transcript", transcript_start);
    let nifs_start = builder.rows();
    builder.begin_encoding_stage(stage::RECURSIVE_NIFS);
    let nifs_outputs = if let Some(header_bundle_wires) = header_bundle_wires {
        enforce_nifs_v_circuit_with_transcript_and_header_bundle_wires(
            builder,
            pp,
            &cfg.nifs,
            &mut transcript,
            header_bundle_wires,
            &inputs.nifs_msg,
        )?
    } else {
        enforce_nifs_v_circuit_with_transcript_and_header_bundle(
            builder,
            pp,
            &cfg.nifs,
            &mut transcript,
            &inputs.nifs_msg,
            sw.pi_ccs_header_bundle,
        )?
    };
    builder.record_row_family("fprime.recursive.nifs", nifs_start);
    let prior_link_start = builder.rows();
    let prior_link_first_column = builder.cols();

    builder.begin_encoding_stage(stage::RECURSIVE_PRIOR_LINK_DIGEST);
    // ── HyperNova recursive link: u_i.public == bits(prior_x_out) ───────
    //
    // The fresh CCS instance's public input MUST encode the previous F'
    // step's `x_out` digest. Raw digest lanes are not low-norm under b=2,
    // so the public input carries canonical Goldilocks bits instead.
    let prior_x_out_inputs = StateXOutDigestInputs {
        mode: cfg.state_x_out_digest_mode,
        vk_fs_digest: sw.vk_fs,
        pi_ccs_header_bundle: sw.pi_ccs_header_bundle,
        structure_digest: sw.pi_ccs_header_bundle,
        chunk_count: sw.chunk_count_in,
        step_count: sw.step_count_in,
        initial_boundary: sw.z_0,
        current_boundary: sw.z_i_in,
        pc: sw.pc,
        semantic_acc: sw.semantic_state_digest_in,
        construction2_acc: sw.acc_digest_in,
        public_trace: sw.public_trace_in,
    };
    if matches!(cfg.state_x_out_digest_mode, StateXOutDigestMode::Stateless) {
        enforce_digest_eq(builder, &sw.semantic_state_digest_in, &sw.acc_digest_in);
    }
    let prior_x_out = match nebula_lane_in_digest {
        None => enforce_state_x_out_digest_circuit(builder, &prior_x_out_inputs),
        Some(lane_digest) => enforce_state_x_out_digest_with_nebula_circuit(builder, &prior_x_out_inputs, lane_digest),
    };

    if nifs_outputs.fresh_x.is_empty() {
        return Err(Error::Inner("strict F' missing fresh u_i".into()));
    }
    // Recursive input link, routed through the source image:
    //   1) source_image[prior_x_out_bits] == enc_inst(prior_x_out)
    //      — pins source bits to canonical 64-bit decomposition of the
    //      in-circuit-recomputed prior_x_out digest.
    //   2) for every fresh u_i in the chunk:
    //        fresh[i].x[0]     == 1                                 (CCS one-slot)
    //        fresh[i].x[1..]   == source_image[prior_x_out_bits]    (recursive link)
    //      The same prior_x_out witness bits are reused for every fresh:
    //      the whole batch is one SuperNeo chunk rooted at a single prior
    //      Construction-2 state, so every fresh CCS instance's public
    //      input encodes the same prior `x_out`. The NIFS proof's
    //      `fresh[i].x` must agree with the source image; tampering
    //      either side breaks the link.
    let prior_bits = source_wires.range(inputs.prior_x_out_bits);
    builder.begin_encoding_stage(stage::RECURSIVE_PRIOR_LINK_ENC_INST);
    enforce_public_bits_encode_digest(builder, prior_bits, &prior_x_out)?;
    let mut fresh_public_suffixes = Vec::with_capacity(nifs_outputs.fresh_x.len());
    for (idx, fresh_x) in nifs_outputs.fresh_x.iter().enumerate() {
        if fresh_x.len() != expected_public_input_len {
            return Err(Error::Inner(format!(
                "strict F' fresh[{idx}] public input length {} != configured length {expected_public_input_len}",
                fresh_x.len(),
            )));
        }
        // CCS constant-one slot.
        builder.enforce_eq(
            &Lc::from_var(fresh_x[F_PRIME_PUBLIC_ONE_OFFSET]),
            &Lc::from_const(F::ONE),
        );
        let link_end = F_PRIME_ENC_INST_OFFSET + F_PRIME_ENC_INST_BITS;
        for (fresh_bit, source_bit) in fresh_x[F_PRIME_ENC_INST_OFFSET..link_end]
            .iter()
            .zip(prior_bits.iter())
        {
            builder.enforce_eq(&Lc::from_var(*fresh_bit), &Lc::from_var(*source_bit));
        }
        fresh_public_suffixes
            .push(fresh_x[cfg.public_input_layout.suffix_offset()..cfg.public_input_layout.suffix_end()].to_vec());
    }
    builder.begin_encoding_stage(stage::RECURSIVE_PRIOR_LINK_CARRIER_PADDING);
    for fresh_x in &nifs_outputs.fresh_x {
        for &padding in &fresh_x[cfg.public_input_layout.carrier_padding_offset()..expected_public_input_len] {
            builder.enforce_eq(&Lc::from_var(padding), &Lc::zero());
        }
    }
    builder.record_row_family("fprime.recursive.prior_link", prior_link_start);
    let prior_link = FPrimePriorLinkWires {
        row_start: prior_link_start,
        row_end: builder.rows(),
        first_allocated_column: prior_link_first_column,
        digest: prior_x_out,
        encoded_bits: prior_bits.to_vec(),
        fresh_public_inputs: nifs_outputs.fresh_x.clone(),
    };

    // Nebula Construction 2 follows the same one-step delay as HyperNova:
    // this F' invocation consumes the previous fresh claim's public memory
    // data and split witness commitment while producing the next claim.
    let nebula_start = builder.rows();
    builder.begin_encoding_stage(stage::RECURSIVE_NEBULA);
    let new_nebula_lane = match (cfg.nebula, sw.nebula.as_ref()) {
        (None, None) => None,
        (Some(nebula_cfg), Some(lane)) => {
            let delayed =
                decode_delayed_nebula_public_suffix_circuit(builder, &fresh_public_suffixes[0], nebula_cfg.stacks)
                    .map_err(|e| Error::Inner(format!("delayed Nebula suffix: {e}")))?;
            let adv = nifs_outputs.fresh_adv[0]
                .as_ref()
                .ok_or_else(|| Error::Inner("delayed Nebula fresh claim is missing adv".into()))?;
            let plan_digest = alloc_4_const(builder, nebula_cfg.plan_digest);
            let context = NebulaOpenContextWires {
                vk_fs: sw.vk_fs,
                z_i: sw.z_i_in,
                acc_digest: sw.acc_digest_in,
                plan_digest,
            };
            let transition = enforce_delayed_nebula_claim_circuit(
                builder,
                lane,
                &delayed,
                adv,
                &context,
                nebula_cfg.steps_per_segment,
                nebula_cfg.seg_max,
            )
            .map_err(|e| Error::Inner(format!("delayed Nebula transition: {e}")))?;
            Some(transition.lane)
        }
        _ => unreachable!("presence checked before allocation"),
    };
    builder.record_row_family("fprime.recursive.nebula", nebula_start);

    // ── Bind acc_digest_in to digest(running) ──────────────────────────
    //
    // Without this, the prior accumulator digest could be disconnected
    // from the actual `running` being folded — a malicious prover could
    // claim `acc_digest_in` matches the digest of one accumulator while
    // NIFS.V folds a different `running`.
    //
    // The digest is computed once inside the SplitNc Π_CCS verifier as
    // the ME-input accumulator handle (`nifs_outputs.running_acc_digest`)
    // and reused here. Both consumers — the transcript absorb and this
    // binding — see the *same* digest wires, so a tampered `running` is
    // caught either via the absorbed handle (Fiat-Shamir) or via this
    // wire-level equality.
    let accumulator_start = builder.rows();
    builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR);
    builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR_INPUT);
    let running_acc_digest = nifs_outputs.running_acc_digest;
    for k in 0..DIGEST_LEN {
        builder.enforce_eq(&Lc::from_var(running_acc_digest[k]), &Lc::from_var(sw.acc_digest_in[k]));
    }

    // ── State advance: bind outgoing accumulator handle ────────────────
    //
    // HyperNova Construction 2 outputs `hash(..., U_{i+1}, ...)`. In this
    // codebase `U_{i+1}` is the exact ordered NIFS.V child vector; its checked
    // parent is a recomposition cache. Compute the child handle here before
    // absorbing it into `state_x_out`.
    builder.begin_encoding_stage(stage::RECURSIVE_ACCUMULATOR_OUTPUT);
    let claimed_acc_digest = alloc_4(builder, inputs.acc_digest_out);
    let new_acc_digest = accumulator::enforce_nifs_output_acc_digest(
        builder,
        &nifs_outputs.children,
        nifs_outputs.outgoing_pending_projection.as_ref(),
    )?;
    enforce_digest_eq(builder, &claimed_acc_digest, &new_acc_digest);
    let new_semantic_state_digest = alloc_4(builder, inputs.semantic_state_digest_out);
    builder.record_row_family("fprime.recursive.accumulator", accumulator_start);

    builder.begin_encoding_stage(stage::RECURSIVE_COUNTERS);
    // Counter advance.
    //
    // `rows_in_chunk` is the size of the *current* chunk being deposited
    // as the new latest (mirrors native `advance_state(..., fresh_count,
    // ...)` where `fresh_count == next_latest.len()`). Note: this is
    // **not** `inputs.nifs_msg.fresh.len()` — that's the previous batch
    // being folded by NIFS.V at this step. The two only coincide when
    // every step's batch has the same size.
    let counter_start = builder.rows();
    let rows_in_chunk = inputs.rows_in_chunk;
    let new_chunk_count_val = inputs
        .state
        .chunk_count_in
        .checked_add(1)
        .ok_or_else(|| Error::Inner("strict F' recursive: chunk_count overflow".into()))?;
    let new_step_count_val = inputs
        .state
        .step_count_in
        .checked_add(rows_in_chunk)
        .ok_or_else(|| Error::Inner("strict F' recursive: step_count overflow".into()))?;
    let counter_outputs = enforce_f_prime_recursive_counter_transition(
        builder,
        sw.chunk_count_in,
        sw.step_count_in,
        &counter_inputs,
        rows_in_chunk,
        new_chunk_count_val,
        new_step_count_val,
    );
    let new_chunk_count = counter_outputs.chunk_count_out;
    let new_step_count = counter_outputs.step_count_out;
    builder.record_row_family("fprime.recursive.counter", counter_start);

    let output_start = builder.rows();
    builder.begin_encoding_stage(stage::RECURSIVE_OUTPUT);
    let (x_out, new_z_i, new_public_trace) = build_x_out(
        builder,
        cfg.state_x_out_digest_mode,
        &sw,
        chunk_digest,
        new_semantic_state_digest,
        new_acc_digest,
        new_chunk_count,
        new_step_count,
        new_nebula_lane.as_ref(),
    );

    let expected_bits = source_wires.range(inputs.public_x_out_bits);
    let x_out_bits = enforce_x_out_public_bit_wires(builder, expected_bits, &x_out)?;

    let state_in = state_in_to_wires(&sw);
    let state_out = state_out_wires(
        &sw,
        new_chunk_count,
        new_step_count,
        new_z_i,
        new_public_trace,
        new_semantic_state_digest,
        new_acc_digest,
        new_nebula_lane,
    );
    builder.record_row_family("fprime.recursive.output", output_start);
    builder.record_row_family("fprime.recursive.total", recursive_start);
    Ok(FPrimeStepOutput {
        x_out,
        x_out_bits,
        prior_link: Some(prior_link),
        state_in,
        state_out,
        nifs_running: Some(nifs_outputs.running),
        nifs_running_parent_authority: nifs_outputs.running_parent_authority,
        nifs_parent: Some(nifs_outputs.parent),
        nifs_children: Some(nifs_outputs.children),
        fresh_public_suffixes,
        fresh_adv: nifs_outputs.fresh_adv,
    })
}

// ── helpers ──────────────────────────────────────────────────────────────

fn alloc_4(builder: &mut R1csBuilder, vals: [F; DIGEST_LEN]) -> [Var; DIGEST_LEN] {
    let mut out = [Var::ONE; DIGEST_LEN];
    for (slot, v) in out.iter_mut().zip(vals.iter()) {
        *slot = builder.alloc(*v);
    }
    out
}

fn alloc_4_const(builder: &mut R1csBuilder, vals: [F; DIGEST_LEN]) -> [Var; DIGEST_LEN] {
    let mut out = [Var::ONE; DIGEST_LEN];
    for (slot, v) in out.iter_mut().zip(vals.iter()) {
        let var = builder.alloc(*v);
        builder.enforce_eq(&Lc::from_var(var), &Lc::from_const(*v));
        *slot = var;
    }
    out
}

//! F' R1CS step — the in-circuit augmented function from
//! Hypernova §6.3 / SuperNeo Construction 2.
//!
//! Two entry points, mirroring the paper's case split:
//!   - [`enforce_f_prime_base_step_circuit`] (i = 0). No NIFS.V; enforces
//!     `z_i = z_0`, `chunk_count_in = 0`, and `acc_digest_in = empty_acc`.
//!     `acc_digest_out` is the same empty-acc constant.
//!   - [`enforce_f_prime_recursive_step_circuit`] (i ≥ 1). Runs NIFS.V to
//!     fold `u_i` into `U_i`, enforces the HyperNova recursive link
//!     `u_i.public == bits(prior_x_out)`, and binds `acc_digest_in` to
//!     the digest of the actual `running` accumulator.
//!
//! Both functions return the `x_out` wires; the caller pins them against
//! the public-input slot.
//!
//! ## Math (Construction 2)
//!
//! ```text
//!   z_{i+1}         = boundary_update(z_i, chunk_digest)
//!   public_trace'   = public_trace_update(public_trace, chunk_digest)
//!   chunk_count'    = chunk_count + 1
//!   step_count'     = step_count + K
//!   pc'             = pc                    (TRIVIAL_PC in this build)
//!
//!   acc_digest' = base:      accumulator_digest_from_claims(b, &[])
//!                recursive:  accumulator_digest(NIFS.V.parent.c_data)
//!
//!   x_out = state_x_out_digest(
//!       vk_fs, structure, chunk_count', step_count',
//!       z_0, z_{i+1}, pc', semantic_state_digest', acc_digest', public_trace')
//! ```
//!
//! ## Bindings enforced (recursive case)
//!
//! 1. **HyperNova recursive public link**: every fresh `u_i.x` in the
//!    batch is `[1 || enc_inst(prior_x_out)]` — same prior `x_out` for
//!    the whole chunk, mirroring the SuperNeo "one chunk rooted at a
//!    single prior Construction-2 state" semantics. The raw
//!    `prior_x_out` digest is an ordinary computed field value; it is
//!    bit-encoded here only because it becomes the public input of a
//!    *fresh SuperNeo CCS instance*, and that public input is part of
//!    the low-norm assignment `z = [x, w]` (`‖z‖_∞ < b = 2`). Poseidon2
//!    itself does not need low norm. See `encoding.md` for the
//!    distinction between this public-instance encoding `enc_inst(h)`
//!    and the unresolved private-witness encoding `enc(F')`.
//! 2. **Running accumulator binding**: `acc_digest_in == digest(running)`,
//!    where `digest(running)` is computed in-circuit by
//!    [`enforce_accumulator_digest_from_children_circuit`]. Without this,
//!    `u_i.public` could bind to one accumulator while NIFS.V folds a
//!    different one.
//! 3. **`pc == TRIVIAL_PC`** (ℓ = 1 in this build).
//! 4. **Strict-mode shape**: non-empty fresh batch, every fresh `u_i`
//!    has `m_in == F_PRIME_PUBLIC_INPUT_LEN` and `x.len() ==
//!    F_PRIME_PUBLIC_INPUT_LEN`. Enforced against the message shape
//!    (the SplitNc config carries `params` + `structure` references,
//!    not redundant integer copies).

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::poseidon2::DIGEST_LEN;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::paper::construction2::TRIVIAL_PC;
use crate::paper::digest::{accumulator_digest_from_claims, digest32_as_fields};
use crate::paper::f_prime::digest_circuit::{
    enforce_boundary_update_digest_circuit, enforce_public_trace_update_digest_circuit,
    enforce_state_x_out_digest_circuit, StateXOutDigestInputs,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use crate::paper::f_prime::source_image_circuit::{enforce_goldilocks_word_canonical, SourceImageWires};
use crate::paper::nifs::circuit::{enforce_nifs_v_circuit_with_transcript, NifsVCircuitConfig, NifsVCircuitMessages};
use crate::paper::params::Params;
use crate::paper::reductions::accumulator_digest_circuit::enforce_accumulator_digest_from_parent_circuit;

/// Canonical bits per `x_out` digest lane. Goldilocks canonical form fits
/// in 64 bits.
pub const X_OUT_BITS_PER_LANE: usize = 64;

/// Number of `enc_inst(x_out)` bits — the bit-decomposed digest body.
pub const F_PRIME_ENC_INST_BITS: usize = DIGEST_LEN * X_OUT_BITS_PER_LANE;

/// Index of the constant-one slot in the F' CCS public input.
pub const F_PRIME_PUBLIC_ONE_OFFSET: usize = 0;

/// First index of the `enc_inst(x_out)` body inside the F' public input.
pub const F_PRIME_ENC_INST_OFFSET: usize = 1;

/// Full F' CCS public-input length: `[1, enc_inst(x_out)…]`.
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
/// is not committable as a real `u_i` (matches `neo-fold-prototype` which
/// used `public_input_len = 257`).
///
/// HyperNova §6.3 has `u_{i+1}.public == enc_inst(F'.x_out)` where
/// `enc_inst` is the protocol's mapping from the raw augmented-step
/// output into the CCS instance public-input shape. For SuperNeo under
/// `b = 2`, the body is "canonical 64-bit decomposition of each of the
/// four Goldilocks digest lanes". See `encoding.md`.
pub const F_PRIME_PUBLIC_INPUT_LEN: usize = 1 + F_PRIME_ENC_INST_BITS;

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
pub fn encode_f_prime_public_input(x_out: [F; DIGEST_LEN]) -> Vec<F> {
    let mut out = Vec::with_capacity(F_PRIME_PUBLIC_INPUT_LEN);
    out.push(F::ONE);
    out.extend(encode_x_out_public_bits(x_out));
    out
}

/// In-circuit inverse of [`encode_x_out_public_bits`]: assert that the
/// length-`F_PRIME_ENC_INST_BITS` `public_bits` wires are the canonical
/// little-endian 64-bit decomposition of `digest[lane]` for each lane.
/// Each lane is canonicity-checked inside [`decompose_var_to_u64_bits`].
fn enforce_public_bits_encode_digest(
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
}

/// Common state-in fields shared between base and recursive F' steps.
#[derive(Clone)]
pub struct FPrimeStateIn {
    pub vk_fs_digest: [F; DIGEST_LEN],
    pub structure_digest: [F; DIGEST_LEN],
    pub chunk_count_in: u64,
    pub step_count_in: u64,
    pub z_0: [F; DIGEST_LEN],
    pub z_i_in: [F; DIGEST_LEN],
    pub pc: u64,
    pub semantic_state_digest_in: [F; DIGEST_LEN],
    pub acc_digest_in: [F; DIGEST_LEN],
    pub public_trace_in: [F; DIGEST_LEN],
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
    /// is *not* yet the full private `enc(F')` witness — internal F'
    /// computation slots remain ordinary field values. See `encoding.md`
    /// for the planned scope of `enc(F')`.
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
///   wire-for-wire (not just by accumulator-digest equality).
pub struct FPrimeStepOutput {
    pub x_out: [Var; DIGEST_LEN],
    pub x_out_bits: Vec<Var>,
    pub state_in: FPrimeStateWires,
    pub state_out: FPrimeStateWires,
    pub nifs_running: Option<Vec<crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires>>,
    pub nifs_running_parent_authority:
        Option<crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires>,
    pub nifs_parent: Option<crate::paper::reductions::pi_dec_circuit::CeClaimWires>,
    pub nifs_children: Option<Vec<crate::paper::reductions::pi_dec_circuit::CeClaimWires>>,
}

/// Full state wire bundle — used both for pre-step (`state_in`) and
/// post-step (`state_out`) views. Fields that don't change across a step
/// (vk_fs_digest, structure_digest, z_0, pc) share wires between `state_in`
/// and `state_out`.
#[derive(Clone, Copy)]
pub struct FPrimeStateWires {
    pub vk_fs_digest: [Var; DIGEST_LEN],
    pub structure_digest: [Var; DIGEST_LEN],
    pub chunk_count: Var,
    pub step_count: Var,
    pub z_0: [Var; DIGEST_LEN],
    pub z_i: [Var; DIGEST_LEN],
    pub pc: Var,
    pub semantic_state_digest: [Var; DIGEST_LEN],
    pub acc_digest: [Var; DIGEST_LEN],
    pub public_trace: [Var; DIGEST_LEN],
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

impl From<crate::paper::reductions::accumulator_digest_circuit::AccumulatorDigestError> for Error {
    fn from(e: crate::paper::reductions::accumulator_digest_circuit::AccumulatorDigestError) -> Self {
        Self::Inner(format!("accumulator digest: {e}"))
    }
}

/// Allocate state-in wires + the `pc == TRIVIAL_PC` guard. Shared between
/// base and recursive F' entry points.
struct StateInWires {
    vk_fs: [Var; DIGEST_LEN],
    structure: [Var; DIGEST_LEN],
    chunk_count_in: Var,
    step_count_in: Var,
    z_0: [Var; DIGEST_LEN],
    z_i_in: [Var; DIGEST_LEN],
    pc: Var,
    semantic_state_digest_in: [Var; DIGEST_LEN],
    acc_digest_in: [Var; DIGEST_LEN],
    public_trace_in: [Var; DIGEST_LEN],
}

fn alloc_state_in(builder: &mut R1csBuilder, state: &FPrimeStateIn) -> StateInWires {
    let pc = builder.alloc(F::from_u64(state.pc));
    builder.enforce_eq(&Lc::from_var(pc), &Lc::from_const(F::from_u64(TRIVIAL_PC)));
    StateInWires {
        vk_fs: alloc_4(builder, state.vk_fs_digest),
        structure: alloc_4(builder, state.structure_digest),
        chunk_count_in: builder.alloc(F::from_u64(state.chunk_count_in)),
        step_count_in: builder.alloc(F::from_u64(state.step_count_in)),
        z_0: alloc_4(builder, state.z_0),
        z_i_in: alloc_4(builder, state.z_i_in),
        pc,
        semantic_state_digest_in: alloc_4(builder, state.semantic_state_digest_in),
        acc_digest_in: alloc_4(builder, state.acc_digest_in),
        public_trace_in: alloc_4(builder, state.public_trace_in),
    }
}

/// Repackage [`StateInWires`] (private, pre-step view) as the public
/// [`FPrimeStateWires`] returned to callers.
fn state_in_to_wires(sw: &StateInWires) -> FPrimeStateWires {
    FPrimeStateWires {
        vk_fs_digest: sw.vk_fs,
        structure_digest: sw.structure,
        chunk_count: sw.chunk_count_in,
        step_count: sw.step_count_in,
        z_0: sw.z_0,
        z_i: sw.z_i_in,
        pc: sw.pc,
        semantic_state_digest: sw.semantic_state_digest_in,
        acc_digest: sw.acc_digest_in,
        public_trace: sw.public_trace_in,
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
) -> FPrimeStateWires {
    FPrimeStateWires {
        vk_fs_digest: sw.vk_fs,
        structure_digest: sw.structure,
        chunk_count,
        step_count,
        z_0: sw.z_0,
        z_i,
        pc: sw.pc,
        semantic_state_digest,
        acc_digest,
        public_trace,
    }
}

/// Build the `x_out` wires for the post-step state. Common tail between
/// base and recursive entry points. Returns both the `x_out` digest and
/// the chained `new_z_i` / `new_public_trace` wires so callers can
/// thread the post-step state into the next step (decider chain) or
/// into the terminal-fold gate.
fn build_x_out(
    builder: &mut R1csBuilder,
    sw: &StateInWires,
    chunk_digest: [Var; DIGEST_LEN],
    new_semantic_state_digest: [Var; DIGEST_LEN],
    new_acc_digest: [Var; DIGEST_LEN],
    new_chunk_count: Var,
    new_step_count: Var,
) -> ([Var; DIGEST_LEN], [Var; DIGEST_LEN], [Var; DIGEST_LEN]) {
    let new_z_i = enforce_boundary_update_digest_circuit(builder, sw.z_i_in, chunk_digest);
    let new_public_trace = enforce_public_trace_update_digest_circuit(builder, sw.public_trace_in, chunk_digest);
    let x_out_inputs = StateXOutDigestInputs {
        vk_fs_digest: sw.vk_fs,
        structure_digest: sw.structure,
        chunk_count: new_chunk_count,
        step_count: new_step_count,
        initial_boundary: sw.z_0,
        current_boundary: new_z_i,
        pc: sw.pc,
        semantic_acc: new_semantic_state_digest,
        construction2_acc: new_acc_digest,
        public_trace: new_public_trace,
    };
    let x_out = enforce_state_x_out_digest_circuit(builder, &x_out_inputs);
    (x_out, new_z_i, new_public_trace)
}

/// F' **base** step (i = 0). No NIFS.V. Enforces `chunk_count_in == 0`,
/// `z_i_in == z_0`, and `acc_digest_in == accumulator_digest_from_claims(b, &[])`.
/// Sets `acc_digest_out = empty_acc_digest`. Returns the `x_out` wires.
pub fn enforce_f_prime_base_step_circuit(
    builder: &mut R1csBuilder,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeBaseInputs<'_>,
) -> Result<FPrimeStepOutput, Error> {
    let sw = alloc_state_in(builder, &inputs.state);
    let chunk_digest = alloc_4(builder, inputs.chunk_digest);

    // Allocate source-image bits once for the whole F' step. Each
    // coordinate becomes a bit-constrained witness wire; the public
    // x_out bits are sliced from this image below.
    let source_wires = SourceImageWires::alloc(builder, inputs.source_image);

    // Bind u64 counters to source-image words (Step 4). The `Var`s
    // remain in use downstream; this just pins them to the canonical
    // low-norm bit decomposition stored in the source image.
    enforce_var_matches_source_word64(builder, &source_wires, inputs.chunk_count_in_word, sw.chunk_count_in);
    enforce_var_matches_source_word64(builder, &source_wires, inputs.step_count_in_word, sw.step_count_in);
    enforce_var_matches_source_word64(builder, &source_wires, inputs.pc_word, sw.pc);

    // Base pre-state: chunk_count_in == 0, step_count_in == 0, z_i_in == z_0.
    builder.enforce_eq(&Lc::from_var(sw.chunk_count_in), &Lc::zero());
    builder.enforce_eq(&Lc::from_var(sw.step_count_in), &Lc::zero());
    for k in 0..DIGEST_LEN {
        builder.enforce_eq(&Lc::from_var(sw.z_i_in[k]), &Lc::from_var(sw.z_0[k]));
    }

    // acc_digest_in must equal the empty-accumulator digest constant.
    let empty_acc = digest32_as_fields(accumulator_digest_from_claims(cfg.b, &[]));
    for k in 0..DIGEST_LEN {
        builder.enforce_eq(&Lc::from_var(sw.acc_digest_in[k]), &Lc::from_const(empty_acc[k]));
    }

    // Output acc_digest is also the empty-acc constant (running stays ⊥).
    let new_acc_digest = alloc_4_const(builder, empty_acc);
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

    let (x_out, new_z_i, new_public_trace) = build_x_out(
        builder,
        &sw,
        chunk_digest,
        new_semantic_state_digest,
        new_acc_digest,
        new_chunk_count,
        new_step_count,
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
    );
    Ok(FPrimeStepOutput {
        x_out,
        x_out_bits,
        state_in,
        state_out,
        // Base step does not run NIFS.V; no children/running wires.
        nifs_running: None,
        nifs_running_parent_authority: None,
        nifs_parent: None,
        nifs_children: None,
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
///   - every fresh `u_i,j` has `m_in == F_PRIME_PUBLIC_INPUT_LEN` and
///     `x.len() == F_PRIME_PUBLIC_INPUT_LEN`. Public input encodes the
///     prior `x_out` digest as `DIGEST_LEN * 64` low-norm bits — raw
///     Goldilocks digest lanes can't be CCS public inputs under `b = 2`.
///   - all K fresh public inputs are bound to the **same** prior
///     `x_out` source-image bits; the whole batch is treated as one
///     SuperNeo chunk rooted at one prior Construction-2 state.
pub fn enforce_f_prime_recursive_step_circuit(
    builder: &mut R1csBuilder,
    pp: &Params,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeRecursiveInputs<'_>,
) -> Result<FPrimeStepOutput, Error> {
    // ── Strict-mode shape ──────────────────────────────────────────────
    if inputs.nifs_msg.fresh.is_empty() {
        return Err(Error::Inner(
            "strict F' recursive: fresh batch must be non-empty".into(),
        ));
    }
    if inputs.rows_in_chunk == 0 {
        return Err(Error::Inner(
            "strict F' recursive: rows_in_chunk must be \u{2265} 1 (new chunk must be non-empty)".into(),
        ));
    }
    for (idx, fresh) in inputs.nifs_msg.fresh.iter().enumerate() {
        if fresh.m_in != F_PRIME_PUBLIC_INPUT_LEN || fresh.x.len() != F_PRIME_PUBLIC_INPUT_LEN {
            return Err(Error::Inner(format!(
                "strict F' recursive: fresh[{idx}] public input must be [1 | {F_PRIME_ENC_INST_BITS} enc_inst bits], \
                 got m_in={}, x.len={}",
                fresh.m_in,
                fresh.x.len()
            )));
        }
    }

    // Recursive means i ≥ 1; chunk_count_in must be nonzero.
    if inputs.state.chunk_count_in == 0 {
        return Err(Error::Inner(
            "strict F' recursive: chunk_count_in must be nonzero".into(),
        ));
    }

    let sw = alloc_state_in(builder, &inputs.state);
    let chunk_digest = alloc_4(builder, inputs.chunk_digest);

    // Allocate source-image bits once for the whole F' step. Each
    // coordinate becomes a bit-constrained witness wire; the public
    // x_out bits are sliced from this image at the end of the step.
    let source_wires = SourceImageWires::alloc(builder, inputs.source_image);

    // Bind u64 counters to source-image words (Step 4).
    enforce_var_matches_source_word64(builder, &source_wires, inputs.chunk_count_in_word, sw.chunk_count_in);
    enforce_var_matches_source_word64(builder, &source_wires, inputs.step_count_in_word, sw.step_count_in);
    enforce_var_matches_source_word64(builder, &source_wires, inputs.pc_word, sw.pc);

    // In-circuit nonzero gadget: prove chunk_count_in · inv == 1 with
    // `inv = chunk_count_in^{-1}`. If a malicious prover sets the wire to
    // zero, no inverse exists and the constraint fails. The native-side
    // check above ensures honest witness construction.
    {
        use p3_field::Field;
        let cc_in_f = F::from_u64(inputs.state.chunk_count_in);
        let inv = builder.alloc(cc_in_f.inverse());
        builder.enforce(
            &Lc::from_var(sw.chunk_count_in),
            &Lc::from_var(inv),
            &Lc::from_const(F::ONE),
        );
    }

    // ── NIFS.V composition ──────────────────────────────────────────────
    let mut transcript = TranscriptGadget::new(builder, cfg.transcript_label);
    transcript.append_fields(builder, b"f_prime/vk_fs", &sw.vk_fs);
    transcript.append_fields(builder, b"f_prime/structure", &sw.structure);
    transcript.append_fields(builder, b"f_prime/z_0", &sw.z_0);
    transcript.append_fields(builder, b"f_prime/z_i_in", &sw.z_i_in);
    transcript.append_fields(builder, b"f_prime/public_trace_in", &sw.public_trace_in);
    transcript.append_fields(builder, b"f_prime/chunk_digest", &chunk_digest);

    let nifs_outputs =
        enforce_nifs_v_circuit_with_transcript(builder, pp, &cfg.nifs, &mut transcript, &inputs.nifs_msg)?;

    // ── HyperNova recursive link: u_i.public == bits(prior_x_out) ───────
    //
    // The fresh CCS instance's public input MUST encode the previous F'
    // step's `x_out` digest. Raw digest lanes are not low-norm under b=2,
    // so the public input carries canonical Goldilocks bits instead.
    let prior_x_out_inputs = StateXOutDigestInputs {
        vk_fs_digest: sw.vk_fs,
        structure_digest: sw.structure,
        chunk_count: sw.chunk_count_in,
        step_count: sw.step_count_in,
        initial_boundary: sw.z_0,
        current_boundary: sw.z_i_in,
        pc: sw.pc,
        semantic_acc: sw.semantic_state_digest_in,
        construction2_acc: sw.acc_digest_in,
        public_trace: sw.public_trace_in,
    };
    let prior_x_out = enforce_state_x_out_digest_circuit(builder, &prior_x_out_inputs);

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
    enforce_public_bits_encode_digest(builder, prior_bits, &prior_x_out)?;
    for (idx, fresh_x) in nifs_outputs.fresh_x.iter().enumerate() {
        if fresh_x.len() != F_PRIME_PUBLIC_INPUT_LEN {
            return Err(Error::Inner(format!(
                "strict F' fresh[{idx}] public input length {} != F_PRIME_PUBLIC_INPUT_LEN ({F_PRIME_PUBLIC_INPUT_LEN})",
                fresh_x.len()
            )));
        }
        // CCS constant-one slot.
        builder.enforce_eq(
            &Lc::from_var(fresh_x[F_PRIME_PUBLIC_ONE_OFFSET]),
            &Lc::from_const(F::ONE),
        );
        for (fresh_bit, source_bit) in fresh_x[F_PRIME_ENC_INST_OFFSET..]
            .iter()
            .zip(prior_bits.iter())
        {
            builder.enforce_eq(&Lc::from_var(*fresh_bit), &Lc::from_var(*source_bit));
        }
    }

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
    let running_acc_digest = nifs_outputs.running_acc_digest;
    for k in 0..DIGEST_LEN {
        builder.enforce_eq(&Lc::from_var(running_acc_digest[k]), &Lc::from_var(sw.acc_digest_in[k]));
    }

    // ── State advance: acc_digest_out from NIFS.V's parent ─────────────
    //
    // `k_carry` comes from the children produced by Π_DEC (= the new
    // running accumulator's claim count). The accumulator-digest gadget
    // needs that count to fan out the parent commitment correctly.
    let k_carry = inputs.nifs_msg.children.len();
    let new_acc_digest = enforce_accumulator_digest_from_parent_circuit(builder, k_carry, &nifs_outputs.parent_c_data);
    let new_semantic_state_digest = alloc_4(builder, inputs.semantic_state_digest_out);

    // Counter advance.
    //
    // `rows_in_chunk` is the size of the *current* chunk being deposited
    // as the new latest (mirrors native `advance_state(..., fresh_count,
    // ...)` where `fresh_count == next_latest.len()`). Note: this is
    // **not** `inputs.nifs_msg.fresh.len()` — that's the previous batch
    // being folded by NIFS.V at this step. The two only coincide when
    // every step's batch has the same size.
    let rows_in_chunk = inputs.rows_in_chunk;
    let new_chunk_count_val = inputs.state.chunk_count_in + 1;
    let new_step_count_val = inputs.state.step_count_in + rows_in_chunk;
    let new_chunk_count = builder.alloc(F::from_u64(new_chunk_count_val));
    let new_step_count = builder.alloc(F::from_u64(new_step_count_val));
    let mut chunk_sum = Lc::from_var(sw.chunk_count_in);
    chunk_sum.add_constant(F::ONE);
    builder.enforce_eq(&Lc::from_var(new_chunk_count), &chunk_sum);
    let mut step_sum = Lc::from_var(sw.step_count_in);
    step_sum.add_constant(F::from_u64(rows_in_chunk));
    builder.enforce_eq(&Lc::from_var(new_step_count), &step_sum);

    let (x_out, new_z_i, new_public_trace) = build_x_out(
        builder,
        &sw,
        chunk_digest,
        new_semantic_state_digest,
        new_acc_digest,
        new_chunk_count,
        new_step_count,
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
    );
    Ok(FPrimeStepOutput {
        x_out,
        x_out_bits,
        state_in,
        state_out,
        nifs_running: Some(nifs_outputs.running),
        nifs_running_parent_authority: nifs_outputs.running_parent_authority,
        nifs_parent: Some(nifs_outputs.parent),
        nifs_children: Some(nifs_outputs.children),
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

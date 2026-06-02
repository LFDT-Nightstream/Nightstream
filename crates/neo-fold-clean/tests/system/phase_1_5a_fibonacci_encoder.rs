//! Phase 1.5a — Milestone 4 Fibonacci F' encoder.
//!
//! Drives the new `encode_f_prime_step` end-to-end on the same
//! kind of recursive fixture Phase 1.4f assembles by hand. The encoder
//! consumes "real prover output" (plan + state digests + chunk digest +
//! NIFS payload views + Poseidon traces + boundary public-x_out bits) and
//! produces an `enc(F'_i)` instance: image + structure + satisfying
//! witness.
//!
//! Tests:
//! 1. Honest: an honest recursive-step input round-trips through the
//!    encoder and the returned witness satisfies the structure.
//! 2. The boundary public-x_out bits the encoder emits match the canonical
//!    little-endian decomposition the F' R1CS emitter uses
//!    (`encode_x_out_public_bits` on the state_x_out digest).
//! 3. Coherently tampering the encoded witness's chunk-digest lane
//!    trips the direct `chunk_digest -> new_z_i` mirror. Bit-validity
//!    still holds.
//! 4. Coherently tampering a boundary public-x_out lane in the encoded witness
//!    trips the state_x_out → boundary digest binding.
//! 5. Strict-low-norm invariant: the encoded witness is exactly
//!    `image.values`, `ccs.m == layout.end`, and every committed
//!    coordinate is in `{0, 1}` (except `z[0] = 1`).
//!
//! Phase 1.5b — encoded F' → foldable `CcsInstance`:
//! 6. The encoder's strict low-norm witness commits cleanly through
//!    `EncodedFPrimeStep::to_ccs_instance`, producing a CCS
//!    instance with witness length equal to the encoded witness.
//! 7. The resulting `CcsInstance` folds through SuperNeo NIFS.P /
//!    NIFS.V end-to-end (verifier accepts; prover and verifier agree on
//!    the new running accumulator). Proves the encoded F' bit witness is
//!    foldable by the existing folding machinery, with no lifecycle
//!    changes.

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use neo_fold_clean::frontends::f_prime::encoder::encode_f_prime_step;
use neo_fold_clean::paper::f_prime::r1cs::encode_x_out_public_bits;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use support::fibonacci_f_prime::{build_honest_step_input, BOUNDARY_BITS};

// State-lane base indices inside `FPrimeLaneSlots::state_lanes`:
// state_in occupies lanes 0..28, state_out lanes 28..46, chunk_digest lanes 46..50.
const STATE_LANE_CHUNK_DIGEST_BASE: usize = 46;

/// Recompose a 64-bit lane to its canonical-u64 F value from the
/// committed bits.
fn decode_lane(z: &[F], lane_bit_start: usize) -> F {
    let mut acc = F::ZERO;
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        if z[lane_bit_start + i] == F::ONE {
            acc += F::from_u64(1u64 << i);
        }
    }
    acc
}

/// Coherently rewrite a lane's bits to encode `new_value`. Bit validity
/// remains satisfied; only constraints that USE the source value see a
/// mismatch.
fn flip_lane_bits_to(z: &mut [F], lane_bit_start: usize, new_value: F) {
    let v = new_value.as_canonical_u64();
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        z[lane_bit_start + i] = if ((v >> i) & 1) == 1 { F::ONE } else { F::ZERO };
    }
}

#[test]
fn phase_1_5a_recursive_fixture_encoder_satisfies_structure() {
    let (input, _) = build_honest_step_input();
    // The encoder itself asserts `is_satisfied` internally; if this
    // returns, it satisfied the structure. Re-check defensively.
    let encoded = encode_f_prime_step(input);
    assert!(
        encoded.structure.is_satisfied(&encoded.witness),
        "encoded honest step must satisfy its structure (first failing row: {:?})",
        encoded.structure.first_unsatisfied_row(&encoded.witness),
    );
}

#[test]
fn phase_1_5a_encoder_public_x_out_matches_f_prime_emitter() {
    let (input, state_x_out_digest) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);

    // The encoder's boundary public-x_out bits must match the canonical
    // little-endian decomposition the F' R1CS emitter uses.
    let expected = encode_x_out_public_bits(state_x_out_digest);
    let boundary_start = encoded.image.layout.boundary.offset;
    let actual = &encoded.image.values[boundary_start..boundary_start + expected.len()];
    assert_eq!(
        actual, expected,
        "encoder boundary public-x_out bits must equal encode_x_out_public_bits(state_x_out_digest)"
    );
}

#[test]
fn phase_1_5a_encoder_rejects_tampered_chunk_digest() {
    let (input, _) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);
    let mut z = encoded.witness.clone();
    assert!(encoded.structure.is_satisfied(&z), "baseline must satisfy");

    // Chunk digest lane 0 is mirrored into state_out.new_z_i. Coherent
    // tamper keeps bit/decode rows satisfied but must trip that mirror.
    let lane = encoded.structure.lane_slots.state_lanes[STATE_LANE_CHUNK_DIGEST_BASE];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !encoded.structure.is_satisfied(&z),
        "coherent chunk_digest tamper must trip the chunk_digest -> new_z_i mirror"
    );
}

#[test]
fn phase_1_5a_encoder_rejects_tampered_public_x_out() {
    let (input, _) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);
    let mut z = encoded.witness.clone();
    assert!(encoded.structure.is_satisfied(&z), "baseline must satisfy");

    // The first public-x_out lane is bound to state_x_out's digest field 0.
    // Coherent tamper must trip the state_x_out → boundary digest binding.
    let lane = encoded.structure.lane_slots.public_x_out_binding_lanes[0][0];
    let new_value = decode_lane(&z, lane.bit_start) + F::ONE;
    flip_lane_bits_to(&mut z, lane.bit_start, new_value);

    assert!(
        !encoded.structure.is_satisfied(&z),
        "coherent public-x_out lane tamper must trip the state_x_out → boundary digest binding"
    );
}

#[test]
fn phase_1_5a_encoder_witness_is_strict_low_norm_image_bits() {
    let (input, _) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);

    // The encoded witness is exactly the committed image bits.
    assert_eq!(
        encoded.witness, encoded.image.values,
        "encoded witness must equal image.values verbatim"
    );
    assert_eq!(
        encoded.structure.ccs.m, encoded.image.layout.end,
        "structure.ccs.m must equal layout.end (no decoded lane columns)"
    );
    assert_eq!(encoded.witness.len(), encoded.image.layout.end);
    assert_eq!(encoded.witness[0], F::ONE, "z[0] must be the CCS constant slot");

    // Every other coordinate must be in {0, 1} so the witness is
    // foldable under SuperNeo's b = 2 norm bound.
    for (i, &v) in encoded.witness.iter().enumerate().skip(1) {
        assert!(
            v == F::ZERO || v == F::ONE,
            "encoded witness entry {i} = {v:?} violates the strict low-norm invariant"
        );
    }
}

// ── Phase 1.5b — encoded F' → foldable CcsInstance ───────────────────────

/// F' public-input length: the CCS constant slot `z[0] = 1` plus the
/// `BOUNDARY_BITS` boundary lanes the recursive F' verifier exposes as
/// `enc_inst(x_out)`. Everything past this is private witness `w`.
const ENCODED_F_PRIME_M_IN: usize = 1 + BOUNDARY_BITS;

#[test]
fn phase_1_5b_encoded_f_prime_converts_to_ccs_instance() {
    use neo_fold_clean::config;
    use neo_fold_clean::frontends::direct_ccs::ajtai;

    let (input, _) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);

    let structure = encoded.structure.ccs.clone();
    let params = config::r1cs_params(structure.n, structure.m).expect("encoded F' params");
    let log = ajtai::setup_seeded(&params, &structure, 0xF15B_0001);

    let instance = encoded
        .to_ccs_instance(&params, &log, ENCODED_F_PRIME_M_IN)
        .expect("encoded F' image should convert to a CCS instance");

    assert_eq!(instance.claim.m_in, ENCODED_F_PRIME_M_IN);
    assert_eq!(instance.claim.x.len(), ENCODED_F_PRIME_M_IN);
    assert_eq!(instance.witness.w.len(), encoded.witness.len() - ENCODED_F_PRIME_M_IN);
    // The first field element is the CCS constant slot; the remaining
    // ENCODED_F_PRIME_M_IN - 1 entries are the boundary public-x_out bits.
    assert_eq!(instance.claim.x[0], F::ONE);
    assert_eq!(instance.claim.x, encoded.witness[..ENCODED_F_PRIME_M_IN]);
}

#[test]
fn phase_1_5b_encoded_f_prime_instance_folds_through_nifs() {
    use neo_fold_clean::config;
    use neo_fold_clean::engine::transcript::Transcript;
    use neo_fold_clean::frontends::direct_ccs::ajtai;
    use neo_fold_clean::frontends::direct_ccs::{ajtai_dec_mixer, ajtai_rlc_mixer};
    use neo_fold_clean::paper::construction2::RunningInstance;
    use neo_fold_clean::paper::nifs;
    use neo_fold_clean::paper::relations::superneo_inactive_x_zero;

    const LABEL: &[u8] = b"neo.fold.clean/test/encoded-f-prime-nifs/v1";

    let (input, _) = build_honest_step_input();
    let encoded = encode_f_prime_step(input);

    let structure = encoded.structure.ccs.clone();
    let params = config::r1cs_params(structure.n, structure.m).expect("encoded F' params");
    let log = ajtai::setup_seeded(&params, &structure, 0xF15B_0002);
    let cache = neo_reductions::optimized_engine::OptimizedStructureCache::build(&structure).expect("cache build");

    let instance = encoded
        .to_ccs_instance(&params, &log, ENCODED_F_PRIME_M_IN)
        .expect("encoded F' image should convert to a CCS instance");

    let running = RunningInstance::default();

    let mut prover_tr = Transcript::with_label(LABEL);
    let (next_running, proof) = nifs::prove(
        &mut prover_tr,
        &params,
        &structure,
        &cache,
        &log,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        vec![instance.clone()],
        &running,
    )
    .expect("NIFS.P folds encoded F' instance");

    let mut verifier_tr = Transcript::with_label(LABEL);
    let verified = nifs::verify(
        &mut verifier_tr,
        &params,
        &structure,
        &cache,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        &[instance.claim.clone()],
        &running,
        &proof,
    )
    .expect("NIFS.V verifies encoded F' fold");

    assert_eq!(verified.claims, next_running.claims);
    assert_eq!(verified.parent_authority, next_running.parent_authority);

    // Regression gate: every CE claim coming out of Π_DEC (in
    // `next_running`) and Π_CCS (in `verified`) must have all-zero
    // entries in the inactive columns of `X`. This is the soundness
    // invariant `pi_dec::validate_inactive_x_zero` checks, and
    // re-asserting it here pins the encoded-F' path to honor it for
    // non-trivial `m_in`.
    for claim in &next_running.claims {
        assert!(
            superneo_inactive_x_zero(&claim.X, claim.m_in),
            "running claim X must be zero past its active columns (m_in = {})",
            claim.m_in
        );
    }
    for claim in &verified.claims {
        assert!(
            superneo_inactive_x_zero(&claim.X, claim.m_in),
            "verified claim X must be zero past its active columns (m_in = {})",
            claim.m_in
        );
    }
}

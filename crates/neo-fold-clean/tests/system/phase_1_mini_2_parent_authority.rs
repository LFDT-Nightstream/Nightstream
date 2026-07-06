//! Phase 1.1-mini-2 — parent_authority CE digest trace.
//!
//! Reuses the production [`poseidon_trace`] module from mini-1 verbatim;
//! adds a test-local preimage builder mirroring
//! [`paper::digest::ce_claim_digest`] for the parent_authority claim,
//! then asserts the decoded digest equals the production reference and
//! every committed coord is a bit.
//!
//! Out of scope:
//! - NIFS circuit rewiring or any change to `nifs/circuit.rs`.
//! - Lifecycle migration.
//! - A generic encoder over all claims.
//! - Touching the three `ivc_invariants` tests; they remain red.
//!
//! The fixture is a real `parent_authority` produced by `nifs::prove`
//! over `tests/support/toy_preprocessing` + two `toy_instance`s. This
//! ensures the preimage covers every CeClaim field
//! (`c`, `X`, `r`, `y_ring`, `m_in`, `fold_digest`) with values an
//! honest NIFS would emit.

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::{ce_claim_digest, digest32_as_fields};
use neo_fold_clean::paper::f_prime::poseidon_trace::{
    assert_committed_coords_are_bits, decode_digest_lanes, encode_poseidon_trace, PoseidonTraceLayout,
    BITS_PER_PERMUTATION,
};
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::relations::superneo_public_x_cols;
use neo_fold_clean::CeClaim;
use neo_math::F;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

// ── Test-local preimage construction (mirrors ce_claim_digest) ───────────

/// Mirror of the private `paper::digest::pack_bytes_as_fields`. Tag
/// length-prefix followed by 7-byte-per-field little-endian chunks.
fn pack_bytes_as_fields(bytes: &[u8]) -> Vec<F> {
    const BYTES_PER_LIMB: usize = 7;
    let mut out = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_LIMB));
    out.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(F::from_u64(u64::from_le_bytes(limb)));
    }
    out
}

/// Build the preimage `paper::digest::ce_claim_digest` would hash for
/// the given parent_authority CeClaim. Kept here (and not in production)
/// per Phase 1.1-mini-2 scope: prove the trace module works for this
/// digest without exposing a new public preimage-builder API yet.
fn parent_authority_ce_digest_preimage(parent: &CeClaim) -> Vec<F> {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/ce_claim_digest/v2");

    // Commitment shape + data.
    preimage.push(F::from_u64(parent.c.d as u64));
    preimage.push(F::from_u64(parent.c.kappa as u64));
    preimage.push(F::from_u64(parent.c.data.len() as u64));
    preimage.extend_from_slice(&parent.c.data);

    // Public-input matrix X: shape + active columns only.
    let active_x_cols = superneo_public_x_cols(parent.m_in);
    preimage.push(F::from_u64(parent.X.rows() as u64));
    preimage.push(F::from_u64(parent.X.cols() as u64));
    preimage.push(F::from_u64(active_x_cols as u64));
    for r in 0..parent.X.rows() {
        for c in 0..active_x_cols {
            preimage.push(parent.X[(r, c)]);
        }
    }

    // r evaluation point (K-element vector, split into base-field limbs).
    preimage.push(F::from_u64(parent.r.len() as u64));
    for r in &parent.r {
        for limb in r.as_basis_coefficients_slice() {
            preimage.push(*limb);
        }
    }

    // y_ring evaluations: outer-length + per-row (inner-length, flattened limbs).
    preimage.push(F::from_u64(parent.y_ring.len() as u64));
    for row in &parent.y_ring {
        preimage.push(F::from_u64(row.len() as u64));
        for v in row {
            for limb in v.as_basis_coefficients_slice() {
                preimage.push(*limb);
            }
        }
    }

    preimage.push(F::from_u64(parent.m_in as u64));
    preimage.extend(digest32_as_fields(parent.fold_digest));
    preimage
}

// ── Fixture: NIFS-produced parent_authority claim ────────────────────────

fn build_parent_authority_fixture() -> CeClaim {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, 7), support::toy_instance(&prep, 11)];
    let mut prover_tr = Transcript::session();
    let (next_running, _proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &RunningInstance::default(),
    )
    .expect("NIFS.P should produce a parent_authority over toy inputs");
    next_running
        .parent_authority
        .expect("non-empty running must carry the Pi_RLC parent_authority")
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn phase_1_mini_2_parent_authority_committed_coords_are_bits() {
    let parent = build_parent_authority_fixture();
    let preimage = parent_authority_ce_digest_preimage(&parent);
    let image = encode_poseidon_trace(&preimage);
    assert_committed_coords_are_bits(&image.values);
}

#[test]
fn phase_1_mini_2_parent_authority_decoded_digest_matches_ce_claim_digest() {
    let parent = build_parent_authority_fixture();

    // Reference: production `ce_claim_digest`. By construction this is
    // the digest the Π_CCS verifier will absorb into the Fiat-Shamir
    // transcript at step 4-5 (see `pi_ccs_split_nc_circuit/verifier.rs`).
    let reference = ce_claim_digest(&parent);

    let preimage = parent_authority_ce_digest_preimage(&parent);
    let image = encode_poseidon_trace(&preimage);
    let decoded = decode_digest_lanes(&image);

    assert_eq!(
        decoded, image.digest_native,
        "decoded digest bits must match the bit-backed builder's reported digest"
    );
    assert_eq!(
        image.digest_native, reference,
        "bit-backed builder's digest must match production ce_claim_digest"
    );
    assert_eq!(
        decoded, reference,
        "decoded digest must match production ce_claim_digest (transitive parity)"
    );
}

#[test]
fn phase_1_mini_2_parent_authority_layout_sanity() {
    let parent = build_parent_authority_fixture();
    let preimage = parent_authority_ce_digest_preimage(&parent);
    let layout = PoseidonTraceLayout::from_preimage_len(preimage.len());

    eprintln!(
        "parent_authority preimage: {} F values, {} absorbs, {} trace bits (~{:.2} KiB)",
        preimage.len(),
        layout.absorbs,
        layout.trace_len,
        layout.trace_len as f64 / 8.0 / 1024.0,
    );

    // Layout invariants — same family of asserts as mini-1, applied to a
    // substantially larger preimage so we exercise the layout for a real
    // soundness-critical digest, not just the small state_x_out case.
    assert_eq!(layout.constant_slot, 0);
    assert!(layout.trace_start > layout.constant_slot);
    assert!(layout.end() > layout.trace_start);
    assert_eq!(layout.trace_len, layout.absorbs * BITS_PER_PERMUTATION);
    assert!(layout.final_state_start() >= layout.trace_start);
    assert!(layout.final_state_start() < layout.end());

    // The parent_authority preimage MUST be strictly larger than the
    // state_x_out preimage (~32 F). For any non-degenerate CCS claim
    // the commitment data (`c.d * c.kappa`) plus the K-extension r and
    // y_ring fields alone exceed that.
    assert!(
        preimage.len() > 32,
        "parent_authority preimage ({}) must exceed state_x_out's (~32 F)",
        preimage.len()
    );

    // Encoder must agree with the layout descriptor on size.
    let image = encode_poseidon_trace(&preimage);
    assert_eq!(image.values.len(), layout.end());
}

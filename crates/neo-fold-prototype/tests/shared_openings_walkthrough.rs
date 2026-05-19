#![allow(non_snake_case)]
//! Ajtai commitment walkthrough: one test per Dory-vs-SuperNeo capability row.
//!
//! Each test exercises the actual lattice-based primitives from neo-ajtai that give
//! Nightstream the same batching/opening capabilities Jolt gets from Dory's pairing-based
//! PCS — through fundamentally different cryptographic mechanisms.
//!
//! Row | Dory (pairing-based)               | Ajtai/SuperNeo (lattice-based)             | Test
//! ----|-------------------------------------|--------------------------------------------|------
//!  1  | C_joint = sum gamma^i * C_i          | rho_i * C_i  (S-module hom, Thm 2)         | row_1_s_module_commitment_homomorphism
//!  2  | IPA opening proof per claim          | Thm 5 eval hom -> no per-claim proof needed | row_2_evaluation_homomorphism_eliminates_opening_proofs
//!  3  | batch via gamma^i linear combination | Pi_RLC: N ring challenges -> 1 joint claim  | row_3_pi_rlc_batches_n_claims_to_one
//!  4  | Stage 7 all-to-one-point convergence | per-point batching (convergence deferred)   | row_4_per_point_batching_groups_by_evaluation_point
//!  5  | pairing equation + IPA               | unification sumcheck ties groups             | row_5_unification_sumcheck_ties_groups
//!  6  | small coefficients in IPA            | Pi_DEC: split_b + verify_split_open          | row_6_pi_dec_norm_management
//!  7  | structured reference string (SRS)    | public randomness only (transparent)         | row_7_transparent_setup
//!  8  | discrete-log binding                 | Module-SIS binding (post-quantum)             | row_8_binding_from_module_sis
//!  9  | O(1) group elements                  | O(kappa*d) field elts, independent of m       | row_9_commitment_size_independent_of_witness
//! 10  | N/A                                  | additive hom -> chunk-wise commit              | row_10_streaming_commitment_via_additive_homomorphism

use neo_ajtai::{
    assert_range_b, commit, s_lincomb, sample_uniform_rq, setup, split_b, verify_open, verify_split_open, Commitment,
    DecompStyle, PP,
};
use neo_fold_prototype::core::opening::time::{prove_time_opening, verify_time_opening};
use neo_fold_prototype::core::opening::{OpeningClaim, OpeningDomain, OpeningSource};
use neo_math::ring::Rq;
use neo_math::s_action::SAction;
use neo_math::{Fq, D, K};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// Parameters and helpers
// ---------------------------------------------------------------------------

const KAPPA: usize = 4;
const M: usize = 8;

fn test_pp() -> PP<Rq> {
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD);
    setup(&mut rng, D, KAPPA, M).expect("setup")
}

/// Column-major d x m witness with entries in {0, 1}.
fn small_witness(seed: u64) -> Vec<Fq> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..D * M)
        .map(|_| if rng.next_u64() & 1 == 1 { Fq::ONE } else { Fq::ZERO })
        .collect()
}

/// Apply the S-action of a ring element rho to a d x m column-major witness.
/// S_rho(Z) applies rot(rho) independently to each column of Z.
fn s_action_on_witness(rho: &Rq, z: &[Fq]) -> Vec<Fq> {
    let s = SAction::from_ring(*rho);
    let m = z.len() / D;
    let mut result = vec![Fq::ZERO; D * m];
    for col in 0..m {
        let src: [Fq; D] = z[col * D..(col + 1) * D].try_into().unwrap();
        let dst = s.apply_vec(&src);
        result[col * D..(col + 1) * D].copy_from_slice(&dst);
    }
    result
}

/// Synthetic K-valued evaluation point for time-opening tests.
fn pt(tag: u64) -> Vec<K> {
    vec![K::from(Fq::from_u64(tag))]
}

/// Synthetic opening claim.
fn oclaim(
    source: OpeningSource,
    domain: OpeningDomain,
    point: Vec<K>,
    ordinal: u64,
    col: u32,
    tag: u8,
) -> OpeningClaim {
    OpeningClaim {
        source,
        domain,
        point,
        ordinal,
        column_ids: vec![col],
        digest: [tag; 32],
    }
}

// ---------------------------------------------------------------------------
// Row 1: S-module commitment homomorphism (SuperNeo Theorem 2)
// ---------------------------------------------------------------------------

#[test]
fn row_1_s_module_commitment_homomorphism() {
    // Dory:      gamma^i * C_i  (field-scalar homomorphism over pairing groups)
    // Ajtai:     rho_i  * C_i   (ring-scalar S-module homomorphism)
    //
    // SuperNeo Theorem 2: the Ajtai commitment L is an R_F-module homomorphism.
    //   rho_1 * L(Z_1) + rho_2 * L(Z_2) = L( S_{rho_1}(Z_1) + S_{rho_2}(Z_2) )
    //
    // The ring structure (Rq, not just Fq) gives a richer batching alphabet
    // than Dory's field-scalar gamma^i. Each ring challenge is a D-dimensional
    // object, providing more entropy per challenge.
    let pp = test_pp();
    let Z1 = small_witness(1);
    let Z2 = small_witness(2);
    let C1 = commit(&pp, &Z1);
    let C2 = commit(&pp, &Z2);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let rho1 = sample_uniform_rq(&mut rng);
    let rho2 = sample_uniform_rq(&mut rng);

    // LHS: rho_1 * C_1 + rho_2 * C_2  (operate on commitments)
    let lhs = s_lincomb(&[rho1, rho2], &[C1, C2]).expect("s_lincomb");

    // RHS: L( S_{rho_1}(Z_1) + S_{rho_2}(Z_2) )  (operate on witnesses, then commit)
    let rho1_Z1 = s_action_on_witness(&rho1, &Z1);
    let rho2_Z2 = s_action_on_witness(&rho2, &Z2);
    let combined: Vec<Fq> = rho1_Z1
        .iter()
        .zip(rho2_Z2.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    let rhs = commit(&pp, &combined);

    assert_eq!(
        lhs, rhs,
        "S-module homomorphism: rho*L(Z) on commitments = L(S_rho(Z)) on witnesses"
    );
}

// ---------------------------------------------------------------------------
// Row 2: Evaluation homomorphism (SuperNeo Theorem 5)
// ---------------------------------------------------------------------------

#[test]
fn row_2_evaluation_homomorphism_eliminates_opening_proofs() {
    // Dory:  each evaluation claim needs an inner-product argument (opening proof).
    // Ajtai: Theorem 5 says ring linear combinations of committed vectors
    //        preserve evaluations:  M_hat_z*(r) = sum rho_i * M_hat_{z_i}(r).
    //
    // At the commitment level this means: the combined commitment opens
    // correctly to the combined witness. NO per-claim opening proof is needed.
    // The evaluation homomorphism is inherited for free from the S-module
    // structure (Theorem 2).
    let pp = test_pp();
    let witnesses: Vec<Vec<Fq>> = (0..4).map(|i| small_witness(10 + i)).collect();
    let commitments: Vec<Commitment> = witnesses.iter().map(|Z| commit(&pp, Z)).collect();

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let rhos: Vec<Rq> = (0..4).map(|_| sample_uniform_rq(&mut rng)).collect();

    // Combine commitments via S-module action
    let C_joint = s_lincomb(&rhos, &commitments).expect("s_lincomb");

    // Combine witnesses via S-action
    let mut Z_joint = vec![Fq::ZERO; D * M];
    for (rho, Z) in rhos.iter().zip(witnesses.iter()) {
        let acted = s_action_on_witness(rho, Z);
        for (dst, &src) in Z_joint.iter_mut().zip(acted.iter()) {
            *dst += src;
        }
    }

    // The combined commitment opens to the combined witness — no extra proof.
    // This is the evaluation homomorphism in action: every evaluation claim
    // is absorbed into the combined opening for free.
    assert!(
        verify_open(&pp, &C_joint, &Z_joint),
        "evaluation homomorphism: combined commitment must open to combined witness"
    );
}

// ---------------------------------------------------------------------------
// Row 3: Pi_RLC batches N evaluation claims to 1
// ---------------------------------------------------------------------------

#[test]
fn row_3_pi_rlc_batches_n_claims_to_one() {
    // Dory:  batch N claims via gamma^i field scalars, one combined IPA.
    // Ajtai: Pi_RLC batches N claims via rho_i ring challenges, one joint
    //        commitment, one opening verification.
    //
    // Given N independent committed vectors (N separate claims), the verifier
    // samples N ring challenges and the prover/verifier compute:
    //   C_joint = sum_i  rho_i * C_i
    //   Z_joint = sum_i  S_{rho_i}(Z_i)
    //
    // Then verify_open(pp, C_joint, Z_joint) is the ONLY check.
    // N claims -> 1 verification.
    let pp = test_pp();
    let n = 6;

    let witnesses: Vec<Vec<Fq>> = (0..n).map(|i| small_witness(100 + i as u64)).collect();
    let commitments: Vec<Commitment> = witnesses.iter().map(|Z| commit(&pp, Z)).collect();

    // Each commitment verifies individually (N separate checks before batching)
    for (C, Z) in commitments.iter().zip(witnesses.iter()) {
        assert!(verify_open(&pp, C, Z));
    }

    // Verifier samples N ring challenges
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);
    let rhos: Vec<Rq> = (0..n).map(|_| sample_uniform_rq(&mut rng)).collect();

    // Joint commitment + witness
    let C_joint = s_lincomb(&rhos, &commitments).expect("s_lincomb");
    let mut Z_joint = vec![Fq::ZERO; D * M];
    for (rho, Z) in rhos.iter().zip(witnesses.iter()) {
        let acted = s_action_on_witness(rho, Z);
        for (dst, &src) in Z_joint.iter_mut().zip(acted.iter()) {
            *dst += src;
        }
    }

    // ONE verification replaces N separate checks
    assert!(
        verify_open(&pp, &C_joint, &Z_joint),
        "Pi_RLC: one joint verification must cover all {} claims",
        n
    );
}

// ---------------------------------------------------------------------------
// Row 4: Per-point batching groups claims by evaluation point
// ---------------------------------------------------------------------------

#[test]
fn row_4_per_point_batching_groups_by_evaluation_point() {
    // Dory (Jolt Stage 7): converges ALL evaluation points into one via
    //   Hamming Weight Claim Reduction. One Dory proof at that single point.
    //
    // Ajtai/Nightstream: per-point batching. Claims at the same point batch
    //   into one group via Pi_RLC. Different points remain separate groups.
    //   This defers the "converge to one point" optimization while still
    //   capturing significant savings within each group.
    //
    // 7 claims at 2 distinct points -> 2 groups -> 2 batched verifications.
    let claims = vec![
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(100), 0, 0, 1),
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(100), 0, 1, 2),
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(100), 0, 2, 3),
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(100), 0, 3, 4),
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(200), 1, 0, 5),
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(200), 1, 1, 6),
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(200), 1, 2, 7),
    ];

    let summary = prove_time_opening(&claims, &[]).expect("prove");
    verify_time_opening(&claims, &[], &Some(summary.clone())).expect("verify");

    // 2 groups: 7 claims batched into 2 opening proofs (not 7).
    assert_eq!(summary.groups.len(), 2);

    let group_4 = summary
        .groups
        .iter()
        .find(|g| g.claim_indices.len() == 4)
        .unwrap();
    assert_eq!(group_4.point, pt(100));
    assert_eq!(group_4.coefficients.len(), 4);

    let group_3 = summary
        .groups
        .iter()
        .find(|g| g.claim_indices.len() == 3)
        .unwrap();
    assert_eq!(group_3.point, pt(200));
    assert_eq!(group_3.coefficients.len(), 3);

    // Different points -> not trivially unifiable (convergence deferred).
    assert!(!summary.can_unify);
}

// ---------------------------------------------------------------------------
// Row 5: Unification sumcheck ties groups into one proof
// ---------------------------------------------------------------------------

#[test]
fn row_5_unification_sumcheck_ties_groups() {
    // Dory:  pairing equation + IPA ties all claims.
    // Ajtai: unification sumcheck (ceil_log2(groups) rounds) binds all
    //        per-point groups into a single transcript-bound proof.
    //
    // 4 groups at 4 different points -> 2 sumcheck rounds.
    let claims = vec![
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(10), 0, 0, 1),
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(20), 1, 0, 2),
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(30), 2, 0, 3),
        oclaim(OpeningSource::MainLane, OpeningDomain::Cpu, pt(40), 3, 0, 4),
    ];

    let summary = prove_time_opening(&claims, &[]).expect("prove");
    verify_time_opening(&claims, &[], &Some(summary.clone())).expect("verify");

    assert_eq!(summary.groups.len(), 4);
    // ceil_log2(4) = 2 sumcheck rounds
    assert_eq!(summary.unification.round_polys.len(), 2);
    assert_eq!(summary.unification.r_unify.len(), 2);

    // Tamper: corrupt a sumcheck round polynomial -> verifier rejects.
    let mut tampered = summary.clone();
    tampered.unification.round_polys[0][0] += K::ONE;
    let err = verify_time_opening(&claims, &[], &Some(tampered)).expect_err("tampered must fail");
    assert!(format!("{err}").contains("unification"));

    // Tamper: corrupt the claimed sum -> also rejected.
    let mut tampered_sum = summary;
    tampered_sum.unification.claimed_sum += K::ONE;
    let err = verify_time_opening(&claims, &[], &Some(tampered_sum)).expect_err("tampered sum must fail");
    assert!(format!("{err}").contains("unification"));
}

// ---------------------------------------------------------------------------
// Row 6: Pi_DEC norm management via split_b + verify_split_open
// ---------------------------------------------------------------------------

#[test]
fn row_6_pi_dec_norm_management() {
    // Dory:  inner-product argument coefficients are naturally small.
    // Ajtai: after ring linear combination (Pi_RLC), witness norms grow.
    //        Pi_DEC uses split_b to decompose: Z -> (Z_1,..,Z_k) with ||Z_i||_inf < b.
    //        verify_split_open confirms sum b^{i-1} C_i = C, and Z recomposes.
    //
    // This is Nightstream's norm management primitive (Def. 11, Sec. 3.2-3.3).
    let pp = test_pp();

    // Witness with entries in {0..7} to exercise multi-level decomposition.
    let mut rng = ChaCha8Rng::seed_from_u64(0x42);
    let Z: Vec<Fq> = (0..D * M)
        .map(|_| Fq::from_u64(rng.next_u64() % 8))
        .collect();
    let C = commit(&pp, &Z);
    assert!(verify_open(&pp, &C, &Z));

    // Decompose: base b=2, depth k=4 (handles entries up to 2^4 = 16 > 8).
    let b = 2u32;
    let k = 4usize;
    let Z_pieces = split_b(&Z, b, D, M, k, DecompStyle::NonNegative);
    assert_eq!(Z_pieces.len(), k);

    // Each piece has bounded infinity norm.
    for piece in &Z_pieces {
        assert_range_b(piece, b).expect("each piece must have ||.||_inf < b");
    }

    // Commit each piece independently.
    let C_pieces: Vec<Commitment> = Z_pieces.iter().map(|Zi| commit(&pp, Zi)).collect();

    // Verify the split opening: original commitment = sum b^{i-1} * C_i
    // and the recomposed witness matches.
    assert!(
        verify_split_open(&pp, &C, b, &C_pieces, &Z_pieces),
        "Pi_DEC: split opening must verify against original commitment"
    );

    // Explicit recomposition check: sum b^{i-1} * Z_i = Z.
    let mut Z_recomposed = vec![Fq::ZERO; D * M];
    let mut pow = Fq::ONE;
    let b_fq = Fq::from_u64(b as u64);
    for piece in &Z_pieces {
        for (dst, &src) in Z_recomposed.iter_mut().zip(piece.iter()) {
            *dst += src * pow;
        }
        pow *= b_fq;
    }
    assert_eq!(Z, Z_recomposed, "split_b recomposition must recover original witness");
}

// ---------------------------------------------------------------------------
// Row 7: Transparent setup (no trusted ceremony)
// ---------------------------------------------------------------------------

#[test]
fn row_7_transparent_setup() {
    // Dory:  requires a structured reference string (SRS) from a trusted ceremony.
    //        The SRS contains pairing-compatible group elements.
    //
    // Ajtai: setup samples M <- R_q^{kappa x m} uniformly from public randomness.
    //        No trusted ceremony. Same seed -> same PP (deterministic, auditable).
    let mut rng1 = ChaCha8Rng::seed_from_u64(12345);
    let mut rng2 = ChaCha8Rng::seed_from_u64(12345);

    let pp1 = setup(&mut rng1, D, KAPPA, M).expect("setup 1");
    let pp2 = setup(&mut rng2, D, KAPPA, M).expect("setup 2");

    // Same seed -> identical public parameters (deterministic, auditable).
    assert_eq!(pp1.m_rows, pp2.m_rows);

    // Different seed -> different public parameters.
    let mut rng3 = ChaCha8Rng::seed_from_u64(99999);
    let pp3 = setup(&mut rng3, D, KAPPA, M).expect("setup 3");
    assert_ne!(pp1.m_rows, pp3.m_rows);

    // A commitment under pp1 verifies under pp1 but NOT under pp3.
    let Z = small_witness(7);
    let C = commit(&pp1, &Z);
    assert!(verify_open(&pp1, &C, &Z));
    assert!(!verify_open(&pp3, &C, &Z));
}

// ---------------------------------------------------------------------------
// Row 8: Binding from Module-SIS (post-quantum)
// ---------------------------------------------------------------------------

#[test]
fn row_8_binding_from_module_sis() {
    // Dory:  binding relies on the discrete-log assumption in pairing groups.
    //        Vulnerable to quantum computers via Shor's algorithm.
    //
    // Ajtai: binding relies on Module-SIS (Short Integer Solution).
    //        Finding Z' != Z with L(Z') = L(Z) requires solving M-SIS,
    //        which is believed hard even for quantum adversaries.
    let pp = test_pp();

    let Z1 = small_witness(1);
    let Z2 = small_witness(2);
    assert_ne!(Z1, Z2, "witnesses must differ");

    let C1 = commit(&pp, &Z1);
    let C2 = commit(&pp, &Z2);

    // Different witnesses -> different commitments (binding property).
    assert_ne!(
        C1, C2,
        "binding: different witnesses must produce different commitments"
    );

    // Cross-verification fails: C1 does not open to Z2.
    assert!(!verify_open(&pp, &C1, &Z2));
    assert!(!verify_open(&pp, &C2, &Z1));

    // Even a single coefficient flip breaks the opening.
    let mut Z1_tampered = Z1.clone();
    Z1_tampered[0] += Fq::ONE;
    assert!(!verify_open(&pp, &C1, &Z1_tampered));
}

// ---------------------------------------------------------------------------
// Row 9: Commitment size independent of witness dimension
// ---------------------------------------------------------------------------

#[test]
fn row_9_commitment_size_independent_of_witness() {
    // Dory:  O(1) group elements (~64-128 bytes for BN254). Extremely compact.
    //
    // Ajtai: O(kappa * d) field elements.
    //        For Nightstream: commitment = kappa x d x 8 bytes (Goldilocks).
    //        Independent of witness dimension m — the commitment is compact.
    //        Larger than Dory but no pairings and post-quantum.
    let expected_elts = KAPPA * D;
    let expected_bytes = expected_elts * 8; // Goldilocks = 8 bytes

    // Commitment size is the same regardless of witness dimension m.
    for &m in &[8, 64, 256] {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let pp = setup(&mut rng, D, KAPPA, m).expect("setup");
        let Z: Vec<Fq> = (0..D * m).map(|i| Fq::from_u64((i % 2) as u64)).collect();
        let C = commit(&pp, &Z);

        assert_eq!(C.d, D);
        assert_eq!(C.kappa, KAPPA);
        assert_eq!(
            C.data.len(),
            expected_elts,
            "commitment has kappa*d = {} field elements regardless of m={}",
            expected_elts,
            m
        );
    }

    // For Nightstream's production parameters (kappa=8):
    //   8 * 54 * 8 = 3456 bytes — compact enough for on-chain verification.
    // Test parameters (kappa=4): 4 * 54 * 8 = 1728 bytes.
    assert_eq!(expected_bytes, KAPPA * D * 8);
}

// ---------------------------------------------------------------------------
// Row 10: Streaming commitment via additive homomorphism
// ---------------------------------------------------------------------------

#[test]
fn row_10_streaming_commitment_via_additive_homomorphism() {
    // Dory:  no direct streaming support; the SRS must match the full witness size.
    //
    // Ajtai: additive homomorphism enables chunk-by-chunk commitment.
    //        Commit(Z) = Commit(Z_chunk1 || 0) + Commit(0 || Z_chunk2) + ...
    //
    // This is critical for Nightstream's chunked folding: each chunk is committed
    // independently, and the fold accumulates commitments via the S-module structure.
    let pp = test_pp();
    let Z_full = small_witness(55);
    let C_full = commit(&pp, &Z_full);

    // Stream column by column: zero-padded single-column witnesses.
    let mut C_streaming = Commitment::zeros(D, KAPPA);
    for col in 0..M {
        let mut Z_col = vec![Fq::ZERO; D * M];
        Z_col[col * D..(col + 1) * D].copy_from_slice(&Z_full[col * D..(col + 1) * D]);
        let C_col = commit(&pp, &Z_col);
        C_streaming.add_inplace(&C_col);
    }

    assert_eq!(C_full, C_streaming, "streaming: Commit(Z_full) = sum_col Commit(Z_col)");

    // Also works for two-chunk split (first half / second half of columns).
    let half = M / 2;
    let mut Z_a = vec![Fq::ZERO; D * M];
    let mut Z_b = vec![Fq::ZERO; D * M];
    for col in 0..half {
        Z_a[col * D..(col + 1) * D].copy_from_slice(&Z_full[col * D..(col + 1) * D]);
    }
    for col in half..M {
        Z_b[col * D..(col + 1) * D].copy_from_slice(&Z_full[col * D..(col + 1) * D]);
    }

    let mut C_halves = commit(&pp, &Z_a);
    C_halves.add_inplace(&commit(&pp, &Z_b));
    assert_eq!(C_full, C_halves, "two-chunk streaming matches full commit");

    // The full witness still verifies against the streamed commitment.
    assert!(verify_open(&pp, &C_streaming, &Z_full));
}

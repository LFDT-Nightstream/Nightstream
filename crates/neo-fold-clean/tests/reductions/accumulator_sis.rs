//! Native/circuit parity and low-norm-lowering pins for SIS bulk bindings.
//!
//! | Boundary | Evidence |
//! |---|---|
//! | Native ↔ R1CS | Commitment and digest outputs agree |
//! | Canonical opening | Boundary values accept; `x+p` and auxiliary mutations reject |
//! | Selective lowering | One 41-coordinate word is reused |
//! | Gadget-native lowering | Candidate-preserving decode and retained-row rejection |
//! | Seeded matrix | Seed changes alter authoritative row identity |

#[path = "../gadgets/checked_program_artifact_support.rs"]
#[allow(dead_code)]
mod checked_program_artifact_support;
#[path = "../system/full_history_manifest_identity_support.rs"]
#[allow(dead_code)]
mod full_history_manifest_identity_support;

use neo_ajtai::{commit_row_major_seeded, Commitment};
use neo_ccs::{CeClaim, Mat};
use neo_fold_clean::engine::r1cs_circuit::builder::{RowFamilyRange, Var, BALANCED_TERNARY_DIGITS};
use neo_fold_clean::engine::r1cs_circuit::{KVar, R1csBuilder};
use neo_fold_clean::frontends::f_prime::gadget_native::encode_r1cs_gadget_native;
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment, lower_field_r1cs,
};
use neo_fold_clean::paper::construction2::{PendingProjectionState, RunningInstance};
use neo_fold_clean::paper::digest::{
    accumulator_ce_claim_digest, digest32_as_fields, pending_accumulator_family_digest,
    pending_accumulator_family_preimage, PendingAccumulatorFamilyState, PENDING_ACCUMULATOR_FAMILY_CHILDREN,
    PENDING_ACCUMULATOR_FAMILY_COLUMN_POINT, PENDING_ACCUMULATOR_FAMILY_DOMAIN, PENDING_ACCUMULATOR_FAMILY_MATRICES,
    PENDING_ACCUMULATOR_FAMILY_M_IN, PENDING_ACCUMULATOR_FAMILY_ROW_POINT,
};
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::{
    accumulator_digest, commit_fields, enforce_accumulator_digest, enforce_commit_fields, SisAccumulatorConfig,
    SisAccumulatorError, ACCUMULATOR_CE_CLAIM_SIS_CONFIG, CCS_CLAIM_SIS_CONFIG, CE_CLAIM_SIS_CONFIG,
    DIGEST_COMPRESSION_MAX_MESSAGE_COLS, NEBULA_LEAF_SIS_CONFIG, PENDING_ACCUMULATOR_FAMILY_SIS_CONFIG,
    PI_CCS_OUTPUTS_SIS_CONFIG, PI_RLC_PROJECTION_SIS_CONFIG, PROTOCOL_BINDING_KAPPA, PROTOCOL_BINDING_MAX_MESSAGE_COLS,
    SIS_DIGEST_COMPRESSION_CONFIG,
};
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::{
    encode_pending_accumulator_family_preimage, PendingAccumulatorFamilyChildInputs,
    PendingAccumulatorFamilyDigestInputs, PendingAccumulatorFamilyStateInputs,
};
use neo_math::{KExtensions, D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xA7; 32],
    kappa: 1,
    domain: 0x5349_5354_4553_5431,
};

type TestCeClaim = CeClaim<Commitment, F, K>;

#[test]
fn pending_projection_state_is_typed_and_survives_public_projection() {
    let old_block = std::array::from_fn(|index| K::from(F::from_u64(index as u64 + 1)));
    let parent_y_zcol = std::array::from_fn(|lane| K::from(F::from_u64(100 + lane as u64)));
    let pending = PendingProjectionState::new(old_block, parent_y_zcol);
    let running = RunningInstance::new(Vec::new(), Vec::new(), None, Some(pending.clone()));

    assert!(!running.is_empty(), "pending state must keep the carrier active");
    assert_eq!(running.pending_projection(), Some(&pending));
    assert_eq!(running.claims_only().pending_projection(), Some(&pending));
}

fn accumulator_ce_golden_claim() -> TestCeClaim {
    let c = Commitment {
        d: D,
        kappa: PROTOCOL_BINDING_KAPPA,
        data: (0..D * PROTOCOL_BINDING_KAPPA)
            .map(|index| F::from_u64(index as u64 + 1))
            .collect(),
    };
    let m_in = 4;
    let mut x = Mat::zero(D, m_in, F::ZERO);
    for row in 0..D {
        x.set(row, 0, F::from_u64(1_000 + row as u64));
    }
    let r = (0..3)
        .map(|index| K::from_coeffs([F::from_u64(2_000 + 2 * index), F::from_u64(2_001 + 2 * index)]))
        .collect();
    let s_col = (0..2)
        .map(|index| K::from_coeffs([F::from_u64(3_000 + 2 * index), F::from_u64(3_001 + 2 * index)]))
        .collect();
    let y_ring: Vec<Vec<K>> = (0..3)
        .map(|row| {
            (0..4)
                .map(|col| {
                    let base = 4_000 + 8 * row + 2 * col;
                    K::from_coeffs([F::from_u64(base), F::from_u64(base + 1)])
                })
                .collect()
        })
        .collect();
    let ct = y_ring.iter().map(|row| row[0]).collect();
    let y_zcol = (0..4)
        .map(|index| K::from_coeffs([F::from_u64(5_000 + 2 * index), F::from_u64(5_001 + 2 * index)]))
        .collect();
    let mut fold_digest = [0u8; 32];
    for lane in 0..4 {
        fold_digest[8 * lane] = 0x30 + lane as u8;
    }

    TestCeClaim {
        c,
        X: x,
        r,
        s_col,
        y_ring,
        ct,
        aux_openings: Vec::new(),
        y_zcol,
        m_in,
        fold_digest,
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
        adv: None,
    }
}

fn reference_pack_bytes_as_fields(bytes: &[u8]) -> Vec<F> {
    let mut fields = Vec::with_capacity(1 + bytes.len().div_ceil(7));
    fields.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(7) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        fields.push(F::from_u64(u64::from_le_bytes(limb)));
    }
    fields
}

fn reference_append_k_slice(fields: &mut Vec<F>, values: &[K]) {
    fields.push(F::from_u64(values.len() as u64));
    for value in values {
        fields.extend_from_slice(&value.as_coeffs());
    }
}

fn reference_append_k_values(fields: &mut Vec<F>, values: &[K]) {
    for value in values {
        fields.extend_from_slice(&value.as_coeffs());
    }
}

fn pending_family_claim(child: usize, kappa: usize) -> TestCeClaim {
    let c = Commitment {
        d: D,
        kappa,
        data: (0..D * kappa)
            .map(|index| F::from_u64(10_000 * child as u64 + index as u64 + 1))
            .collect(),
    };
    let m_in = 270;
    let mut x = Mat::zero(D, m_in, F::ZERO);
    for column in 0..m_in.div_ceil(D) {
        for lane in 0..D {
            x[(lane, column)] = F::from_u64(100_000 * child as u64 + (column * D + lane) as u64 + 1);
        }
    }
    let r = (0..PENDING_ACCUMULATOR_FAMILY_ROW_POINT)
        .map(|index| K::from_coeffs([F::from_usize(200_000 + 2 * index), F::from_usize(200_001 + 2 * index)]))
        .collect();
    let s_col = (0..PENDING_ACCUMULATOR_FAMILY_COLUMN_POINT)
        .map(|index| {
            let index = index as u64;
            K::from_coeffs([F::from_u64(300_000 + 2 * index), F::from_u64(300_001 + 2 * index)])
        })
        .collect();
    let y_ring: Vec<Vec<K>> = (0..13)
        .map(|matrix| {
            let mut row = vec![K::ZERO; D.next_power_of_two()];
            for (lane, value) in row[..D].iter_mut().enumerate() {
                let base = 1_000_000 * child as u64 + 10_000 * matrix as u64 + 2 * lane as u64 + 1;
                *value = K::from_coeffs([F::from_u64(base), F::from_u64(base + 1)]);
            }
            row
        })
        .collect();
    let ct = y_ring.iter().map(|row| row[0]).collect();
    let y_zcol = (0..D.next_power_of_two())
        .map(|lane| {
            let base = 9_000_000 * child as u64 + 2 * lane as u64 + 1;
            K::from_coeffs([F::from_u64(base), F::from_u64(base + 1)])
        })
        .collect();
    let mut fold_digest = [0u8; 32];
    for lane in 0..4 {
        fold_digest[8 * lane..8 * lane + 8].copy_from_slice(&(400_000 + lane as u64).to_le_bytes());
    }

    TestCeClaim {
        c,
        X: x,
        r,
        s_col,
        y_ring,
        ct,
        aux_openings: Vec::new(),
        y_zcol,
        m_in,
        fold_digest,
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
        adv: None,
    }
}

fn pending_family_claims(kappa: usize) -> Vec<TestCeClaim> {
    (0..14)
        .map(|child| pending_family_claim(child, kappa))
        .collect()
}

fn alloc_k(builder: &mut R1csBuilder, value: K) -> KVar {
    let [c0, c1] = value.as_coeffs();
    KVar::alloc(builder, c0, c1)
}

struct OwnedPendingFamilyChildWires {
    c_data: Vec<Var>,
    x_active_column_major: Vec<Var>,
    y_ring_active: Vec<Vec<KVar>>,
}

fn reference_pending_family_preimage(
    claims: &[TestCeClaim],
    pending: Option<PendingAccumulatorFamilyState<'_>>,
) -> Vec<F> {
    let mut fields = reference_pack_bytes_as_fields(b"neo.fold.clean/f_prime/accumulator/pending_family_digest/v1");
    fields.extend([
        F::from_u64(14),
        F::from_u64(PENDING_ACCUMULATOR_FAMILY_ROW_POINT as u64),
        F::from_u64(PENDING_ACCUMULATOR_FAMILY_COLUMN_POINT as u64),
    ]);
    reference_append_k_values(&mut fields, &claims[0].r);
    reference_append_k_values(&mut fields, &claims[0].s_col);
    fields.push(F::from_u64(270));
    for chunk in claims[0].fold_digest.chunks_exact(8) {
        fields.push(F::from_u64(u64::from_le_bytes(
            chunk.try_into().expect("fold digest lane"),
        )));
    }
    for claim in claims {
        fields.extend_from_slice(&claim.c.data);
        for column in 0..5 {
            for lane in 0..D {
                fields.push(claim.X[(lane, column)]);
            }
        }
        for row in &claim.y_ring {
            reference_append_k_values(&mut fields, &row[..D]);
        }
    }
    match pending {
        None => {
            fields.push(F::ZERO);
            fields.extend(std::iter::repeat_n(F::ZERO, 2 * (19 + D)));
        }
        Some(pending) => {
            fields.push(F::ONE);
            reference_append_k_values(&mut fields, pending.old_block);
            reference_append_k_values(&mut fields, pending.parent_y_zcol);
        }
    }
    fields
}

/// Independent serialization pin for the validated clean accumulator CE core.
///
/// `y_zcol` is deliberately absent: its authority belongs to the delayed raw-NC
/// relation, not to this Construction-2 child digest.
fn reference_accumulator_ce_claim_v2_preimage(claim: &TestCeClaim) -> Vec<F> {
    assert!(claim.aux_openings.is_empty());
    assert!(claim.c_step_coords.is_empty());
    assert_eq!(claim.u_offset, 0);
    assert_eq!(claim.u_len, 0);
    assert!(claim.adv.is_none());

    let mut fields = reference_pack_bytes_as_fields(b"neo.fold.clean/accumulator_ce_claim_digest/v2");
    fields.extend([
        F::from_u64(claim.c.d as u64),
        F::from_u64(claim.c.kappa as u64),
        F::from_u64(claim.c.data.len() as u64),
    ]);
    fields.extend_from_slice(&claim.c.data);

    let active_x_cols = claim.m_in.div_ceil(D);
    assert!(active_x_cols <= claim.X.cols());
    fields.extend([
        F::from_u64(claim.X.rows() as u64),
        F::from_u64(claim.X.cols() as u64),
        F::from_u64(active_x_cols as u64),
    ]);
    for row in 0..claim.X.rows() {
        for col in 0..active_x_cols {
            fields.push(claim.X[(row, col)]);
        }
    }

    reference_append_k_slice(&mut fields, &claim.r);
    reference_append_k_slice(&mut fields, &claim.s_col);
    fields.push(F::from_u64(claim.y_ring.len() as u64));
    for row in &claim.y_ring {
        reference_append_k_slice(&mut fields, row);
    }
    reference_append_k_slice(&mut fields, &claim.ct);
    fields.push(F::ZERO); // aux_openings.len
    fields.push(F::from_u64(claim.m_in as u64));
    for chunk in claim.fold_digest.chunks_exact(8) {
        fields.push(F::from_u64(u64::from_le_bytes(
            chunk.try_into().expect("eight-byte fold-digest lane"),
        )));
    }
    fields.extend([F::ZERO, F::ZERO, F::ZERO]); // c_step_coords.len, u_offset, u_len
    fields
}

#[test]
fn seeded_phi81_blocks_change_authoritative_row_identity() {
    fn fixture(seed: [u8; 32]) -> R1csBuilder {
        let mut builder = R1csBuilder::new();
        let fields = builder.alloc_vec(&[F::from_u64(3), F::from_u64(5), F::from_u64(8)]);
        enforce_commit_fields(
            &mut builder,
            SisAccumulatorConfig {
                seed,
                kappa: 1,
                domain: 0x5345_4544_5F49_4454,
            },
            &fields,
        )
        .expect("seeded Phi81 fixture");
        assert!(builder.is_satisfied());
        builder
    }

    let left = fixture([0x51; 32]);
    let right = fixture([0xA2; 32]);
    assert_eq!(
        left.sparse_triplets(),
        right.sparse_triplets(),
        "the legacy sparse-only surface erases the seeded A coefficient source"
    );
    assert_eq!(left.seeded_phi81_a_blocks().len(), 1);
    assert_eq!(right.seeded_phi81_a_blocks().len(), 1);

    let left_range = RowFamilyRange {
        name: "seeded-identity-regression",
        row_start: 0,
        row_end: left.rows(),
    };
    let right_range = RowFamilyRange {
        row_end: right.rows(),
        ..left_range
    };
    assert_ne!(
        full_history_manifest_identity_support::range_hash(&left, &left_range),
        full_history_manifest_identity_support::range_hash(&right, &right_range),
        "the authoritative range hash must bind implicit seeded A rows"
    );

    let left_program = checked_program_artifact_support::normalize(&left);
    let right_program = checked_program_artifact_support::normalize(&right);
    assert_ne!(
        left_program.instructions, right_program.instructions,
        "checked-program normalization must materialize implicit seeded A coefficients"
    );
    let seeded_row = left.seeded_phi81_a_blocks()[0].row_start();
    match &left_program.instructions[seeded_row] {
        checked_program_artifact_support::Instruction::Define(checked_program_artifact_support::Definition {
            rhs: checked_program_artifact_support::Rhs::Product(a, b),
            ..
        }) => {
            assert!(!a.is_empty(), "seeded A row must be present");
            assert_eq!(b, &[(0, 1)], "seeded row multiplier");
        }
        checked_program_artifact_support::Instruction::Check(row) => {
            assert!(!row.a.is_empty(), "seeded A row must be present");
            assert_eq!(row.b, [(0, 1)], "seeded row multiplier");
        }
        instruction => panic!("unexpected seeded row normalization: {instruction:?}"),
    }
}

#[test]
fn rank_two_ajtai_maps_use_exact_rows_and_share_message_openings() {
    const FIRST: SisAccumulatorConfig = SisAccumulatorConfig {
        seed: [0x31; 32],
        kappa: 2,
        domain: 0x414a_5441_495f_3031,
    };
    const SECOND: SisAccumulatorConfig = SisAccumulatorConfig {
        seed: [0x32; 32],
        kappa: 2,
        domain: 0x414a_5441_495f_3032,
    };

    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let fields = builder.alloc_vec(&[F::from_u64(3), F::from_u64(5), F::from_u64(8)]);

    let first = enforce_commit_fields(&mut builder, FIRST, &fields).expect("first rank-two map");
    let first_block = builder.seeded_phi81_a_blocks()[0].clone();
    assert_eq!(first.data.len(), 2 * D);
    assert_eq!(first_block.row_end() - first_block.row_start(), 2 * D);
    assert_eq!(first.data.len() * BALANCED_TERNARY_DIGITS, 4_428);

    let opening_count = builder.encoding_trace().balanced_ternary_openings().len();
    let rows_before_second = builder.rows();
    let columns_before_second = builder.cols();
    let second = enforce_commit_fields(&mut builder, SECOND, &fields).expect("second rank-two map");
    let second_block = builder.seeded_phi81_a_blocks()[1].clone();

    assert_eq!(
        builder.encoding_trace().balanced_ternary_openings().len(),
        opening_count,
        "the second map must reuse the same canonical message openings"
    );
    assert_eq!(builder.rows() - rows_before_second, 2 + 2 * D);
    assert_eq!(builder.cols() - columns_before_second, 2 + 2 * D);
    assert_eq!(second_block.row_start(), rows_before_second + 2);
    assert_eq!(second_block.row_end() - second_block.row_start(), 2 * D);
    assert_eq!(second_block.word_starts(), first_block.word_starts());

    let snapshot = builder.snapshot();
    for (offset, output) in second.data.iter().enumerate() {
        let row = second_block.row_start() + offset;
        assert_eq!(snapshot.b_row(row), [(Var::ONE.col(), F::ONE)]);
        assert_eq!(snapshot.c_row(row), [(output.col(), F::ONE)]);
    }

    let changed = 37;
    let mut invalid = snapshot.witness().to_vec();
    invalid[second.data[changed].col()] += F::ONE;
    assert_eq!(
        snapshot.first_unsatisfied_row(&invalid),
        Some(second_block.row_start() + changed)
    );
}

#[test]
fn repeated_rank_two_maps_reuse_the_same_compact_selective_openings() {
    const FIRST: SisAccumulatorConfig = SisAccumulatorConfig {
        seed: [0x41; 32],
        kappa: 2,
        domain: 0x414a_5441_495f_3131,
    };
    const SECOND: SisAccumulatorConfig = SisAccumulatorConfig {
        seed: [0x42; 32],
        kappa: 2,
        domain: 0x414a_5441_495f_3132,
    };

    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&[F::from_u64(3), F::from_u64(5), F::from_u64(8)]);
    enforce_commit_fields(&mut builder, FIRST, &fields).expect("first rank-two map");
    enforce_commit_fields(&mut builder, SECOND, &fields).expect("second rank-two map");

    let lowered = lower_field_r1cs(builder, &[]).expect("lower repeated rank-two maps");
    let (shape, _) = lowered.into_parts();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape], 0, D, 0)
        .expect("selective repeated rank-two maps");
    let audit = relation
        .selective_compiler_audit()
        .expect("selective compiler audit");

    for openings in audit.canonical_openings() {
        assert_eq!(
            openings.len(),
            fields.len(),
            "commitment reuse must not allocate a second opening for the same field"
        );
        assert!(openings.iter().all(|opening| {
            opening.digit_coordinates().len() == 41
                && opening.borrow_coordinates().len() == 20
                && opening.coordinate_count() == 61
                && opening.emitted_rows().len() == 21
        }));
    }
}

#[test]
fn protocol_binding_maps_match_estimated_two_level_profile() {
    let long_maps = [
        CCS_CLAIM_SIS_CONFIG,
        CE_CLAIM_SIS_CONFIG,
        ACCUMULATOR_CE_CLAIM_SIS_CONFIG,
        PENDING_ACCUMULATOR_FAMILY_SIS_CONFIG,
        PI_CCS_OUTPUTS_SIS_CONFIG,
        PI_RLC_PROJECTION_SIS_CONFIG,
        NEBULA_LEAF_SIS_CONFIG,
    ];
    for config in long_maps {
        assert_eq!(config.kappa, PROTOCOL_BINDING_KAPPA);
    }
    assert_eq!(SIS_DIGEST_COMPRESSION_CONFIG.kappa, 1);

    let all_maps = [
        long_maps[0],
        long_maps[1],
        long_maps[2],
        long_maps[3],
        long_maps[4],
        long_maps[5],
        long_maps[6],
        SIS_DIGEST_COMPRESSION_CONFIG,
    ];
    for (index, config) in all_maps.iter().enumerate() {
        assert!(
            all_maps[..index]
                .iter()
                .all(|prior| prior.seed != config.seed && prior.domain != config.domain),
            "every estimated map must have an independent seed and domain"
        );
    }
}

#[test]
fn protocol_binding_widths_stop_at_the_estimated_security_boundary() {
    let rank_two_max_fields = PROTOCOL_BINDING_MAX_MESSAGE_COLS * D / BALANCED_TERNARY_DIGITS;
    let rank_one_max_fields = DIGEST_COMPRESSION_MAX_MESSAGE_COLS * D / BALANCED_TERNARY_DIGITS;
    assert_eq!(rank_two_max_fields, 66_342);
    assert_eq!(rank_one_max_fields, 108);

    let rank_two_error = commit_fields(CCS_CLAIM_SIS_CONFIG, &vec![F::ZERO; rank_two_max_fields + 1])
        .expect_err("rank-two message above the estimated width must fail");
    assert!(matches!(
        rank_two_error,
        SisAccumulatorError::MessageTooWide {
            kappa: PROTOCOL_BINDING_KAPPA,
            field_count: 66_343,
            max_field_count: 66_342,
            max_message_cols: PROTOCOL_BINDING_MAX_MESSAGE_COLS,
        }
    ));

    let rank_one_error = commit_fields(SIS_DIGEST_COMPRESSION_CONFIG, &vec![F::ZERO; rank_one_max_fields + 1])
        .expect_err("rank-one message above the estimated width must fail");
    assert!(matches!(
        rank_one_error,
        SisAccumulatorError::MessageTooWide {
            kappa: 1,
            field_count: 109,
            max_field_count: 108,
            max_message_cols: DIGEST_COMPRESSION_MAX_MESSAGE_COLS,
        }
    ));

    let folding_error = commit_fields(
        SisAccumulatorConfig {
            kappa: 18,
            ..CCS_CLAIM_SIS_CONFIG
        },
        &[F::ZERO],
    )
    .expect_err("the rank-18 folding commitment must use its own module");
    assert!(matches!(
        folding_error,
        SisAccumulatorError::UnsupportedKappa { kappa: 18 }
    ));
}

#[test]
fn pending_accumulator_family_matches_exact_lean_order_and_count() {
    const KAPPA: usize = 4;
    assert_eq!(PENDING_ACCUMULATOR_FAMILY_DOMAIN.len(), 59);
    assert_eq!(PENDING_ACCUMULATOR_FAMILY_CHILDREN, 14);
    assert_eq!(PENDING_ACCUMULATOR_FAMILY_ROW_POINT, 23);
    assert_eq!(PENDING_ACCUMULATOR_FAMILY_COLUMN_POINT, 19);
    assert_eq!(PENDING_ACCUMULATOR_FAMILY_M_IN, 270);
    assert_eq!(PENDING_ACCUMULATOR_FAMILY_MATRICES, 13);

    let claims = pending_family_claims(KAPPA);
    let reference_none = reference_pending_family_preimage(&claims, None);
    let production_none = pending_accumulator_family_preimage(&claims, KAPPA, None).expect("valid fixed family");
    assert_eq!(production_none, reference_none);
    assert_eq!(production_none.len(), 26_709, "Lean bounded_field_count");
    let production_claims = pending_family_claims(18);
    assert_eq!(
        pending_accumulator_family_preimage(&production_claims, 18, None)
            .expect("valid production-width family")
            .len(),
        37_293,
        "Lean production_field_count"
    );

    let old_block: Vec<K> = (0..19)
        .map(|index| K::from_coeffs([F::from_u64(500_000 + 2 * index), F::from_u64(500_001 + 2 * index)]))
        .collect();
    let parent_y_zcol: Vec<K> = (0..D)
        .map(|index| {
            K::from_coeffs([
                F::from_u64(600_000 + 2 * index as u64),
                F::from_u64(600_001 + 2 * index as u64),
            ])
        })
        .collect();
    let pending = PendingAccumulatorFamilyState {
        old_block: &old_block,
        parent_y_zcol: &parent_y_zcol,
    };
    let reference_some = reference_pending_family_preimage(&claims, Some(pending));
    let production_some =
        pending_accumulator_family_preimage(&claims, KAPPA, Some(pending)).expect("valid pending family");
    assert_eq!(production_some, reference_some);
    assert_eq!(production_some.len(), 26_709, "pending option is fixed width");
    assert_ne!(production_none, production_some, "discriminator binds absence");

    let reference_digest = accumulator_digest(PENDING_ACCUMULATOR_FAMILY_SIS_CONFIG, &reference_some)
        .expect("nonempty independent pending-family preimage");
    assert_eq!(
        pending_accumulator_family_digest(&claims, KAPPA, Some(pending)).expect("valid pending family"),
        reference_digest
    );
}

#[test]
fn pending_accumulator_family_rejects_omitted_field_aliases() {
    const KAPPA: usize = 4;
    let claims = pending_family_claims(KAPPA);
    let baseline = pending_accumulator_family_preimage(&claims, KAPPA, None).expect("valid fixed family");

    let mut sidecar_only = claims.clone();
    sidecar_only[3].y_zcol[0] += K::ONE;
    assert_eq!(
        pending_accumulator_family_preimage(&sidecar_only, KAPPA, None).expect("y_zcol is transport only"),
        baseline,
        "prover-carried child y_zcol must not become family authority"
    );

    let mut permuted = claims.clone();
    permuted.swap(0, 1);
    assert_ne!(
        pending_accumulator_family_preimage(&permuted, KAPPA, None).expect("ordered children remain valid"),
        baseline,
        "child order is semantic"
    );

    let mut inactive_x = claims.clone();
    inactive_x[2].X[(0, 5)] = F::ONE;
    assert!(pending_accumulator_family_preimage(&inactive_x, KAPPA, None).is_err());

    let mut padded_y = claims.clone();
    padded_y[4].y_ring[1][D] = K::ONE;
    assert!(pending_accumulator_family_preimage(&padded_y, KAPPA, None).is_err());

    let mut wrong_ct = claims.clone();
    wrong_ct[5].ct[2] += K::ONE;
    assert!(pending_accumulator_family_preimage(&wrong_ct, KAPPA, None).is_err());

    let mut different_point = claims.clone();
    different_point[6].r[0] += K::ONE;
    assert!(pending_accumulator_family_preimage(&different_point, KAPPA, None).is_err());

    let mut different_fold = claims.clone();
    different_fold[7].fold_digest[0] ^= 1;
    assert!(pending_accumulator_family_preimage(&different_fold, KAPPA, None).is_err());

    let mut unsupported = claims.clone();
    unsupported[8].aux_openings.push(K::ONE);
    assert!(pending_accumulator_family_preimage(&unsupported, KAPPA, None).is_err());

    let short_old_block = vec![K::ZERO; 18];
    let parent_y_zcol = vec![K::ZERO; D];
    assert!(pending_accumulator_family_preimage(
        &claims,
        KAPPA,
        Some(PendingAccumulatorFamilyState {
            old_block: &short_old_block,
            parent_y_zcol: &parent_y_zcol,
        }),
    )
    .is_err());
}

#[test]
fn pending_accumulator_family_circuit_preimage_matches_native_without_sis_lowering() {
    const KAPPA: usize = 4;
    let claims = pending_family_claims(KAPPA);
    let old_block: Vec<K> = (0..19)
        .map(|index| K::from_coeffs([F::from_u64(700_000 + 2 * index), F::from_u64(700_001 + 2 * index)]))
        .collect();
    let parent_y_zcol: Vec<K> = (0..D)
        .map(|index| {
            K::from_coeffs([
                F::from_u64(800_000 + 2 * index as u64),
                F::from_u64(800_001 + 2 * index as u64),
            ])
        })
        .collect();
    let native_pending = PendingAccumulatorFamilyState {
        old_block: &old_block,
        parent_y_zcol: &parent_y_zcol,
    };
    let native = pending_accumulator_family_preimage(&claims, KAPPA, Some(native_pending)).expect("valid family");

    let mut builder = R1csBuilder::new();
    let row_point: Vec<KVar> = claims[0]
        .r
        .iter()
        .copied()
        .map(|value| alloc_k(&mut builder, value))
        .collect();
    let column_point: Vec<KVar> = claims[0]
        .s_col
        .iter()
        .copied()
        .map(|value| alloc_k(&mut builder, value))
        .collect();
    let fold_digest_vec = builder.alloc_vec(&digest32_as_fields(claims[0].fold_digest));
    let fold_digest_fields: [Var; 4] = fold_digest_vec.try_into().expect("four fold lanes");
    let owned_children: Vec<OwnedPendingFamilyChildWires> = claims
        .iter()
        .map(|claim| {
            let c_data = builder.alloc_vec(&claim.c.data);
            let x_values: Vec<F> = (0..5)
                .flat_map(|column| (0..D).map(move |lane| claim.X[(lane, column)]))
                .collect();
            let x_active_column_major = builder.alloc_vec(&x_values);
            let y_ring_active = claim
                .y_ring
                .iter()
                .map(|row| {
                    row[..D]
                        .iter()
                        .copied()
                        .map(|value| alloc_k(&mut builder, value))
                        .collect()
                })
                .collect();
            OwnedPendingFamilyChildWires {
                c_data,
                x_active_column_major,
                y_ring_active,
            }
        })
        .collect();
    let child_inputs: Vec<PendingAccumulatorFamilyChildInputs<'_>> = owned_children
        .iter()
        .map(|child| PendingAccumulatorFamilyChildInputs {
            c_data: &child.c_data,
            x_active_column_major: &child.x_active_column_major,
            y_ring_active: &child.y_ring_active,
        })
        .collect();
    let old_block_wires: Vec<KVar> = old_block
        .iter()
        .copied()
        .map(|value| alloc_k(&mut builder, value))
        .collect();
    let parent_y_zcol_wires: Vec<KVar> = parent_y_zcol
        .iter()
        .copied()
        .map(|value| alloc_k(&mut builder, value))
        .collect();
    let inputs = PendingAccumulatorFamilyDigestInputs {
        verifier_rows: KAPPA,
        row_point: &row_point,
        column_point: &column_point,
        m_in: 270,
        fold_digest_fields,
        children: &child_inputs,
        pending: Some(PendingAccumulatorFamilyStateInputs {
            old_block: &old_block_wires,
            parent_y_zcol: &parent_y_zcol_wires,
        }),
    };
    let encoded =
        encode_pending_accumulator_family_preimage(&mut builder, &inputs).expect("valid circuit family shape");
    let circuit: Vec<F> = encoded
        .iter()
        .map(|wire| builder.witness()[wire.col()])
        .collect();
    assert_eq!(circuit, native);
    assert_eq!(circuit.len(), 26_709);
    assert!(builder.is_satisfied());
    assert!(
        builder.rows() < 200,
        "preimage parity must not accidentally lower the 26k-field SIS map"
    );
}

#[test]
fn accumulator_ce_claim_digest_v2_golden_pins_validated_core_serialization() {
    let claim = accumulator_ce_golden_claim();
    assert!(
        !claim.y_zcol.is_empty(),
        "fixture must expose the omitted y_zcol boundary"
    );

    let preimage = reference_accumulator_ce_claim_v2_preimage(&claim);
    assert_eq!(preimage.len(), 232, "validated CE-core v2 field count");
    let reference = accumulator_digest(ACCUMULATOR_CE_CLAIM_SIS_CONFIG, &preimage)
        .expect("nonempty independent accumulator CE preimage");
    let production = accumulator_ce_claim_digest(&claim);
    assert_eq!(
        production, reference,
        "production must preserve the independently serialized v2 CE-core field order"
    );
    assert_eq!(
        production.map(|lane| lane.as_canonical_u64()),
        [
            8_082_636_512_356_632_764,
            9_418_715_576_585_056_888,
            6_232_849_576_044_046_739,
            8_716_910_379_429_202_171,
        ],
        "deterministic v2 accumulator CE digest golden vector"
    );

    let mut changed_sidecar = claim.clone();
    changed_sidecar.y_zcol[0] += K::ONE;
    assert_eq!(
        reference_accumulator_ce_claim_v2_preimage(&changed_sidecar),
        preimage,
        "the CE-core golden vector must not imply y_zcol authority"
    );
}

#[test]
fn accumulator_ce_claim_digest_v2_rejects_sis_config_substitution() {
    let claim = accumulator_ce_golden_claim();
    let preimage = reference_accumulator_ce_claim_v2_preimage(&claim);
    let baseline = accumulator_ce_claim_digest(&claim);

    assert_eq!(ACCUMULATOR_CE_CLAIM_SIS_CONFIG.seed, [0xC7; 32]);
    assert_eq!(ACCUMULATOR_CE_CLAIM_SIS_CONFIG.kappa, PROTOCOL_BINDING_KAPPA);
    assert_eq!(ACCUMULATOR_CE_CLAIM_SIS_CONFIG.domain, 0x4143_4345_5F43_4C4D);

    let changed_seed = SisAccumulatorConfig {
        seed: [0xC8; 32],
        ..ACCUMULATOR_CE_CLAIM_SIS_CONFIG
    };
    let changed_domain = SisAccumulatorConfig {
        domain: ACCUMULATOR_CE_CLAIM_SIS_CONFIG.domain ^ 1,
        ..ACCUMULATOR_CE_CLAIM_SIS_CONFIG
    };
    for (label, config) in [
        ("changed C7 seed", changed_seed),
        ("changed accumulator domain", changed_domain),
        ("CE-claim SIS config", CE_CLAIM_SIS_CONFIG),
    ] {
        let substituted =
            accumulator_digest(config, &preimage).expect("nonempty independently serialized accumulator CE preimage");
        assert_ne!(substituted, baseline, "{label} must not preserve the digest");
    }
}

#[test]
fn sis_accumulator_matches_native_and_rejects_tampering() {
    let values = [F::from_u64(3), F::from_u64(F::ORDER_U64 / 2), -F::ONE];
    let native = commit_fields(CONFIG, &values).expect("native SIS accumulator");
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    let commitment = enforce_commit_fields(&mut builder, CONFIG, &fields).expect("SIS accumulator circuit");
    let circuit_data: Vec<F> = commitment
        .data
        .iter()
        .map(|wire| builder.witness()[wire.col()])
        .collect();

    assert_eq!(circuit_data, native.data);
    assert!(builder.is_satisfied());
    assert_eq!(
        builder.cols(),
        426,
        "three fields, canonical shifted-base-3 trits, borrow witnesses, and one D-wide output"
    );
    assert_eq!(
        builder.rows(),
        428,
        "canonical shifted-base-3 decompositions plus D output equations"
    );
    assert!(
        builder.nonzero_entries() < 10_000,
        "the seeded ring map must not materialize its rotated coefficients"
    );

    let input_col = fields[1].col();
    let input = builder.witness()[input_col];
    builder.tamper_witness(input_col, input + F::ONE);
    assert!(
        !builder.is_satisfied(),
        "input mutation must break canonical recomposition"
    );
    builder.tamper_witness(input_col, input);

    let output_col = commitment.data[0].col();
    let output = builder.witness()[output_col];
    builder.tamper_witness(output_col, output + F::ONE);
    assert!(!builder.is_satisfied(), "output mutation must break the Ajtai equation");
}

fn forge_alternate_shifted_base3_opening(
    builder: &mut R1csBuilder,
    field: neo_fold_clean::engine::r1cs_circuit::Var,
    commitment_data: &[neo_fold_clean::engine::r1cs_circuit::Var],
) {
    let midpoint = F::ORDER_U64 / 2;
    // If N is the canonical shifted-base-3 representative, N+p encodes the
    // same field residue and still fits in 41 trits for this fixture. Recompute
    // every auxiliary and the Ajtai output so only the terminal borrow remains
    // false. A reconstruction-only circuit accepts this forged opening.
    let modulus = F::ORDER_U64 as u128;
    let shift = (3u128.pow(41) - 1) / 2;
    let canonical_n = (midpoint as u128 + shift) % modulus;
    let mut remaining = canonical_n + modulus;
    assert!(remaining < 3u128.pow(41), "alternate representative fits in 41 trits");
    let alternate: [F; 41] = core::array::from_fn(|_| {
        let trit = remaining % 3;
        remaining /= 3;
        match trit {
            0 => -F::ONE,
            1 => F::ZERO,
            2 => F::ONE,
            _ => unreachable!("base-3 digit"),
        }
    });
    assert_eq!(remaining, 0);

    let digit_columns = builder
        .balanced_ternary_digit_columns(field)
        .expect("recorded balanced-ternary decomposition");
    assert_eq!(digit_columns.len(), 41);
    let negative_start = digit_columns[40] + 1;
    let borrow_start = negative_start + 41;
    for (index, (&column, &digit)) in digit_columns.iter().zip(&alternate).enumerate() {
        builder.tamper_witness(column, digit);
        builder.tamper_witness(negative_start + index, if digit == -F::ONE { F::ONE } else { F::ZERO });
    }

    let mut bound = F::ORDER_U64 - 1;
    let mut borrow = false;
    for (index, &digit) in alternate.iter().enumerate() {
        let trit = if digit == -F::ONE {
            0
        } else if digit == F::ZERO {
            1
        } else {
            2
        };
        let next = trit + u64::from(borrow) > bound % 3;
        bound /= 3;
        if index + 1 < 41 {
            builder.tamper_witness(borrow_start + index, if next { F::ONE } else { F::ZERO });
        } else {
            assert!(next, "N+p must leave a terminal borrow");
        }
        borrow = next;
    }

    let mut message = Mat::zero(D, 1, F::ZERO);
    for (row, &digit) in alternate.iter().enumerate() {
        message[(row, 0)] = digit;
    }
    let forged = commit_row_major_seeded(CONFIG.seed, D, CONFIG.kappa, 1, &message);
    for (wire, value) in commitment_data.iter().zip(forged.data) {
        builder.tamper_witness(wire.col(), value);
    }
}

#[test]
fn sis_accumulator_rejects_noncanonical_shifted_base3_opening() {
    let midpoint = F::ORDER_U64 / 2;
    let mut builder = R1csBuilder::new();
    let field = builder.alloc(F::from_u64(midpoint));
    let commitment = enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("SIS commitment circuit");
    assert!(builder.is_satisfied(), "canonical shifted-base-3 opening");
    forge_alternate_shifted_base3_opening(&mut builder, field, &commitment.data);

    assert!(
        !builder.is_satisfied(),
        "a field residue must have exactly one accepted shifted-base-3 opening"
    );
    assert_eq!(
        builder.first_unsatisfied_row(),
        Some(125),
        "after recomputing every forged auxiliary, only the terminal borrow row must reject"
    );
}

#[test]
fn selective_sis_lowering_rejects_noncanonical_shifted_base3_opening() {
    let mut builder = R1csBuilder::new();
    let field = builder.alloc(F::from_u64(F::ORDER_U64 / 2));
    let commitment = enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("SIS commitment circuit");
    forge_alternate_shifted_base3_opening(&mut builder, field, &commitment.data);
    assert!(!builder.is_satisfied(), "source R1CS must reject the forged opening");

    let lowered = lower_field_r1cs(builder, &[]).expect("field lowering");
    let (shape, assignment) = lowered.into_parts();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape], 0, 1, 0)
        .expect("selective low-norm lowering");
    let encoded = relation
        .encode(0, &assignment)
        .expect("encode forged source assignment");
    assert!(
        !relation.is_satisfied(&encoded),
        "direct shifted-ternary rows must reject x+p after every source auxiliary and Ajtai output is recomputed"
    );
}

#[test]
fn gadget_native_sis_lowering_preserves_and_rejects_the_noncanonical_opening() {
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let field = builder.alloc(F::from_u64(F::ORDER_U64 / 2));
    let commitment = enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("SIS commitment circuit");
    forge_alternate_shifted_base3_opening(&mut builder, field, &commitment.data);
    assert!(!builder.is_satisfied(), "source R1CS must reject the forged opening");

    let source = builder.snapshot();
    let encoded = encode_r1cs_gadget_native(&source, builder.encoding_trace(), &[])
        .expect("gadget-native balanced-ternary lowering");
    assert_eq!(
        encoded
            .plan
            .decode_source(&encoded.assignment)
            .expect("decode forged candidate"),
        source.witness(),
        "materialization must preserve the forged digits instead of normalizing from the field residue"
    );
    assert!(
        !encoded.is_satisfied(),
        "the retained terminal-borrow row must reject the x+p opening"
    );
}

#[test]
fn sis_shifted_base3_boundaries_accept_and_auxiliaries_are_load_bearing() {
    let shift = ((3u128.pow(41) - 1) / 2) as u64;
    let values = [
        0,
        1,
        F::ORDER_U64 - 1,
        F::ORDER_U64 - shift,
        F::ORDER_U64 - 1 - shift,
        F::ORDER_U64 / 2,
        F::ORDER_U64 / 2 + 1,
        0xdead_beef_cafe_babe % F::ORDER_U64,
    ];

    for value in values {
        let mut builder = R1csBuilder::new();
        let field = builder.alloc(F::from_u64(value));
        enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("boundary SIS commitment");
        assert!(
            builder.is_satisfied(),
            "canonical shifted-base-3 opening must accept residue {value}"
        );
    }

    // `x = p-1-M` maps to the exact comparator boundary N=p-1.
    let mut builder = R1csBuilder::new();
    let field = builder.alloc(F::from_u64(F::ORDER_U64 - 1 - shift));
    enforce_commit_fields(&mut builder, CONFIG, &[field]).expect("N=p-1 SIS commitment");
    assert!(builder.is_satisfied(), "N=p-1 boundary");
    let digits = builder
        .balanced_ternary_digit_columns(field)
        .expect("recorded boundary decomposition");
    let negative_start = digits[40] + 1;
    let borrow_start = negative_start + 41;

    for column in negative_start..borrow_start + 40 {
        let original = builder.witness()[column];
        let tampered = if original == F::ZERO { F::ONE } else { F::ZERO };
        builder.tamper_witness(column, tampered);
        assert!(
            !builder.is_satisfied(),
            "canonicality auxiliary column {column} must be load-bearing"
        );
        builder.tamper_witness(column, original);
        assert!(builder.is_satisfied());
    }

    let first_digit = digits[0];
    let original = builder.witness()[first_digit];
    builder.tamper_witness(first_digit, F::from_u64(2));
    assert!(!builder.is_satisfied(), "a trit outside {{-1,0,1}} must be rejected");
    builder.tamper_witness(first_digit, original);
    assert!(builder.is_satisfied());
}

#[test]
fn sis_accumulator_hash_then_fs_matches_native_and_exposes_cost() {
    let values = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
    let native = accumulator_digest(CONFIG, &values).expect("native SIS digest");
    assert_ne!(
        native,
        accumulator_digest(CONFIG, &[values[0], values[1], values[2], F::ZERO]).expect("length-separated SIS digest"),
        "the SIS digest must bind the input field count"
    );
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    let wires = enforce_accumulator_digest(&mut builder, CONFIG, &fields).expect("SIS digest circuit");
    let circuit_digest: [F; 4] = std::array::from_fn(|lane| builder.witness()[wires.digest[lane].col()]);

    assert_eq!(circuit_digest, native);
    assert!(builder.is_satisfied());

    for wire in [
        wires.commitment.data[0],
        wires.digest_compression.data[0],
        wires.digest[0],
    ] {
        let original = builder.witness()[wire.col()];
        builder.tamper_witness(wire.col(), original + F::ONE);
        assert!(!builder.is_satisfied(), "every binding layer must be load-bearing");
        builder.tamper_witness(wire.col(), original);
        assert!(builder.is_satisfied());
    }
    eprintln!(
        "SIS accumulator (3 fields, kappa=1): rows={}, cols={}, nnz={}",
        builder.rows(),
        builder.cols(),
        builder.nonzero_entries()
    );
}

#[test]
fn sis_accumulator_reuses_authoritative_trits_in_selective_lowering() {
    let values = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    enforce_accumulator_digest(&mut builder, CONFIG, &fields).expect("SIS digest circuit");
    let lowered = lower_field_r1cs(builder, &[]).expect("field lowering");
    let (shape, field_assignment) = lowered.into_parts();
    let first_private_field = shape.m_in;
    let arms = [shape.clone(), shape];
    let relation =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&arms, 0, 1, 0).expect("selective low-norm lowering");
    for openings in relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .canonical_openings()
    {
        assert_eq!(
            openings.len(),
            values.len() + D,
            "the input fields and first-map outputs each need one canonical opening"
        );
        assert!(openings
            .iter()
            .all(|opening| opening.coordinate_count() == 61 && opening.emitted_rows().len() == 21));
    }
    let mut encoded = relation
        .encode(0, &field_assignment)
        .expect("encoded SIS arm");

    assert!(relation.is_satisfied(&encoded));
    let source_slot = relation
        .field_slot(0, first_private_field)
        .expect("SIS source field slot");
    assert_eq!(source_slot.1, 41, "SIS must reuse one balanced-ternary field slot");
    eprintln!(
        "selective SIS accumulator (3 fields, kappa=1): rows={}, committed_coordinates={}",
        relation.structure().n,
        relation.structure().m
    );
    assert!(
        relation.structure().n < 100_000 && relation.structure().m < 100_000,
        "shifted-base-3 canonicality must not reintroduce the 400k-coordinate lowering blow-up"
    );

    encoded[source_slot.0] = F::from_u64(2);
    assert!(!relation.is_satisfied(&encoded), "a non-unit SIS trit must be rejected");
}

#[test]
fn rank_two_digest_chain_counts_each_distinct_opening_once() {
    let values = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
    let mut builder = R1csBuilder::new();
    let fields = builder.alloc_vec(&values);
    let wires =
        enforce_accumulator_digest(&mut builder, CCS_CLAIM_SIS_CONFIG, &fields).expect("rank-two SIS digest circuit");
    let rank_two_outputs = wires
        .commitment
        .data
        .iter()
        .map(|wire| wire.col())
        .collect::<Vec<_>>();
    let rank_one_outputs = wires
        .digest_compression
        .data
        .iter()
        .map(|wire| wire.col())
        .collect::<Vec<_>>();
    let lowered = lower_field_r1cs(builder, &[]).expect("field lowering");
    let (shape, _) = lowered.into_parts();
    let relation = build_multi_branch_selective_low_norm_r1cs_with_alignment(&[shape.clone(), shape], 0, D, 0)
        .expect("selective rank-two digest");

    for openings in relation
        .selective_compiler_audit()
        .expect("selective compiler audit")
        .canonical_openings()
    {
        assert_eq!(
            openings.len(),
            values.len() + 2 * D,
            "three source fields plus the 108 rank-two output fields"
        );
        assert_eq!(
            openings
                .iter()
                .map(|opening| opening.coordinate_count())
                .sum::<usize>(),
            61 * (values.len() + 2 * D)
        );
        assert_eq!(
            openings
                .iter()
                .map(|opening| opening.emitted_rows().len())
                .sum::<usize>(),
            21 * (values.len() + 2 * D)
        );
        assert!(rank_two_outputs.iter().all(|output| openings
            .iter()
            .any(|opening| opening.source_field() == *output)));
        assert!(rank_one_outputs.iter().all(|output| openings
            .iter()
            .all(|opening| opening.source_field() != *output)));
    }
    for arm in 0..2 {
        assert!(rank_two_outputs.iter().all(|output| relation
            .field_slot(arm, *output)
            .is_some_and(|slot| slot.1 == 41)));
        assert!(rank_one_outputs.iter().all(|output| relation
            .field_slot(arm, *output)
            .is_some_and(|slot| slot.1 == 41)));
    }
}

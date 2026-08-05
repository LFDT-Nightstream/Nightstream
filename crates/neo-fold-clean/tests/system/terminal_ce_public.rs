#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term};
use neo_fold_clean::engine::decider::__test_isolation::{
    enforce_terminal_ce_public_from_children_against, enforce_terminal_ce_public_pinned_against,
    enforce_terminal_ce_verify_from_children_against,
};
use neo_fold_clean::paper::digest::{terminal_ce_public_digest, terminal_ce_relation_digest, terminal_children_digest};
use neo_fold_clean::paper::terminal_ce::{
    TerminalCeProof, TerminalCePublic, TerminalCePublicError, TerminalCeVerifyError,
};
use neo_fold_clean::{config, preprocess, CeClaim, Params, Preprocessing};
use neo_math::{KExtensions, D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn f(value: u64) -> F {
    F::from_u64(value)
}

fn k(c0: u64, c1: u64) -> K {
    K::from_coeffs([f(c0), f(c1)])
}

fn changed_terminal_ce_structure_fixture() -> neo_fold_clean::Structure {
    let mut m0 = Mat::identity(D);
    m0[(0, 0)] = f(2);
    CcsStructure::new(vec![m0], SparsePoly::new(1, vec![])).expect("changed terminal CE structure fixture")
}

fn changed_terminal_ce_polynomial_fixture() -> neo_fold_clean::Structure {
    CcsStructure::new(
        vec![Mat::identity(D)],
        SparsePoly::new(
            1,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1],
            }],
        ),
    )
    .expect("changed terminal CE polynomial fixture")
}

fn two_col_terminal_ce_preprocessing() -> Preprocessing {
    let mut m0 = Mat::zero(1, 2, F::ZERO);
    m0[(0, 0)] = F::ONE;
    let structure =
        CcsStructure::new(vec![m0], SparsePoly::new(1, vec![])).expect("two-column terminal CE structure fixture");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("two-column terminal CE params");
    support::install_ajtai_module(&params, &structure);
    preprocess(params, structure, Some(1)).expect("two-column terminal CE preprocessing")
}

fn terminal_child_fixture() -> CeClaim {
    let mut X = Mat::zero(D, D, F::ZERO);
    for row in 0..D {
        X[(row, 0)] = f(100 + row as u64);
    }
    let kappa = Params::production().kappa() as usize;
    let d_pad = D.next_power_of_two();
    let padded_row = |start: u64| {
        let mut row = (0..d_pad)
            .map(|idx| k(start + idx as u64 * 2, start + 1 + idx as u64 * 2))
            .collect::<Vec<_>>();
        row[D..].fill(K::ZERO);
        row
    };
    let y0 = padded_row(11);
    let y1 = padded_row(151);
    let point_len = D.next_power_of_two().trailing_zeros() as usize;
    CeClaim {
        adv: None,
        c: Commitment {
            d: D,
            kappa,
            data: (0..(D * kappa)).map(|idx| f(10 + idx as u64)).collect(),
        },
        X,
        r: (0..point_len)
            .map(|index| k(1 + 2 * index as u64, 2 + 2 * index as u64))
            .collect(),
        y_ring: vec![y0.clone(), y1.clone()],
        ct: vec![y0[0], y1[0]],
        m_in: D,
        fold_digest: [42u8; 32],
    }
}

fn supported_terminal_child_fixture() -> CeClaim {
    terminal_child_fixture()
}

fn second_supported_terminal_child_fixture() -> CeClaim {
    let mut claim = supported_terminal_child_fixture();
    claim.c.data[0] += F::from_u64(7);
    claim.X[(0, 0)] += F::from_u64(11);
    claim.r[0] += k(13, 14);
    claim.y_ring[0][0] += k(17, 18);
    claim.ct[0] = claim.y_ring[0][0];
    claim.fold_digest[0] ^= 0x55;
    claim
}

fn proof_for_children(
    params: &Params,
    structure: &neo_fold_clean::Structure,
    children: &[CeClaim],
    bytes: Vec<u8>,
) -> TerminalCeProof {
    let public = TerminalCePublic::from_terminal_children(params, structure, children)
        .expect("terminal CE proof helper requires supported terminal children");
    TerminalCeProof::new_unchecked(public.digest(), bytes)
}

#[test]
fn terminal_children_digest_binds_every_terminal_child_public_field() {
    let base_claim = terminal_child_fixture();
    let base = terminal_children_digest(&[base_claim.clone()]);

    let mut cases: Vec<(&str, CeClaim)> = Vec::new();

    let mut claim = base_claim.clone();
    claim.c.data[0] += F::ONE;
    cases.push(("commitment data", claim));

    let mut claim = base_claim.clone();
    claim.c.kappa += 1;
    cases.push(("commitment metadata", claim));

    let mut claim = base_claim.clone();
    claim.X[(0, 0)] += F::ONE;
    cases.push(("active X", claim));

    let mut claim = base_claim.clone();
    claim.r[0] += K::ONE;
    cases.push(("r", claim));

    let mut claim = base_claim.clone();
    claim.r[0] += k(0, 1);
    cases.push(("r c1 limb", claim));

    let mut claim = base_claim.clone();
    claim.y_ring[0][0] += K::ONE;
    cases.push(("y_ring", claim));

    let mut claim = base_claim.clone();
    claim.y_ring[0][0] += k(0, 1);
    cases.push(("y_ring c1 limb", claim));

    let mut claim = base_claim.clone();
    claim.ct[0] += K::ONE;
    cases.push(("ct", claim));

    let mut claim = base_claim.clone();
    claim.ct[0] += k(0, 1);
    cases.push(("ct c1 limb", claim));

    let mut claim = base_claim.clone();
    claim.m_in += 1;
    cases.push(("m_in", claim));

    let mut claim = base_claim.clone();
    claim.fold_digest[0] ^= 1;
    cases.push(("fold_digest", claim));

    for (label, claim) in cases {
        let digest = terminal_children_digest(&[claim]);
        assert_ne!(digest, base, "terminal children digest did not bind {label}");
    }

    let digest_with_extra_claim = terminal_children_digest(&[base_claim.clone(), base_claim]);
    assert_ne!(
        digest_with_extra_claim, base,
        "terminal children digest must bind claim count"
    );
}

#[test]
fn terminal_ce_public_binds_structure_params_and_terminal_children() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let child = supported_terminal_child_fixture();
    let public = TerminalCePublic::from_terminal_children(params, structure, &[child.clone()])
        .expect("supported terminal child");
    assert_eq!(
        public.relation_digest,
        terminal_ce_relation_digest(),
        "TerminalCePublic must bind the terminal CE relation contract"
    );

    let changed_structure = changed_terminal_ce_structure_fixture();
    let changed_structure = TerminalCePublic::from_terminal_children(params, &changed_structure, &[child.clone()])
        .expect("supported terminal child with changed structure");
    assert_ne!(
        changed_structure, public,
        "TerminalCePublic must bind the verifier-owned structure digest"
    );

    let changed_polynomial = changed_terminal_ce_polynomial_fixture();
    let changed_polynomial = TerminalCePublic::from_terminal_children(params, &changed_polynomial, &[child.clone()])
        .expect("supported terminal child with changed structure polynomial");
    assert_ne!(
        changed_polynomial, public,
        "TerminalCePublic must bind the full SuperNeo structure, including f"
    );

    let mut raw_params = NeoParams::goldilocks_paper_b2();
    raw_params.lambda -= 1;
    let changed_params = Params::test_only_from_neo_params(raw_params);
    let changed_params_public = TerminalCePublic::from_terminal_children(&changed_params, structure, &[child.clone()])
        .expect("supported terminal child with changed params");
    assert_ne!(
        changed_params_public, public,
        "TerminalCePublic must bind the SuperNeo parameter set"
    );

    let mut changed_child = child.clone();
    changed_child.r[0] += K::ONE;
    let changed_child_public = TerminalCePublic::from_terminal_children(params, structure, &[changed_child])
        .expect("supported terminal child with changed r");
    assert_ne!(
        changed_child_public, public,
        "TerminalCePublic must bind the NIFS-derived terminal children"
    );

    let changed_count_public = TerminalCePublic::from_terminal_children(params, structure, &[child.clone(), child])
        .expect("supported terminal children with changed count");
    assert_ne!(
        changed_count_public, public,
        "TerminalCePublic must bind terminal child count"
    );
    assert_eq!(public.claim_count, 1);
    assert_eq!(changed_count_public.claim_count, 2);
}

#[test]
fn terminal_ce_public_digest_binds_every_public_statement_field() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let child = supported_terminal_child_fixture();
    let public = TerminalCePublic::from_terminal_children(params, structure, &[child]).expect("terminal CE public");
    let base = public.digest();
    assert_eq!(
        base,
        terminal_ce_public_digest(
            public.relation_digest,
            public.structure_digest,
            public.params_digest,
            public.terminal_children_digest,
            public.claim_count,
        ),
        "TerminalCePublic::digest must use the canonical digest helper"
    );

    let mut changed = public.clone();
    changed.relation_digest[0] += F::ONE;
    assert_ne!(changed.digest(), base, "public digest must bind relation_digest");

    let mut changed = public.clone();
    changed.structure_digest[0] += F::ONE;
    assert_ne!(changed.digest(), base, "public digest must bind structure_digest");

    let mut changed = public.clone();
    changed.params_digest[0] += F::ONE;
    assert_ne!(changed.digest(), base, "public digest must bind params_digest");

    let mut changed = public.clone();
    changed.terminal_children_digest[0] += F::ONE;
    assert_ne!(
        changed.digest(),
        base,
        "public digest must bind terminal_children_digest"
    );

    let mut changed = public;
    changed.claim_count += 1;
    assert_ne!(changed.digest(), base, "public digest must bind claim_count");
}

#[test]
fn terminal_ce_public_rejects_locally_malformed_ce_shapes() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();

    let mut bad_d = supported_terminal_child_fixture();
    bad_d.c.d -= 1;
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_d]).unwrap_err(),
        TerminalCePublicError::CommitmentD {
            index: 0,
            expected: D,
            got: D - 1
        }
    );

    let mut bad_kappa = supported_terminal_child_fixture();
    bad_kappa.c.kappa += 1;
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_kappa]).unwrap_err(),
        TerminalCePublicError::CommitmentKappa {
            index: 0,
            expected: params.kappa() as usize,
            got: params.kappa() as usize + 1
        }
    );

    let mut bad_c_len = supported_terminal_child_fixture();
    bad_c_len.c.data.pop();
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_c_len]).unwrap_err(),
        TerminalCePublicError::CommitmentDataLen {
            index: 0,
            expected: D * params.kappa() as usize,
            got: D * params.kappa() as usize - 1
        }
    );

    let mut bad_y_ring = supported_terminal_child_fixture();
    bad_y_ring.y_ring[0].pop();
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_y_ring]).unwrap_err(),
        TerminalCePublicError::YRingLaneCount {
            index: 0,
            matrix_index: 0,
            expected: D.next_power_of_two(),
            got: D.next_power_of_two() - 1
        }
    );

    let mut bad_ct = supported_terminal_child_fixture();
    bad_ct.ct.pop();
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_ct]).unwrap_err(),
        TerminalCePublicError::CtLen {
            index: 0,
            expected: 2,
            got: 1
        }
    );

    let mut bad_m_in = supported_terminal_child_fixture();
    bad_m_in.m_in = structure.m + 1;
    bad_m_in.X = Mat::zero(D, bad_m_in.m_in, F::ZERO);
    bad_m_in.X[(0, 0)] = F::ONE;
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_m_in]).unwrap_err(),
        TerminalCePublicError::MInExceedsStructureM {
            index: 0,
            expected: structure.m,
            got: structure.m + 1,
        }
    );

    let mut bad_r = supported_terminal_child_fixture();
    bad_r.r.pop();
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_r]).unwrap_err(),
        TerminalCePublicError::RLen {
            index: 0,
            expected: 6,
            got: 5,
        }
    );

    let mut bad_y_ring_count = supported_terminal_child_fixture();
    bad_y_ring_count
        .y_ring
        .push(bad_y_ring_count.y_ring[0].clone());
    bad_y_ring_count.ct.push(bad_y_ring_count.y_ring[1][0]);
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_y_ring_count]).unwrap_err(),
        TerminalCePublicError::YRingCount {
            index: 0,
            expected: 2,
            got: 3,
        }
    );
}

#[test]
fn terminal_ce_public_rejects_denormalized_ct_and_y_ring_padding() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();

    let mut bad_ct = supported_terminal_child_fixture();
    bad_ct.ct[0] += K::ONE;
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_ct]).unwrap_err(),
        TerminalCePublicError::CtMismatch {
            index: 0,
            matrix_index: 0
        }
    );

    let mut bad_padding = supported_terminal_child_fixture();
    bad_padding.y_ring[0][D] = K::ONE;
    assert_eq!(
        TerminalCePublic::from_terminal_children(params, structure, &[bad_padding]).unwrap_err(),
        TerminalCePublicError::YRingPaddingNonZero {
            index: 0,
            matrix_index: 0,
            lane: D
        }
    );
}

#[test]
fn terminal_ce_public_rejects_malformed_active_x_shape() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();

    let mut bad_rows = supported_terminal_child_fixture();
    bad_rows.X = Mat::zero(D - 1, 1, F::ZERO);
    let err = TerminalCePublic::from_terminal_children(params, structure, &[bad_rows])
        .expect_err("terminal CE public statement must reject malformed X row count");
    assert_eq!(
        err,
        TerminalCePublicError::XRows {
            index: 0,
            expected: D,
            got: D - 1
        }
    );

    let mut bad_cols = supported_terminal_child_fixture();
    bad_cols.m_in = D + 1;
    let err = TerminalCePublic::from_terminal_children(params, structure, &[bad_cols])
        .expect_err("terminal CE public statement must reject X.cols that drift from m_in");
    assert_eq!(
        err,
        TerminalCePublicError::XCols {
            index: 0,
            expected: D + 1,
            got: D
        }
    );
}

#[test]
fn terminal_ce_public_circuit_rejects_locally_malformed_ce_shapes() {
    let prep = support::toy_preprocessing();

    let cases = [
        ("commitment d", {
            let mut claim = supported_terminal_child_fixture();
            claim.c.d -= 1;
            claim
        }),
        ("commitment kappa", {
            let mut claim = supported_terminal_child_fixture();
            claim.c.kappa += 1;
            claim
        }),
        ("commitment data length", {
            let mut claim = supported_terminal_child_fixture();
            claim.c.data.pop();
            claim
        }),
        ("y_ring row", {
            let mut claim = supported_terminal_child_fixture();
            claim.y_ring[0].pop();
            claim
        }),
        ("ct length", {
            let mut claim = supported_terminal_child_fixture();
            claim.ct.pop();
            claim
        }),
        ("m_in", {
            let mut claim = supported_terminal_child_fixture();
            claim.m_in = prep.structure().m + 1;
            claim.X = Mat::zero(D, claim.m_in, F::ZERO);
            claim.X[(0, 0)] = F::ONE;
            claim
        }),
        ("r length", {
            let mut claim = supported_terminal_child_fixture();
            claim.r.pop();
            claim
        }),
        ("y_ring length", {
            let mut claim = supported_terminal_child_fixture();
            claim.y_ring.push(claim.y_ring[0].clone());
            claim.ct.push(claim.y_ring[1][0]);
            claim
        }),
    ];

    for (label, claim) in cases {
        let err = enforce_terminal_ce_public_from_children_against(&prep, &[claim])
            .err()
            .expect("malformed terminal child shape must fail before digesting");
        assert!(
            err.contains(label),
            "expected circuit public constructor error containing {label:?}, got: {err}"
        );
    }
}

#[test]
fn terminal_ce_public_circuit_rejects_denormalized_ct_and_y_ring_padding() {
    let prep = support::toy_preprocessing();

    let mut bad_ct = supported_terminal_child_fixture();
    bad_ct.ct[0] += K::ONE;
    let ct_output = enforce_terminal_ce_public_from_children_against(&prep, &[bad_ct])
        .expect("same-shape ct mismatch should synthesize equality rows");
    assert!(
        !ct_output.builder.is_satisfied(),
        "terminal CE public circuit accepted ct not equal to y_ring lane zero"
    );

    let mut bad_padding = supported_terminal_child_fixture();
    bad_padding.y_ring[0][D] = K::ONE;
    let padding_output = enforce_terminal_ce_public_from_children_against(&prep, &[bad_padding])
        .expect("same-shape padding mismatch should synthesize zero rows");
    assert!(
        !padding_output.builder.is_satisfied(),
        "terminal CE public circuit accepted nonzero y_ring padding"
    );
}

#[test]
fn terminal_ce_public_rejects_x_cols_not_matching_m_in() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let mut child = supported_terminal_child_fixture();
    child.X = Mat::zero(D, child.m_in + 1, F::ZERO);

    let err = TerminalCePublic::from_terminal_children(params, structure, &[child.clone()])
        .expect_err("terminal CE public statement must use the reference X.cols == m_in shape");
    assert_eq!(
        err,
        TerminalCePublicError::XCols {
            index: 0,
            expected: child.m_in,
            got: child.m_in + 1,
        }
    );

    let err = enforce_terminal_ce_public_from_children_against(&prep, &[child])
        .err()
        .expect("circuit public constructor must reject X.cols != m_in before digesting");
    assert!(err.contains("X.cols"), "expected X.cols shape error, got: {err}");
}

#[test]
fn terminal_ce_public_rejects_inactive_x_columns_not_zero() {
    let prep = two_col_terminal_ce_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let mut child = supported_terminal_child_fixture();
    child.m_in = 2;
    let active = child.X[(0, 0)];
    child.X = Mat::zero(D, 2, F::ZERO);
    child.X[(0, 0)] = active;
    child.X[(0, 1)] = F::ONE;

    let err = TerminalCePublic::from_terminal_children(params, structure, &[child.clone()])
        .expect_err("native public statement must reject unabsorbed inactive X data");
    assert_eq!(
        err,
        TerminalCePublicError::InactiveXNonZero {
            index: 0,
            row: 0,
            col: 1,
        }
    );

    let circuit = enforce_terminal_ce_public_from_children_against(&prep, &[child])
        .expect("circuit public constructor should emit inactive-X zero rows");
    assert!(
        !circuit.builder.is_satisfied(),
        "circuit public constructor must reject unabsorbed inactive X data"
    );
}

#[test]
fn terminal_ce_public_circuit_constructor_matches_native_public_statement() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let child = supported_terminal_child_fixture();
    let native = TerminalCePublic::from_terminal_children(params, structure, &[child.clone()])
        .expect("supported terminal child");

    let circuit =
        enforce_terminal_ce_public_from_children_against(&prep, &[child]).expect("circuit public constructor");
    assert!(
        circuit.builder.is_satisfied(),
        "terminal CE public constructor rows must satisfy for supported children"
    );
    assert_eq!(circuit.structure_digest, native.structure_digest);
    assert_eq!(circuit.relation_digest, native.relation_digest);
    assert_eq!(circuit.params_digest, native.params_digest);
    assert_eq!(circuit.terminal_children_digest, native.terminal_children_digest);
    assert_eq!(circuit.public_digest, native.digest());
    assert_eq!(circuit.claim_count, native.claim_count);
}

#[test]
fn terminal_ce_public_circuit_constructor_matches_native_after_same_shape_c1_relabel() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let honest_child = supported_terminal_child_fixture();
    let honest_public = TerminalCePublic::from_terminal_children(params, structure, &[honest_child.clone()])
        .expect("supported terminal child");

    let mut relabeled_child = honest_child;
    relabeled_child.r[0] += k(0, 1);
    relabeled_child.y_ring[0][0] += k(0, 3);
    relabeled_child.ct[0] = relabeled_child.y_ring[0][0];

    let native = TerminalCePublic::from_terminal_children(params, structure, &[relabeled_child.clone()])
        .expect("same-shape c1 relabel remains a supported terminal public statement");
    assert_ne!(
        native.digest(),
        honest_public.digest(),
        "same-shape c1 relabel must change the compact terminal CE public digest"
    );

    let circuit = enforce_terminal_ce_public_from_children_against(&prep, &[relabeled_child])
        .expect("circuit public constructor");
    assert!(
        circuit.builder.is_satisfied(),
        "same-shape c1 relabel should satisfy public-constructor rows"
    );
    assert_eq!(circuit.structure_digest, native.structure_digest);
    assert_eq!(circuit.relation_digest, native.relation_digest);
    assert_eq!(circuit.params_digest, native.params_digest);
    assert_eq!(circuit.terminal_children_digest, native.terminal_children_digest);
    assert_eq!(circuit.public_digest, native.digest());
    assert_eq!(circuit.claim_count, native.claim_count);
}

#[test]
fn terminal_ce_public_pinned_constructor_rejects_relation_digest_mismatch() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let child = supported_terminal_child_fixture();
    let mut expected = TerminalCePublic::from_terminal_children(params, structure, &[child.clone()])
        .expect("supported terminal child");
    expected.relation_digest[0] += F::ONE;

    let output = enforce_terminal_ce_public_pinned_against(&prep, &[child], &expected)
        .expect("wrong relation digest should still synthesize pinning rows");
    assert!(
        !output.builder.is_satisfied(),
        "terminal CE public constructor accepted an expected public statement with the wrong relation digest"
    );
}

#[test]
fn terminal_ce_public_pinned_constructor_rejects_terminal_child_wire_tamper() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let child = supported_terminal_child_fixture();
    let expected = TerminalCePublic::from_terminal_children(params, structure, &[child.clone()])
        .expect("supported terminal child");

    let output = enforce_terminal_ce_public_pinned_against(&prep, &[child], &expected)
        .expect("pinned circuit public constructor");
    let mut builder = output.builder;
    assert!(
        builder.is_satisfied(),
        "pinned terminal CE public constructor must satisfy before tampering"
    );

    let cases = [
        ("commitment data", output.probes.c_data),
        ("active X", output.probes.x),
        ("r", output.probes.r_c0),
        ("r c1", output.probes.r_c1),
        ("y_ring", output.probes.y_ring_limb),
        ("y_ring c1", output.probes.y_ring_c1),
        ("ct", output.probes.ct_c0),
        ("ct c1", output.probes.ct_c1),
        ("fold_digest", output.probes.fold_digest_field),
    ];
    for (label, col) in cases {
        let original = builder.witness()[col];
        builder.tamper_witness(col, original + F::ONE);
        assert!(
            !builder.is_satisfied(),
            "pinned terminal CE public constructor did not reject {label} wire tamper"
        );
        builder.tamper_witness(col, original);
        assert!(
            builder.is_satisfied(),
            "restoring {label} wire should restore satisfiability"
        );
    }
}

#[test]
fn terminal_ce_public_pinned_constructor_rejects_child_order_and_count_mismatch() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let child_a = supported_terminal_child_fixture();
    let child_b = second_supported_terminal_child_fixture();
    let expected = TerminalCePublic::from_terminal_children(params, structure, &[child_a.clone(), child_b.clone()])
        .expect("supported terminal children");

    let honest = enforce_terminal_ce_public_pinned_against(&prep, &[child_a.clone(), child_b.clone()], &expected)
        .expect("honest child order should synthesize");
    assert!(
        honest.builder.is_satisfied(),
        "expected child order must satisfy the pinned terminal public statement"
    );

    let swapped = enforce_terminal_ce_public_pinned_against(&prep, &[child_b.clone(), child_a.clone()], &expected)
        .expect("swapped child order should synthesize but violate the pinned digest");
    assert!(
        !swapped.builder.is_satisfied(),
        "swapped terminal children must not satisfy the pinned terminal public statement"
    );

    let err = enforce_terminal_ce_public_pinned_against(&prep, &[child_a], &expected)
        .err()
        .expect("terminal child count mismatch must fail closed");
    assert!(
        err.contains("claim_count mismatch"),
        "expected claim_count mismatch, got: {err}"
    );
}

#[test]
fn terminal_ce_circuit_verifier_rejects_until_real_proof_verifier_exists() {
    let prep = support::toy_preprocessing();
    let child = supported_terminal_child_fixture();
    let proof = proof_for_children(&prep.params, prep.structure(), &[child.clone()], Vec::new());

    let (builder, result) = enforce_terminal_ce_verify_from_children_against(&prep, &[child], &proof);
    let err = result.expect_err("digest-only or empty terminal CE proof must not verify");
    assert_eq!(err, TerminalCeVerifyError::Unsupported);
    assert!(
        builder.is_satisfied(),
        "fail-closed unsupported verifier must not emit unsatisfied placeholder rows"
    );
}

#[test]
fn terminal_ce_circuit_verifier_does_not_pseudo_verify_proof_bytes() {
    let prep = support::toy_preprocessing();
    let child = supported_terminal_child_fixture();
    let proof_a = proof_for_children(&prep.params, prep.structure(), &[child.clone()], vec![1, 2, 3, 4]);
    let proof_b = proof_for_children(&prep.params, prep.structure(), &[child.clone()], vec![9, 8, 7, 6, 5]);

    let (builder_a, result_a) = enforce_terminal_ce_verify_from_children_against(&prep, &[child.clone()], &proof_a);
    let (builder_b, result_b) = enforce_terminal_ce_verify_from_children_against(&prep, &[child], &proof_b);

    assert_eq!(
        result_a.expect_err("fake terminal CE proof bytes must fail closed"),
        TerminalCeVerifyError::Unsupported
    );
    assert_eq!(
        result_b.expect_err("different fake terminal CE proof bytes must fail closed"),
        TerminalCeVerifyError::Unsupported
    );
    assert!(
        builder_a.is_satisfied() && builder_b.is_satisfied(),
        "unsupported verifier should only emit public-statement binding rows today"
    );
    assert_eq!(
        builder_a.rows(),
        builder_b.rows(),
        "unsupported verifier must not emit byte-dependent verifier rows before a real proof backend exists"
    );
    assert_eq!(
        builder_a.witness().len(),
        builder_b.witness().len(),
        "unsupported verifier must not allocate byte-dependent witness columns before a real proof backend exists"
    );
}

#[test]
fn terminal_ce_circuit_verifier_rejects_same_shape_child_with_fake_proof_bytes() {
    let prep = support::toy_preprocessing();
    let params = &prep.params;
    let structure = prep.structure();
    let mut child = supported_terminal_child_fixture();
    child.r[0] += K::ONE;
    let _public = TerminalCePublic::from_terminal_children(params, structure, &[child.clone()])
        .expect("same-shape terminal child should still form a public statement");
    let proof = proof_for_children(params, structure, &[child.clone()], vec![1, 2, 3, 4]);

    let (builder, result) = enforce_terminal_ce_verify_from_children_against(&prep, &[child], &proof);
    let err = result.expect_err("well-shaped terminal children plus arbitrary proof bytes must not verify");
    assert_eq!(err, TerminalCeVerifyError::Unsupported);
    assert!(
        builder.is_satisfied(),
        "public-statement rows should satisfy; rejection must come from fail-closed proof verification, \
         not from relying on shape errors"
    );
}

#[test]
fn terminal_ce_circuit_verifier_constrains_proof_public_digest_to_recomputed_statement() {
    let prep = support::toy_preprocessing();
    let child = supported_terminal_child_fixture();
    let public = TerminalCePublic::from_terminal_children(&prep.params, prep.structure(), &[child.clone()])
        .expect("terminal CE public");
    let mut wrong_digest = public.digest();
    wrong_digest[0] += F::ONE;
    let proof = TerminalCeProof::new_unchecked(wrong_digest, vec![1, 2, 3, 4]);

    let (builder, result) = enforce_terminal_ce_verify_from_children_against(&prep, &[child], &proof);
    let err = result.expect_err("unsupported verifier still fails closed");
    assert_eq!(err, TerminalCeVerifyError::Unsupported);
    assert!(
        !builder.is_satisfied(),
        "compact terminal CE verifier seam accepted proof material bound to the wrong \
         terminal public digest. The proof's declared public digest must equal the digest \
        recomputed from NIFS terminal-child wires before any backend-specific verifier can run."
    );
}

#[test]
fn terminal_ce_circuit_verifier_rejects_proof_for_different_same_shape_children() {
    let prep = support::toy_preprocessing();
    let child = supported_terminal_child_fixture();
    let proof = proof_for_children(&prep.params, prep.structure(), &[child.clone()], vec![1, 2, 3, 4]);

    let mut tampered_child = child;
    tampered_child.r[0] += K::ONE;

    let (builder, result) = enforce_terminal_ce_verify_from_children_against(&prep, &[tampered_child], &proof);
    let err = result.expect_err("unsupported verifier still fails closed");
    assert_eq!(err, TerminalCeVerifyError::Unsupported);
    assert!(
        !builder.is_satisfied(),
        "compact terminal CE verifier seam accepted proof material bound to a different \
         same-shape terminal child. The verifier must recompute the public digest from \
         the actual terminal-child wires, not trust the proof's declared statement."
    );
}

#[test]
fn terminal_ce_circuit_verifier_rejects_proof_for_same_shape_c1_limb_relabel() {
    let prep = support::toy_preprocessing();
    let child = supported_terminal_child_fixture();
    let proof = proof_for_children(&prep.params, prep.structure(), &[child.clone()], vec![1, 2, 3, 4]);

    let mut relabeled_child = child;
    relabeled_child.r[0] += k(0, 1);
    relabeled_child.y_ring[0][0] += k(0, 3);
    relabeled_child.ct[0] = relabeled_child.y_ring[0][0];

    let (builder, result) = enforce_terminal_ce_verify_from_children_against(&prep, &[relabeled_child], &proof);
    let err = result.expect_err("unsupported verifier still fails closed");
    assert_eq!(err, TerminalCeVerifyError::Unsupported);
    assert!(
        !builder.is_satisfied(),
        "compact terminal CE verifier seam accepted proof material after same-shape \
         K-extension c1 limb relabeling. The public statement must bind both limbs \
         of every terminal-child K field, not only c0."
    );
}

#[test]
fn terminal_ce_circuit_verifier_rejects_proof_for_different_structure_context() {
    let prep = support::toy_preprocessing();
    let child = supported_terminal_child_fixture();
    let changed_structure = changed_terminal_ce_structure_fixture();
    let changed_public = TerminalCePublic::from_terminal_children(&prep.params, &changed_structure, &[child.clone()])
        .expect("same child is well-shaped under changed terminal CE structure");
    let proof = TerminalCeProof::new_unchecked(changed_public.digest(), vec![1, 2, 3, 4]);

    let (builder, result) = enforce_terminal_ce_verify_from_children_against(&prep, &[child], &proof);
    let err = result.expect_err("unsupported verifier still fails closed");
    assert_eq!(err, TerminalCeVerifyError::Unsupported);
    assert!(
        !builder.is_satisfied(),
        "compact terminal CE verifier seam accepted proof material bound to a different \
         verifier-owned structure. The verifier must recompute the structure digest from \
         preprocessing, not accept the proof's declared context."
    );
}

#[test]
fn terminal_ce_circuit_verifier_rejects_proof_for_different_structure_polynomial() {
    let prep = support::toy_preprocessing();
    let child = supported_terminal_child_fixture();
    let changed_structure = changed_terminal_ce_polynomial_fixture();
    let changed_public = TerminalCePublic::from_terminal_children(&prep.params, &changed_structure, &[child.clone()])
        .expect("same child is well-shaped under changed terminal CE structure polynomial");
    let proof = TerminalCeProof::new_unchecked(changed_public.digest(), vec![1, 2, 3, 4]);

    let (builder, result) = enforce_terminal_ce_verify_from_children_against(&prep, &[child], &proof);
    let err = result.expect_err("unsupported verifier still fails closed");
    assert_eq!(err, TerminalCeVerifyError::Unsupported);
    assert!(
        !builder.is_satisfied(),
        "compact terminal CE verifier seam accepted proof material bound to a different \
         SuperNeo structure polynomial f. The verifier-owned structure digest must bind \
         the full CCS structure, not only matrix shape and entries."
    );
}

#[test]
fn terminal_ce_circuit_verifier_rejects_proof_for_different_params_context() {
    let prep = support::toy_preprocessing();
    let child = supported_terminal_child_fixture();
    let mut raw_params = NeoParams::goldilocks_paper_b2();
    raw_params.lambda -= 1;
    let changed_params = Params::test_only_from_neo_params(raw_params);
    let changed_public = TerminalCePublic::from_terminal_children(&changed_params, prep.structure(), &[child.clone()])
        .expect("same child is well-shaped under changed terminal CE params");
    let proof = TerminalCeProof::new_unchecked(changed_public.digest(), vec![1, 2, 3, 4]);

    let (builder, result) = enforce_terminal_ce_verify_from_children_against(&prep, &[child], &proof);
    let err = result.expect_err("unsupported verifier still fails closed");
    assert_eq!(err, TerminalCeVerifyError::Unsupported);
    assert!(
        !builder.is_satisfied(),
        "compact terminal CE verifier seam accepted proof material bound to a different \
         SuperNeo parameter digest. The verifier must recompute params from preprocessing, \
         not accept the proof's declared context."
    );
}

#[test]
fn terminal_ce_circuit_verifier_recomputes_public_statement_from_terminal_children_before_failing_closed() {
    let prep = support::toy_preprocessing();
    let mut child = supported_terminal_child_fixture();
    child.c.kappa += 1;
    let proof = TerminalCeProof::new_unchecked([F::ZERO; 4], Vec::new());

    let (builder, result) = enforce_terminal_ce_verify_from_children_against(&prep, &[child], &proof);
    let err = result.expect_err("malformed terminal children must be rejected before proof verification");
    match err {
        TerminalCeVerifyError::PublicStatement(msg) => {
            assert!(
                msg.contains("commitment kappa"),
                "unexpected public-statement error: {msg}"
            );
        }
        other => panic!("expected public-statement rejection, got {other:?}"),
    }
    assert!(
        builder.is_satisfied(),
        "shape rejection should fail closed before emitting placeholder unsatisfied rows"
    );
}

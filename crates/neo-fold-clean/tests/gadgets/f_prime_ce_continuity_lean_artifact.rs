//! Exact one-claim CE child/running continuity artifact and drift gate.

#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::engine::decider::__test_isolation::{
    enforce_ce_continuity_against_self, enforce_ce_continuity_between, CeContinuityProbeWires,
};
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::CeClaim;
use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrime/FPrimeCeContinuityArtifact.lean";

fn f(value: u64) -> F {
    F::from_u64(value)
}

fn k(c0: u64, c1: u64) -> K {
    K::from_coeffs([f(c0), f(c1)])
}

fn claim_fixture() -> CeClaim {
    let mut x = Mat::zero(D, 1, F::ZERO);
    for row in 0..D {
        x[(row, 0)] = f(100 + row as u64);
    }
    let kappa = Params::production().kappa() as usize;
    let d_pad = D.next_power_of_two();
    let mut y_ring = (0..d_pad)
        .map(|idx| k(11 + idx as u64 * 2, 12 + idx as u64 * 2))
        .collect::<Vec<_>>();
    let mut y_zcol = (0..d_pad)
        .map(|idx| k(211 + idx as u64 * 2, 212 + idx as u64 * 2))
        .collect::<Vec<_>>();
    for lane in D..d_pad {
        y_ring[lane] = K::ZERO;
        y_zcol[lane] = K::ZERO;
    }
    CeClaim {
        adv: None,
        c: Commitment {
            d: D,
            kappa,
            data: (0..D * kappa).map(|idx| f(10 + idx as u64)).collect(),
        },
        X: x,
        r: vec![k(1, 2)],
        s_col: vec![k(5, 6)],
        y_ring: vec![y_ring.clone()],
        ct: vec![y_ring[0]],
        aux_openings: Vec::new(),
        y_zcol,
        m_in: 1,
        fold_digest: [42u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn build() -> (R1csBuilder, CeContinuityProbeWires) {
    enforce_ce_continuity_against_self(&claim_fixture()).expect("emit CE continuity")
}

fn equality_pairs(builder: &R1csBuilder) -> Vec<(usize, usize)> {
    let (a, b, c) = builder.sparse_triplets();
    (0..builder.rows())
        .filter_map(|row| {
            let a_terms = a
                .iter()
                .filter(|&&(candidate, _, _)| candidate == row)
                .map(|&(_, column, coefficient)| (column, coefficient))
                .collect::<Vec<_>>();
            let b_terms = b
                .iter()
                .filter(|&&(candidate, _, _)| candidate == row)
                .map(|&(_, column, coefficient)| (column, coefficient))
                .collect::<Vec<_>>();
            assert_eq!(b_terms, vec![(0, F::ONE)]);
            assert!(c.iter().all(|&(candidate, _, _)| candidate != row));
            match a_terms.as_slice() {
                // Shape metadata is allocated through `alloc_usize`, which
                // emits a verifier-owned constant pin before continuity.
                [(_, coefficient)] => {
                    assert_eq!(*coefficient, F::ONE);
                    None
                }
                [(left, left_coefficient), (right, right_coefficient)] if *right == 0 => {
                    assert_eq!(*left_coefficient, F::ONE);
                    assert_ne!(*right_coefficient, F::ZERO);
                    None
                }
                [(left, left_coefficient), (right, right_coefficient)] => {
                    assert_eq!(*left_coefficient, F::ONE);
                    assert_eq!(*right_coefficient, -F::ONE);
                    Some((*left, *right))
                }
                _ => panic!("CE continuity row {row} is neither a constant pin nor direct equality"),
            }
        })
        .collect()
}

fn lean_pairs(pairs: &[(usize, usize)]) -> String {
    format!(
        "[{}]",
        pairs
            .iter()
            .map(|&(left, right)| format!("({left}, {right})"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn pair_runs(pairs: &[(usize, usize)]) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < pairs.len() {
        let mut end = start + 1;
        while end < pairs.len() && pairs[end].0 == pairs[end - 1].0 + 1 && pairs[end].1 == pairs[end - 1].1 + 1 {
            end += 1;
        }
        runs.push((pairs[start].0, pairs[start].1, end - start));
        start = end;
    }
    runs
}

fn lean_runs(runs: &[(usize, usize, usize)]) -> String {
    format!(
        "[{}]",
        runs.iter()
            .map(|&(left, right, count)| format!("⟨{left}, {right}, {count}⟩"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn artifact_hashes(builder: &R1csBuilder, forged: &[F], pairs: &[(usize, usize)]) -> (String, String) {
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/f-prime-ce-continuity\n\
         source=enforce_child_core_equal_running\npairs={}\nleft_cols={}\nrows={}\ncols={}\n{}",
        lean_pairs(pairs),
        lean_nat_list(pairs.iter().map(|pair| pair.0)),
        builder.rows(),
        builder.cols(),
        lean_rows(builder),
    );
    let witness_payload = format!(
        "{}\n{}",
        lean_witness("honestWitness", builder.witness()),
        lean_witness("forgedWitness", forged),
    );
    (sha256_hex(&row_payload), sha256_hex(&witness_payload))
}

#[test]
fn ce_continuity_accepts_honest_and_has_only_direct_equalities() {
    let (builder, _) = build();
    let pairs = equality_pairs(&builder);
    assert!(!pairs.is_empty());
    assert!(pairs.len() < builder.rows(), "shape metadata pins must remain explicit");
    assert!(builder.unconstrained_columns().is_empty());
    assert!(builder.is_satisfied());
}

#[test]
fn ce_continuity_rejects_each_authority_family() {
    let selectors: [fn(&CeContinuityProbeWires) -> usize; 14] = [
        |w| w.c_data0.col(),
        |w| w.x0.col(),
        |w| w.c_d.col(),
        |w| w.c_kappa.col(),
        |w| w.x_rows.col(),
        |w| w.x_cols.col(),
        |w| w.m_in.col(),
        |w| w.r_c0.col(),
        |w| w.r_c1.col(),
        |w| w.s_col_c0.col(),
        |w| w.s_col_c1.col(),
        |w| w.ct_c1.col(),
        |w| w.y_ring_c1.col(),
        |w| w.fold_digest0.col(),
    ];
    for select in selectors {
        let (mut builder, probes) = build();
        let column = select(&probes);
        builder.tamper_witness(column, builder.witness()[column] + F::ONE);
        assert!(!builder.is_satisfied(), "CE continuity disconnected column {column}");
    }
}

#[test]
fn ce_continuity_does_not_read_child_or_running_y_zcol() {
    let claim = claim_fixture();
    let (baseline, _) = enforce_ce_continuity_between(&claim, &claim).expect("emit baseline");
    let baseline = baseline.snapshot();

    let mut child_mutation = claim.clone();
    child_mutation.y_zcol[0] += K::ONE;
    let (child, _) = enforce_ce_continuity_between(&child_mutation, &claim).expect("emit child mutation");
    let child = child.snapshot();
    assert!(baseline.has_same_relation(&child));
    assert_eq!(baseline.witness(), child.witness());

    let mut running_mutation = claim.clone();
    running_mutation.y_zcol[0] += K::ONE;
    let (running, _) = enforce_ce_continuity_between(&claim, &running_mutation).expect("emit running mutation");
    let running = running.snapshot();
    assert!(baseline.has_same_relation(&running));
    assert_eq!(baseline.witness(), running.witness());
}

#[test]
fn lean_ce_continuity_artifact_matches_committed_file() {
    let (honest, probes) = build();
    let pairs = equality_pairs(&honest);
    let (mut forged, _) = build();
    let column = probes.y_ring_c1.col();
    forged.tamper_witness(column, forged.witness()[column] + F::ONE);
    let (row_hash, witness_hash) = artifact_hashes(&honest, forged.witness(), &pairs);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), ARTIFACT_REL_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    let expected_rows = format!("def artifactSha256 : String := \"{row_hash}\"");
    let expected_witnesses = format!("def witnessSha256 : String := \"{witness_hash}\"");
    let expected_runs = format!("def pairRuns : List PairRun :=\n  {}", lean_runs(&pair_runs(&pairs)));
    let expected_row_count = format!("def rowCount : Nat := {}", honest.rows());
    let expected_col_count = format!("def colCount : Nat := {}", honest.cols());
    let compact = |value: &str| {
        value
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<String>()
    };
    let compact_committed = compact(&committed);
    if !committed.contains(&expected_rows)
        || !committed.contains(&expected_witnesses)
        || !compact_committed.contains(&compact(&expected_runs))
        || !committed.contains(&expected_row_count)
        || !committed.contains(&expected_col_count)
    {
        let expected_path = format!("{path}.expected");
        std::fs::write(
            &expected_path,
            format!(
                "{expected_rows}\n{expected_witnesses}\n{expected_row_count}\n{expected_col_count}\n{expected_runs}\n"
            ),
        )
        .expect("write .expected CE-continuity metadata");
        panic!("generated Lean CE-continuity artifact drifted. Wrote {expected_path}; inspect and copy it");
    }
}

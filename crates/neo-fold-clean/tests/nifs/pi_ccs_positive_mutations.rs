//! Mutations of a caller-supplied positive PiCCS proof.
//! The caller first checks the complete Lean and Rust positive results.

use std::sync::atomic::{AtomicUsize, Ordering};

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{from_complex, KExtensions, D, F, K};
use neo_params::NeoParams;
use neo_reductions::{
    engines::paper_exact_engine::paper_exact_verify_with_trace, optimized_engine::optimized_verify_with_trace,
    PiCcsProof,
};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

const MODULUS: u64 = 0xffff_ffff_0000_0001;

pub fn check_proof_mutations(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &CcsClaim<Commitment, F>,
    running: &[CeClaim<Commitment, F, K>],
    outputs: &[CeClaim<Commitment, F, K>],
    proof: &PiCcsProof,
) {
    let rejects = |changed: &PiCcsProof, label: &str| {
        let mut paper_transcript = Poseidon2Transcript::new_v1_1();
        let paper = paper_exact_verify_with_trace(
            &mut paper_transcript,
            params,
            structure,
            std::slice::from_ref(fresh),
            running,
            outputs,
            changed,
        );
        assert!(!matches!(paper, Ok((true, _))), "PaperExact accepted {label}");
        let mut optimized_transcript = Poseidon2Transcript::new_v1_1();
        let optimized = optimized_verify_with_trace(
            &mut optimized_transcript,
            params,
            structure,
            std::slice::from_ref(fresh),
            running,
            outputs,
            changed,
        );
        assert!(!matches!(optimized, Ok((true, _))), "optimized accepted {label}");
    };
    assert_eq!(proof.sumcheck_rounds.len(), 28);
    assert!(proof.sumcheck_rounds.iter().all(|round| round.len() == 10));
    let mut changed = proof.clone();
    changed.sumcheck_rounds.pop();
    rejects(&changed, "missing round");
    let mut changed = proof.clone();
    changed.sumcheck_rounds[0].pop();
    rejects(&changed, "missing round coefficient");
    let mut checked = 2;
    for round in 0..proof.sumcheck_rounds.len() {
        for coefficient in 0..proof.sumcheck_rounds[round].len() {
            for limb in 0..2 {
                let mut changed = proof.clone();
                let mut words: [u64; 2] = changed.sumcheck_rounds[round][coefficient]
                    .to_limbs_u64()
                    .into();
                words[limb] = if words[limb] == MODULUS - 1 { 0 } else { words[limb] + 1 };
                changed.sumcheck_rounds[round][coefficient] =
                    from_complex(F::from_u64(words[0]), F::from_u64(words[1]));
                rejects(
                    &changed,
                    &format!("round {round}, coefficient {coefficient}, limb {limb}"),
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 2 + 28 * 10 * 2);
    println!("positive_pi_ccs_proof_mutations_rejected={checked} engines=paper_exact,optimized");
}

fn change_digest(digest: &mut [u8; 32], lane: usize) {
    let start = lane * 8;
    let word = u64::from_le_bytes(digest[start..start + 8].try_into().expect("digest lane"));
    let next = if word == MODULUS - 1 { 0 } else { word + 1 };
    digest[start..start + 8].copy_from_slice(&next.to_le_bytes());
}

pub fn check_claim_mutations(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh: &CcsClaim<Commitment, F>,
    running: &[CeClaim<Commitment, F, K>],
    outputs: &[CeClaim<Commitment, F, K>],
    proof: &PiCcsProof,
    group: &str,
) {
    assert_eq!(running.len(), 16);
    assert_eq!(outputs.len(), 17);
    let checked = AtomicUsize::new(0);
    let rejects = |fresh: &CcsClaim<Commitment, F>,
                   running: &[CeClaim<Commitment, F, K>],
                   outputs: &[CeClaim<Commitment, F, K>],
                   label: &str| {
        let mut paper_transcript = Poseidon2Transcript::new_v1_1();
        let paper = paper_exact_verify_with_trace(
            &mut paper_transcript,
            params,
            structure,
            std::slice::from_ref(fresh),
            running,
            outputs,
            proof,
        );
        assert!(!matches!(paper, Ok((true, _))), "PaperExact accepted {label}");
        let mut optimized_transcript = Poseidon2Transcript::new_v1_1();
        let optimized = optimized_verify_with_trace(
            &mut optimized_transcript,
            params,
            structure,
            std::slice::from_ref(fresh),
            running,
            outputs,
            proof,
        );
        assert!(!matches!(optimized, Ok((true, _))), "optimized accepted {label}");
        checked.fetch_add(1, Ordering::Relaxed);
    };

    match group {
        "point-mutations" => {
            // This group is for the nonzero-running fixture. Zero openings
            // stay valid at every common point and use the parent hash gate.
            assert!(running
                .iter()
                .any(|claim| claim.eval_k.iter().any(|&value| value != K::ZERO)));
            for coordinate in 0..28 {
                for limb in 0..2 {
                    let mut changed = running.to_vec();
                    let mut words: [u64; 2] = changed[0].r[coordinate].to_limbs_u64().into();
                    words[limb] = if words[limb] == MODULUS - 1 { 0 } else { words[limb] + 1 };
                    let point = from_complex(F::from_u64(words[0]), F::from_u64(words[1]));
                    for claim in &mut changed {
                        claim.r[coordinate] = point;
                    }
                    rejects(
                        fresh,
                        &changed,
                        outputs,
                        &format!("common prior point {coordinate}, limb {limb}"),
                    );
                }
            }
            assert_eq!(checked.load(Ordering::Relaxed), 28 * 2);
        }
        "statement-mutations" => {
            // Zero running openings remain valid at every common point.
            // The parent pilot hashes that point; its preimage-mutation gate
            // owns rejection of a changed point with the old public digest.
            for lane in 0..4 {
                let mut changed = running.to_vec();
                for claim in &mut changed {
                    change_digest(&mut claim.fold_digest, lane);
                }
                rejects(fresh, &changed, outputs, &format!("shared prior-digest {lane}"));
            }
            let mut changed = running.to_vec();
            changed[1].r[0] += K::ONE;
            rejects(fresh, &changed, outputs, "one inconsistent prior-point");
            let mut changed = running.to_vec();
            change_digest(&mut changed[1].fold_digest, 0);
            rejects(fresh, &changed, outputs, "one inconsistent prior-digest");
            let mut changed = running.to_vec();
            changed[0].eval_k[D] = K::ONE;
            rejects(fresh, &changed, outputs, "running Eval_K nonzero padding");
            let mut changed = running.to_vec();
            changed[0].eval_a[0][D] = K::ONE;
            rejects(fresh, &changed, outputs, "running Eval_A nonzero padding");
            for source in 0..running.len() {
                for family in 0..17 {
                    let mut changed = running.to_vec();
                    match family {
                        0 => changed[source].c.data[0] += F::ONE,
                        1 => changed[source].X[(0, 0)] += F::ONE,
                        2 => changed[source].eval_k[0] += K::ONE,
                        _ => changed[source].eval_a[family - 3][0] += K::ONE,
                    }
                    rejects(
                        fresh,
                        &changed,
                        outputs,
                        &format!("running source {source}, family {family}"),
                    );
                }
            }
            let mut changed = fresh.clone();
            changed.c.data[0] += F::ONE;
            rejects(&changed, running, outputs, "fresh commitment");
            let mut changed = fresh.clone();
            changed.x[0] += F::ONE;
            rejects(&changed, running, outputs, "fresh public input");
            assert_eq!(checked.load(Ordering::Relaxed), 4 + 4 + 16 * 17 + 2);
        }
        "output-mutations" => {
            (0..outputs.len()).into_par_iter().for_each(|source| {
                for family in 0..17 {
                    let mut changed = outputs.to_vec();
                    match family {
                        0 => changed[source].c.data[0] += F::ONE,
                        1 => changed[source].X[(0, 0)] += F::ONE,
                        2 => changed[source].eval_k[0] += K::ONE,
                        _ => changed[source].eval_a[family - 3][0] += K::ONE,
                    }
                    rejects(
                        fresh,
                        running,
                        &changed,
                        &format!("output source {source}, family {family}"),
                    );
                }
                for coordinate in 0..28 {
                    let mut changed = outputs.to_vec();
                    changed[source].r[coordinate] += K::ONE;
                    rejects(
                        fresh,
                        running,
                        &changed,
                        &format!("output source {source}, point {coordinate}"),
                    );
                }
                for lane in 0..4 {
                    let mut changed = outputs.to_vec();
                    change_digest(&mut changed[source].fold_digest, lane);
                    rejects(
                        fresh,
                        running,
                        &changed,
                        &format!("output source {source}, digest {lane}"),
                    );
                }
            });
            for shape in 0..10 {
                let mut changed = outputs.to_vec();
                match shape {
                    0 => {
                        changed.pop();
                    }
                    1 => {
                        changed[0].c.data.pop();
                    }
                    2 => {
                        changed[0].r.pop();
                    }
                    3 => {
                        changed[0].eval_k.pop();
                    }
                    4 => changed[0].eval_k[D] = K::ONE,
                    5 => {
                        changed[0].eval_a.pop();
                    }
                    6 => {
                        changed[0].eval_a[0].pop();
                    }
                    7 => changed[0].eval_a[0][D] = K::ONE,
                    8 => changed[0].X = neo_ccs::Mat::zero(D, 4, F::ZERO),
                    9 => changed[0].m_in -= 1,
                    _ => unreachable!(),
                }
                rejects(fresh, running, &changed, &format!("malformed output shape {shape}"));
            }
            assert_eq!(checked.load(Ordering::Relaxed), 17 * (17 + 28 + 4) + 10);
        }
        _ => panic!("unknown claim mutation group"),
    }
    println!(
        "positive_pi_ccs_{group}_rejected={} engines=paper_exact,optimized",
        checked.load(Ordering::Relaxed)
    );
}

//! Assemble saved D proof inputs and run the existing complete observation path.
//! Recomputed split values and commitments bind the digit witnesses. Saved
//! openings remain proof inputs; normal verification checks their public
//! relations. Their earlier computation is retained execution evidence.

use std::{path::Path, time::Instant};

use neo_ccs::{Mat, V1_1Evaluations};
use neo_fold_clean::paper::{
    construction2::RunningInstance, nifs::NifsProof, pi_ccs, pi_dec, pi_rlc, relations::ajtai_dec_mixer,
};
use neo_math::{D, F, K};
use neo_reductions::common::validate_superneo_witness_mat;
use nightstream_fprime::PI_DEC_V1_1_CHILD_COUNT;
use p3_field::PrimeCharacteristicRing;

use super::super::{stage1_actual::ActualBase, Observed};
use super::{commit_parent, load, load_verified_parent, CommittedSplit, SavedChildOpening, SavedSplit};

/// Read the exact ordered saved material, construct the normal D proof, and
/// reuse full NIFS verification, mutations and Observed conversion. This does
/// not recalculate saved openings or certify them as witness evaluations.
pub fn assemble(
    package_path: &Path,
    fixture_path: &Path,
    parent_dir: &Path,
    material_dir: &Path,
    opening_dir: &Path,
) -> Observed {
    let started = Instant::now();
    let (actual, saved) = load_verified_parent(package_path, fixture_path, parent_dir);
    let ActualBase {
        package,
        params,
        fresh,
        running,
    } = actual;
    let fresh_claim = fresh.claim;
    let prior = running.claims_only();
    drop((fresh.witness, running));
    println!(
        "actual_selected_assembly_parent_replayed_elapsed={:?}",
        started.elapsed()
    );

    let count = PI_DEC_V1_1_CHILD_COUNT;
    assert_eq!(params.k_rho() as usize, count);
    let metadata: SavedSplit = load(&material_dir.join("split.json"));
    assert_eq!(metadata.schema, 1);
    assert_eq!(metadata.parent, saved.rlc_parent, "same replayed R parent");
    assert_eq!(metadata.nonzero.len(), count, "complete ordered digit flags");
    assert_eq!(metadata.commitments.len(), count, "complete ordered digit commitments");

    let parent: Mat<F> = load(&parent_dir.join("parent-witness.json"));
    validate_superneo_witness_mat(&parent, package.structure().m).expect("selected R witness carrier");
    let CommittedSplit {
        digits: expected_digits,
        nonzero,
        commitments,
    } = commit_parent(&params, &parent, &saved.rlc_parent.c);
    drop(parent);
    assert_eq!(expected_digits.len(), count);
    assert_eq!(
        metadata.nonzero, nonzero,
        "activity recomputed from the exact parent split"
    );
    assert_eq!(
        metadata.commitments, commitments,
        "ordered fixed-key digit commitments recomputed"
    );
    println!(
        "actual_selected_assembly_split_committed_elapsed={:?}",
        started.elapsed()
    );

    let mut digits = Vec::with_capacity(count);
    let mut openings = Vec::with_capacity(count);
    for (child, expected_digit) in expected_digits.into_iter().enumerate() {
        let digit: Mat<F> = load(&material_dir.join(format!("digit-{child}.json")));
        validate_superneo_witness_mat(&digit, package.structure().m).expect("selected D digit carrier");
        assert!(digit == expected_digit, "saved digit equals the canonical parent split");
        drop(expected_digit);

        let opening = if nonzero[child] {
            let record: SavedChildOpening = load(&opening_dir.join(format!("child-{child}.json")));
            assert_eq!(record.schema, 1);
            assert_eq!(record.child, child, "exact child order");
            assert_eq!(
                record.parent, saved.rlc_parent,
                "child opening belongs to the replayed parent and point"
            );
            assert_eq!(
                record.commitment, commitments[child],
                "opening record uses the recomputed digit commitment"
            );
            assert_eq!(record.transcript_state, saved.transcript_state, "replayed C/R state");
            assert_eq!(
                record.transcript_absorbed, saved.transcript_absorbed,
                "replayed C/R sponge position"
            );
            assert_eq!(record.opening.eval_k.len(), D, "complete Pad coefficients");
            assert_eq!(
                record.opening.eval_a.len(),
                package.structure().t(),
                "complete matrix families"
            );
            assert!(
                record.opening.eval_a.iter().all(|family| family.len() == D),
                "complete matrix coefficients"
            );
            record.opening
        } else {
            // Exact zero follows from the recomputed split and Mat equality.
            // This is the same inactive-digit branch used by normal D.
            V1_1Evaluations {
                eval_k: vec![K::ZERO; D],
                eval_a: vec![vec![K::ZERO; D]; package.structure().t()],
            }
        };
        digits.push(digit);
        openings.push(opening);
    }

    let (children, d_proof) = pi_dec::prove_from_split_material(
        &params,
        package.structure(),
        None,
        None,
        ajtai_dec_mixer,
        &saved.rlc_parent,
        digits,
        nonzero,
        commitments,
        openings,
    )
    .expect("normal D construction and consistency checks on saved proof inputs");
    let next = RunningInstance::new(children.claims, children.witnesses, Some(saved.rlc_parent.clone()));
    let proof = NifsProof {
        pi_ccs: pi_ccs::Proof {
            sumcheck: saved.sumcheck,
            outputs: saved.ccs_outputs,
        },
        pi_rlc: pi_rlc::Proof {
            combined: saved.rlc_parent,
        },
        pi_dec: d_proof,
    };
    super::super::observe(&package, &params, &fresh_claim, &prior, next, proof, started)
}

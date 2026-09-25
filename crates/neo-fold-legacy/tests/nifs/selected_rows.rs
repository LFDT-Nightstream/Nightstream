//! Low-level resource dispatch and transcript replay; empty rows are not the selected application.

use neo_ajtai::nightstream_fprime_setup::{
    commit_production_signed_unit_matrix, PRODUCTION_CARRIER_WIDTH, PRODUCTION_MESSAGE_COLUMNS,
};
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly, V1_1Evaluations};
use neo_math::{D, F, K};
use neo_reductions::{
    common::split_b_matrix_k_with_nonzero_flags,
    superneo_eval::{SuperneoEvalCacheBuilder, SuperneoZBlocks},
};
use nightstream_fprime::{PI_CCS_V1_1_MATRIX_COUNT, PI_CCS_V1_1_SOURCE_COUNT, PI_DEC_V1_1_CHILD_COUNT};
use p3_field::PrimeCharacteristicRing;

use crate::engine::transcript::Transcript;
use crate::paper::{
    construction2::{LaneCommitmentMode, RunningInstance},
    nifs,
    params::Params,
    pi_dec,
    relations::{ajtai_dec_mixer, ajtai_rlc_mixer, CcsInstance},
};

#[test]
fn selected_rows_resources_preserve_normal_nifs_replay() {
    let started = std::time::Instant::now();
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    // Shape-only header: the row builder supplies all actual rows. Neither
    // this header nor its digest grants authority to the prover's row data.
    let structure = CcsStructure::new_verifier_artifact_header(
        1,
        PRODUCTION_CARRIER_WIDTH,
        PI_CCS_V1_1_MATRIX_COUNT,
        SparsePoly::new(PI_CCS_V1_1_MATRIX_COUNT, vec![]),
    )
    .unwrap();
    let mut builder = SuperneoEvalCacheBuilder::new(1, structure.m, structure.t()).unwrap();
    for matrix in 0..structure.t() {
        builder.push_row(matrix, 0, []).unwrap();
    }
    let cache = builder.finish().unwrap();
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree()).unwrap();

    let mut positive = vec![0_u64; columns];
    let mut negative = vec![0_u64; columns];
    positive[0] = 1;
    negative[columns - 1] = 1 << (D - 1);
    let witness = Mat::<F>::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap();
    drop((positive, negative));
    let mut public = vec![F::ZERO; D];
    public[0] = F::ONE;
    let fresh = CcsInstance {
        claim: CcsClaim {
            c: commit_production_signed_unit_matrix(&witness).unwrap(),
            x: public,
            m_in: D,
            adv: None,
        },
        witness: CcsWitness {
            w: Vec::new(),
            Z: witness,
        },
    };
    let fresh_claim = fresh.claim.clone();
    let running = RunningInstance::canonical_zero(&params, &structure, D, LaneCommitmentMode::Plain).unwrap();
    let running_public = running.claims_only();
    let mut parent_transcript = Transcript::session();
    let (parent_ccs, parent) = super::prove_parent_with_rows(
        &mut parent_transcript,
        &params,
        &structure,
        &cache,
        vec![fresh.clone()],
        running.clone(),
    )
    .unwrap();
    let (digits, flags) =
        split_b_matrix_k_with_nonzero_flags(&parent.witness, params.k_rho() as usize, params.b()).unwrap();
    assert!(flags.iter().any(|&active| active), "nonzero child is covered");
    assert!(flags.iter().any(|&active| !active), "zero child is covered");
    let commitments = digits
        .iter()
        .map(commit_production_signed_unit_matrix)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let openings = digits
        .iter()
        .zip(&flags)
        .map(|(digit, &nonzero)| {
            if nonzero {
                let blocks = SuperneoZBlocks::from_witness_mat(digit, structure.m).unwrap();
                let mut evaluated = cache
                    .eval_real_v1_1_openings(&parent.claim.r, std::slice::from_ref(&blocks))
                    .unwrap();
                assert_eq!(evaluated.len(), 1);
                evaluated.pop().unwrap()
            } else {
                V1_1Evaluations {
                    eval_k: vec![K::ZERO; D],
                    eval_a: vec![vec![K::ZERO; D]; structure.t()],
                }
            }
        })
        .collect::<Vec<_>>();
    let (staged_children, staged_d) = pi_dec::prove_from_split_material(
        &params,
        &structure,
        None,
        None,
        ajtai_dec_mixer,
        &parent.claim,
        digits,
        flags,
        commitments,
        openings.clone(),
    )
    .unwrap();
    let parent_claim = parent.claim;
    drop((parent.witness, parent.projection));
    let mut prover_transcript = Transcript::session();
    let (next, proof) = super::prove_owned_with_rows(
        &mut prover_transcript,
        &params,
        &structure,
        &cache,
        vec![fresh],
        running,
    )
    .unwrap();
    let mut verifier_transcript = Transcript::session();
    let verified = nifs::verify(
        &mut verifier_transcript,
        &params,
        &structure,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        &[fresh_claim],
        &running_public,
        &proof,
    )
    .unwrap();
    assert_eq!(next.claims, verified.claims);
    assert_eq!(staged_d, proof.pi_dec, "staged D preserves every normal child field");
    assert_eq!(staged_children.claims, next.claims);
    assert!(
        staged_children.witnesses == next.witnesses,
        "staged and normal D return identical digit witnesses"
    );
    for (opening, child) in openings.iter().zip(&proof.pi_dec.children) {
        assert_eq!(opening.eval_k.as_slice(), &child.eval_k[..D], "all 54 Pad coefficients");
        assert_eq!(opening.eval_a.len(), structure.t());
        for (expected, actual) in opening.eval_a.iter().zip(&child.eval_a) {
            assert_eq!(expected.as_slice(), &actual[..D], "all 54 coefficients of every matrix");
        }
    }
    assert_eq!(next.parent_authority, verified.parent_authority);
    assert_eq!(next.parent_authority, Some(proof.pi_rlc.combined.clone()));
    assert_eq!(next.claims.len(), PI_DEC_V1_1_CHILD_COUNT);
    assert_eq!(next.witnesses.len(), PI_DEC_V1_1_CHILD_COUNT);
    assert_eq!(proof.pi_ccs.outputs.len(), PI_CCS_V1_1_SOURCE_COUNT);
    assert_eq!(prover_transcript.snapshot(), verifier_transcript.snapshot());
    assert_eq!(parent_ccs, proof.pi_ccs, "staged prefix preserves the actual C proof");
    assert_eq!(
        parent_claim, proof.pi_rlc.combined,
        "staged prefix preserves the actual R parent"
    );
    assert_eq!(
        parent_transcript.snapshot(),
        prover_transcript.snapshot(),
        "D does not change the C/R transcript"
    );
    println!("selected_rows_normal_nifs_elapsed={:?}", started.elapsed());
}

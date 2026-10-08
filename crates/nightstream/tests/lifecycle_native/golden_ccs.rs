//! Compare all PiCCS fields and mutation groups with the current optimized verifier.

use super::*;
use crate::folding::{pi_ccs, Params};
use crate::lifecycle::tests::{claim, commitment, extensions, fields as input_fields, frame};
use neo_reductions::optimized_engine::optimized_verify_with_trace;
use neo_transcript::Poseidon2Transcript;

#[path = "golden_ccs_mutations.rs"]
mod mutations;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Check {
    Accept,
    ProofMutations,
    StatementMutations,
    OutputMutations,
    PointMutations,
}

pub(super) fn check(package_path: &Path, identity: [u64; 4], input_path: &Path, lean_path: &Path, check: Check) {
    let bytes = fs::read(package_path).expect("candidate package");
    let package =
        nightstream_fprime::load_per_application_package(&bytes, identity).expect("candidate structural identity");
    let structure = package.ccs_structure_header().unwrap();
    drop((package, bytes));
    let level = neo_params::NeoParams::nightstream_goldilocks_k16()
        .padded_row_security_summary_for_shape(
            structure.n,
            structure.m,
            structure.t(),
            structure.max_degree(),
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )
        .unwrap()
        .security_bits;
    // Use the fixture's derived level; this checker sets no application policy.
    let params = Params::for_ccs_shape(
        structure.domain_rows(),
        structure.m,
        structure.t(),
        structure.max_degree(),
        level,
    )
    .unwrap();
    let input = read(input_path.to_path_buf());
    let lean = read(lean_path.to_path_buf());
    assert_eq!(input.as_array().unwrap().len(), 7);
    assert_eq!(input[0], 2);
    assert_eq!(lean.as_array().unwrap().len(), 6);
    assert_eq!(lean[0], 1);
    assert_eq!(lean[1], input, "complete PiCCS input");
    assert_eq!(lean[5].as_array().unwrap().len(), 15);
    assert_eq!(lean[5][0], 1, "positive Lean result required before mutations");
    let fresh = CcsClaim {
        c: commitment(&input[1]),
        x: input_fields(&input[2]),
        m_in: 270,
        adv: None,
    };
    assert_eq!(fresh.x.len(), 270);
    assert_eq!(fresh.x[0], F::ONE);
    assert!(fresh.x[257..].iter().all(|&value| value == F::ZERO));
    let prior_digest: [u64; 4] = std::array::from_fn(|lane| {
        let word = (0..64).fold(0u64, |word, bit| {
            let digit = fresh.x[1 + lane * 64 + bit].as_canonical_u64();
            assert!(digit <= 1);
            word | (digit << bit)
        });
        assert!(word < F::ORDER_U64);
        word
    });
    let running: Vec<_> = (0..16)
        .map(|source| {
            claim(
                &json!([
                    input[6][1][source],
                    input[6][2][source],
                    input[6][0],
                    input[6][3][source],
                    input[6][4][source]
                ]),
                frame(prior_digest),
            )
        })
        .collect();
    let outgoing = input_fields(&lean[5][14]);
    assert_eq!(outgoing.len(), 8);
    let digest = frame(std::array::from_fn(|lane| outgoing[lane].as_canonical_u64()));
    let outputs = (0..17)
        .map(|source| {
            let (commitment, public) = if source == 0 {
                (&input[1], &input[2])
            } else {
                (&input[6][1][source - 1], &input[6][2][source - 1])
            };
            claim(
                &json!([commitment, public, lean[5][6], input[4][source], input[5][source]]),
                digest,
            )
        })
        .collect();
    let proof = pi_ccs::Proof {
        sumcheck: pi_ccs::SumcheckProof::new(
            input[3]
                .as_array()
                .unwrap()
                .iter()
                .map(extensions)
                .collect(),
        ),
        outputs,
    };
    assert_eq!(
        ccs_input(&fresh, &running, &proof),
        input,
        "complete native input decoding"
    );
    assert_eq!(running_value(&running), lean[2], "complete running statement");
    assert_eq!(
        lean[3],
        json!([prior_digest, input[1], input[2]]),
        "public transcript blocks"
    );
    let mut claimed = Vec::new();
    for coefficient in 0..D {
        for source in 0..16 {
            claimed.extend(words(&[running[source].eval_k[coefficient]])[0]);
        }
    }
    for coefficient in 0..D {
        for matrix in 0..4 {
            for source in 0..16 {
                claimed.extend(words(&[running[source].eval_a[matrix][coefficient]])[0]);
            }
        }
    }
    assert_eq!(
        lean[4],
        json!([
            input[6][0]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|pair| pair.as_array().unwrap())
                .collect::<Vec<_>>(),
            claimed
        ]),
        "complete verifier evaluation blocks"
    );
    let (accepted, trace) = optimized_verify_with_trace(
        &mut Poseidon2Transcript::new_v1_1(),
        params.inner(),
        &structure,
        std::slice::from_ref(&fresh),
        &running,
        &proof.outputs,
        &proof.sumcheck,
    )
    .expect("complete current verifier trace");
    assert!(accepted);
    assert_eq!(
        ccs_phase(&proof, &trace),
        lean[5],
        "all 15 Lean/native PiCCS result fields"
    );
    println!("pi_ccs_complete_phase_values=passed accepted=true engine=optimized");
    match check {
        Check::Accept => (),
        Check::ProofMutations => mutations::check_proof_mutations(
            params.inner(),
            &structure,
            &fresh,
            &running,
            &proof.outputs,
            &proof.sumcheck,
        ),
        check => {
            let group = match check {
                Check::StatementMutations => "statement-mutations",
                Check::OutputMutations => "output-mutations",
                Check::PointMutations => "point-mutations",
                _ => unreachable!(),
            };
            mutations::check_claim_mutations(
                params.inner(),
                &structure,
                &fresh,
                &running,
                &proof.outputs,
                &proof.sumcheck,
                group,
            );
        }
    }
}

//! Rejects noncanonical public outputs for the selected padded-row PiCCS protocol.

use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule, Commitment};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly, Term};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::optimized_engine::PiCcsProof;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn identity_left(n: usize, m: usize) -> Mat<F> {
    let mut matrix = Mat::zero(n, m, F::ZERO);
    for index in 0..n.min(m) {
        matrix.set(index, index, F::ONE);
    }
    matrix
}

type OutputClaim = CeClaim<Commitment, F, K>;

struct HonestProof {
    label: &'static [u8],
    params: NeoParams,
    structure: CcsStructure<F>,
    claim: CcsClaim<Commitment, F>,
    outputs: Vec<OutputClaim>,
    proof: PiCcsProof,
}

fn honest_proof(label: &'static [u8]) -> HonestProof {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(4).expect("params");
    let structure = CcsStructure::new(
        vec![identity_left(4, D)],
        SparsePoly::new(
            1,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1],
            }],
        ),
    )
    .expect("structure");

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(983);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, structure.m / D).expect("Ajtai setup");
    let commitment_scheme = AjtaiSModule::new(Arc::new(pp));
    let z = Mat::from_row_major(D, structure.m / D, vec![F::ZERO; structure.m]);
    let claim = CcsClaim {
        c: commitment_scheme.commit(&z),
        x: vec![],
        m_in: 0,
        adv: None,
    };
    let witness = CcsWitness {
        w: vec![F::ZERO; structure.m],
        Z: z,
    };

    let mut prover_transcript = Poseidon2Transcript::new(label);
    let (outputs, proof) = neo_reductions::api::prove(
        FoldingMode::Optimized,
        &mut prover_transcript,
        &params,
        &structure,
        core::slice::from_ref(&claim),
        core::slice::from_ref(&witness),
        &[],
        &[],
        &commitment_scheme,
    )
    .expect("honest proof");

    HonestProof {
        label,
        params,
        structure,
        claim,
        outputs,
        proof,
    }
}

fn raw_accepts(fixture: &HonestProof, outputs: &[OutputClaim], proof: &PiCcsProof) -> bool {
    let mut verifier_transcript = Poseidon2Transcript::new(fixture.label);
    neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut verifier_transcript,
        &fixture.params,
        &fixture.structure,
        core::slice::from_ref(&fixture.claim),
        &[],
        outputs,
        proof,
    )
    .unwrap_or(false)
}

#[test]
fn raw_pi_ccs_rejects_nonzero_fresh_output_eval_k_padding() {
    let fixture = honest_proof(b"redteam/raw-pi-ccs/eval-k-padding");
    let mut malformed = fixture.outputs.clone();
    assert!(D < malformed[0].eval_k.len(), "fixture must expose Eval_K padding");
    malformed[0].eval_k[D] = K::ONE;

    assert!(
        !raw_accepts(&fixture, &malformed, &fixture.proof),
        "raw PiCCS accepted a nonzero fresh-output Eval_K padding lane"
    );
}

#[test]
fn raw_pi_ccs_rejects_noncanonical_output_widths() {
    let fixture = honest_proof(b"redteam/raw-pi-ccs/output-widths");
    assert!(
        raw_accepts(&fixture, &fixture.outputs, &fixture.proof),
        "baseline proof must verify"
    );

    let mut extra_matrix = fixture.outputs.clone();
    extra_matrix[0]
        .eval_a
        .push(vec![K::ZERO; D.next_power_of_two()]);
    assert!(
        !raw_accepts(&fixture, &extra_matrix, &fixture.proof),
        "raw PiCCS accepted an extra Eval_A matrix"
    );

    let mut extra_pad_coordinate = fixture.outputs.clone();
    extra_pad_coordinate[0].eval_k.push(K::ZERO);
    assert!(
        !raw_accepts(&fixture, &extra_pad_coordinate, &fixture.proof),
        "raw PiCCS accepted an extra Eval_K coordinate"
    );
}

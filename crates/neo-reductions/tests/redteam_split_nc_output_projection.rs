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
fn raw_pi_ccs_rejects_unsupported_output_sidecars() {
    let fixture = honest_proof(b"redteam/raw-pi-ccs/output-sidecars");
    assert!(
        raw_accepts(&fixture, &fixture.outputs, &fixture.proof),
        "baseline proof must verify"
    );

    let mutations: [(&str, fn(&mut OutputClaim)); 4] = [
        ("aux_openings", |output| output.aux_openings.push(K::ONE)),
        ("c_step_coords", |output| output.c_step_coords.push(F::ONE)),
        ("u_offset", |output| output.u_offset = 1),
        ("u_len", |output| output.u_len = 1),
    ];
    let mut accepted = Vec::new();

    for (name, mutate) in mutations {
        let mut malformed = fixture.outputs.clone();
        mutate(&mut malformed[0]);
        if raw_accepts(&fixture, &malformed, &fixture.proof) {
            accepted.push(name);
        }
    }

    assert!(
        accepted.is_empty(),
        "raw Pi_CCS accepted unsupported output sidecars: {accepted:?}"
    );
}

#[test]
fn raw_pi_ccs_rejects_nonzero_fresh_output_y_ring_padding() {
    let fixture = honest_proof(b"redteam/raw-pi-ccs/y-ring-padding");
    let mut malformed = fixture.outputs.clone();
    assert!(D < malformed[0].y_ring[0].len(), "fixture must expose padding");
    malformed[0].y_ring[0][D] = K::ONE;

    assert!(
        !raw_accepts(&fixture, &malformed, &fixture.proof),
        "raw Pi_CCS accepted a nonzero fresh-output y_ring padding lane"
    );
}

//! The PiCCS engine checks every running claim before its transcript: the
//! commitment payload must match its declared shape, and auxiliary lane
//! commitments are rejected until a reduction binds them.

use neo_ajtai::{commit_row_major, setup, Commitment};
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, LaneCommitments, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::{pi_ccs_prove, pi_ccs_verify, PiCcsProof};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

mod zero_running;

type Output = CeClaim<Commitment, F, K>;

struct Fixture {
    params: NeoParams,
    structure: CcsStructure<F>,
    fresh: CcsClaim<Commitment, F>,
    running: Vec<Output>,
    outputs: Vec<Output>,
    proof: PiCcsProof,
}

fn transcript() -> Poseidon2Transcript {
    Poseidon2Transcript::new_v1_2()
}

fn fixture() -> Fixture {
    let structure = CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).unwrap();
    let params = NeoParams::goldilocks_auto_rectangular_r1cs_ccs(D, D).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0x0072_756e_6e69_6e67);
    let committer = setup(&mut rng, D, params.kappa() as usize, 1).unwrap();
    let zero = Mat::zero(D, 1, F::ZERO);
    let fresh = CcsClaim {
        c: commit_row_major(&committer, &zero),
        x: vec![F::ZERO; D],
        m_in: D,
        adv: None,
    };
    let witness = CcsWitness { Z: zero };
    let (running, running_witnesses) = zero_running::zero_running(&params, &structure, 1, D);
    let (outputs, proof) = pi_ccs_prove(
        &mut transcript(),
        &params,
        &structure,
        std::slice::from_ref(&fresh),
        std::slice::from_ref(&witness),
        &running,
        &running_witnesses,
    )
    .unwrap();
    Fixture {
        params,
        structure,
        fresh,
        running,
        outputs,
        proof,
    }
}

fn accepts(fixture: &Fixture, running: &[Output], outputs: &[Output]) -> bool {
    matches!(
        pi_ccs_verify(
            &mut transcript(),
            &fixture.params,
            &fixture.structure,
            std::slice::from_ref(&fixture.fresh),
            running,
            outputs,
            &fixture.proof,
        ),
        Ok(true)
    )
}

#[test]
fn pi_ccs_rejects_auxiliary_lane_commitments() {
    let fixture = fixture();
    let lanes = LaneCommitments {
        ops: Commitment::zeros(fixture.params.kappa() as usize),
        is: Commitment::zeros(fixture.params.kappa() as usize),
        fs: Commitment::zeros(fixture.params.kappa() as usize),
    };
    let mut running = fixture.running.clone();
    let mut outputs = fixture.outputs.clone();
    running[0].adv = Some(lanes);
    outputs[1].adv = running[0].adv.clone();
    assert!(
        !accepts(&fixture, &running, &outputs),
        "PiCCS accepted auxiliary lane commitments that no reduction binds"
    );
}

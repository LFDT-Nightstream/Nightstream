use neo_ajtai::Commitment;
use neo_ccs::{Mat, SparsePoly};
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use super::{
    enforce_nifs_v_circuit_with_transcript_and_header_bundle, Error, NifsVCircuitConfig, NifsVCircuitMessages,
};
use crate::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget, Var};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs;
use crate::paper::reductions::pi_ccs_circuit::{PiCcsVerifierConfig, PiCcsVerifierRelation};
use crate::paper::relations::CeClaim;

#[test]
fn native_header_entry_rejects_before_emission() {
    let params = Params::production();
    let config = NifsVCircuitConfig {
        pi_ccs: PiCcsVerifierConfig {
            params: &params,
            structure: PiCcsVerifierRelation::from_parts(1, D, SparsePoly::new(14, Vec::new())),
            matrix_digest: [F::ZERO; 4],
        },
    };
    let proof = pi_ccs::Proof {
        sumcheck: pi_ccs::SumcheckProof::new(Vec::new()),
        outputs: Vec::new(),
    };
    let combined = CeClaim {
        c: Commitment {
            d: D,
            kappa: params.kappa() as usize,
            data: vec![F::ZERO; D * params.kappa() as usize],
        },
        X: Mat::zero(D, 1, F::ZERO),
        r: Vec::new(),
        eval_k: vec![K::ZERO; D.next_power_of_two()],
        eval_a: vec![vec![K::ZERO; D.next_power_of_two()]; 14],
        m_in: D,
        fold_digest: [0; 32],
        adv: None,
    };
    // The disabled entry must reject before inspecting these empty messages.
    // The former body entered compressed PiCCS and returned a shape error.
    let messages = NifsVCircuitMessages {
        fresh: &[],
        running: &[],
        running_parent_authority: None,
        pi_ccs: &proof,
        combined: &combined,
        children: &[],
    };
    let mut builder = R1csBuilder::new();
    builder.enable_encoding_trace();
    let mut transcript = TranscriptGadget::from_native_state(
        &mut builder,
        core::array::from_fn(|index| F::from_u64(index as u64 + 3)),
        2,
    );
    let rows = builder.rows();
    let witness = builder.witness().to_vec();
    let (a, b, c) = builder.sparse_triplets();
    let constraints = (a.to_vec(), b.to_vec(), c.to_vec());
    let trace = format!("{:?}", builder.encoding_trace());
    let state = transcript.variable_state();
    let cursor = transcript.absorbed();
    let bindings = transcript.constant_bindings().to_vec();

    let result = enforce_nifs_v_circuit_with_transcript_and_header_bundle(
        &mut builder,
        &params,
        &config,
        &mut transcript,
        &messages,
        [Var::ONE; 4],
    );
    match result {
        Err(Error::Inner(message)) => assert_eq!(
            message,
            "native NIFS circuit is unavailable: compressed PiCCS is not a SuperNeo v1.1 relation"
        ),
        Err(error) => panic!("unexpected native circuit error: {error}"),
        Ok(_) => panic!("compressed native NIFS circuit remained available"),
    }
    assert_eq!(builder.rows(), rows);
    assert_eq!(builder.cols(), witness.len());
    assert_eq!(builder.witness(), witness);
    assert_eq!(
        builder.sparse_triplets(),
        (
            constraints.0.as_slice(),
            constraints.1.as_slice(),
            constraints.2.as_slice()
        )
    );
    assert_eq!(format!("{:?}", builder.encoding_trace()), trace);
    assert_eq!(transcript.variable_state(), state);
    assert_eq!(transcript.absorbed(), cursor);
    assert_eq!(transcript.constant_bindings(), bindings);
}

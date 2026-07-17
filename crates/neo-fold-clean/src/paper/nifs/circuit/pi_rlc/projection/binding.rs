//! Transcript binding for Π_RLC combined values and quotient advice.
//!
//! **Owns:** the exact production projection-preimage order, native quotient recomputation,
//! SIS compression, and beta squeeze. **Does not own:** beta-polynomial
//! evaluation or projection identities. **Emits constraints:** constant-label
//! bindings, SIS digest rows, and transcript rows. **Authority boundary:** beta
//! is sampled only after every combined value and the exact advice wires later
//! consumed by the identities have entered the transcript. The preimage mixes
//! paper-public fields with separately classified Nebula/delayed-NC extensions.
//! The SIS digest is transcript compression, never replacement authority for
//! the bound wires.
//!
//! | Stage child | Bound data |
//! | --- | --- |
//! | `domain` | Projection-binding domain separator |
//! | `combined.*` | Paper commitment/X/y_ring outputs plus adv/y_zcol extensions |
//! | `quotient.*` | Matching division-quotient advice |
//! | `sis_digest` | Compressed binding preimage |
//! | `transcript_beta` | SIS digest absorption and beta squeeze |

use neo_ccs::LaneCommitments;
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::ring_action::PROJECTION_QUOTIENT_LEN;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::f_prime::nebula_lane_circuit::enforce_nebula_lane_leaf_digests_circuit;
use crate::paper::reductions::accumulator_sis_circuit::{
    enforce_accumulator_digest as enforce_sis_accumulator_digest, PI_RLC_PROJECTION_SIS_CONFIG,
};
use crate::paper::reductions::pi_dec_circuit::DecInputWires;
use crate::paper::reductions::pi_rlc;
use crate::paper::reductions::pi_rlc_circuit::{
    alloc_rlc_projection_quotient_advice, rlc_projection_quotients, stage, RlcCommitmentWires, RlcPaddedKVectorWires,
    RlcXWires,
};
use crate::paper::relations::superneo_public_x_cols;

use super::super::super::Error;
use super::super::fold_wires::FoldWires;

pub(super) struct BindingOutputs {
    pub(super) beta: [Var; 2],
    pub(super) commitment_q: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    pub(super) adv_q: Option<LaneCommitments<Vec<[Var; PROJECTION_QUOTIENT_LEN]>>>,
    pub(super) x_q: Vec<[Var; PROJECTION_QUOTIENT_LEN]>,
    pub(super) y_ring_q: Vec<[[Var; PROJECTION_QUOTIENT_LEN]; 2]>,
    pub(super) y_zcol_q: [[Var; PROJECTION_QUOTIENT_LEN]; 2],
}

pub(super) fn enforce(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    dec_wires: &DecInputWires,
    folds: &FoldWires,
    kappa: usize,
) -> Result<BindingOutputs, Error> {
    let binding_start = builder.rows();
    builder.begin_encoding_stage(stage::PROJECTION_BINDING);
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_DOMAIN);
    let mut preimage = alloc_binding_domain(builder);

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED);
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_COMMITMENT);
    append_fields(
        builder,
        &mut preimage,
        pi_rlc::PI_RLC_PROJECTION_COMBINED_C_LABEL,
        &dec_wires.parent.c_data[..D * kappa],
    );

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT);
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_COMMITMENT);
    let commitment_q = alloc_commitment_quotients(builder, &folds.commitment)?;
    for quotient in &commitment_q {
        append_fields(
            builder,
            &mut preimage,
            pi_rlc::PI_RLC_PROJECTION_QUOTIENTS_LABEL,
            quotient,
        );
    }

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_ADV);
    if folds.adv.is_some() {
        let combined = dec_wires
            .parent
            .adv
            .as_ref()
            .ok_or_else(|| Error::Inner("Pi_RLC adv projection has inputs but no combined coordinate".into()))?;
        for leaf in enforce_nebula_lane_leaf_digests_circuit(
            builder,
            combined.ops.d,
            combined.ops.kappa,
            &combined.ops.data,
            &combined.is.data,
            &combined.fs.data,
        ) {
            append_fields(
                builder,
                &mut preimage,
                pi_rlc::PI_RLC_PROJECTION_COMBINED_ADV_LABEL,
                &leaf,
            );
        }
    }

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_ADV);
    let adv_q = if let Some(adv) = &folds.adv {
        let ops = alloc_commitment_quotients(builder, &adv.ops)?;
        let is = alloc_commitment_quotients(builder, &adv.is)?;
        let fs = alloc_commitment_quotients(builder, &adv.fs)?;
        for quotient in ops.iter().chain(&is).chain(&fs) {
            append_fields(
                builder,
                &mut preimage,
                pi_rlc::PI_RLC_PROJECTION_ADV_QUOTIENTS_LABEL,
                quotient,
            );
        }
        Some(LaneCommitments { ops, is, fs })
    } else {
        None
    };

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_X);
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_X);
    let x_q = alloc_x_advice(builder, &mut preimage, &folds.x)?;

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_Y_RING);
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_Y_RING);
    let mut y_ring_q = Vec::with_capacity(folds.y_ring.len());
    for wires in &folds.y_ring {
        y_ring_q.push(alloc_y_ring_advice(builder, &mut preimage, wires)?);
    }

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_Y_ZCOL);
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_Y_ZCOL);
    let y_zcol_q = alloc_y_zcol_advice(builder, &mut preimage, &folds.y_zcol)?;

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_SIS_DIGEST);
    let digest = enforce_sis_accumulator_digest(builder, PI_RLC_PROJECTION_SIS_CONFIG, &preimage)
        .map_err(|error| Error::Inner(format!("Pi_RLC projection SIS binding: {error}")))?
        .digest;

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_TRANSCRIPT_BETA);
    transcript.append_fields(builder, pi_rlc::PI_RLC_PROJECTION_BINDING_DIGEST_LABEL, &digest);
    let beta = transcript.challenge_fields(builder, pi_rlc::PI_RLC_PROJECTION_BETA_LABEL, 2);
    builder.record_row_family("nifs.pi_rlc.projection_binding", binding_start);

    Ok(BindingOutputs {
        beta: [beta[0], beta[1]],
        commitment_q,
        adv_q,
        x_q,
        y_ring_q,
        y_zcol_q,
    })
}

fn alloc_commitment_quotients(
    builder: &mut R1csBuilder,
    wires: &RlcCommitmentWires,
) -> Result<Vec<[Var; PROJECTION_QUOTIENT_LEN]>, Error> {
    let rhos = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|index| builder.witness()[pair.rho_coeffs[index].col()]))
        .collect::<Vec<_>>();
    let commitments = wires
        .inputs
        .iter()
        .map(|pair| neo_ajtai::Commitment {
            d: D,
            kappa: pair.kappa,
            data: pair
                .c_data
                .iter()
                .map(|wire| builder.witness()[wire.col()])
                .collect(),
        })
        .collect::<Vec<_>>();
    Ok(rlc_projection_quotients(&rhos, &commitments)?
        .iter()
        .map(|lane| core::array::from_fn(|index| builder.alloc(lane.q[index])))
        .collect())
}

fn alloc_x_advice(
    builder: &mut R1csBuilder,
    preimage: &mut Vec<Var>,
    wires: &RlcXWires,
) -> Result<Vec<[Var; PROJECTION_QUOTIENT_LEN]>, Error> {
    let active_cols = superneo_public_x_cols(wires.m_in);
    let rhos = wires
        .inputs
        .iter()
        .map(|pair| pair.rho_coeffs)
        .collect::<Vec<_>>();
    let mut quotients = Vec::with_capacity(active_cols);
    for column in 0..active_cols {
        let inputs = wires
            .inputs
            .iter()
            .map(|pair| core::array::from_fn(|row| pair.x_flat[row * wires.m_in + column]))
            .collect::<Vec<_>>();
        let output: [Var; D] = core::array::from_fn(|row| wires.combined_x_flat[row * wires.m_in + column]);
        builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_X);
        append_fields(builder, preimage, pi_rlc::PI_RLC_PROJECTION_COMBINED_X_LABEL, &output);
        builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_X);
        let quotient = alloc_rlc_projection_quotient_advice(builder, &rhos, &inputs)?;
        append_fields(
            builder,
            preimage,
            pi_rlc::PI_RLC_PROJECTION_X_QUOTIENTS_LABEL,
            &quotient,
        );
        quotients.push(quotient);
    }
    Ok(quotients)
}

fn alloc_y_ring_advice(
    builder: &mut R1csBuilder,
    preimage: &mut Vec<Var>,
    wires: &RlcPaddedKVectorWires,
) -> Result<[[Var; PROJECTION_QUOTIENT_LEN]; 2], Error> {
    let rhos = wires
        .inputs
        .iter()
        .map(|pair| pair.rho_coeffs)
        .collect::<Vec<_>>();
    let inputs_c0 = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|lane| pair.y_c0[lane]))
        .collect::<Vec<_>>();
    let inputs_c1 = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|lane| pair.y_c1[lane]))
        .collect::<Vec<_>>();
    let output_c0: [Var; D] = core::array::from_fn(|lane| wires.combined_c0[lane]);
    let output_c1: [Var; D] = core::array::from_fn(|lane| wires.combined_c1[lane]);

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_Y_RING);
    append_fields(
        builder,
        preimage,
        pi_rlc::PI_RLC_PROJECTION_COMBINED_Y_RING_LABEL,
        &output_c0,
    );
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_Y_RING);
    let quotient_c0 = alloc_rlc_projection_quotient_advice(builder, &rhos, &inputs_c0)?;
    append_fields(
        builder,
        preimage,
        pi_rlc::PI_RLC_PROJECTION_Y_RING_QUOTIENTS_LABEL,
        &quotient_c0,
    );

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_Y_RING);
    append_fields(
        builder,
        preimage,
        pi_rlc::PI_RLC_PROJECTION_COMBINED_Y_RING_LABEL,
        &output_c1,
    );
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_Y_RING);
    let quotient_c1 = alloc_rlc_projection_quotient_advice(builder, &rhos, &inputs_c1)?;
    append_fields(
        builder,
        preimage,
        pi_rlc::PI_RLC_PROJECTION_Y_RING_QUOTIENTS_LABEL,
        &quotient_c1,
    );
    Ok([quotient_c0, quotient_c1])
}

fn alloc_y_zcol_advice(
    builder: &mut R1csBuilder,
    preimage: &mut Vec<Var>,
    wires: &RlcPaddedKVectorWires,
) -> Result<[[Var; PROJECTION_QUOTIENT_LEN]; 2], Error> {
    let rhos = wires
        .inputs
        .iter()
        .map(|pair| pair.rho_coeffs)
        .collect::<Vec<_>>();
    let inputs_c0 = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|lane| pair.y_c0[lane]))
        .collect::<Vec<_>>();
    let inputs_c1 = wires
        .inputs
        .iter()
        .map(|pair| core::array::from_fn(|lane| pair.y_c1[lane]))
        .collect::<Vec<_>>();
    let output_c0: [Var; D] = core::array::from_fn(|lane| wires.combined_c0[lane]);
    let output_c1: [Var; D] = core::array::from_fn(|lane| wires.combined_c1[lane]);

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_Y_ZCOL);
    append_fields(
        builder,
        preimage,
        pi_rlc::PI_RLC_PROJECTION_COMBINED_Y_ZCOL_LABEL,
        &output_c0,
    );
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_Y_ZCOL);
    let quotient_c0 = alloc_rlc_projection_quotient_advice(builder, &rhos, &inputs_c0)?;
    append_fields(
        builder,
        preimage,
        pi_rlc::PI_RLC_PROJECTION_Y_ZCOL_QUOTIENTS_LABEL,
        &quotient_c0,
    );

    builder.begin_encoding_stage(stage::PROJECTION_BINDING_COMBINED_Y_ZCOL);
    append_fields(
        builder,
        preimage,
        pi_rlc::PI_RLC_PROJECTION_COMBINED_Y_ZCOL_LABEL,
        &output_c1,
    );
    builder.begin_encoding_stage(stage::PROJECTION_BINDING_QUOTIENT_Y_ZCOL);
    let quotient_c1 = alloc_rlc_projection_quotient_advice(builder, &rhos, &inputs_c1)?;
    append_fields(
        builder,
        preimage,
        pi_rlc::PI_RLC_PROJECTION_Y_ZCOL_QUOTIENTS_LABEL,
        &quotient_c1,
    );
    Ok([quotient_c0, quotient_c1])
}

fn alloc_binding_domain(builder: &mut R1csBuilder) -> Vec<Var> {
    crate::paper::digest::pack_bytes_as_fields(pi_rlc::PI_RLC_PROJECTION_BINDING_DOMAIN)
        .into_iter()
        .map(|value| alloc_constant(builder, value))
        .collect()
}

fn append_fields(builder: &mut R1csBuilder, preimage: &mut Vec<Var>, label: &[u8], fields: &[Var]) {
    preimage.extend(
        crate::paper::digest::pack_bytes_as_fields(label)
            .into_iter()
            .map(|value| alloc_constant(builder, value)),
    );
    preimage.push(alloc_constant(builder, F::from_u64(fields.len() as u64)));
    preimage.extend_from_slice(fields);
}

fn alloc_constant(builder: &mut R1csBuilder, value: F) -> Var {
    let wire = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    wire
}

//! Deterministic shape synthesis for the two distinct Nebula F' relations.
//!
//! Placeholder messages carry exact protocol dimensions but no authority.
//! They are used only to emit verifier matrices during preprocessing.

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

use super::{
    enforce_nebula_application_f_prime_base_step, enforce_nebula_application_f_prime_recursive_step,
    enforce_nebula_f_prime_base_step, enforce_nebula_f_prime_recursive_step, NebulaFPrimeFieldArmShape,
    NebulaFPrimeFieldShapeAudit, NebulaFPrimeRelationError,
};
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::frontends::nebula::application::NebulaApplication;
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, SparseR1cs};
use crate::paper::construction2::{running::zero_lane_commitments, NebulaConfig, NebulaLane};
use crate::paper::digest::{AccumulatorHandle, StateXOutDigestMode};
use crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use crate::paper::f_prime::r1cs::{
    FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig,
    F_PRIME_ENC_INST_BITS, F_PRIME_PUBLIC_INPUT_LEN,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage};
use crate::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs;
use crate::paper::reductions::pi_ccs_circuit::{PiCcsVerifierConfig, PiCcsVerifierRelation};
use crate::paper::relations::{CcsClaim, CeClaim, Structure};

pub(super) struct ArmShapes {
    pub base: SparseR1cs,
    pub recursive: SparseR1cs,
}

struct SynthesizedArm {
    shape: SparseR1cs,
}

struct ShapeContext<'a> {
    params: &'a Params,
    plan: &'a NebulaPlan,
    config: NebulaConfig,
    matrix_digest: [F; 4],
    folded: &'a PiCcsVerifierRelation,
    application: Option<&'a NebulaApplication>,
    joint_variables: usize,
    joint_degree: usize,
}

pub(super) fn synthesize_arm_shapes(
    params: &Params,
    folded: &PiCcsVerifierRelation,
    plan: &NebulaPlan,
    application: Option<&NebulaApplication>,
) -> Result<ArmShapes, NebulaFPrimeRelationError> {
    let context = shape_context(params, folded, plan, application)?;

    #[cfg(feature = "perf-timers")]
    let arm_started = std::time::Instant::now();
    let base = synthesize_base(&context)?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[fprime-arm-shape] branch=base rows={} columns={} total={:.3}s",
        base.shape.n,
        base.shape.m,
        arm_started.elapsed().as_secs_f64(),
    );
    #[cfg(feature = "perf-timers")]
    let arm_started = std::time::Instant::now();
    let recursive = synthesize_recursive(&context)?;
    #[cfg(feature = "perf-timers")]
    eprintln!(
        "[fprime-arm-shape] branch=recursive_shared rows={} columns={} total={:.3}s",
        recursive.shape.n,
        recursive.shape.m,
        arm_started.elapsed().as_secs_f64(),
    );
    Ok(ArmShapes {
        base: base.shape,
        recursive: recursive.shape,
    })
}

pub(super) fn audit_arm_shapes(
    params: &Params,
    folded: &Structure,
    plan: &NebulaPlan,
) -> Result<NebulaFPrimeFieldShapeAudit, NebulaFPrimeRelationError> {
    let folded_relation = PiCcsVerifierRelation::from_structure(folded);
    let context = shape_context(params, &folded_relation, plan, None)?;
    let base = arm_shape(synthesize_base(&context)?.shape);
    let recursive = arm_shape(synthesize_recursive(&context)?.shape);
    let bootstrap_recursive = recursive;
    Ok(NebulaFPrimeFieldShapeAudit {
        verifier_rows: folded.n,
        verifier_columns: folded.m,
        base,
        bootstrap_recursive,
        recursive,
    })
}

fn shape_context<'a>(
    params: &'a Params,
    folded: &'a PiCcsVerifierRelation,
    plan: &'a NebulaPlan,
    application: Option<&'a NebulaApplication>,
) -> Result<ShapeContext<'a>, NebulaFPrimeRelationError> {
    let dims = neo_reductions::engines::pi_ccs_joint::build_joint_dims_for_shape(
        params.inner(),
        folded.n(),
        folded.m(),
        folded.t(),
        folded.max_degree(),
        1,
        params.k_rho() as usize,
    )
    .map_err(|error| NebulaFPrimeRelationError::Geometry(format!("verifier dimensions: {error}")))?;
    Ok(ShapeContext {
        params,
        plan,
        config: super::relation_config(plan, application),
        matrix_digest: [F::ZERO; 4],
        folded,
        application,
        joint_variables: dims.variables,
        joint_degree: dims.degree,
    })
}

fn arm_shape(shape: SparseR1cs) -> NebulaFPrimeFieldArmShape {
    let audit = NebulaFPrimeFieldArmShape {
        rows: shape.n,
        columns: shape.m,
        public_columns: shape.m_in,
        poseidon2_permutations: shape.poseidon2_permutations(),
    };
    drop(shape);
    audit
}

fn synthesize_base(context: &ShapeContext<'_>) -> Result<SynthesizedArm, NebulaFPrimeRelationError> {
    let application_assignment = shape_application_assignment(context.application);
    let semantic = application_semantic_values(context.application, &application_assignment)?;
    let empty = AccumulatorHandle::empty().digest_fields();
    let mut source = FPrimeSourceImage::new();
    let chunk_count_in_word = source.push_u64_le(0);
    let step_count_in_word = source.push_u64_le(0);
    let pc_word = source.push_u64_le(1);
    let public_x_out_bits = source.push_enc_inst([F::ZERO; 4]);
    let inputs = FPrimeBaseInputs {
        state: shape_state(context, false, semantic.input.unwrap_or(empty), empty),
        chunk_digest: [F::ZERO; 4],
        semantic_state_digest_out: semantic.output.unwrap_or(empty),
        rows_in_chunk: 1,
        source_image: &source,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        public_x_out_bits,
    };
    let mut builder = R1csBuilder::new();
    let output = match context.application {
        Some(application) => enforce_nebula_application_f_prime_base_step(
            &mut builder,
            context.plan.circuit(),
            &shape_s_mem_assignment(context.plan),
            application,
            &application_assignment,
            Some([[F::ZERO; 4]; 3]),
            &step_config(context),
            &inputs,
        )?,
        None => enforce_nebula_f_prime_base_step(
            &mut builder,
            context.plan.circuit(),
            &shape_s_mem_assignment(context.plan),
            Some([[F::ZERO; 4]; 3]),
            &step_config(context),
            &inputs,
        )?,
    };
    Ok(SynthesizedArm {
        shape: lower_field_r1cs(builder, &output.public_outputs())?
            .into_parts()
            .0,
    })
}

fn synthesize_recursive(context: &ShapeContext<'_>) -> Result<SynthesizedArm, NebulaFPrimeRelationError> {
    let application_assignment = shape_application_assignment(context.application);
    let semantic = application_semantic_values(context.application, &application_assignment)?;
    let public_input_len =
        FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(context.config.stacks)).total_len();
    let ce = zero_ce_claim(context, public_input_len);
    let running = vec![ce.clone(); context.params.k_rho() as usize];
    let running_parent = Some(ce.clone());
    let fresh = [zero_fresh_claim(context, public_input_len)];
    let outputs = vec![ce.clone(); fresh.len() + running.len()];
    let sumcheck = pi_ccs::SumcheckProof::new(vec![vec![K::ZERO; context.joint_degree + 1]; context.joint_variables]);
    let proof = pi_ccs::Proof { sumcheck, outputs };
    let combined = ce.clone();
    let children = vec![ce; context.params.k_rho() as usize];
    let nifs_msg = NifsVCircuitMessages {
        fresh: &fresh,
        running: &running,
        running_parent_authority: running_parent.as_ref(),
        pi_ccs: &proof,
        combined: &combined,
        children: &children,
    };

    let running_digest =
        AccumulatorHandle::from_running_parts(context.params.b(), &running, running_parent.as_ref()).digest_fields();
    let output_digest =
        AccumulatorHandle::from_running_parts(context.params.b(), &children, Some(&combined)).digest_fields();
    let mut source = FPrimeSourceImage::new();
    let chunk_count_in_word = source.push_u64_le(1);
    let step_count_in_word = source.push_u64_le(1);
    let pc_word = source.push_u64_le(1);
    let prior_public = source.push_f_prime_public_input([F::ZERO; 4]);
    let prior_x_out_bits = BitRange::new(prior_public.start() + 1, F_PRIME_ENC_INST_BITS);
    let public_x_out_bits = source.push_enc_inst([F::ZERO; 4]);
    let inputs = FPrimeRecursiveInputs {
        state: shape_state(context, true, semantic.input.unwrap_or(running_digest), running_digest),
        chunk_digest: [F::ZERO; 4],
        semantic_state_digest_out: semantic.output.unwrap_or(output_digest),
        acc_digest_out: output_digest,
        nifs_msg,
        rows_in_chunk: 1,
        source_image: &source,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        prior_x_out_bits,
        public_x_out_bits,
    };
    let mut builder = R1csBuilder::new();
    let output = match context.application {
        Some(application) => enforce_nebula_application_f_prime_recursive_step(
            &mut builder,
            context.params,
            context.plan.circuit(),
            &shape_s_mem_assignment(context.plan),
            application,
            &application_assignment,
            Some([[F::ZERO; 4]; 3]),
            &step_config(context),
            &inputs,
        )?,
        None => enforce_nebula_f_prime_recursive_step(
            &mut builder,
            context.params,
            context.plan.circuit(),
            &shape_s_mem_assignment(context.plan),
            Some([[F::ZERO; 4]; 3]),
            &step_config(context),
            &inputs,
        )?,
    };
    Ok(SynthesizedArm {
        shape: lower_field_r1cs(builder, &output.public_outputs())?
            .into_parts()
            .0,
    })
}

fn step_config<'a>(context: &'a ShapeContext<'a>) -> FPrimeStepConfig<'a> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: PiCcsVerifierConfig {
                params: context.params,
                structure: context.folded.clone(),
                matrix_digest: context.matrix_digest,
            },
        },
        b: context.params.b(),
        transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
        public_input_layout: FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(
            context.config.stacks,
        )),
        nebula: Some(&context.config),
        state_x_out_digest_mode: context
            .application
            .map_or(StateXOutDigestMode::Stateless, |application| {
                crate::frontends::r1cs_f_prime::ivc::shape::digest_mode(application.recursive_plan())
            }),
    }
}

fn shape_state(
    context: &ShapeContext<'_>,
    recursive: bool,
    semantic_digest: [F; 4],
    acc_digest: [F; 4],
) -> FPrimeStateIn {
    FPrimeStateIn {
        vk_fs_digest: [F::ZERO; 4],
        pi_ccs_header_bundle: context.matrix_digest,
        chunk_count_in: u64::from(recursive),
        step_count_in: u64::from(recursive),
        z_0: [F::ZERO; 4],
        z_i_in: [F::ZERO; 4],
        pc: 1,
        semantic_state_digest_in: semantic_digest,
        acc_digest_in: acc_digest,
        public_trace_in: [F::ZERO; 4],
        nebula: Some(NebulaLane::base(&context.config)),
    }
}

fn shape_s_mem_assignment(plan: &NebulaPlan) -> Vec<F> {
    let mut assignment = vec![F::ZERO; plan.circuit().cols()];
    assignment[0] = F::ONE;
    assignment
}

fn shape_application_assignment(application: Option<&NebulaApplication>) -> Vec<F> {
    let mut assignment = vec![F::ZERO; application.map_or(0, |app| app.shape().m())];
    if let Some(one) = assignment.first_mut() {
        *one = F::ONE;
    }
    assignment
}

fn application_semantic_values(
    application: Option<&NebulaApplication>,
    assignment: &[F],
) -> Result<crate::frontends::r1cs_f_prime::ivc::shape::SemanticValues, NebulaFPrimeRelationError> {
    match application {
        Some(application) => Ok(crate::frontends::r1cs_f_prime::ivc::shape::semantic_values(
            application.recursive_plan(),
            assignment,
        )?),
        None => Ok(crate::frontends::r1cs_f_prime::ivc::shape::SemanticValues {
            input: None,
            output: None,
        }),
    }
}

fn zero_fresh_claim(context: &ShapeContext<'_>, m_in: usize) -> CcsClaim {
    let mut x = vec![F::ZERO; m_in];
    x[0] = F::ONE;
    x[F_PRIME_PUBLIC_INPUT_LEN + context.config.stacks.x_bits()] = F::ONE;
    CcsClaim {
        c: Commitment::zeros(D, context.params.kappa() as usize),
        x,
        m_in,
        adv: Some(zero_lane_commitments(context.params)),
    }
}

fn zero_ce_claim(context: &ShapeContext<'_>, m_in: usize) -> CeClaim {
    let d_pad = D.next_power_of_two();
    CeClaim {
        c: Commitment::zeros(D, context.params.kappa() as usize),
        X: Mat::zero(D, crate::paper::relations::superneo_public_x_cols(m_in), F::ZERO),
        r: vec![K::ZERO; context.joint_variables],
        eval_k: vec![K::ZERO; d_pad],
        eval_a: vec![vec![K::ZERO; d_pad]; context.folded.t()],
        m_in,
        fold_digest: [0u8; 32],
        adv: Some(zero_lane_commitments(context.params)),
    }
}

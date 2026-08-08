//! Deterministic field-R1CS synthesis for the generic implementation IVC arms.
//!
//! Owns: fixed-point shape witnesses and the physical ordering of application,
//! F-prime step, and semantic-link rows used by the IVC frontend.
//!
//! Does not own: paper semantics, selective low-norm lowering, or permission to
//! remove any emitted row.
//!
//! | Stage path | Mathematical obligation | Rust owner |
//! |---|---|---|
//! | `fprime.{base,recursive}.finalize.application` | Enforce the application R1CS and derived semantic digests | `enforce_*_application` |
//! | `fprime.{base,recursive}.step.*` | Enforce the selected F-prime transition | `paper::f_prime::r1cs` |
//! | `fprime.{base,recursive}.finalize.semantic_links` | Bind application semantics to the transition output | `bind_semantic_state` |

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

use super::R1csIvcError;
use crate::engine::r1cs_circuit::{enforce_poseidon2_hash, Lc, R1csBuilder, Var};
use crate::frontends::f_prime::recursive_plan::{
    semantic_state_app_public_header, semantic_state_field_header, RecursiveStepImagePlan,
};
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, R1csShape, SparseR1cs};
use crate::paper::construction2::SemanticStateMode;
use crate::paper::digest::{digest32_as_fields, AccumulatorHandle, StateXOutDigestMode};
use crate::paper::f_prime::digest_circuit::alloc_constant;
use crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use crate::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use crate::paper::f_prime::r1cs::{
    enforce_f_prime_base_step_circuit, enforce_f_prime_recursive_step_circuit, FPrimeBaseInputs,
    FPrimePublicInputLayout, FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig, FPrimeStepOutput,
    F_PRIME_ENC_INST_BITS,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage};
use crate::paper::f_prime::stage as fprime_stage;
use crate::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs;
use crate::paper::reductions::pi_ccs_circuit::{PiCcsVerifierConfig, PiCcsVerifierRelation};
use crate::paper::relations::{CcsClaim, CeClaim};

pub(super) struct ArmShapes {
    pub base: SparseR1cs,
    pub bootstrap_recursive: SparseR1cs,
    pub recursive: SparseR1cs,
}

#[derive(Clone, Copy)]
pub(crate) struct SemanticValues {
    pub input: Option<[F; 4]>,
    pub output: Option<[F; 4]>,
}

struct ShapeContext<'a> {
    params: &'a Params,
    app: &'a R1csShape,
    plan: &'a RecursiveStepImagePlan,
    folded: &'a PiCcsVerifierRelation,
    folded_public_input_len: usize,
    matrix_digest: [F; 4],
    joint_variables: usize,
    joint_degree: usize,
}

/// One complete synthesis round.
pub(super) struct SynthesizedArmShapes {
    pub arms: [SparseR1cs; 3],
}

pub(super) fn synthesize_arm_shapes(
    params: &Params,
    folded: &PiCcsVerifierRelation,
    folded_public_input_len: usize,
    app: &R1csShape,
    plan: &RecursiveStepImagePlan,
) -> Result<SynthesizedArmShapes, R1csIvcError> {
    let context = shape_context(params, folded, folded_public_input_len, app, plan)?;
    let bootstrap_recursive = synthesize_recursive(&context)?;
    let recursive = synthesize_recursive(&context)?;
    let arms = ArmShapes {
        base: synthesize_base(&context)?,
        bootstrap_recursive,
        recursive,
    };
    Ok(SynthesizedArmShapes {
        arms: [arms.base, arms.bootstrap_recursive, arms.recursive],
    })
}

fn shape_context<'a>(
    params: &'a Params,
    folded: &'a PiCcsVerifierRelation,
    folded_public_input_len: usize,
    app: &'a R1csShape,
    plan: &'a RecursiveStepImagePlan,
) -> Result<ShapeContext<'a>, R1csIvcError> {
    if folded_public_input_len > folded.m() {
        return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
            format!(
                "folded public carrier width {folded_public_input_len} exceeds relation width {}",
                folded.m()
            ),
        )));
    }
    let dims = neo_reductions::engines::pi_ccs_joint::build_joint_dims_for_shape(
        params.inner(),
        folded.n(),
        folded.m(),
        folded.t(),
        folded.max_degree(),
        1,
        params.k_rho() as usize,
    )
    .map_err(|error| {
        R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(format!(
            "verifier dimensions: {error}"
        )))
    })?;
    Ok(ShapeContext {
        params,
        app,
        plan,
        folded,
        folded_public_input_len,
        matrix_digest: [F::ZERO; 4],
        joint_variables: dims.variables,
        joint_degree: dims.degree,
    })
}

fn synthesize_base(context: &ShapeContext<'_>) -> Result<SparseR1cs, R1csIvcError> {
    let assignment = shape_app_assignment(context.app);
    let semantic = semantic_values(context.plan, &assignment)?;
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
    let cfg = step_config(context);
    let mut builder = R1csBuilder::new();
    let output = enforce_base_application(&mut builder, context.app, &assignment, context.plan, &cfg, &inputs)?;
    Ok(lower_field_r1cs(builder, &output.x_out_bits)?
        .into_parts()
        .0)
}

fn synthesize_recursive(context: &ShapeContext<'_>) -> Result<SparseR1cs, R1csIvcError> {
    let assignment = shape_app_assignment(context.app);
    let semantic = semantic_values(context.plan, &assignment)?;
    let ce = zero_ce_claim(context);
    let running = vec![ce.clone(); context.params.k_rho() as usize];
    let running_parent = Some(ce.clone());
    let fresh = [zero_fresh_claim(context.params, context.folded_public_input_len)];
    let outputs = vec![ce.clone(); fresh.len() + running.len()];
    let sumcheck = pi_ccs::SumcheckProof::new(vec![vec![K::ZERO; context.joint_degree + 1]; context.joint_variables]);
    let outputs_digest = crate::paper::digest::pi_ccs_outputs_digest(&outputs);
    let proof = pi_ccs::Proof {
        sumcheck,
        outputs,
        outputs_digest,
    };
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

    let running_digest = AccumulatorHandle::from_running_parts(&running, running_parent.as_ref()).digest_fields();
    let output_digest = AccumulatorHandle::from_running_parts(&children, Some(&combined)).digest_fields();
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
    let cfg = step_config(context);
    let mut builder = R1csBuilder::new();
    let output = enforce_recursive_application(
        &mut builder,
        context.params,
        context.app,
        &assignment,
        context.plan,
        &cfg,
        &inputs,
    )?;
    let arm = lower_field_r1cs(builder, &output.x_out_bits)?
        .into_parts()
        .0;
    Ok(arm)
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
        public_input_layout: FPrimePublicInputLayout::plain(),
        nebula: None,
        state_x_out_digest_mode: digest_mode(context.plan),
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
        nebula: None,
    }
}

fn zero_fresh_claim(params: &Params, public_input_len: usize) -> CcsClaim {
    let mut x = vec![F::ZERO; public_input_len];
    x[0] = F::ONE;
    CcsClaim {
        c: Commitment::zeros(D, params.kappa() as usize),
        x,
        m_in: public_input_len,
        adv: None,
    }
}

fn zero_ce_claim(context: &ShapeContext<'_>) -> CeClaim {
    let d_pad = D.next_power_of_two();
    CeClaim {
        c: Commitment::zeros(D, context.params.kappa() as usize),
        X: Mat::zero(
            D,
            crate::paper::relations::superneo_public_x_cols(context.folded_public_input_len),
            F::ZERO,
        ),
        r: vec![K::ZERO; context.joint_variables],
        y_ring: vec![vec![K::ZERO; d_pad]; context.folded.t() + 1],
        ct: vec![K::ZERO; context.folded.t() + 1],
        m_in: context.folded_public_input_len,
        fold_digest: [0u8; 32],
        adv: None,
    }
}

fn shape_app_assignment(app: &R1csShape) -> Vec<F> {
    let mut assignment = vec![F::ZERO; app.m()];
    if let Some(one) = assignment.first_mut() {
        *one = F::ONE;
    }
    assignment
}

pub(crate) fn semantic_values(plan: &RecursiveStepImagePlan, assignment: &[F]) -> Result<SemanticValues, R1csIvcError> {
    let Some(state) = plan.state_x_out.as_ref() else {
        return Ok(SemanticValues {
            input: None,
            output: None,
        });
    };
    let input = (!state.semantic_state_in_var_indices.is_empty())
        .then(|| semantic_state_digest_for_assignment(assignment, &state.semantic_state_in_var_indices));
    let output = if !state.semantic_state_out_var_indices.is_empty() {
        Some(semantic_state_digest_for_assignment(
            assignment,
            &state.semantic_state_out_var_indices,
        ))
    } else if !state.app_public_input_var_indices.is_empty() || !state.app_public_input_bit_var_indices.is_empty() {
        let preimage = app_public_semantic_preimage_for_assignment(plan, assignment)?;
        Some(encode_poseidon_trace(&preimage).digest_native)
    } else {
        None
    };
    Ok(SemanticValues { input, output })
}

fn semantic_state_digest_for_assignment(assignment: &[F], indices: &[usize]) -> [F; 4] {
    let values = indices
        .iter()
        .map(|&index| assignment[index])
        .collect::<Vec<_>>();
    encode_poseidon_trace(&crate::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields(&values))
        .digest_native
}

fn app_public_semantic_preimage_for_assignment(
    plan: &RecursiveStepImagePlan,
    assignment: &[F],
) -> Result<Vec<F>, R1csIvcError> {
    let Some(state) = plan.state_x_out.as_ref() else {
        return Ok(Vec::new());
    };
    let mut preimage = semantic_state_app_public_header(
        state.app_public_input_var_indices.len(),
        state.app_public_input_bit_var_indices.len(),
    );
    preimage.extend(
        state
            .app_public_input_var_indices
            .iter()
            .map(|&index| assignment[index]),
    );
    for chunk in state.app_public_input_bit_var_indices.chunks(64) {
        let mut packed = 0u64;
        for (bit, &index) in chunk.iter().enumerate() {
            let value = assignment[index];
            if value == F::ZERO {
                continue;
            }
            if value == F::ONE {
                packed |= 1 << bit;
                continue;
            }
            return Err(R1csIvcError::PackedPublicInputNotBit { index, value });
        }
        preimage.push(F::from_u64(packed));
    }
    Ok(preimage)
}

pub(crate) fn digest_mode(plan: &RecursiveStepImagePlan) -> StateXOutDigestMode {
    let mode = super::super::semantic_state_mode_for_plan(plan);
    match mode {
        SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    }
}

pub(super) fn enforce_base_application(
    builder: &mut R1csBuilder,
    app: &R1csShape,
    assignment: &[F],
    plan: &RecursiveStepImagePlan,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeBaseInputs<'_>,
) -> Result<FPrimeStepOutput, R1csIvcError> {
    builder.begin_encoding_stage(fprime_stage::BASE_ROOT);
    builder.begin_encoding_stage(fprime_stage::BASE_APPLICATION);
    let app_vars = app.enforce_in_f_prime(builder, assignment, pin_app_constant(plan))?;
    let semantic = enforce_semantic_digests(builder, plan, assignment, &app_vars)?;
    let output = enforce_f_prime_base_step_circuit(builder, cfg, inputs)?;
    builder.begin_encoding_stage(fprime_stage::BASE_SEMANTIC_LINKS);
    bind_semantic_state(builder, plan, &output, semantic, true);
    Ok(output)
}

pub(super) fn enforce_recursive_application(
    builder: &mut R1csBuilder,
    params: &Params,
    app: &R1csShape,
    assignment: &[F],
    plan: &RecursiveStepImagePlan,
    cfg: &FPrimeStepConfig<'_>,
    inputs: &FPrimeRecursiveInputs<'_>,
) -> Result<FPrimeStepOutput, R1csIvcError> {
    builder.begin_encoding_stage(fprime_stage::RECURSIVE_ROOT);
    builder.begin_encoding_stage(fprime_stage::RECURSIVE_APPLICATION);
    let app_vars = app.enforce_in_f_prime(builder, assignment, pin_app_constant(plan))?;
    let semantic = enforce_semantic_digests(builder, plan, assignment, &app_vars)?;
    let output = enforce_f_prime_recursive_step_circuit(builder, params, cfg, inputs)?;
    builder.begin_encoding_stage(fprime_stage::RECURSIVE_SEMANTIC_LINKS);
    bind_semantic_state(builder, plan, &output, semantic, false);
    Ok(output)
}

pub(crate) struct SemanticWires {
    input: Option<[Var; 4]>,
    output: Option<[Var; 4]>,
}

pub(crate) fn enforce_semantic_digests(
    builder: &mut R1csBuilder,
    plan: &RecursiveStepImagePlan,
    assignment: &[F],
    app_vars: &[Var],
) -> Result<SemanticWires, R1csIvcError> {
    let Some(state) = plan.state_x_out.as_ref() else {
        return Ok(SemanticWires {
            input: None,
            output: None,
        });
    };
    let input = (!state.semantic_state_in_var_indices.is_empty()).then(|| {
        semantic_field_digest_wires(
            builder,
            state
                .semantic_state_in_var_indices
                .iter()
                .map(|&index| app_vars[index]),
        )
    });
    let output = if !state.semantic_state_out_var_indices.is_empty() {
        Some(semantic_field_digest_wires(
            builder,
            state
                .semantic_state_out_var_indices
                .iter()
                .map(|&index| app_vars[index]),
        ))
    } else if !state.app_public_input_var_indices.is_empty() || !state.app_public_input_bit_var_indices.is_empty() {
        let mut values: Vec<Var> = state
            .app_public_input_var_indices
            .iter()
            .map(|&index| app_vars[index])
            .collect();
        for chunk in state.app_public_input_bit_var_indices.chunks(64) {
            let mut packed_value = 0u64;
            let mut packed_lc = Lc::zero();
            let mut coefficient = F::ONE;
            for (bit, &index) in chunk.iter().enumerate() {
                if assignment[index] == F::ONE {
                    packed_value |= 1u64 << bit;
                }
                packed_lc.add_term(app_vars[index], coefficient);
                coefficient += coefficient;
            }
            let packed = builder.alloc(F::from_u64(packed_value));
            builder.enforce_eq(&Lc::from_var(packed), &packed_lc);
            values.push(packed);
        }
        Some(semantic_app_public_digest_wires(
            builder,
            state.app_public_input_var_indices.len(),
            state.app_public_input_bit_var_indices.len(),
            values,
        ))
    } else {
        None
    };
    Ok(SemanticWires { input, output })
}

fn semantic_field_digest_wires(builder: &mut R1csBuilder, values: impl IntoIterator<Item = Var>) -> [Var; 4] {
    let values: Vec<Var> = values.into_iter().collect();
    semantic_digest_wires(builder, semantic_state_field_header(values.len()), values)
}

fn semantic_app_public_digest_wires(
    builder: &mut R1csBuilder,
    field_count: usize,
    bit_count: usize,
    values: impl IntoIterator<Item = Var>,
) -> [Var; 4] {
    semantic_digest_wires(
        builder,
        semantic_state_app_public_header(field_count, bit_count),
        values,
    )
}

fn semantic_digest_wires(builder: &mut R1csBuilder, header: Vec<F>, values: impl IntoIterator<Item = Var>) -> [Var; 4] {
    let mut preimage: Vec<Var> = header
        .into_iter()
        .map(|value| alloc_constant(builder, value))
        .collect();
    preimage.extend(values);
    enforce_poseidon2_hash(builder, &preimage)
}

pub(crate) fn bind_semantic_state(
    builder: &mut R1csBuilder,
    plan: &RecursiveStepImagePlan,
    output: &FPrimeStepOutput,
    semantic: SemanticWires,
    base: bool,
) {
    if let Some(input) = semantic.input {
        bind_digest(builder, &output.state_in.semantic_state_digest, &input);
    }
    if let Some(out) = semantic.output {
        bind_digest(builder, &output.state_out.semantic_state_digest, &out);
    }
    if base {
        let anchor = plan
            .state_x_out
            .as_ref()
            .and_then(|state| state.initial_semantic_state_digest_anchor)
            .unwrap_or_else(crate::paper::digest::empty_semantic_state_digest);
        let anchor = digest32_as_fields(anchor);
        for lane in 0..4 {
            builder.enforce_eq(
                &Lc::from_var(output.state_in.semantic_state_digest[lane]),
                &Lc::from_const(anchor[lane]),
            );
        }
    }
}

fn bind_digest(builder: &mut R1csBuilder, left: &[Var; 4], right: &[Var; 4]) {
    for lane in 0..4 {
        builder.enforce_eq(&Lc::from_var(left[lane]), &Lc::from_var(right[lane]));
    }
}

pub(crate) fn pin_app_constant(plan: &RecursiveStepImagePlan) -> bool {
    let Some(state) = plan.state_x_out.as_ref() else {
        return plan.app_private_var_widths.iter().any(|&width| width < 64);
    };
    // One semantic role can use z[0] as an ordinary value. Zero roles or
    // the same lane on both transition sides select the conventional
    // constant-one role, which must be constrained directly.
    let mut zero_semantic_roles = usize::from(state.semantic_state_in_var_indices.contains(&0))
        + usize::from(state.semantic_state_out_var_indices.contains(&0));
    if state.semantic_state_out_var_indices.is_empty()
        && (state.app_public_input_var_indices.contains(&0) || state.app_public_input_bit_var_indices.contains(&0))
    {
        zero_semantic_roles += 1;
    }
    plan.app_private_var_widths.iter().any(|&width| width < 64)
        || !state.app_public_input_bit_var_indices.is_empty()
        || (state.initial_semantic_state_digest_anchor.is_some() && zero_semantic_roles != 1)
}

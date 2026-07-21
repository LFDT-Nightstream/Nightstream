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
use crate::frontends::f_prime::recursive_plan::{build_semantic_state_preimage_fields, RecursiveStepImagePlan};
use crate::frontends::r1cs_f_prime::compiler::{
    semantic_state_digest_for_assignment, semantic_state_digest_for_fields,
    state_x_out_app_preimage_lanes_for_assignment,
};
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, R1csShape, SparseR1cs};
use crate::paper::construction2::{PendingProjectionState, SemanticStateMode};
use crate::paper::digest::{
    digest32_as_fields, pending_accumulator_family_digest, AccumulatorHandle, PendingAccumulatorFamilyState,
    StateXOutDigestMode,
};
use crate::paper::f_prime::digest_circuit::alloc_constant;
use crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use crate::paper::f_prime::r1cs::{
    enforce_f_prime_base_step_circuit, enforce_f_prime_recursive_step_circuit, FPrimeBaseInputs,
    FPrimePublicInputLayout, FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig, FPrimeStepOutput,
    F_PRIME_ENC_INST_BITS, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN,
};
use crate::paper::f_prime::source_image::{BitRange, FPrimeSourceImage};
use crate::paper::f_prime::stage as fprime_stage;
use crate::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs;
use crate::paper::reductions::pi_ccs_split_nc_circuit::{SplitNcPiCcsVConfig, SplitNcVerifierRelation};
use crate::paper::relations::{CcsClaim, CeClaim};
use neo_reductions::optimized_engine::oracle::BLOCK_LANE_NC_ROUND_COEFFICIENTS;

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
    folded: &'a SplitNcVerifierRelation,
    folded_public_input_len: usize,
    header_bundle: [F; 4],
    ell_d: usize,
    ell_n: usize,
    ell_m: usize,
    d_sc: usize,
}

/// Exact source-field column occupied by one packed incoming running-claim
/// assignment coordinate after public-output normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RawRunningSourceColumn {
    pub child: usize,
    pub logical_column: usize,
    pub source_column: usize,
}

/// Exact normalized source-field column occupied by one coordinate of the
/// single fresh public-X input consumed by the steady recursive arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FreshSourceColumn {
    pub logical_column: usize,
    pub source_column: usize,
}

/// One synthesis round plus the audit-only source decoders for its steady
/// recursive arm.
pub(super) struct SynthesizedArmShapes {
    pub arms: [SparseR1cs; 3],
    pub raw_running_source_columns: Vec<RawRunningSourceColumn>,
    pub fresh_source_columns: Vec<FreshSourceColumn>,
}

pub(super) fn synthesize_arm_shapes(
    params: &Params,
    folded: &SplitNcVerifierRelation,
    folded_public_input_len: usize,
    app: &R1csShape,
    plan: &RecursiveStepImagePlan,
) -> Result<SynthesizedArmShapes, R1csIvcError> {
    let context = shape_context(params, folded, folded_public_input_len, app, plan)?;
    let (bootstrap_recursive, bootstrap_raw_running, _) = synthesize_recursive(&context, false)?;
    if !bootstrap_raw_running.is_empty() {
        return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
            "bootstrap recursive arm unexpectedly exposed running-assignment columns".into(),
        )));
    }
    let (recursive, raw_running_source_columns, fresh_source_columns) = synthesize_recursive(&context, true)?;
    let arms = ArmShapes {
        base: synthesize_base(&context)?,
        bootstrap_recursive,
        recursive,
    };
    Ok(SynthesizedArmShapes {
        arms: [arms.base, arms.bootstrap_recursive, arms.recursive],
        raw_running_source_columns,
        fresh_source_columns,
    })
}

fn shape_context<'a>(
    params: &'a Params,
    folded: &'a SplitNcVerifierRelation,
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
    let dims = neo_reductions::engines::utils::build_dims_and_policy_for_shape(
        params.inner(),
        folded.n(),
        folded.m(),
        folded.t(),
        folded.max_degree(),
    )
    .map_err(|error| {
        R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(format!(
            "verifier dimensions: {error}"
        )))
    })?;
    if folded.variant() == neo_reductions::optimized_engine::PiCcsProofVariant::BlockLaneNcDelayedV1
        && dims.ell_n != crate::paper::digest::PENDING_ACCUMULATOR_FAMILY_ROW_POINT
    {
        return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
            format!(
                "delayed production relation needs {} row challenges, but {rows} rows produce {}",
                crate::paper::digest::PENDING_ACCUMULATOR_FAMILY_ROW_POINT,
                dims.ell_n,
                rows = folded.n(),
            ),
        )));
    }
    Ok(ShapeContext {
        params,
        app,
        plan,
        folded,
        folded_public_input_len,
        header_bundle: [F::ZERO; 4],
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
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

fn synthesize_recursive(
    context: &ShapeContext<'_>,
    steady: bool,
) -> Result<(SparseR1cs, Vec<RawRunningSourceColumn>, Vec<FreshSourceColumn>), R1csIvcError> {
    let block_mode =
        context.folded.variant() == neo_reductions::optimized_engine::PiCcsProofVariant::BlockLaneNcDelayedV1;
    let assignment = shape_app_assignment(context.app);
    let semantic = semantic_values(context.plan, &assignment)?;
    let ce = zero_ce_claim(context);
    let running = if steady {
        vec![ce.clone(); context.params.k_rho() as usize]
    } else {
        Vec::new()
    };
    let running_parent = steady.then(|| ce.clone());
    let fresh = [zero_fresh_claim(context.params, context.folded_public_input_len)];
    let outputs = vec![ce.clone(); fresh.len() + running.len()];
    let mut sumcheck = pi_ccs::SumcheckProof::new(
        vec![vec![K::ZERO; context.d_sc + 1]; context.ell_n + context.ell_d],
        None,
    );
    let nc_column_coefficients = match context.params.b() {
        2 => 5,
        3 => 7,
        _ => context.d_sc + 1,
    };
    sumcheck.sumcheck_rounds_nc = if block_mode {
        vec![
            vec![K::ZERO; BLOCK_LANE_NC_ROUND_COEFFICIENTS];
            crate::paper::digest::PENDING_ACCUMULATOR_FAMILY_COLUMN_POINT + context.ell_d
        ]
    } else {
        (0..context.ell_m)
            .map(|_| vec![K::ZERO; nc_column_coefficients])
            .chain((0..context.ell_d).map(|_| vec![K::ZERO; context.d_sc + 1]))
            .collect()
    };
    if block_mode {
        sumcheck.variant = neo_reductions::optimized_engine::PiCcsProofVariant::BlockLaneNcDelayedV1;
    }
    sumcheck.header_digest = vec![0u8; 32];
    let outputs_digest = crate::paper::digest::pi_ccs_outputs_digest(&outputs);
    let proof = pi_ccs::Proof {
        sumcheck,
        outputs,
        outputs_digest,
    };
    let combined = ce.clone();
    let children = vec![ce; context.params.k_rho() as usize];
    let running_pending_projection =
        (steady && block_mode).then(|| PendingProjectionState::new([K::ZERO; 19], [K::ZERO; D]));
    let nifs_msg = NifsVCircuitMessages {
        fresh: &fresh,
        running: &running,
        running_parent_authority: running_parent.as_ref(),
        running_pending_projection: running_pending_projection.as_ref(),
        pi_ccs: &proof,
        combined: &combined,
        children: &children,
    };

    let running_digest = if steady && block_mode {
        pending_accumulator_family_digest(
            &running,
            context.params.kappa() as usize,
            running_pending_projection
                .as_ref()
                .map(|pending| PendingAccumulatorFamilyState {
                    old_block: pending.old_block(),
                    parent_y_zcol: pending.parent_y_zcol(),
                }),
        )
        .map_err(|error| {
            R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(format!(
                "shape running pending-family digest: {error}"
            )))
        })?
    } else if steady {
        AccumulatorHandle::from_running_parts(&running, running_parent.as_ref()).digest_fields()
    } else {
        AccumulatorHandle::empty().digest_fields()
    };
    let outgoing_pending = block_mode.then(|| PendingProjectionState::new([K::ZERO; 19], [K::ZERO; D]));
    let output_digest = if block_mode {
        pending_accumulator_family_digest(
            &children,
            context.params.kappa() as usize,
            outgoing_pending
                .as_ref()
                .map(|pending| PendingAccumulatorFamilyState {
                    old_block: pending.old_block(),
                    parent_y_zcol: pending.parent_y_zcol(),
                }),
        )
        .map_err(|error| {
            R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(format!(
                "shape outgoing pending-family digest: {error}"
            )))
        })?
    } else {
        AccumulatorHandle::from_running_parts(&children, Some(&combined)).digest_fields()
    };
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
    let raw_running_source_columns = raw_running_source_columns(
        builder.cols(),
        &output.x_out_bits,
        output.nifs_running.as_deref().unwrap_or_default(),
    )?;
    let fresh_source_columns = fresh_source_columns(builder.cols(), &output.x_out_bits, output.prior_link.as_ref())?;
    let arm = lower_field_r1cs(builder, &output.x_out_bits)?
        .into_parts()
        .0;
    if raw_running_source_columns
        .iter()
        .any(|entry| entry.source_column >= arm.m)
    {
        return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
            "normalized raw running-assignment column exceeds the recursive source arm".into(),
        )));
    }
    if fresh_source_columns
        .iter()
        .any(|entry| entry.source_column >= arm.m)
    {
        return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
            "normalized fresh public-X column exceeds the recursive source arm".into(),
        )));
    }
    Ok((arm, raw_running_source_columns, fresh_source_columns))
}

fn fresh_source_columns(
    source_column_count: usize,
    public_outputs: &[Var],
    prior_link: Option<&crate::paper::f_prime::r1cs::FPrimePriorLinkWires>,
) -> Result<Vec<FreshSourceColumn>, R1csIvcError> {
    let prior_link = prior_link.ok_or_else(|| {
        R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
            "recursive fresh public-X audit is missing prior-link wires".into(),
        ))
    })?;
    let [fresh] = prior_link.fresh_public_inputs.as_slice() else {
        return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
            format!(
                "recursive fresh public-X audit expected one source, found {}",
                prior_link.fresh_public_inputs.len()
            ),
        )));
    };
    if fresh.len() != F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN {
        return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
            format!(
                "recursive fresh public-X audit has {} coordinates, expected {F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN}",
                fresh.len()
            ),
        )));
    }

    let mut source_columns = std::collections::BTreeSet::new();
    let mut records = Vec::with_capacity(F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN);
    for (logical_column, wire) in fresh.iter().copied().enumerate() {
        let source = wire.col();
        let source_column = normalized_target_column(source_column_count, public_outputs, source).ok_or_else(|| {
            R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(format!(
                "fresh[0].x coordinate {logical_column} references invalid builder column {source}"
            )))
        })?;
        if crate::frontends::r1cs_f_prime::lowering::normalized_source_column(
            source_column_count,
            public_outputs,
            source_column,
        ) != Some(source)
        {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                format!("fresh[0].x coordinate {logical_column} failed source-column normalization round trip"),
            )));
        }
        if !source_columns.insert(source_column) {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                format!("fresh[0].x coordinate {logical_column} repeats normalized source column {source_column}"),
            )));
        }
        records.push(FreshSourceColumn {
            logical_column,
            source_column,
        });
    }
    Ok(records)
}

fn raw_running_source_columns(
    source_column_count: usize,
    public_outputs: &[Var],
    running: &[crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires],
) -> Result<Vec<RawRunningSourceColumn>, R1csIvcError> {
    let public_output_columns = public_outputs
        .iter()
        .map(|output| output.col())
        .collect::<std::collections::BTreeSet<_>>();
    if public_output_columns.len() != public_outputs.len()
        || public_output_columns.contains(&Var::ONE.col())
        || public_output_columns
            .iter()
            .any(|&column| column >= source_column_count)
    {
        return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
            "raw running-assignment audit received invalid public-output normalization columns".into(),
        )));
    }
    let mut records = Vec::new();
    for (child, claim) in running.iter().enumerate() {
        if claim.x_rows != D
            || claim.x_cols == 0
            || claim.m_in > claim.x_rows * claim.x_cols
            || claim.x.len() != claim.x_rows * claim.x_cols
        {
            return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                format!(
                    "running[{child}] raw assignment has invalid geometry: rows={}, cols={}, m_in={}, x.len={}",
                    claim.x_rows,
                    claim.x_cols,
                    claim.m_in,
                    claim.x.len()
                ),
            )));
        }
        for logical_column in 0..claim.m_in {
            let source = claim.x[(logical_column % D) * claim.x_cols + (logical_column / D)].col();
            let source_column =
                normalized_target_column(source_column_count, public_outputs, source).ok_or_else(|| {
                    R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(format!(
                    "running[{child}].x packed coordinate {logical_column} references invalid builder column {source}"
                )))
                })?;
            if crate::frontends::r1cs_f_prime::lowering::normalized_source_column(
                source_column_count,
                public_outputs,
                source_column,
            ) != Some(source)
            {
                return Err(R1csIvcError::Composition(crate::paper::f_prime::r1cs::Error::Inner(
                    format!(
                        "running[{child}].x packed coordinate {logical_column} failed source-column normalization round trip"
                    ),
                )));
            }
            records.push(RawRunningSourceColumn {
                child,
                logical_column,
                source_column,
            });
        }
    }
    Ok(records)
}

/// Forward spelling of the exact normalization permutation used by
/// `lower_field_r1cs`. The inverse is checked above for every exported wire,
/// so this audit cannot silently drift from the actual source-arm ordering.
fn normalized_target_column(source_columns: usize, public_outputs: &[Var], source: usize) -> Option<usize> {
    if source >= source_columns {
        return None;
    }
    if source == Var::ONE.col() {
        return Some(0);
    }
    if let Some(public_index) = public_outputs
        .iter()
        .position(|output| output.col() == source)
    {
        return Some(public_index + 1);
    }
    let public_before = public_outputs
        .iter()
        .filter(|output| output.col() < source)
        .count();
    Some(1 + public_outputs.len() + (source - 1 - public_before))
}

fn step_config<'a>(context: &'a ShapeContext<'a>) -> FPrimeStepConfig<'a> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: SplitNcPiCcsVConfig {
                params: context.params,
                structure: context.folded.clone(),
                header_bundle: context.header_bundle,
                ell_d: context.ell_d,
                ell_n: context.ell_n,
                ell_m: context.ell_m,
                d_sc: context.d_sc,
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
        pi_ccs_header_bundle: context.header_bundle,
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
    let d_pad = 1usize << context.ell_d;
    CeClaim {
        c: Commitment::zeros(D, context.params.kappa() as usize),
        X: Mat::zero(D, context.folded_public_input_len, F::ZERO),
        r: vec![K::ZERO; context.ell_n],
        s_col: vec![
            K::ZERO;
            if context.folded.variant() == neo_reductions::optimized_engine::PiCcsProofVariant::BlockLaneNcDelayedV1 {
                crate::paper::digest::PENDING_ACCUMULATOR_FAMILY_COLUMN_POINT
            } else {
                context.ell_m
            }
        ],
        y_ring: vec![vec![K::ZERO; d_pad]; context.folded.t()],
        ct: vec![K::ZERO; context.folded.t()],
        aux_openings: Vec::new(),
        y_zcol: vec![K::ZERO; d_pad],
        m_in: context.folded_public_input_len,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
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
        let fields = state_x_out_app_preimage_lanes_for_assignment(plan, assignment)?;
        Some(semantic_state_digest_for_fields(&fields))
    } else {
        None
    };
    Ok(SemanticValues { input, output })
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
        semantic_digest_wires(
            builder,
            state
                .semantic_state_in_var_indices
                .iter()
                .map(|&index| app_vars[index]),
        )
    });
    let output = if !state.semantic_state_out_var_indices.is_empty() {
        Some(semantic_digest_wires(
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
        Some(semantic_digest_wires(builder, values))
    } else {
        None
    };
    Ok(SemanticWires { input, output })
}

fn semantic_digest_wires(builder: &mut R1csBuilder, values: impl IntoIterator<Item = Var>) -> [Var; 4] {
    let mut preimage: Vec<Var> = build_semantic_state_preimage_fields(&[])
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
    let explicit_semantic_output = !state.semantic_state_out_var_indices.is_empty();
    let zero_absorbed = state.semantic_state_in_var_indices.contains(&0)
        || state.semantic_state_out_var_indices.contains(&0)
        || (!explicit_semantic_output
            && (state.app_public_input_var_indices.contains(&0)
                || state.app_public_input_bit_var_indices.contains(&0)));
    plan.app_private_var_widths.iter().any(|&width| width < 64)
        || !state.app_public_input_bit_var_indices.is_empty()
        || (state.initial_semantic_state_digest_anchor.is_some() && !zero_absorbed)
}

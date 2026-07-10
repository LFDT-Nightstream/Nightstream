//! Deterministic shape synthesis for the three authoritative Nebula F' arms.
//!
//! Placeholder messages carry exact protocol dimensions but no authority.
//! They are used only to emit verifier matrices during preprocessing.

use neo_ajtai::Commitment;
use neo_ccs::{LaneCommitments, Mat};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

use super::{
    enforce_nebula_f_prime_base_step, enforce_nebula_f_prime_recursive_step, NebulaFPrimeFieldArmShape,
    NebulaFPrimeFieldShapeAudit, NebulaFPrimeRelationError,
};
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::frontends::nebula::plan::NebulaPlan;
use crate::frontends::r1cs_f_prime::{lower_field_r1cs, SparseR1cs};
use crate::paper::construction2::{NebulaConfig, NebulaLane};
use crate::paper::digest::{AccumulatorHandle, StateXOutDigestMode};
use crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL;
use crate::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use crate::paper::f_prime::r1cs::{
    FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeRecursiveInputs, FPrimeStateIn, FPrimeStepConfig,
    F_PRIME_PUBLIC_INPUT_LEN,
};
use crate::paper::f_prime::source_image::FPrimeSourceImage;
use crate::paper::nifs::circuit::{NifsVCircuitConfig, NifsVCircuitMessages};
use crate::paper::params::Params;
use crate::paper::reductions::pi_ccs;
use crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use crate::paper::relations::{CcsClaim, CeClaim, Structure};

pub(super) struct ArmShapes {
    pub base: SparseR1cs,
    pub bootstrap_recursive: SparseR1cs,
    pub recursive: SparseR1cs,
}

struct ShapeContext<'a> {
    params: &'a Params,
    plan: &'a NebulaPlan,
    config: NebulaConfig,
    header_bundle: [F; 4],
    ell_d: usize,
    ell_n: usize,
    ell_m: usize,
    d_sc: usize,
    folded: &'a Structure,
}

pub(super) fn synthesize_arm_shapes(
    params: &Params,
    folded: &Structure,
    plan: &NebulaPlan,
) -> Result<ArmShapes, NebulaFPrimeRelationError> {
    let context = shape_context(params, folded, plan)?;

    Ok(ArmShapes {
        base: synthesize_base(&context)?,
        bootstrap_recursive: synthesize_recursive(&context, false)?,
        recursive: synthesize_recursive(&context, true)?,
    })
}

pub(super) fn audit_arm_shapes(
    params: &Params,
    folded: &Structure,
    plan: &NebulaPlan,
) -> Result<NebulaFPrimeFieldShapeAudit, NebulaFPrimeRelationError> {
    let context = shape_context(params, folded, plan)?;
    let base = arm_shape(synthesize_base(&context)?);
    let bootstrap_recursive = arm_shape(synthesize_recursive(&context, false)?);
    let recursive = arm_shape(synthesize_recursive(&context, true)?);
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
    folded: &'a Structure,
    plan: &'a NebulaPlan,
) -> Result<ShapeContext<'a>, NebulaFPrimeRelationError> {
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params.inner(), folded)
        .map_err(|error| NebulaFPrimeRelationError::Geometry(format!("verifier dimensions: {error}")))?;
    let matrix_digest = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(folded, None);
    let header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        params.inner(),
        folded,
        dims,
        &matrix_digest,
    )
    .map_err(|error| NebulaFPrimeRelationError::Geometry(format!("verifier header: {error}")))?;
    Ok(ShapeContext {
        params,
        plan,
        config: plan.config(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
        folded,
    })
}

fn arm_shape(shape: SparseR1cs) -> NebulaFPrimeFieldArmShape {
    let audit = NebulaFPrimeFieldArmShape {
        rows: shape.n,
        columns: shape.m,
        public_columns: shape.m_in,
    };
    drop(shape);
    audit
}

fn synthesize_base(context: &ShapeContext<'_>) -> Result<SparseR1cs, NebulaFPrimeRelationError> {
    let mut source = FPrimeSourceImage::new();
    let chunk_count_in_word = source.push_u64_le(0);
    let step_count_in_word = source.push_u64_le(0);
    let pc_word = source.push_u64_le(1);
    let public_x_out_bits = source.push_enc_inst([F::ZERO; 4]);
    let inputs = FPrimeBaseInputs {
        state: shape_state(context, false, AccumulatorHandle::empty().digest_fields()),
        chunk_digest: [F::ZERO; 4],
        semantic_state_digest_out: AccumulatorHandle::empty().digest_fields(),
        rows_in_chunk: 1,
        source_image: &source,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        public_x_out_bits,
    };
    let mut builder = R1csBuilder::new();
    let output = enforce_nebula_f_prime_base_step(
        &mut builder,
        context.plan.circuit(),
        &shape_s_mem_assignment(context.plan),
        Some([[F::ZERO; 4]; 3]),
        &step_config(context),
        &inputs,
    )?;
    Ok(lower_field_r1cs(builder, &output.public_outputs())?
        .into_parts()
        .0)
}

fn synthesize_recursive(context: &ShapeContext<'_>, steady: bool) -> Result<SparseR1cs, NebulaFPrimeRelationError> {
    let public_input_len = F_PRIME_PUBLIC_INPUT_LEN + delayed_nebula_public_suffix_len(context.config.stacks);
    let ce = zero_ce_claim(context, public_input_len);
    let running = if steady {
        vec![ce.clone(); context.params.k_rho() as usize]
    } else {
        Vec::new()
    };
    let running_parent = steady.then(|| ce.clone());
    let fresh = [zero_fresh_claim(context, public_input_len)];
    let outputs = vec![ce.clone(); fresh.len() + running.len()];
    let mut sumcheck = pi_ccs::SumcheckProof::new(
        vec![vec![K::ZERO; context.d_sc + 1]; context.ell_n + context.ell_d],
        None,
    );
    sumcheck.sumcheck_rounds_nc = vec![vec![K::ZERO; context.d_sc + 1]; context.ell_m + context.ell_d];
    sumcheck.header_digest = vec![0u8; 32];
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

    let running_digest = if steady {
        AccumulatorHandle::from_running_parts(&running, running_parent.as_ref()).digest_fields()
    } else {
        AccumulatorHandle::empty().digest_fields()
    };
    let output_digest = AccumulatorHandle::from_running_parts(&children, Some(&combined)).digest_fields();
    let mut source = FPrimeSourceImage::new();
    let chunk_count_in_word = source.push_u64_le(1);
    let step_count_in_word = source.push_u64_le(1);
    let pc_word = source.push_u64_le(1);
    let prior_x_out_bits = source.push_enc_inst([F::ZERO; 4]);
    let public_x_out_bits = source.push_enc_inst([F::ZERO; 4]);
    let inputs = FPrimeRecursiveInputs {
        state: shape_state(context, true, running_digest),
        chunk_digest: [F::ZERO; 4],
        semantic_state_digest_out: output_digest,
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
    let output = enforce_nebula_f_prime_recursive_step(
        &mut builder,
        context.params,
        context.plan.circuit(),
        &shape_s_mem_assignment(context.plan),
        Some([[F::ZERO; 4]; 3]),
        &step_config(context),
        &inputs,
    )?;
    Ok(lower_field_r1cs(builder, &output.public_outputs())?
        .into_parts()
        .0)
}

fn step_config<'a>(context: &'a ShapeContext<'a>) -> FPrimeStepConfig<'a> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: SplitNcPiCcsVConfig {
                params: context.params,
                structure: context.folded,
                header_bundle: context.header_bundle,
                ell_d: context.ell_d,
                ell_n: context.ell_n,
                ell_m: context.ell_m,
                d_sc: context.d_sc,
            },
        },
        b: context.params.b(),
        transcript_label: F_PRIME_STEP_TRANSCRIPT_LABEL,
        public_input_layout: FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(
            context.config.stacks,
        )),
        nebula: Some(&context.config),
        state_x_out_digest_mode: StateXOutDigestMode::Stateless,
    }
}

fn shape_state(context: &ShapeContext<'_>, recursive: bool, acc_digest: [F; 4]) -> FPrimeStateIn {
    FPrimeStateIn {
        vk_fs_digest: [F::ZERO; 4],
        pi_ccs_header_bundle: context.header_bundle,
        chunk_count_in: u64::from(recursive),
        step_count_in: u64::from(recursive),
        z_0: [F::ZERO; 4],
        z_i_in: [F::ZERO; 4],
        pc: 1,
        semantic_state_digest_in: acc_digest,
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

fn zero_fresh_claim(context: &ShapeContext<'_>, m_in: usize) -> CcsClaim {
    let mut x = vec![F::ZERO; m_in];
    x[0] = F::ONE;
    x[F_PRIME_PUBLIC_INPUT_LEN + context.config.stacks.x_bits()] = F::ONE;
    CcsClaim {
        c: Commitment::zeros(D, context.params.kappa() as usize),
        x,
        m_in,
        adv: Some(zero_adv(context.params.kappa() as usize)),
    }
}

fn zero_ce_claim(context: &ShapeContext<'_>, m_in: usize) -> CeClaim {
    let d_pad = 1usize << context.ell_d;
    CeClaim {
        c: Commitment::zeros(D, context.params.kappa() as usize),
        X: Mat::zero(D, m_in, F::ZERO),
        r: vec![K::ZERO; context.ell_n],
        s_col: vec![K::ZERO; context.ell_m],
        y_ring: vec![vec![K::ZERO; d_pad]; context.folded.t()],
        ct: vec![K::ZERO; context.folded.t()],
        aux_openings: Vec::new(),
        y_zcol: vec![K::ZERO; d_pad],
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
        adv: Some(zero_adv(context.params.kappa() as usize)),
    }
}

fn zero_adv(kappa: usize) -> LaneCommitments<Commitment> {
    LaneCommitments {
        ops: Commitment::zeros(D, kappa),
        is: Commitment::zeros(D, kappa),
        fs: Commitment::zeros(D, kappa),
    }
}

//! Authoritative complete augmented `F'` relation for one uniform IVC step.
//!
//! Owns: verifier-owned context validation, base/recursive branch construction,
//! application composition, public-output binding, and fixed-language encoding.
//!
//! Does not own: Construction-2 branch algebra, NIFS internals, application
//! semantics, or low-level slot equations.
//!
//! Emits constraints: yes; it composes the complete authoritative field R1CS
//! and hands that relation to the selected exact low-norm lowering.
//!
//! Authority boundary: native checks and profiler metadata never participate
//! in acceptance. Public context, branch selector, application relation, and
//! all carried outputs are constrained inside the composed relation.
//!
//! | Child phase | Mathematical obligation | Emits constraints? | Rust owner | Lean owner |
//! |---|---|---|---|---|
//! | Verifier key/context | Fix one relation language and transcript header | yes | this file | full-relation refinement open |
//! | Base/recursive branch | Enforce exactly one Construction-2 transition | yes | `paper/f_prime/r1cs.rs` | FPrime semantics |
//! | Application | Enforce the configured state transition | yes | configured `R1csRelation` | application-specific |
//! | Public links | Bind semantic hashes, state, and exact `x_out` | yes | this file | full-relation refinement open |
//! | Gadget-native lowering | Preserve and encode every source row | yes | `frontends/f_prime/gadget_native/` | per-family model/refinement files |

use std::sync::Arc;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::builder::RowFamilyRange;
use crate::engine::r1cs_circuit::poseidon2::{enforce_poseidon2_hash, DIGEST_LEN};
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, R1csEncodingTrace, R1csRelation, R1csSnapshot, Var};
use crate::frontends::direct_ccs::FrontendError;
use crate::frontends::f_prime::gadget_native::{
    encode_r1cs_gadget_native, estimate_r1cs_gadget_native, estimate_selector_gated_r1cs_gadget_native,
    EncodedGadgetNativeR1cs, GadgetNativeError, GadgetNativeEstimate, SelectorGatedGadgetNativeEstimate,
};
use crate::frontends::f_prime::low_norm_r1cs::{
    encode_r1cs_derived, encode_r1cs_oracle, EncodedLowNormR1cs, LowNormR1csError,
};
use crate::frontends::r1cs_f_prime::structure::{r1cs_coeff_rows, R1csShape};
use crate::paper::construction2::verifier_key::VerifierKeyError;
use crate::paper::construction2::VerifierKey;
use crate::paper::digest::{digest32_as_fields, digest_fields_as_digest32, pack_bytes_as_fields, StateXOutDigestMode};
use crate::paper::f_prime::digest_circuit::enforce_vk_fs_digest_circuit;
use crate::paper::f_prime::r1cs::{
    enforce_construction2_f_prime_base_step_circuit, enforce_f_prime_recursive_step_circuit_with_header_bundle_wires,
    Error as FPrimeError, FPrimeBaseInputs, FPrimeRecursiveInputs, FPrimeStepConfig, FPrimeStepOutput,
};
use crate::paper::f_prime::stage as fprime_stage;
use crate::paper::params::Params;

const SEMANTIC_STATE_FIELDS_TAG: &[u8] = b"neo.fold.clean/semantic_state/fields/v2";
const SEMANTIC_STATE_FIELDS_SCHEMA: u64 = 1;

/// Verifier-owned constants that identify one fixed `F'` relation language.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FullFPrimeContext {
    vk_fs_digest: [F; DIGEST_LEN],
    structure_digest: [F; DIGEST_LEN],
    pi_ccs_header_bundle: [F; DIGEST_LEN],
    ajtai_pp_digest: [F; DIGEST_LEN],
    initial_semantic_state_digest: [F; DIGEST_LEN],
}

impl FullFPrimeContext {
    /// Derive the Construction-2 context from authoritative NIFS inputs.
    pub fn derive(
        pp: &Params,
        structure: &crate::paper::relations::Structure,
        log: &neo_ajtai::AjtaiSModule,
        initial_semantic_state_digest: [F; DIGEST_LEN],
    ) -> Result<Self, FullFPrimeError> {
        let structure_digest = crate::paper::digest::structure_digest(structure);
        let ajtai_pp_digest = crate::paper::digest::ajtai_public_parameters_digest(log)?;
        let vk = VerifierKey::derive(
            pp,
            structure,
            ajtai_pp_digest,
            Some(crate::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN),
            digest_fields_as_digest32(initial_semantic_state_digest),
        )?;
        Ok(Self {
            vk_fs_digest: digest32_as_fields(vk.digest()),
            structure_digest,
            pi_ccs_header_bundle: vk.pi_ccs_header_bundle(),
            ajtai_pp_digest,
            initial_semantic_state_digest,
        })
    }

    pub fn vk_fs_digest(&self) -> [F; DIGEST_LEN] {
        self.vk_fs_digest
    }

    pub fn structure_digest(&self) -> [F; DIGEST_LEN] {
        self.structure_digest
    }

    pub fn pi_ccs_header_bundle(&self) -> [F; DIGEST_LEN] {
        self.pi_ccs_header_bundle
    }

    pub fn ajtai_pp_digest(&self) -> [F; DIGEST_LEN] {
        self.ajtai_pp_digest
    }

    pub fn initial_semantic_state_digest(&self) -> [F; DIGEST_LEN] {
        self.initial_semantic_state_digest
    }
}

/// Verifier-owned augmented-function specification.
///
/// The NIFS configuration, application R1CS, and state-column schema are
/// frozen once. Per-step callers may supply only witnesses for that language.
pub struct FullFPrimeRelation<'a> {
    context: FullFPrimeContext,
    cfg: FPrimeStepConfig<'a>,
    application: &'a R1csShape,
    state_in_columns: Vec<usize>,
    state_out_columns: Vec<usize>,
}

struct ApplicationStep<'a> {
    relation: &'a R1csShape,
    assignment: &'a [F],
    state_in_columns: &'a [usize],
    state_out_columns: &'a [usize],
}

#[derive(Clone, Copy)]
struct FullFPrimeVerifierKeyWires {
    structure_digest: [Var; DIGEST_LEN],
    pi_ccs_header_bundle: [Var; DIGEST_LEN],
    ajtai_pp_digest: [Var; DIGEST_LEN],
    initial_semantic_state_digest: [Var; DIGEST_LEN],
    vk_fs_digest: [Var; DIGEST_LEN],
}

impl<'a> FullFPrimeRelation<'a> {
    pub fn new(
        context: FullFPrimeContext,
        cfg: FPrimeStepConfig<'a>,
        application: &'a R1csShape,
        state_in_columns: Vec<usize>,
        state_out_columns: Vec<usize>,
    ) -> Result<Self, FullFPrimeError> {
        if cfg.state_x_out_digest_mode != StateXOutDigestMode::Stateful {
            return Err(FullFPrimeError::StatelessSchema);
        }
        if cfg.b != cfg.nifs.pi_ccs.params.b() {
            return Err(FullFPrimeError::ConfigBoundMismatch {
                configured: cfg.b,
                params: cfg.nifs.pi_ccs.params.b(),
            });
        }
        validate_nifs_config(&cfg.nifs.pi_ccs, context.pi_ccs_header_bundle)?;
        let configured_vk = VerifierKey::derive_from_structure_digest(
            cfg.nifs.pi_ccs.params,
            &context.structure_digest,
            cfg.nifs.pi_ccs.header_bundle,
            context.ajtai_pp_digest,
            Some(crate::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN),
            digest_fields_as_digest32(context.initial_semantic_state_digest),
        );
        if context.vk_fs_digest != digest32_as_fields(configured_vk.digest()) {
            return Err(FullFPrimeError::ContextMismatch { field: "vk_fs" });
        }
        if context.pi_ccs_header_bundle != configured_vk.pi_ccs_header_bundle() {
            return Err(FullFPrimeError::ContextMismatch {
                field: "Pi_CCS header bundle",
            });
        }
        validate_application_schema(application, &state_in_columns, &state_out_columns)?;
        Ok(Self {
            context,
            cfg,
            application,
            state_in_columns,
            state_out_columns,
        })
    }

    pub fn context(&self) -> FullFPrimeContext {
        self.context
    }

    pub fn build_base(
        &self,
        inputs: &FPrimeBaseInputs<'_>,
        assignment: &[F],
    ) -> Result<FullFPrimeBranchExecution, FullFPrimeError> {
        let application = self.application_step(assignment)?;
        validate_fresh_arity(inputs.rows_in_chunk)?;
        let mut builder = R1csBuilder::new();
        builder.enable_encoding_trace();
        builder.begin_encoding_stage(fprime_stage::BASE_ROOT);
        builder.begin_encoding_stage(fprime_stage::BASE_VERIFIER_KEY);
        let verifier_key = alloc_verifier_key_wires(&mut builder, &self.context, self.cfg.nifs.pi_ccs.params);
        let output = enforce_construction2_f_prime_base_step_circuit(&mut builder, &self.cfg, inputs)?;
        finish_full_relation(&mut builder, BranchKind::Base, &verifier_key, output, &application)
    }

    pub fn build_recursive(
        &self,
        inputs: &FPrimeRecursiveInputs<'_>,
        assignment: &[F],
    ) -> Result<FullFPrimeBranchExecution, FullFPrimeError> {
        let application = self.application_step(assignment)?;
        validate_fresh_arity(inputs.rows_in_chunk)?;
        let pp = self.cfg.nifs.pi_ccs.params;
        validate_fixed_nifs_shape(pp, inputs)?;
        let mut builder = R1csBuilder::new();
        builder.enable_encoding_trace();
        builder.begin_encoding_stage(fprime_stage::RECURSIVE_ROOT);
        builder.begin_encoding_stage(fprime_stage::RECURSIVE_VERIFIER_KEY);
        let verifier_key = alloc_verifier_key_wires(&mut builder, &self.context, pp);
        let output = enforce_f_prime_recursive_step_circuit_with_header_bundle_wires(
            &mut builder,
            pp,
            &self.cfg,
            verifier_key.pi_ccs_header_bundle,
            inputs,
        )?;
        finish_full_relation(&mut builder, BranchKind::Recursive, &verifier_key, output, &application)
    }

    fn application_step<'b>(&'b self, assignment: &'b [F]) -> Result<ApplicationStep<'b>, FullFPrimeError> {
        validate_application_assignment(self.application, assignment)?;
        Ok(ApplicationStep {
            relation: self.application,
            assignment,
            state_in_columns: &self.state_in_columns,
            state_out_columns: &self.state_out_columns,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BranchKind {
    Base,
    Recursive,
}

/// One fully constrained branch before the two Construction-2 cases are
/// combined into a fixed relation.
#[derive(Clone, Debug)]
pub struct FullFPrimeBranchExecution {
    kind: BranchKind,
    snapshot: Arc<R1csSnapshot>,
    encoding_trace: Arc<R1csEncodingTrace>,
    row_family_ranges: Arc<Vec<RowFamilyRange>>,
    public_bit_columns: Vec<usize>,
    application_columns: Vec<usize>,
    verifier_key_columns: Vec<usize>,
}

impl FullFPrimeBranchExecution {
    pub fn snapshot(&self) -> &R1csSnapshot {
        self.snapshot.as_ref()
    }

    pub fn public_bit_columns(&self) -> &[usize] {
        &self.public_bit_columns
    }

    pub fn encoding_trace(&self) -> &R1csEncodingTrace {
        self.encoding_trace.as_ref()
    }

    /// Assurance-only source-row ownership captured before the builder is
    /// frozen into an R1CS snapshot. These ranges never affect acceptance.
    #[doc(hidden)]
    pub fn row_family_ranges(&self) -> &[RowFamilyRange] {
        self.row_family_ranges.as_ref()
    }

    pub fn estimate_gadget_native_encoding(&self) -> Result<GadgetNativeEstimate, GadgetNativeError> {
        estimate_r1cs_gadget_native(&self.snapshot, &self.encoding_trace, &self.public_bit_columns)
    }

    pub fn encode_gadget_native(&self) -> Result<EncodedGadgetNativeR1cs, GadgetNativeError> {
        encode_r1cs_gadget_native(&self.snapshot, &self.encoding_trace, &self.public_bit_columns)
    }

    /// Source-R1CS column for each application assignment coordinate.
    pub fn application_columns(&self) -> &[usize] {
        &self.application_columns
    }

    /// Raw fixed-shape verifier-key coordinates in
    /// `[structure_digest, Pi_CCS_header, initial_state_digest]` order.
    pub fn verifier_key_columns(&self) -> &[usize] {
        &self.verifier_key_columns
    }

    pub fn first_unsatisfied_row(&self) -> Option<usize> {
        self.snapshot.first_unsatisfied_row(self.snapshot.witness())
    }

    pub fn is_satisfied(&self) -> bool {
        self.first_unsatisfied_row().is_none()
    }
}

#[derive(Clone, Debug)]
struct BranchTemplate {
    snapshot: Arc<R1csSnapshot>,
    encoding_trace: Arc<R1csEncodingTrace>,
    public_bit_columns: Vec<usize>,
    application_columns: Vec<usize>,
    verifier_key_columns: Vec<usize>,
}

impl From<&FullFPrimeBranchExecution> for BranchTemplate {
    fn from(execution: &FullFPrimeBranchExecution) -> Self {
        Self {
            snapshot: execution.snapshot.clone(),
            encoding_trace: execution.encoding_trace.clone(),
            public_bit_columns: execution.public_bit_columns.clone(),
            application_columns: execution.application_columns.clone(),
            verifier_key_columns: execution.verifier_key_columns.clone(),
        }
    }
}

/// Verifier-owned fixed shape for the single augmented relation.
///
/// Construction 2 has one `F'` with an internal `i == 0` branch; base and
/// recursive steps are not two independently foldable relations.  This type
/// freezes both branch shapes and composes either witness into the same R1CS
/// language with a constrained selector.
#[derive(Clone, Debug)]
pub struct FullFPrimeShape {
    base: BranchTemplate,
    recursive: BranchTemplate,
    relation: Arc<R1csRelation>,
    layout: UnifiedLayout,
}

#[derive(Clone, Debug)]
struct UnifiedLayout {
    public_bit_columns: Vec<usize>,
    is_base_column: usize,
    canonical_zero_column: usize,
    private_pool_columns: Vec<usize>,
    base_map: Vec<usize>,
    recursive_map: Vec<usize>,
    base_private_columns: Vec<usize>,
    recursive_private_columns: Vec<usize>,
}

impl FullFPrimeShape {
    pub fn new(
        base: &FullFPrimeBranchExecution,
        recursive: &FullFPrimeBranchExecution,
    ) -> Result<Self, FullFPrimeError> {
        if base.kind != BranchKind::Base {
            return Err(FullFPrimeError::WrongBranch {
                expected: "base",
                got: branch_name(base.kind),
            });
        }
        if recursive.kind != BranchKind::Recursive {
            return Err(FullFPrimeError::WrongBranch {
                expected: "recursive",
                got: branch_name(recursive.kind),
            });
        }
        if !base.is_satisfied() {
            return Err(FullFPrimeError::UnsatisfiedTemplate { branch: "base" });
        }
        if !recursive.is_satisfied() {
            return Err(FullFPrimeError::UnsatisfiedTemplate { branch: "recursive" });
        }
        validate_branch_schema(base)?;
        validate_branch_schema(recursive)?;
        if base.public_bit_columns.len() != recursive.public_bit_columns.len() {
            return Err(FullFPrimeError::BranchPublicInputLength {
                base: base.public_bit_columns.len(),
                recursive: recursive.public_bit_columns.len(),
            });
        }
        let base_template = BranchTemplate::from(base);
        let recursive_template = BranchTemplate::from(recursive);
        let (reference, layout) = compile_unified_relation(&base_template, &recursive_template, base);
        debug_assert!(reference.is_satisfied(reference.witness()));
        let relation = reference.relation_arc();
        Ok(Self {
            base: base_template,
            recursive: recursive_template,
            relation,
            layout,
        })
    }

    pub fn execute_base(&self, execution: &FullFPrimeBranchExecution) -> Result<FullFPrimeExecution, FullFPrimeError> {
        self.compose(BranchKind::Base, execution)
    }

    pub fn execute_recursive(
        &self,
        execution: &FullFPrimeBranchExecution,
    ) -> Result<FullFPrimeExecution, FullFPrimeError> {
        self.compose(BranchKind::Recursive, execution)
    }

    /// Per-branch diagnostic before selector composition. The production
    /// fixed-relation lowering uses the same two traced templates.
    pub fn gadget_native_branch_estimates(
        &self,
    ) -> Result<(GadgetNativeEstimate, GadgetNativeEstimate), GadgetNativeError> {
        let base = estimate_r1cs_gadget_native(
            &self.base.snapshot,
            &self.base.encoding_trace,
            &self.base.public_bit_columns,
        )?;
        let recursive = estimate_r1cs_gadget_native(
            &self.recursive.snapshot,
            &self.recursive.encoding_trace,
            &self.recursive.public_bit_columns,
        )?;
        Ok((base, recursive))
    }

    pub fn gadget_native_fixed_estimate(&self) -> Result<SelectorGatedGadgetNativeEstimate, GadgetNativeError> {
        estimate_selector_gated_r1cs_gadget_native(
            &self.base.snapshot,
            &self.base.encoding_trace,
            &self.base.public_bit_columns,
            &self.recursive.snapshot,
            &self.recursive.encoding_trace,
            &self.recursive.public_bit_columns,
        )
    }

    fn compose(
        &self,
        active_kind: BranchKind,
        active: &FullFPrimeBranchExecution,
    ) -> Result<FullFPrimeExecution, FullFPrimeError> {
        if active.kind != active_kind {
            return Err(FullFPrimeError::WrongBranch {
                expected: branch_name(active_kind),
                got: branch_name(active.kind),
            });
        }
        let active_template = match active_kind {
            BranchKind::Base => &self.base,
            BranchKind::Recursive => &self.recursive,
        };
        validate_execution_matches_template(active, active_template)?;
        let witness = build_unified_witness(&self.base, &self.recursive, &self.layout, active_kind, active);
        let active_map = match active_kind {
            BranchKind::Base => &self.layout.base_map,
            BranchKind::Recursive => &self.layout.recursive_map,
        };
        let application_columns = active
            .application_columns
            .iter()
            .map(|&column| active_map[column])
            .collect();
        Ok(FullFPrimeExecution {
            snapshot: Arc::new(R1csSnapshot::from_shared_relation(Arc::clone(&self.relation), witness)),
            public_bit_columns: self.layout.public_bit_columns.clone(),
            application_columns,
            is_base_column: self.layout.is_base_column,
        })
    }
}

/// Frozen witness for the one fixed, selector-composed `F'` relation.
#[derive(Clone, Debug)]
pub struct FullFPrimeExecution {
    snapshot: Arc<R1csSnapshot>,
    public_bit_columns: Vec<usize>,
    application_columns: Vec<usize>,
    is_base_column: usize,
}

impl FullFPrimeExecution {
    pub fn snapshot(&self) -> &R1csSnapshot {
        self.snapshot.as_ref()
    }

    pub fn public_bit_columns(&self) -> &[usize] {
        &self.public_bit_columns
    }

    pub fn application_columns(&self) -> &[usize] {
        &self.application_columns
    }

    pub fn is_base_column(&self) -> usize {
        self.is_base_column
    }

    pub fn first_unsatisfied_row(&self) -> Option<usize> {
        self.snapshot.first_unsatisfied_row(self.snapshot.witness())
    }

    pub fn is_satisfied(&self) -> bool {
        self.first_unsatisfied_row().is_none()
    }

    pub fn encode_oracle(&self) -> Result<EncodedLowNormR1cs, LowNormR1csError> {
        encode_r1cs_oracle(&self.snapshot, &self.public_bit_columns)
    }

    pub fn encode_derived(&self) -> Result<EncodedLowNormR1cs, LowNormR1csError> {
        encode_r1cs_derived(&self.snapshot, &self.public_bit_columns)
    }
}

#[derive(Debug, Error)]
pub enum FullFPrimeError {
    #[error(transparent)]
    VerifierKey(#[from] VerifierKeyError),
    #[error(transparent)]
    AjtaiSetup(#[from] neo_ajtai::AjtaiError),
    #[error(transparent)]
    ApplicationShape(#[from] FrontendError),
    #[error("full F' application assignment length {got} does not match relation width {expected}")]
    AssignmentLength { expected: usize, got: usize },
    #[error("full F' application assignment column zero must be the constant one")]
    ApplicationConstantOne,
    #[error("full F' semantic state must be non-empty and have equal input/output arity (in={input}, out={output})")]
    StateArity { input: usize, output: usize },
    #[error("full F' semantic-state {side} column {column} is out of range for application width {width}")]
    StateColumn {
        side: &'static str,
        column: usize,
        width: usize,
    },
    #[error("full F' semantic-state {side} contains duplicate column {column}")]
    DuplicateStateColumn { side: &'static str, column: usize },
    #[error("full F' requires the stateful x_out schema")]
    StatelessSchema,
    #[error("full F' configured norm bound b={configured} does not match NIFS params b={params}")]
    ConfigBoundMismatch { configured: u32, params: u32 },
    #[error("full F' verifier-context anchor {field} does not match the step witness")]
    ContextMismatch { field: &'static str },
    #[error("full F' Construction-2 relation requires rows_in_chunk = 1, got {got}")]
    FreshArity { got: u64 },
    #[error("full F' recursive NIFS {what}: expected {expected}, got {got}")]
    NifsShape {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("full F' NIFS verifier configuration does not match its authoritative {field}")]
    NifsConfigMismatch { field: &'static str },
    #[error("full F' branch mismatch: expected {expected}, got {got}")]
    WrongBranch {
        expected: &'static str,
        got: &'static str,
    },
    #[error("full F' base/recursive public enc_inst lengths differ ({base} versus {recursive})")]
    BranchPublicInputLength { base: usize, recursive: usize },
    #[error("full F' branch public column {column} is invalid or duplicated")]
    BranchPublicColumn { column: usize },
    #[error("full F' branch application column {column} is not relation-constrained")]
    UnconstrainedApplicationColumn { column: usize },
    #[error("full F' execution does not match its verifier-owned {branch} branch shape")]
    BranchRelationMismatch { branch: &'static str },
    #[error("full F' {branch} template witness does not satisfy the unified relation")]
    UnsatisfiedTemplate { branch: &'static str },
    #[error(transparent)]
    FPrime(#[from] FPrimeError),
    #[error(transparent)]
    GadgetNative(#[from] GadgetNativeError),
}

fn branch_name(kind: BranchKind) -> &'static str {
    match kind {
        BranchKind::Base => "base",
        BranchKind::Recursive => "recursive",
    }
}

fn validate_branch_schema(execution: &FullFPrimeBranchExecution) -> Result<(), FullFPrimeError> {
    let mut seen_public = vec![false; execution.snapshot.cols()];
    let explicit_bits = execution.snapshot.explicitly_boolean_columns();
    for &column in &execution.public_bit_columns {
        if column == 0 || column >= execution.snapshot.cols() || seen_public[column] || !explicit_bits[column] {
            return Err(FullFPrimeError::BranchPublicColumn { column });
        }
        seen_public[column] = true;
    }
    let unconstrained = execution.snapshot.unconstrained_columns();
    for &column in &execution.application_columns {
        if unconstrained.binary_search(&column).is_ok() {
            return Err(FullFPrimeError::UnconstrainedApplicationColumn { column });
        }
    }
    Ok(())
}

fn validate_execution_matches_template(
    execution: &FullFPrimeBranchExecution,
    template: &BranchTemplate,
) -> Result<(), FullFPrimeError> {
    if !execution.snapshot.has_same_relation(&template.snapshot)
        || execution.public_bit_columns != template.public_bit_columns
        || execution.application_columns != template.application_columns
        || execution.verifier_key_columns != template.verifier_key_columns
    {
        return Err(FullFPrimeError::BranchRelationMismatch {
            branch: branch_name(execution.kind),
        });
    }
    Ok(())
}

fn compile_unified_relation(
    base: &BranchTemplate,
    recursive: &BranchTemplate,
    base_execution: &FullFPrimeBranchExecution,
) -> (R1csSnapshot, UnifiedLayout) {
    let public_values = base
        .public_bit_columns
        .iter()
        .map(|&column| base_execution.snapshot.witness()[column])
        .collect::<Vec<_>>();
    let mut builder = R1csBuilder::new();
    let public_vars = public_values
        .into_iter()
        .map(|value| {
            let variable = builder.alloc(value);
            enforce_bit(&mut builder, variable);
            variable
        })
        .collect::<Vec<_>>();
    let is_base = builder.alloc(F::ONE);
    enforce_bit(&mut builder, is_base);
    let canonical_zero = builder.alloc(F::ZERO);
    builder.enforce_eq(&Lc::from_var(canonical_zero), &Lc::zero());

    let base_private = canonical_private_columns(base);
    let recursive_private = canonical_private_columns(recursive);
    let pool_len = base_private.len().max(recursive_private.len());
    let mut private_pool = Vec::with_capacity(pool_len);
    for position in 0..pool_len {
        let value = base_private
            .get(position)
            .map_or(F::ZERO, |&column| base_execution.snapshot.witness()[column]);
        let variable = builder.alloc(value);
        if position >= base_private.len() {
            builder.enforce(&Lc::from_var(variable), &Lc::from_var(is_base), &Lc::zero());
        }
        if position >= recursive_private.len() {
            builder.enforce(&Lc::from_var(variable), &one_minus(is_base), &Lc::zero());
        }
        private_pool.push(variable);
    }
    let base_map = map_branch_columns(base, &base_private, &public_vars, &private_pool, canonical_zero);
    let recursive_map = map_branch_columns(
        recursive,
        &recursive_private,
        &public_vars,
        &private_pool,
        canonical_zero,
    );
    emit_gated_branch_rows(&mut builder, &base.snapshot, &base_map, &Lc::from_var(is_base));
    emit_gated_branch_rows(&mut builder, &recursive.snapshot, &recursive_map, &one_minus(is_base));
    let layout = UnifiedLayout {
        public_bit_columns: public_vars.iter().map(|variable| variable.col()).collect(),
        is_base_column: is_base.col(),
        canonical_zero_column: canonical_zero.col(),
        private_pool_columns: private_pool.iter().map(|variable| variable.col()).collect(),
        base_map: base_map.iter().map(|variable| variable.col()).collect(),
        recursive_map: recursive_map
            .iter()
            .map(|variable| variable.col())
            .collect(),
        base_private_columns: base_private,
        recursive_private_columns: recursive_private,
    };
    (builder.snapshot(), layout)
}

fn build_unified_witness(
    base: &BranchTemplate,
    recursive: &BranchTemplate,
    layout: &UnifiedLayout,
    active_kind: BranchKind,
    active: &FullFPrimeBranchExecution,
) -> Vec<F> {
    let mut witness =
        Vec::with_capacity(layout.private_pool_columns.len() + base.snapshot.rows() + recursive.snapshot.rows() + 3);
    witness.push(F::ONE);
    witness.extend(
        active
            .public_bit_columns
            .iter()
            .map(|&column| active.snapshot.witness()[column]),
    );
    assert_eq!(witness.len(), layout.is_base_column);
    witness.push(if active_kind == BranchKind::Base {
        F::ONE
    } else {
        F::ZERO
    });
    assert_eq!(witness.len(), layout.canonical_zero_column);
    witness.push(F::ZERO);

    let active_private = match active_kind {
        BranchKind::Base => &layout.base_private_columns,
        BranchKind::Recursive => &layout.recursive_private_columns,
    };
    for position in 0..layout.private_pool_columns.len() {
        assert_eq!(witness.len(), layout.private_pool_columns[position]);
        witness.push(
            active_private
                .get(position)
                .map_or(F::ZERO, |&column| active.snapshot.witness()[column]),
        );
    }
    append_branch_residuals(&mut witness, &base.snapshot, &layout.base_map);
    append_branch_residuals(&mut witness, &recursive.snapshot, &layout.recursive_map);
    witness
}

fn append_branch_residuals(witness: &mut Vec<F>, snapshot: &R1csSnapshot, mapping: &[usize]) {
    for row in 0..snapshot.rows() {
        let a = eval_mapped_row(snapshot.a_row(row), mapping, witness);
        let b = eval_mapped_row(snapshot.b_row(row), mapping, witness);
        let c = eval_mapped_row(snapshot.c_row(row), mapping, witness);
        witness.push(a * b - c);
    }
}

fn eval_mapped_row(row: &[(usize, F)], mapping: &[usize], witness: &[F]) -> F {
    row.iter().fold(F::ZERO, |acc, &(column, coefficient)| {
        acc + coefficient * witness[mapping[column]]
    })
}

fn canonical_private_columns(template: &BranchTemplate) -> Vec<usize> {
    let mut excluded = vec![false; template.snapshot.cols()];
    excluded[0] = true;
    for &column in &template.public_bit_columns {
        excluded[column] = true;
    }
    for column in template.snapshot.unconstrained_columns() {
        excluded[column] = true;
    }
    excluded
        .into_iter()
        .enumerate()
        .filter_map(|(column, is_excluded)| (!is_excluded).then_some(column))
        .collect()
}

fn map_branch_columns(
    template: &BranchTemplate,
    private_columns: &[usize],
    public_vars: &[Var],
    private_pool: &[Var],
    canonical_zero: Var,
) -> Vec<Var> {
    let mut public_positions = vec![None; template.snapshot.cols()];
    for (position, &column) in template.public_bit_columns.iter().enumerate() {
        public_positions[column] = Some(position);
    }
    let mut private_positions = vec![None; template.snapshot.cols()];
    for (position, &column) in private_columns.iter().enumerate() {
        private_positions[column] = Some(position);
    }
    let mut mapping = vec![canonical_zero; template.snapshot.cols()];
    mapping[0] = Var::ONE;
    for column in 1..template.snapshot.cols() {
        if let Some(position) = public_positions[column] {
            mapping[column] = public_vars[position];
        } else if let Some(position) = private_positions[column] {
            mapping[column] = private_pool[position];
        } else {
            // Syntactically unused verifier sidecars are projected out of
            // the canonical committed witness.
            mapping[column] = canonical_zero;
        }
    }
    mapping
}

fn emit_gated_branch_rows(builder: &mut R1csBuilder, snapshot: &R1csSnapshot, mapping: &[Var], active: &Lc) {
    for row in 0..snapshot.rows() {
        let a = translate_branch_lc(snapshot.a_row(row), mapping);
        let b = translate_branch_lc(snapshot.b_row(row), mapping);
        let c = translate_branch_lc(snapshot.c_row(row), mapping);
        let residual_value = builder.eval(&a) * builder.eval(&b) - builder.eval(&c);
        let residual = builder.alloc(residual_value);
        let mut c_plus_residual = c;
        c_plus_residual.add_term(residual, F::ONE);
        builder.enforce(&a, &b, &c_plus_residual);
        builder.enforce(&Lc::from_var(residual), active, &Lc::zero());
    }
}

fn translate_branch_lc(row: &[(usize, F)], mapping: &[Var]) -> Lc {
    let mut out = Lc::zero();
    for &(column, coefficient) in row {
        out.add_term(mapping[column], coefficient);
    }
    out
}

fn one_minus(variable: Var) -> Lc {
    Lc::from_const(F::ONE).add_scaled(&Lc::from_var(variable), -F::ONE)
}

/// Native half of the canonical semantic-state encoding used by the complete
/// relation. Arity and representation are absorbed before values.
pub fn semantic_state_digest_fields(values: &[F]) -> [F; DIGEST_LEN] {
    let mut preimage = pack_bytes_as_fields(SEMANTIC_STATE_FIELDS_TAG);
    preimage.push(F::from_u64(SEMANTIC_STATE_FIELDS_SCHEMA));
    preimage.push(F::from_u64(values.len() as u64));
    preimage.extend_from_slice(values);
    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage)
}

fn validate_fresh_arity(rows_in_chunk: u64) -> Result<(), FullFPrimeError> {
    if rows_in_chunk != 1 {
        return Err(FullFPrimeError::FreshArity { got: rows_in_chunk });
    }
    Ok(())
}

fn validate_application_schema(
    relation: &R1csShape,
    state_in_columns: &[usize],
    state_out_columns: &[usize],
) -> Result<(), FullFPrimeError> {
    relation.validate_shape()?;
    let width = relation.m();
    if state_in_columns.is_empty() || state_in_columns.len() != state_out_columns.len() {
        return Err(FullFPrimeError::StateArity {
            input: state_in_columns.len(),
            output: state_out_columns.len(),
        });
    }
    validate_state_columns("input", state_in_columns, width)?;
    validate_state_columns("output", state_out_columns, width)
}

fn validate_application_assignment(relation: &R1csShape, assignment: &[F]) -> Result<(), FullFPrimeError> {
    let width = relation.m();
    if assignment.len() != width {
        return Err(FullFPrimeError::AssignmentLength {
            expected: width,
            got: assignment.len(),
        });
    }
    if assignment.first().copied() != Some(F::ONE) {
        return Err(FullFPrimeError::ApplicationConstantOne);
    }
    Ok(())
}

fn validate_state_columns(side: &'static str, columns: &[usize], width: usize) -> Result<(), FullFPrimeError> {
    let mut seen = vec![false; width];
    for &column in columns {
        if column >= width {
            return Err(FullFPrimeError::StateColumn { side, column, width });
        }
        if seen[column] {
            return Err(FullFPrimeError::DuplicateStateColumn { side, column });
        }
        seen[column] = true;
    }
    Ok(())
}

fn validate_fixed_nifs_shape(pp: &Params, inputs: &FPrimeRecursiveInputs<'_>) -> Result<(), FullFPrimeError> {
    let k = pp.k_rho() as usize;
    let checks = [
        ("fresh CCS count", 1usize, inputs.nifs_msg.fresh.len()),
        ("running CE count", k, inputs.nifs_msg.running.len()),
        ("output CE child count", k, inputs.nifs_msg.children.len()),
    ];
    for (what, expected, got) in checks {
        if got != expected {
            return Err(FullFPrimeError::NifsShape { what, expected, got });
        }
    }
    if inputs.nifs_msg.running_parent_authority.is_none() {
        return Err(FullFPrimeError::NifsShape {
            what: "derived running decomposition parent count",
            expected: 1,
            got: 0,
        });
    }
    Ok(())
}

fn validate_nifs_config(
    cfg: &crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig<'_>,
    expected_header_bundle: [F; DIGEST_LEN],
) -> Result<(), FullFPrimeError> {
    let relation = &cfg.structure;
    let dims = neo_reductions::engines::utils::build_dims_and_policy_for_shape(
        cfg.params.inner(),
        relation.n(),
        relation.m(),
        relation.t(),
        relation.max_degree(),
    )
    .map_err(|_| FullFPrimeError::NifsConfigMismatch { field: "dimensions" })?;
    let expected_dimensions = (dims.ell_d, dims.ell_n, dims.ell_m, dims.d_sc);
    let configured_dimensions = (cfg.ell_d, cfg.ell_n, cfg.ell_m, cfg.d_sc);
    if configured_dimensions != expected_dimensions {
        return Err(FullFPrimeError::NifsConfigMismatch { field: "dimensions" });
    }
    // The SplitNc verifier relation intentionally owns no matrices. Its
    // header must therefore match the full-relation header pinned by context.
    if cfg.header_bundle != expected_header_bundle {
        return Err(FullFPrimeError::NifsConfigMismatch { field: "header bundle" });
    }
    Ok(())
}

fn alloc_verifier_key_wires(
    builder: &mut R1csBuilder,
    context: &FullFPrimeContext,
    params: &Params,
) -> FullFPrimeVerifierKeyWires {
    let structure_digest = context.structure_digest.map(|value| builder.alloc(value));
    let pi_ccs_header_bundle = context
        .pi_ccs_header_bundle
        .map(|value| builder.alloc(value));
    let ajtai_pp_digest = context.ajtai_pp_digest.map(|value| builder.alloc(value));
    let initial_semantic_state_digest = context
        .initial_semantic_state_digest
        .map(|value| builder.alloc(value));
    let vk_fs_digest = enforce_vk_fs_digest_circuit(
        builder,
        params,
        structure_digest,
        pi_ccs_header_bundle,
        ajtai_pp_digest,
        Some(crate::paper::f_prime::r1cs::F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN),
        initial_semantic_state_digest,
    );
    FullFPrimeVerifierKeyWires {
        structure_digest,
        pi_ccs_header_bundle,
        ajtai_pp_digest,
        initial_semantic_state_digest,
        vk_fs_digest,
    }
}

fn finish_full_relation(
    builder: &mut R1csBuilder,
    kind: BranchKind,
    verifier_key: &FullFPrimeVerifierKeyWires,
    output: FPrimeStepOutput,
    application: &ApplicationStep<'_>,
) -> Result<FullFPrimeBranchExecution, FullFPrimeError> {
    let (finalize, context_link, application_stage, semantic_links) = match kind {
        BranchKind::Base => (
            fprime_stage::BASE_FINALIZE,
            fprime_stage::BASE_CONTEXT_LINK,
            fprime_stage::BASE_APPLICATION,
            fprime_stage::BASE_SEMANTIC_LINKS,
        ),
        BranchKind::Recursive => (
            fprime_stage::RECURSIVE_FINALIZE,
            fprime_stage::RECURSIVE_CONTEXT_LINK,
            fprime_stage::RECURSIVE_APPLICATION,
            fprime_stage::RECURSIVE_SEMANTIC_LINKS,
        ),
    };
    builder.begin_encoding_stage(finalize);
    builder.begin_encoding_stage(context_link);
    bind_verifier_key(builder, &output, verifier_key);
    if kind == BranchKind::Base {
        enforce_digest_eq(
            builder,
            &output.state_in.semantic_state_digest,
            &verifier_key.initial_semantic_state_digest,
        );
    }
    builder.begin_encoding_stage(application_stage);
    let application_vars = enforce_application(builder, application);
    builder.begin_encoding_stage(semantic_links);
    bind_application_state(builder, &output, application, &application_vars);
    let public_bit_columns = output.x_out_bits.iter().map(|wire| wire.col()).collect();
    let application_columns = application_vars.iter().map(|wire| wire.col()).collect();
    let verifier_key_columns = verifier_key
        .structure_digest
        .iter()
        .chain(verifier_key.pi_ccs_header_bundle.iter())
        .chain(verifier_key.ajtai_pp_digest.iter())
        .chain(verifier_key.initial_semantic_state_digest.iter())
        .map(|wire| wire.col())
        .collect();
    builder.begin_encoding_stage("complete");
    Ok(FullFPrimeBranchExecution {
        kind,
        snapshot: Arc::new(builder.snapshot()),
        encoding_trace: Arc::new(builder.encoding_trace().clone()),
        row_family_ranges: Arc::new(builder.row_family_ranges().to_vec()),
        public_bit_columns,
        application_columns,
        verifier_key_columns,
    })
}

fn bind_verifier_key(builder: &mut R1csBuilder, output: &FPrimeStepOutput, verifier_key: &FullFPrimeVerifierKeyWires) {
    for lane in 0..DIGEST_LEN {
        builder.enforce_eq(
            &Lc::from_var(output.state_in.vk_fs_digest[lane]),
            &Lc::from_var(verifier_key.vk_fs_digest[lane]),
        );
        builder.enforce_eq(
            &Lc::from_var(output.state_in.pi_ccs_header_bundle[lane]),
            &Lc::from_var(verifier_key.pi_ccs_header_bundle[lane]),
        );
    }
}

fn enforce_application(builder: &mut R1csBuilder, application: &ApplicationStep<'_>) -> Vec<Var> {
    let mut variables = Vec::with_capacity(application.assignment.len());
    variables.push(Var::ONE);
    variables.extend(
        application
            .assignment
            .iter()
            .skip(1)
            .copied()
            .map(|value| builder.alloc(value)),
    );

    let rows = r1cs_coeff_rows(application.relation);
    for row in 0..application.relation.n() {
        builder.enforce(
            &application_lc(&rows.a[row], &variables),
            &application_lc(&rows.b[row], &variables),
            &application_lc(&rows.c[row], &variables),
        );
    }
    variables
}

fn application_lc(row: &[(usize, F)], variables: &[Var]) -> Lc {
    let mut out = Lc::zero();
    for &(column, coefficient) in row {
        out.add_term(variables[column], coefficient);
    }
    out
}

fn bind_application_state(
    builder: &mut R1csBuilder,
    output: &FPrimeStepOutput,
    application: &ApplicationStep<'_>,
    variables: &[Var],
) {
    let input_vars = application
        .state_in_columns
        .iter()
        .map(|&column| variables[column])
        .collect::<Vec<_>>();
    let output_vars = application
        .state_out_columns
        .iter()
        .map(|&column| variables[column])
        .collect::<Vec<_>>();
    let semantic_in = enforce_semantic_state_digest(builder, &input_vars);
    let semantic_out = enforce_semantic_state_digest(builder, &output_vars);

    enforce_digest_eq(builder, &output.state_in.semantic_state_digest, &semantic_in);
    enforce_digest_eq(builder, &output.state_out.semantic_state_digest, &semantic_out);

    // Application state owns only `semantic_state_digest`. The F' step owns
    // `z_i` and `public_trace` as its chunk-shape trace; never alias the two.
}

fn enforce_semantic_state_digest(builder: &mut R1csBuilder, values: &[Var]) -> [Var; DIGEST_LEN] {
    let mut preimage = pack_bytes_as_fields(SEMANTIC_STATE_FIELDS_TAG)
        .into_iter()
        .map(|value| alloc_constant(builder, value))
        .collect::<Vec<_>>();
    preimage.push(alloc_constant(builder, F::from_u64(SEMANTIC_STATE_FIELDS_SCHEMA)));
    preimage.push(alloc_constant(builder, F::from_u64(values.len() as u64)));
    preimage.extend_from_slice(values);
    enforce_poseidon2_hash(builder, &preimage)
}

fn alloc_constant(builder: &mut R1csBuilder, value: F) -> Var {
    let variable = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(variable), &Lc::from_const(value));
    variable
}

fn enforce_digest_eq(builder: &mut R1csBuilder, left: &[Var; DIGEST_LEN], right: &[Var; DIGEST_LEN]) {
    for lane in 0..DIGEST_LEN {
        builder.enforce_eq(&Lc::from_var(left[lane]), &Lc::from_var(right[lane]));
    }
}

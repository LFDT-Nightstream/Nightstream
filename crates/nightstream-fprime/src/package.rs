use std::sync::Arc;

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use thiserror::Error;

use crate::identity::relation_identifier;
use crate::sparse::{eval_sparse_combination, SparseCombination, SparseRow, WitnessInstruction};
use crate::witness::{execute_witness_batch, WitnessBatch};
use crate::WitnessAssignment;

mod assignment_transport;
pub use assignment_transport::{LoadedAssignmentPlan, LogicalAssignment};
mod compact;
pub(crate) mod json;
use compact::{CompactRowInvocation, CompactRowTemplate};
use json::{canonical_field, word_to_usize};
mod matrix_program;
pub use matrix_program::MatrixRun;
pub(crate) mod native_application;
use native_application::PreparedApplication;
mod prepared;
pub use prepared::{load_prepared_package, PREPARED_FIXED_SOURCE_NODE_LIMIT};
mod v1_2;
mod validate;
mod wire;
pub use v1_2::{
    PiCcsV1_2EncodedInputs, PiCcsV1_2OutputEvaluations, PiCcsV1_2PackageInputs, PiDecV1_2PackageInputs,
    DECOMPOSITION_BASE, DECOMPOSITION_K_RHO, DECOMPOSITION_NORM_BOUND, PI_CCS_V1_2_COEFFICIENT_COUNT,
    PI_CCS_V1_2_FRESH_COMMITMENT_WORDS, PI_CCS_V1_2_MATRIX_COUNT, PI_CCS_V1_2_PRIOR_CHILDREN_WORDS,
    PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_2_ROUND_COUNT,
    PI_CCS_V1_2_SOURCE_COUNT, PI_CCS_V1_2_STATE_PREIMAGE_WORDS, PI_CCS_V1_2_VERIFIER_CONTEXT_WORDS,
    PI_DEC_V1_2_CHILD_COUNT, PI_DEC_V1_2_COMMITMENT_WORDS_PER_CHILD, PI_DEC_V1_2_EVAL_A_MATRICES_PER_CHILD,
    PI_DEC_V1_2_EVAL_K_VALUES_PER_CHILD, PI_DEC_V1_2_PUBLIC_INPUT_WORDS_PER_CHILD,
};
mod r1cs;
pub use r1cs::{PackageR1cs, PackageSparseMatrix};
mod relation;
pub use relation::{CcsMatrixSource, PackageCcsRelation, PackagePolynomialTerm};
mod sealed;
mod terminal;
pub use sealed::{
    load_per_application_package, load_poseidon2_hash_chain_v1_package, load_prepared_application_records,
    LoadedApplicationPlan, LoadedPerApplicationPackage, LogicalMatrixEntry, LogicalMatrixRow,
};
pub use terminal::{load_terminal_package, TerminalOutputs};
mod pi_ccs_v1_2_transcript;
mod source_row;
pub use pi_ccs_v1_2_transcript::{derive_pi_ccs_v1_2_transcript, PiCcsV1_2Transcript};

const MAX_JOINT_DOMAIN_VARIABLES: u32 = 28;
const MAX_JOINT_DOMAIN: usize = 1usize << MAX_JOINT_DOMAIN_VARIABLES;

#[derive(Debug, Error)]
pub enum PackageError {
    #[error("failed to read circuit package: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid circuit-package JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("circuit-package bytes are not the canonical Lean encoding")]
    NonCanonicalBytes,
    #[error("malformed circuit package: {0}")]
    Malformed(&'static str),
    #[error("circuit package exceeds a size limit: {0}")]
    TooLarge(&'static str),
    /// A caller argument is invalid; the package is not at fault.
    #[error("invalid input: {0}")]
    Input(&'static str),
    #[error("{what}: expected {expected}, got {actual}")]
    InputLength {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("unsatisfied assertion row {row}")]
    UnsatisfiedAssertionRow { row: usize },
    #[error("noncanonical Goldilocks word in {location}: {value}")]
    NonCanonicalField { location: &'static str, value: u64 },
    #[error("{kind} does not match: expected {expected:?}, computed {computed:?}")]
    IdentityMismatch {
        kind: IdentityKind,
        expected: [u64; 4],
        computed: [u64; 4],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityKind {
    Relation,
    Package,
    VerificationKeyBinding,
    VerifierContext,
}

impl std::fmt::Display for IdentityKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Relation => "relation identifier",
            Self::Package => "final package identity",
            Self::VerificationKeyBinding => "verification-key binding",
            Self::VerifierContext => "verifier-context package identity",
        })
    }
}

pub(crate) fn require_input_length(what: &'static str, expected: usize, actual: usize) -> Result<(), PackageError> {
    if expected != actual {
        return Err(PackageError::InputLength { what, expected, actual });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColumnRef {
    Input(usize),
    Local(usize),
}

#[derive(Clone, Debug)]
struct TemplateTerm {
    column: ColumnRef,
    coefficient: Goldilocks,
}

#[derive(Clone, Debug)]
struct TemplateCombination {
    constant: Goldilocks,
    terms: Vec<TemplateTerm>,
}

#[derive(Clone, Debug)]
struct TemplateRow {
    output_local: usize,
    a: TemplateCombination,
    b: TemplateCombination,
    c: TemplateCombination,
}

#[derive(Clone, Debug)]
struct PermutationTemplate {
    input_count: usize,
    local_column_count: usize,
    output_local_start: usize,
    rows: Vec<TemplateRow>,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    role: u64,
    start: usize,
    length: usize,
}

#[derive(Clone, Debug)]
struct Layout {
    row_count: usize,
    private_column_count: usize,
    constant_column: usize,
    public_column_count: usize,
    total_column_count: usize,
    private_segments: Vec<Segment>,
    public_segments: Vec<Segment>,
}

#[derive(Clone, Copy, Debug)]
struct HashChain {
    phase: u64,
    row_start: usize,
    row_count: usize,
    input_start: usize,
    input_length: usize,
    witness_start: usize,
    witness_length: usize,
    absorb_count: usize,
    digest_length: usize,
    digest_start: usize,
}

#[derive(Clone, Debug)]
struct PermutationInvocation {
    phase: u64,
    row_start: usize,
    witness_start: usize,
    inputs: Vec<SparseCombination>,
}

#[derive(Clone, Copy)]
enum ScheduledInvocation<'a> {
    Hash {
        chain: HashChain,
        ordinal: usize,
        row_start: usize,
        witness_start: usize,
    },
    Explicit(&'a PermutationInvocation),
}

impl ScheduledInvocation<'_> {
    fn row_start(self) -> usize {
        match self {
            Self::Hash { row_start, .. } => row_start,
            Self::Explicit(invocation) => invocation.row_start,
        }
    }

    fn witness_start(self) -> usize {
        match self {
            Self::Hash { witness_start, .. } => witness_start,
            Self::Explicit(invocation) => invocation.witness_start,
        }
    }
}

#[derive(Clone, Copy)]
enum ScheduledWitness<'a> {
    Permutation(ScheduledInvocation<'a>),
    Compact(&'a CompactRowInvocation),
    Generic(&'a WitnessInstruction),
    Application(&'a PreparedApplication),
}

#[derive(Clone, Copy)]
enum ScheduledAssignment<'a> {
    Permutation(ScheduledInvocation<'a>),
    Compact(&'a CompactRowInvocation),
    Batch(&'a WitnessBatch),
    Generic(&'a WitnessInstruction),
    Application(&'a PreparedApplication),
}

impl ScheduledAssignment<'_> {
    fn target_start(self) -> usize {
        match self {
            Self::Permutation(invocation) => invocation.witness_start(),
            Self::Compact(invocation) => invocation.output_column,
            Self::Batch(batch) => batch.start,
            Self::Generic(instruction) => instruction.target,
            Self::Application(application) => application.plan().private_range().start,
        }
    }
}

impl ScheduledWitness<'_> {
    fn row_start(self) -> usize {
        match self {
            Self::Permutation(invocation) => invocation.row_start(),
            Self::Compact(invocation) => invocation.row_start,
            Self::Generic(instruction) => instruction.row_index,
            Self::Application(application) => application.plan().row_range().start,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoadedPackage {
    layout: Layout,
    relation: PackageCcsRelation,
    terminal: Option<LoadedTerminalLayout>,
    permutation: PermutationTemplate,
    hash_chains: Vec<HashChain>,
    permutation_invocations: Vec<PermutationInvocation>,
    compact_templates: Vec<CompactRowTemplate>,
    compact_invocations: Vec<CompactRowInvocation>,
    witness_batches: Vec<WitnessBatch>,
    witness_instructions: Vec<WitnessInstruction>,
    assertion_rows: Vec<SparseRow>,
    native_application: Option<Arc<PreparedApplication>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadedTerminalLayout {
    row_start: usize,
    row_count: usize,
    running_claim_count: usize,
    fresh_claim_count: usize,
}

impl LoadedTerminalLayout {
    pub fn row_start(&self) -> usize {
        self.row_start
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn running_claim_count(&self) -> usize {
        self.running_claim_count
    }

    pub fn fresh_claim_count(&self) -> usize {
        self.fresh_claim_count
    }
}

impl LoadedPackage {
    pub fn row_count(&self) -> usize {
        self.layout.row_count
    }

    pub fn private_column_count(&self) -> usize {
        self.layout.private_column_count
    }

    pub fn private_input_count(&self) -> usize {
        self.layout
            .private_segments
            .iter()
            .filter(|segment| !v1_2::is_witness_role(segment.role))
            .map(|segment| segment.length)
            .sum()
    }

    /// The position of `column` in the private inputs of `execute_witness`,
    /// or `None` when no caller input segment holds it.
    pub fn private_input_index(&self, column: usize) -> Option<usize> {
        let mut offset = 0;
        for segment in &self.layout.private_segments {
            if v1_2::is_witness_role(segment.role) {
                continue;
            }
            if segment.start <= column && column < segment.start + segment.length {
                return Some(offset + column - segment.start);
            }
            offset += segment.length;
        }
        None
    }

    /// The private inputs of `execute_witness` that hold the prior state
    /// preimage, or `None` when no caller input segment has that role.
    pub fn prior_preimage_inputs(&self) -> Option<std::ops::Range<usize>> {
        let mut offset = 0;
        for segment in &self.layout.private_segments {
            if v1_2::is_witness_role(segment.role) {
                continue;
            }
            if segment.role == v1_2::PRIOR_PREIMAGE_ROLE {
                return Some(offset..offset + segment.length);
            }
            offset += segment.length;
        }
        None
    }

    pub fn public_column_count(&self) -> usize {
        self.layout.public_column_count
    }

    pub fn total_column_count(&self) -> usize {
        self.layout.total_column_count
    }

    pub fn ccs_relation(&self) -> &PackageCcsRelation {
        &self.relation
    }

    pub fn terminal(&self) -> Option<&LoadedTerminalLayout> {
        self.terminal.as_ref()
    }

    pub fn execute_witness(
        &self,
        private_inputs: &[u64],
        public_values: &[u64],
    ) -> Result<WitnessAssignment, PackageError> {
        self.execute_witness_with_application(private_inputs, public_values, None)
    }

    fn execute_witness_with_application(
        &self,
        private_inputs: &[u64],
        public_values: &[u64],
        application_values: Option<&[Goldilocks]>,
    ) -> Result<WitnessAssignment, PackageError> {
        if application_values.is_some() && self.native_application.is_none() {
            return Err(PackageError::Input("precomputed values require a native application"));
        }
        require_input_length("private input length", self.private_input_count(), private_inputs.len())?;
        require_input_length(
            "public input length",
            self.layout.public_column_count,
            public_values.len(),
        )?;
        if self
            .layout
            .public_segments
            .iter()
            .map(|segment| segment.length)
            .sum::<usize>()
            != public_values.len()
        {
            return Err(PackageError::Malformed("public input length"));
        }
        for value in private_inputs {
            canonical_field(*value, "private input")?;
        }
        for value in public_values {
            canonical_field(*value, "public input")?;
        }

        let mut assignment = vec![Goldilocks::ZERO; self.layout.total_column_count];
        let mut input_cursor = 0usize;
        for segment in &self.layout.private_segments {
            if v1_2::is_witness_role(segment.role) {
                continue;
            }
            let input_end = input_cursor + segment.length;
            let segment_end = segment.start + segment.length;
            for (target, value) in assignment[segment.start..segment_end]
                .iter_mut()
                .zip(&private_inputs[input_cursor..input_end])
            {
                *target = Goldilocks::from_u64(*value);
            }
            input_cursor = input_end;
        }
        debug_assert_eq!(input_cursor, private_inputs.len());
        assignment[self.layout.constant_column] = Goldilocks::ONE;
        for (target, value) in assignment[self.layout.constant_column + 1..]
            .iter_mut()
            .zip(public_values)
        {
            *target = Goldilocks::from_u64(*value);
        }

        for witness in scheduled_assignments(self)? {
            match witness {
                ScheduledAssignment::Permutation(invocation) => {
                    self.execute_invocation(invocation, &mut assignment)?;
                }
                ScheduledAssignment::Compact(invocation) => {
                    compact::execute_invocation(invocation, &self.compact_templates, &mut assignment)?;
                }
                ScheduledAssignment::Batch(batch) => {
                    execute_witness_batch(batch, &mut assignment);
                }
                ScheduledAssignment::Generic(instruction) => {
                    let left = eval_sparse_combination(&instruction.a, &assignment);
                    let right = eval_sparse_combination(&instruction.b, &assignment);
                    assignment[instruction.target] = left * right;
                }
                ScheduledAssignment::Application(application) => match application_values {
                    Some(values) => application.apply_values(values, &mut assignment)?,
                    None => application.execute_recipes(&mut assignment)?,
                },
            }
        }
        self.check_assertions(&assignment)?;

        Ok(WitnessAssignment {
            private_values: assignment[..self.layout.private_column_count]
                .iter()
                .map(|value| value.as_canonical_u64())
                .collect(),
            public_values: assignment[self.layout.constant_column + 1..]
                .iter()
                .map(|value| value.as_canonical_u64())
                .collect(),
        })
    }

    fn execute_invocation(
        &self,
        invocation: ScheduledInvocation<'_>,
        assignment: &mut [Goldilocks],
    ) -> Result<(), PackageError> {
        for row in &self.permutation.rows {
            let left = self.eval_template_combination(&row.a, invocation, assignment);
            let right = self.eval_template_combination(&row.b, invocation, assignment);
            let output_column = invocation.witness_start() + row.output_local;
            assignment[output_column] = left * right;
            let output = self.eval_template_combination(&row.c, invocation, assignment);
            if assignment[output_column] != output {
                return Err(PackageError::Malformed("unsatisfied witness row"));
            }
        }
        Ok(())
    }

    fn eval_template_combination(
        &self,
        combination: &TemplateCombination,
        invocation: ScheduledInvocation<'_>,
        assignment: &[Goldilocks],
    ) -> Goldilocks {
        combination
            .terms
            .iter()
            .fold(combination.constant, |sum, term| {
                let value = match term.column {
                    ColumnRef::Input(lane) => self.scheduled_input(invocation, lane, assignment),
                    ColumnRef::Local(index) => assignment[invocation.witness_start() + index],
                };
                sum + term.coefficient * value
            })
    }

    fn scheduled_input(
        &self,
        invocation: ScheduledInvocation<'_>,
        lane: usize,
        assignment: &[Goldilocks],
    ) -> Goldilocks {
        match invocation {
            ScheduledInvocation::Hash { chain, ordinal, .. } => self.invocation_input(chain, ordinal, lane, assignment),
            ScheduledInvocation::Explicit(explicit) => eval_sparse_combination(&explicit.inputs[lane], assignment),
        }
    }

    fn invocation_input(
        &self,
        chain: HashChain,
        invocation: usize,
        lane: usize,
        assignment: &[Goldilocks],
    ) -> Goldilocks {
        debug_assert!(lane < self.permutation.input_count);
        let previous = if invocation == 0 {
            Goldilocks::ZERO
        } else {
            assignment[chain.witness_start
                + (invocation - 1) * self.permutation.local_column_count
                + self.permutation.output_local_start
                + lane]
        };
        if invocation < chain.absorb_count {
            let rate = neo_ccs::crypto::poseidon2_goldilocks::RATE;
            let input_offset = invocation * rate + lane;
            let absorbed = if lane < rate && input_offset < chain.input_length {
                assignment[chain.input_start + input_offset]
            } else {
                Goldilocks::ZERO
            };
            previous + absorbed
        } else if lane == 0 {
            previous + Goldilocks::ONE
        } else {
            previous
        }
    }
}

fn scheduled_invocations(package: &LoadedPackage) -> Result<Vec<ScheduledInvocation<'_>>, PackageError> {
    let hash_count = package
        .hash_chains
        .iter()
        .map(|chain| chain.absorb_count + 1)
        .try_fold(0usize, |sum, count| sum.checked_add(count))
        .ok_or(PackageError::Malformed("invocation count overflow"))?;
    let capacity = hash_count
        .checked_add(package.permutation_invocations.len())
        .ok_or(PackageError::Malformed("invocation count overflow"))?;
    let mut scheduled = Vec::with_capacity(capacity);
    for chain in &package.hash_chains {
        for ordinal in 0..=chain.absorb_count {
            scheduled.push(ScheduledInvocation::Hash {
                chain: *chain,
                ordinal,
                row_start: chain.row_start + ordinal * package.permutation.rows.len(),
                witness_start: chain.witness_start + ordinal * package.permutation.local_column_count,
            });
        }
    }
    scheduled.extend(
        package
            .permutation_invocations
            .iter()
            .map(ScheduledInvocation::Explicit),
    );
    scheduled.sort_unstable_by_key(|invocation| invocation.row_start());
    Ok(scheduled)
}

fn scheduled_witnesses(package: &LoadedPackage) -> Result<Vec<ScheduledWitness<'_>>, PackageError> {
    let invocations = scheduled_invocations(package)?;
    let capacity = invocations
        .len()
        .checked_add(package.compact_invocations.len())
        .and_then(|count| count.checked_add(package.witness_instructions.len()))
        .and_then(|count| count.checked_add(usize::from(package.native_application.is_some())))
        .ok_or(PackageError::Malformed("witness schedule overflow"))?;
    let mut scheduled = Vec::with_capacity(capacity);
    scheduled.extend(invocations.into_iter().map(ScheduledWitness::Permutation));
    scheduled.extend(
        package
            .compact_invocations
            .iter()
            .map(ScheduledWitness::Compact),
    );
    scheduled.extend(
        package
            .witness_instructions
            .iter()
            .map(ScheduledWitness::Generic),
    );
    if let Some(application) = package
        .native_application
        .as_deref()
        .filter(|app| app.records().row_count() != 0)
    {
        scheduled.push(ScheduledWitness::Application(application));
    }
    scheduled.sort_unstable_by_key(|witness| witness.row_start());
    Ok(scheduled)
}

fn scheduled_assignments(package: &LoadedPackage) -> Result<Vec<ScheduledAssignment<'_>>, PackageError> {
    let invocations = scheduled_invocations(package)?;
    let capacity = invocations
        .len()
        .checked_add(package.compact_invocations.len())
        .and_then(|count| count.checked_add(package.witness_batches.len()))
        .and_then(|count| count.checked_add(package.witness_instructions.len()))
        .and_then(|count| count.checked_add(usize::from(package.native_application.is_some())))
        .ok_or(PackageError::Malformed("assignment schedule overflow"))?;
    let mut scheduled = Vec::with_capacity(capacity);
    scheduled.extend(
        invocations
            .into_iter()
            .map(ScheduledAssignment::Permutation),
    );
    scheduled.extend(
        package
            .compact_invocations
            .iter()
            .map(ScheduledAssignment::Compact),
    );
    scheduled.extend(
        package
            .witness_batches
            .iter()
            .map(ScheduledAssignment::Batch),
    );
    scheduled.extend(
        package
            .witness_instructions
            .iter()
            .map(ScheduledAssignment::Generic),
    );
    if let Some(application) = package
        .native_application
        .as_deref()
        .filter(|app| app.records().recipe_count() != 0)
    {
        scheduled.push(ScheduledAssignment::Application(application));
    }
    scheduled.sort_unstable_by_key(|witness| witness.target_start());
    Ok(scheduled)
}

fn checked_end(start: usize, length: usize) -> Result<usize, PackageError> {
    start
        .checked_add(length)
        .ok_or(PackageError::Malformed("range overflow"))
}

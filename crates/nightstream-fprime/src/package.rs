use std::{fs, path::Path};

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use wip_spartan::provider::{goldi::F as SpartanField, GoldilocksWhirEngine};

use crate::identity::relation_identifier;
use crate::sparse::{eval_sparse_combination, SparseCombination, SparseRow, SparseTerm, WitnessInstruction};
use crate::witness::{
    execute_witness_batch, validate_witness_batch, validate_witness_batch_order, validate_witness_coverage,
    RawWitnessBatch, WitnessBatch,
};
use crate::{ProofRun, WitnessAssignment};

mod compact;
use compact::{CompactRowInvocation, CompactRowTemplate, RawCompactRowInvocation, RawCompactRowTemplate};
mod permutation_plan;
mod plan;
mod source_map;
mod v1_1;
mod witness_plan;
pub use v1_1::{
    PiCcsV1_1EncodedInputs, PiCcsV1_1OutputEvaluations, PiCcsV1_1PackageInputs, PiDecV1_1PackageInputs,
    PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_FRESH_COMMITMENT_WORDS, PI_CCS_V1_1_MATRIX_COUNT,
    PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT,
    PI_CCS_V1_1_SOURCE_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS, PI_CCS_V1_1_VERIFIER_CONTEXT_WORDS,
    PI_DEC_V1_1_CHILD_COUNT, PI_DEC_V1_1_COMMITMENT_WORDS_PER_CHILD, PI_DEC_V1_1_EVAL_A_MATRICES_PER_CHILD,
    PI_DEC_V1_1_EVAL_K_VALUES_PER_CHILD, PI_DEC_V1_1_PUBLIC_INPUT_WORDS_PER_CHILD,
};
mod r1cs;
use r1cs::expand_matrices;
pub use r1cs::{PackageR1cs, PackageSparseMatrix};
mod relation;
pub use relation::{CcsMatrixSource, PackageCcsRelation, PackagePolynomialTerm};
mod proving;
pub use proving::{PackageProof, PackageProvingKey, PackageVerifyingKey};
mod pi_ccs_v1_1_transcript;
pub use pi_ccs_v1_1_transcript::{derive_pi_ccs_v1_1_transcript, PiCcsV1_1Transcript};

pub(super) const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;
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
    #[error("invalid circuit package: {0}")]
    Invalid(&'static str),
    #[error("unsatisfied assertion row {row}")]
    UnsatisfiedAssertionRow { row: usize },
    #[error("noncanonical Goldilocks word in {location}: {value}")]
    NonCanonicalField { location: &'static str, value: u64 },
    #[error(
        "relation identifier does not match the verifier-owned identity: expected {expected:?}, computed {computed:?}"
    )]
    ExpectedIdentityMismatch {
        expected: [u64; 4],
        computed: [u64; 4],
    },
    #[error("direct Spartan failure: {0}")]
    Spartan(String),
}

type SpartanEngine = GoldilocksWhirEngine;

#[derive(Debug, Deserialize, Serialize)]
struct RawPackage(
    u64,
    RawProfile,
    RawPoseidonSchedule,
    RawPhysicalLayout,
    relation::RawCcsRelation,
    RawPermutationTemplate,
    Vec<RawHashChain>,
    Vec<RawPermutationInvocation>,
    Vec<RawCompactRowTemplate>,
    Vec<RawCompactRowInvocation>,
    Vec<RawWitnessBatch>,
    Vec<RawWitnessInstruction>,
    Vec<RawSparseRow>,
    Vec<Value>,
);

#[derive(Debug, Deserialize)]
struct RawPlan(u64, RawPackage, Vec<Value>, Vec<Value>, Vec<Value>);

#[derive(Debug, Deserialize, Serialize)]
struct RawProfile(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Debug, Deserialize, Serialize)]
struct RawPoseidonSchedule(u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Debug, Deserialize, Serialize)]
struct RawSegment(u64, u64, u64);

#[derive(Debug, Deserialize, Serialize)]
struct RawPhysicalLayout(u64, u64, u64, u64, u64, Vec<RawSegment>, Vec<RawSegment>);

#[derive(Debug, Deserialize, Serialize)]
struct RawColumnRef(u64, u64);

#[derive(Debug, Deserialize, Serialize)]
struct RawTemplateTerm(RawColumnRef, u64);

#[derive(Debug, Deserialize, Serialize)]
struct RawTemplateCombination(u64, Vec<RawTemplateTerm>);

#[derive(Debug, Deserialize, Serialize)]
struct RawTemplateRow(
    u64,
    RawTemplateCombination,
    RawTemplateCombination,
    RawTemplateCombination,
);

#[derive(Debug, Deserialize, Serialize)]
struct RawPermutationTemplate(u64, u64, u64, Vec<RawTemplateRow>);

#[derive(Debug, Deserialize, Serialize)]
struct RawHashChain(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Debug, Deserialize, Serialize)]
struct RawPermutationInvocation(u64, u64, u64, Vec<RawSparseCombination>);

#[derive(Debug, Deserialize, Serialize)]
struct RawWitnessInstruction(u64, u64, RawSparseCombination, RawSparseCombination);

#[derive(Debug, Deserialize, Serialize)]
struct RawSparseTerm(u64, u64);

#[derive(Debug, Deserialize, Serialize)]
struct RawSparseCombination(u64, Vec<RawSparseTerm>);

#[derive(Debug, Deserialize, Serialize)]
struct RawSparseRow(u64, RawSparseCombination, RawSparseCombination, RawSparseCombination);

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
}

#[derive(Clone, Copy)]
enum ScheduledAssignment<'a> {
    Permutation(ScheduledInvocation<'a>),
    Compact(&'a CompactRowInvocation),
    Batch(&'a WitnessBatch),
    Generic(&'a WitnessInstruction),
}

impl ScheduledAssignment<'_> {
    fn target_start(self) -> usize {
        match self {
            Self::Permutation(invocation) => invocation.witness_start(),
            Self::Compact(invocation) => invocation.output_column,
            Self::Batch(batch) => batch.start,
            Self::Generic(instruction) => instruction.target,
        }
    }
}

impl ScheduledWitness<'_> {
    fn row_start(self) -> usize {
        match self {
            Self::Permutation(invocation) => invocation.row_start(),
            Self::Compact(invocation) => invocation.row_start,
            Self::Generic(instruction) => instruction.row_index,
        }
    }
}

/// A package that passed canonical decoding, structural checks, and the
/// verifier-owned Poseidon2 identity comparison.
#[derive(Clone, Debug)]
pub struct LoadedPackage {
    layout: Layout,
    relation: PackageCcsRelation,
    permutation: PermutationTemplate,
    hash_chains: Vec<HashChain>,
    permutation_invocations: Vec<PermutationInvocation>,
    compact_templates: Vec<CompactRowTemplate>,
    compact_invocations: Vec<CompactRowInvocation>,
    witness_batches: Vec<WitnessBatch>,
    witness_instructions: Vec<WitnessInstruction>,
    assertion_rows: Vec<SparseRow>,
    relation_identifier: [u64; 4],
}

impl LoadedPackage {
    pub fn relation_identifier(&self) -> [u64; 4] {
        self.relation_identifier
    }

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
            .filter(|segment| !v1_1::is_witness_role(segment.role))
            .map(|segment| segment.length)
            .sum()
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

    pub fn template_row_count(&self) -> usize {
        self.permutation.rows.len()
    }

    pub fn assertion_row_count(&self) -> usize {
        self.assertion_rows.len()
    }

    pub fn permutation_invocation_count(&self) -> usize {
        self.permutation_invocations.len()
    }

    pub fn compact_template_count(&self) -> usize {
        self.compact_templates.len()
    }

    pub fn compact_invocation_count(&self) -> usize {
        self.compact_invocations.len()
    }

    pub fn witness_instruction_count(&self) -> usize {
        self.witness_instructions.len()
    }

    /// Execute the canonical sparse-row witness program. `private_inputs`
    /// contains every non-witness private segment in package order.
    pub fn execute_witness(
        &self,
        private_inputs: &[u64],
        public_values: &[u64],
    ) -> Result<WitnessAssignment, PackageError> {
        if private_inputs.len() != self.private_input_count() {
            return Err(PackageError::Invalid("private input length"));
        }
        if self
            .layout
            .public_segments
            .iter()
            .map(|segment| segment.length)
            .sum::<usize>()
            != public_values.len()
            || public_values.len() != self.layout.public_column_count
        {
            return Err(PackageError::Invalid("public input length"));
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
            if v1_1::is_witness_role(segment.role) {
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
            }
        }
        for row in &self.assertion_rows {
            let left = eval_sparse_combination(&row.a, &assignment);
            let right = eval_sparse_combination(&row.b, &assignment);
            let output = eval_sparse_combination(&row.c, &assignment);
            if left * right != output {
                return Err(PackageError::UnsatisfiedAssertionRow { row: row.row_index });
            }
        }

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
                return Err(PackageError::Invalid("unsatisfied witness row"));
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
            let input_offset = invocation * 4 + lane;
            let absorbed = if lane < 4 && input_offset < chain.input_length {
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
        .ok_or(PackageError::Invalid("invocation count overflow"))?;
    let capacity = hash_count
        .checked_add(package.permutation_invocations.len())
        .ok_or(PackageError::Invalid("invocation count overflow"))?;
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
        .ok_or(PackageError::Invalid("witness schedule overflow"))?;
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
        .ok_or(PackageError::Invalid("assignment schedule overflow"))?;
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
    scheduled.sort_unstable_by_key(|witness| witness.target_start());
    Ok(scheduled)
}

pub fn load_file(path: impl AsRef<Path>, expected_identity: [u64; 4]) -> Result<LoadedPackage, PackageError> {
    load(&fs::read(path)?, expected_identity)
}

pub fn load(bytes: &[u8], expected_identity: [u64; 4]) -> Result<LoadedPackage, PackageError> {
    bind_expanded_package(decode_plan(bytes)?, expected_identity)
}

/// Strictly decode one compact plan and return both its identity-bound package
/// and the exact canonical schema-7 package produced by the production
/// expander. The caller must supply the verifier-owned expected identity.
pub fn load_with_expanded_package(
    bytes: &[u8],
    expected_identity: [u64; 4],
) -> Result<(LoadedPackage, Vec<u8>), PackageError> {
    let raw = decode_plan(bytes)?;
    let mut expanded_bytes = serde_json::to_vec(&raw)?;
    expanded_bytes.push(b'\n');
    let package = bind_expanded_package(raw, expected_identity)?;
    Ok((package, expanded_bytes))
}

fn decode_plan(bytes: &[u8]) -> Result<RawPackage, PackageError> {
    let value: Value = serde_json::from_slice(bytes)?;
    let mut canonical = serde_json::to_vec(&value)?;
    canonical.push(b'\n');
    if bytes != canonical {
        return Err(PackageError::NonCanonicalBytes);
    }

    let RawPlan(schema, mut raw, permutation_blocks, compact_blocks, witness_blocks): RawPlan =
        serde_json::from_value(value.clone())?;
    if schema != 8 {
        return Err(PackageError::Invalid("plan schema version"));
    }
    if !raw.7.is_empty() {
        return Err(PackageError::Invalid("static permutation invocations"));
    }
    if !raw.9.is_empty() {
        return Err(PackageError::Invalid("static compact invocations"));
    }
    raw.7 = permutation_plan::expand(permutation_blocks)?;
    raw.9 = plan::expand(compact_blocks)?;
    raw.10.extend(witness_plan::expand(witness_blocks)?);
    Ok(raw)
}

fn bind_expanded_package(raw: RawPackage, expected_identity: [u64; 4]) -> Result<LoadedPackage, PackageError> {
    let expanded_value = serde_json::to_value(&raw)?;
    let computed_identity = relation_identifier(&expanded_value)?;
    let mut package = validate_package(raw, [0; 4])?;
    if computed_identity != expected_identity {
        return Err(PackageError::ExpectedIdentityMismatch {
            expected: expected_identity,
            computed: computed_identity,
        });
    }

    package.relation_identifier = computed_identity;
    Ok(package)
}

fn validate_package(raw: RawPackage, relation_identifier: [u64; 4]) -> Result<LoadedPackage, PackageError> {
    let RawPackage(
        schema,
        profile,
        poseidon,
        layout,
        relation,
        permutation,
        chains,
        invocations,
        compact_templates,
        compact_invocations,
        batches,
        instructions,
        assertion_rows,
        terminal,
    ) = raw;
    if schema != 7 {
        return Err(PackageError::Invalid("schema version"));
    }
    validate_profile(profile)?;
    validate_poseidon(poseidon)?;
    validate_terminal(&terminal)?;

    let layout = validate_layout(layout)?;
    let relation = relation::validate(relation, &layout)?;
    let permutation = validate_permutation(permutation)?;
    let hash_chains = chains
        .into_iter()
        .map(|chain| validate_chain(chain, &layout, &permutation))
        .collect::<Result<Vec<_>, _>>()?;
    validate_chain_order(&hash_chains)?;

    let permutation_invocations = invocations
        .into_iter()
        .map(|invocation| validate_permutation_invocation(invocation, &layout, &permutation))
        .collect::<Result<Vec<_>, _>>()?;
    validate_permutation_invocation_order(&permutation_invocations)?;

    let compact_templates = compact::validate_templates(compact_templates)?;

    let witness_segments = layout
        .private_segments
        .iter()
        .copied()
        .filter(|segment| v1_1::is_witness_role(segment.role))
        .collect::<Vec<_>>();
    let witness_start = witness_segments
        .first()
        .ok_or(PackageError::Invalid("witness segment"))?
        .start;
    let compact_invocations =
        compact::validate_invocations(compact_invocations, &compact_templates, &layout, witness_start)?;
    let witness_batches = batches
        .into_iter()
        .map(|batch| {
            validate_witness_batch(
                batch,
                witness_start,
                layout.private_column_count,
                layout.constant_column,
                layout.total_column_count,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_witness_batch_order(&witness_batches)?;

    let witness_instructions = instructions
        .into_iter()
        .map(|instruction| validate_witness_instruction(instruction, &layout))
        .collect::<Result<Vec<_>, _>>()?;
    validate_witness_instruction_order(&witness_instructions)?;

    let assertion_rows = assertion_rows
        .into_iter()
        .map(|row| validate_sparse_row(row, &layout))
        .collect::<Result<Vec<_>, _>>()?;
    validate_assertion_order(&assertion_rows)?;
    validate_row_coverage(
        &layout,
        &permutation,
        &hash_chains,
        &permutation_invocations,
        &compact_templates,
        &compact_invocations,
        &witness_instructions,
        &assertion_rows,
    )?;
    let mut witness_intervals = hash_chains
        .iter()
        .map(|chain| (chain.witness_start, chain.witness_start + chain.witness_length))
        .collect::<Vec<_>>();
    witness_intervals.extend(permutation_invocations.iter().map(|invocation| {
        (
            invocation.witness_start,
            invocation.witness_start + permutation.local_column_count,
        )
    }));
    witness_intervals.extend(
        compact_invocations
            .iter()
            .map(|invocation| (invocation.output_column, invocation.output_column + 1)),
    );
    witness_intervals.extend(compact_invocations.iter().filter_map(|invocation| {
        let local_count = invocation.local_column_count(&compact_templates);
        (local_count != 0).then_some((invocation.local_start, invocation.local_start + local_count))
    }));
    witness_intervals.extend(
        witness_batches
            .iter()
            .map(|batch| (batch.start, batch.end())),
    );
    witness_intervals.extend(
        witness_instructions
            .iter()
            .map(|instruction| (instruction.target, instruction.target + 1)),
    );
    if witness_intervals
        .iter()
        .any(|&(start, end)| !interval_owned_by_witness_segment(start, end, &witness_segments))
    {
        return Err(PackageError::Invalid("witness interval ownership"));
    }
    for segment in &witness_segments {
        let end = segment.start + segment.length;
        let owned = witness_intervals
            .iter()
            .copied()
            .filter(|&(start, interval_end)| segment.start <= start && interval_end <= end)
            .collect();
        validate_witness_coverage(segment.start, segment.length, owned)?;
    }

    Ok(LoadedPackage {
        layout,
        relation,
        permutation,
        hash_chains,
        permutation_invocations,
        compact_templates,
        compact_invocations,
        witness_batches,
        witness_instructions,
        assertion_rows,
        relation_identifier,
    })
}

fn interval_owned_by_witness_segment(start: usize, end: usize, segments: &[Segment]) -> bool {
    end > start
        && segments
            .iter()
            .any(|segment| segment.start <= start && end <= segment.start + segment.length)
}

fn validate_profile(raw: RawProfile) -> Result<(), PackageError> {
    let RawProfile(modulus, base, digits, bound, fresh, running, rlc_inputs, dec_children, matrices, cube) = raw;
    if (
        modulus,
        base,
        digits,
        bound,
        fresh,
        running,
        rlc_inputs,
        dec_children,
        matrices,
        cube,
    ) != (GOLDILOCKS_MODULUS, 2, 16, 65_536, 1, 16, 17, 16, 14, 28)
    {
        return Err(PackageError::Invalid("fixed production profile"));
    }
    Ok(())
}

fn validate_poseidon(raw: RawPoseidonSchedule) -> Result<(), PackageError> {
    let RawPoseidonSchedule(width, rate, digest, initial, partial, terminal, recipes, output_start) = raw;
    if (width, rate, digest, initial, partial, terminal, recipes, output_start) != (8, 4, 4, 4, 22, 4, 592, 584) {
        return Err(PackageError::Invalid("Poseidon2 schedule"));
    }
    Ok(())
}

fn validate_terminal(raw: &[Value]) -> Result<(), PackageError> {
    match raw {
        [Value::Number(tag)] if tag.as_u64() == Some(0) => Ok(()),
        _ => Err(PackageError::Invalid("pilot terminal option")),
    }
}

fn validate_layout(raw: RawPhysicalLayout) -> Result<Layout, PackageError> {
    let RawPhysicalLayout(rows, private, constant, public, total, private_segments, public_segments) = raw;
    let row_count = word_to_usize(rows, "row count")?;
    let private_column_count = word_to_usize(private, "private column count")?;
    let constant_column = word_to_usize(constant, "constant column")?;
    let public_column_count = word_to_usize(public, "public column count")?;
    let total_column_count = word_to_usize(total, "total column count")?;

    if private_column_count != constant_column {
        return Err(PackageError::Invalid("constant column position"));
    }
    if private_column_count
        .checked_add(1)
        .and_then(|value| value.checked_add(public_column_count))
        != Some(total_column_count)
    {
        return Err(PackageError::Invalid("total column count"));
    }
    if row_count.max(total_column_count.saturating_sub(1)) > MAX_JOINT_DOMAIN {
        return Err(PackageError::Invalid("2^28 joint domain"));
    }

    let expected_private_roles = v1_1::private_segment_roles();
    let private_segments = validate_segments(private_segments, 0, private_column_count, &expected_private_roles)?;
    v1_1::validate_private_segments(&private_segments)?;
    let public_segments = validate_segments(public_segments, constant_column + 1, total_column_count, &[4, 5, 10])?;
    v1_1::validate_public_segments(&public_segments)?;

    Ok(Layout {
        row_count,
        private_column_count,
        constant_column,
        public_column_count,
        total_column_count,
        private_segments,
        public_segments,
    })
}

fn validate_segments(
    raw: Vec<RawSegment>,
    first: usize,
    end: usize,
    expected_roles: &[u64],
) -> Result<Vec<Segment>, PackageError> {
    if raw.len() != expected_roles.len() {
        return Err(PackageError::Invalid("layout segment count"));
    }
    let mut cursor = first;
    let mut segments = Vec::with_capacity(raw.len());
    for (raw, expected_role) in raw.into_iter().zip(expected_roles) {
        let RawSegment(role, start, length) = raw;
        let start = word_to_usize(start, "segment start")?;
        let length = word_to_usize(length, "segment length")?;
        if role != *expected_role || start != cursor || length == 0 {
            return Err(PackageError::Invalid("layout segment partition"));
        }
        cursor = cursor
            .checked_add(length)
            .ok_or(PackageError::Invalid("segment end overflow"))?;
        if cursor > end {
            return Err(PackageError::Invalid("segment end"));
        }
        segments.push(Segment { role, start, length });
    }
    if cursor != end {
        return Err(PackageError::Invalid("layout segment coverage"));
    }
    Ok(segments)
}

fn validate_permutation(raw: RawPermutationTemplate) -> Result<PermutationTemplate, PackageError> {
    let RawPermutationTemplate(input_count, local_count, output_start, rows) = raw;
    let input_count = word_to_usize(input_count, "template input count")?;
    let local_column_count = word_to_usize(local_count, "template local count")?;
    let output_local_start = word_to_usize(output_start, "template output start")?;
    if (input_count, local_column_count, output_local_start) != (8, 592, 584) || rows.len() != local_column_count {
        return Err(PackageError::Invalid("permutation template shape"));
    }

    let rows = rows
        .into_iter()
        .enumerate()
        .map(|(ordinal, row)| validate_template_row(row, ordinal, input_count, local_column_count))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PermutationTemplate {
        input_count,
        local_column_count,
        output_local_start,
        rows,
    })
}

fn validate_template_row(
    raw: RawTemplateRow,
    ordinal: usize,
    input_count: usize,
    local_count: usize,
) -> Result<TemplateRow, PackageError> {
    let RawTemplateRow(output, a, b, c) = raw;
    let output_local = word_to_usize(output, "template output")?;
    if output_local != ordinal {
        return Err(PackageError::Invalid("template output order"));
    }
    let a = validate_template_combination(a, input_count, local_count, Some(output_local))?;
    let b = validate_template_combination(b, input_count, local_count, Some(output_local))?;
    let c = validate_template_combination(c, input_count, local_count, None)?;
    if c.constant != Goldilocks::ZERO
        || c.terms.len() != 1
        || c.terms[0].column != ColumnRef::Local(output_local)
        || c.terms[0].coefficient != Goldilocks::ONE
    {
        return Err(PackageError::Invalid("template witness output equation"));
    }
    Ok(TemplateRow { output_local, a, b, c })
}

fn validate_template_combination(
    raw: RawTemplateCombination,
    input_count: usize,
    local_count: usize,
    causal_before: Option<usize>,
) -> Result<TemplateCombination, PackageError> {
    let RawTemplateCombination(constant, terms) = raw;
    canonical_field(constant, "template constant")?;
    let terms = terms
        .into_iter()
        .map(|RawTemplateTerm(RawColumnRef(tag, index), coefficient)| {
            canonical_field(coefficient, "template coefficient")?;
            let index = word_to_usize(index, "template reference")?;
            let column = match tag {
                0 if index < input_count => ColumnRef::Input(index),
                1 if index < local_count => {
                    if causal_before.is_some_and(|bound| index >= bound) {
                        return Err(PackageError::Invalid("noncausal template reference"));
                    }
                    ColumnRef::Local(index)
                }
                _ => return Err(PackageError::Invalid("template column reference")),
            };
            Ok(TemplateTerm {
                column,
                coefficient: Goldilocks::from_u64(coefficient),
            })
        })
        .collect::<Result<Vec<_>, PackageError>>()?;
    Ok(TemplateCombination {
        constant: Goldilocks::from_u64(constant),
        terms,
    })
}

fn validate_chain(
    raw: RawHashChain,
    layout: &Layout,
    template: &PermutationTemplate,
) -> Result<HashChain, PackageError> {
    let RawHashChain(
        phase,
        row_start,
        row_count,
        input_start,
        input_length,
        witness_start,
        witness_length,
        absorb_count,
        digest_length,
        digest_start,
    ) = raw;
    let chain = HashChain {
        phase,
        row_start: word_to_usize(row_start, "chain row start")?,
        row_count: word_to_usize(row_count, "chain row count")?,
        input_start: word_to_usize(input_start, "chain input start")?,
        input_length: word_to_usize(input_length, "chain input length")?,
        witness_start: word_to_usize(witness_start, "chain witness start")?,
        witness_length: word_to_usize(witness_length, "chain witness length")?,
        absorb_count: word_to_usize(absorb_count, "chain absorb count")?,
        digest_length: word_to_usize(digest_length, "chain digest length")?,
        digest_start: word_to_usize(digest_start, "chain digest start")?,
    };

    let expected_absorbs = chain
        .input_length
        .checked_add(3)
        .ok_or(PackageError::Invalid("hash input length overflow"))?
        / 4;
    let expected_witness = chain
        .absorb_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(template.rows.len()))
        .ok_or(PackageError::Invalid("hash witness length overflow"))?;
    if chain.absorb_count != expected_absorbs
        || chain.witness_length != expected_witness
        || chain.digest_length > 4
        || chain.row_count != chain.witness_length + chain.digest_length
    {
        return Err(PackageError::Invalid("hash chain dimensions"));
    }
    if checked_end(chain.row_start, chain.row_count)? > layout.row_count
        || checked_end(chain.input_start, chain.input_length)? > layout.private_column_count
        || checked_end(chain.witness_start, chain.witness_length)? > layout.private_column_count
        || (chain.digest_length != 0
            && (chain.digest_start <= layout.constant_column
                || checked_end(chain.digest_start, chain.digest_length)? > layout.total_column_count))
    {
        return Err(PackageError::Invalid("hash chain range"));
    }
    Ok(chain)
}

fn validate_chain_order(chains: &[HashChain]) -> Result<(), PackageError> {
    if chains.len() != 2 || chains[0].phase != 1 || chains[1].phase != 2 {
        return Err(PackageError::Invalid("pilot phase order"));
    }
    Ok(())
}

fn validate_permutation_invocation(
    raw: RawPermutationInvocation,
    layout: &Layout,
    template: &PermutationTemplate,
) -> Result<PermutationInvocation, PackageError> {
    let RawPermutationInvocation(phase, row_start, witness_start, inputs) = raw;
    let row_start = word_to_usize(row_start, "invocation row start")?;
    let witness_start = word_to_usize(witness_start, "invocation witness start")?;
    if phase == 0
        || inputs.len() != template.input_count
        || checked_end(row_start, template.rows.len())? > layout.row_count
        || checked_end(witness_start, template.local_column_count)? > layout.private_column_count
    {
        return Err(PackageError::Invalid("permutation invocation shape"));
    }
    let inputs = inputs
        .into_iter()
        .map(|input| validate_sparse_combination(input, layout))
        .collect::<Result<Vec<_>, _>>()?;
    if inputs
        .iter()
        .flat_map(|input| &input.terms)
        .any(|term| term.column < layout.constant_column && term.column >= witness_start)
    {
        return Err(PackageError::Invalid("noncausal invocation input"));
    }
    Ok(PermutationInvocation {
        phase,
        row_start,
        witness_start,
        inputs,
    })
}

fn validate_permutation_invocation_order(invocations: &[PermutationInvocation]) -> Result<(), PackageError> {
    if invocations
        .windows(2)
        .any(|pair| pair[0].row_start >= pair[1].row_start || pair[0].phase > pair[1].phase)
    {
        return Err(PackageError::Invalid("permutation invocation order"));
    }
    Ok(())
}

fn validate_witness_instruction(
    raw: RawWitnessInstruction,
    layout: &Layout,
) -> Result<WitnessInstruction, PackageError> {
    let RawWitnessInstruction(row_index, target, a, b) = raw;
    let row_index = word_to_usize(row_index, "witness row index")?;
    let target = word_to_usize(target, "witness target")?;
    let witness_start = layout
        .private_segments
        .iter()
        .find(|segment| segment.role == 3)
        .ok_or(PackageError::Invalid("witness segment"))?
        .start;
    if row_index >= layout.row_count || target < witness_start || target >= layout.private_column_count {
        return Err(PackageError::Invalid("witness instruction range"));
    }
    let a = validate_sparse_combination(a, layout)?;
    let b = validate_sparse_combination(b, layout)?;
    if a.terms
        .iter()
        .chain(&b.terms)
        .any(|term| term.column < layout.constant_column && term.column >= target)
    {
        return Err(PackageError::Invalid("noncausal witness instruction"));
    }
    Ok(WitnessInstruction {
        row_index,
        target,
        a,
        b,
    })
}

fn validate_witness_instruction_order(instructions: &[WitnessInstruction]) -> Result<(), PackageError> {
    if instructions
        .windows(2)
        .any(|pair| pair[0].row_index >= pair[1].row_index)
    {
        return Err(PackageError::Invalid("witness instruction order"));
    }
    Ok(())
}

fn validate_sparse_row(raw: RawSparseRow, layout: &Layout) -> Result<SparseRow, PackageError> {
    let RawSparseRow(row_index, a, b, c) = raw;
    let row_index = word_to_usize(row_index, "assertion row index")?;
    if row_index >= layout.row_count {
        return Err(PackageError::Invalid("assertion row range"));
    }
    Ok(SparseRow {
        row_index,
        a: validate_sparse_combination(a, layout)?,
        b: validate_sparse_combination(b, layout)?,
        c: validate_sparse_combination(c, layout)?,
    })
}

fn validate_sparse_combination(raw: RawSparseCombination, layout: &Layout) -> Result<SparseCombination, PackageError> {
    let RawSparseCombination(constant, terms) = raw;
    canonical_field(constant, "assertion constant")?;
    let terms = terms
        .into_iter()
        .map(|RawSparseTerm(column, coefficient)| {
            canonical_field(coefficient, "assertion coefficient")?;
            let column = word_to_usize(column, "assertion column")?;
            if column >= layout.total_column_count || column == layout.constant_column {
                return Err(PackageError::Invalid("assertion column range"));
            }
            Ok(SparseTerm {
                column,
                coefficient: Goldilocks::from_u64(coefficient),
            })
        })
        .collect::<Result<Vec<_>, PackageError>>()?;
    Ok(SparseCombination {
        constant: Goldilocks::from_u64(constant),
        terms,
    })
}

fn validate_assertion_order(rows: &[SparseRow]) -> Result<(), PackageError> {
    if rows
        .windows(2)
        .any(|pair| pair[0].row_index >= pair[1].row_index)
    {
        return Err(PackageError::Invalid("assertion row order"));
    }
    Ok(())
}

fn validate_row_coverage(
    layout: &Layout,
    template: &PermutationTemplate,
    chains: &[HashChain],
    invocations: &[PermutationInvocation],
    compact_templates: &[CompactRowTemplate],
    compact_invocations: &[CompactRowInvocation],
    instructions: &[WitnessInstruction],
    assertions: &[SparseRow],
) -> Result<(), PackageError> {
    let mut intervals = chains
        .iter()
        .map(|chain| (chain.row_start, chain.row_start + chain.witness_length))
        .collect::<Vec<_>>();
    intervals.extend(
        invocations
            .iter()
            .map(|invocation| (invocation.row_start, invocation.row_start + template.rows.len())),
    );
    intervals.extend(compact_invocations.iter().map(|invocation| {
        (
            invocation.row_start,
            invocation.row_start + invocation.row_count(compact_templates),
        )
    }));
    intervals.extend(
        instructions
            .iter()
            .map(|instruction| (instruction.row_index, instruction.row_index + 1)),
    );
    intervals.extend(
        assertions
            .iter()
            .map(|row| (row.row_index, row.row_index + 1)),
    );
    intervals.sort_unstable();
    let mut cursor = 0usize;
    for (start, end) in intervals {
        if start != cursor || end <= start {
            return Err(PackageError::Invalid("physical row coverage"));
        }
        cursor = end;
    }
    if cursor != layout.row_count {
        return Err(PackageError::Invalid("physical row count coverage"));
    }
    Ok(())
}

fn canonical_field(value: u64, location: &'static str) -> Result<(), PackageError> {
    if value >= GOLDILOCKS_MODULUS {
        Err(PackageError::NonCanonicalField { location, value })
    } else {
        Ok(())
    }
}

fn word_to_usize(value: u64, location: &'static str) -> Result<usize, PackageError> {
    usize::try_from(value).map_err(|_| PackageError::Invalid(location))
}

fn checked_end(start: usize, length: usize) -> Result<usize, PackageError> {
    start
        .checked_add(length)
        .ok_or(PackageError::Invalid("range overflow"))
}

//! Checks that turn the wire tuples into the loaded package model.
//!
//! Owns every structural check of the inner package: profile, schedule,
//! layout, templates, producers, ordering, and row and column coverage. It
//! does not execute witnesses.

use std::sync::Arc;

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::native_application::PreparedApplication;
use super::wire::{
    RawColumnRef, RawHashChain, RawPackage, RawPermutationInvocation, RawPermutationTemplate, RawPhysicalLayout,
    RawPoseidonSchedule, RawProfile, RawSegment, RawSparseCombination, RawSparseRow, RawSparseTerm,
    RawTemplateCombination, RawTemplateRow, RawTemplateTerm, RawWitnessInstruction,
};
use super::{
    canonical_field, checked_end, compact, relation, sealed, v1_2, word_to_usize, ColumnRef, HashChain, Layout,
    LoadedPackage, LoadedTerminalLayout, PackageError, PermutationInvocation, PermutationTemplate, Segment,
    TemplateCombination, TemplateRow, TemplateTerm, DECOMPOSITION_BASE, DECOMPOSITION_K_RHO, DECOMPOSITION_NORM_BOUND,
    MAX_JOINT_DOMAIN, PI_CCS_V1_2_MATRIX_COUNT, PI_CCS_V1_2_ROUND_COUNT, PI_CCS_V1_2_SOURCE_COUNT,
    PI_DEC_V1_2_CHILD_COUNT,
};
use crate::sparse::{SparseCombination, SparseRow, SparseTerm, WitnessInstruction};
use crate::witness::{validate_witness_batch, validate_witness_batch_order, validate_witness_coverage};

pub(super) fn validate_package(
    raw: RawPackage,
    native_application: Option<Arc<PreparedApplication>>,
) -> Result<LoadedPackage, PackageError> {
    let (mut package, terminal) = decode_components(raw, native_application)?;
    package.terminal = validate_terminal(&terminal, package.relation.row_count())?;
    validate_chain_order(&package.hash_chains)?;
    validate_row_coverage(&package)?;
    let witness_segments = witness_segments(&package.layout);
    let witness_intervals = owned_witness_intervals(&package, &witness_segments)?;
    for segment in &witness_segments {
        let end = segment.start + segment.length;
        let owned = witness_intervals
            .iter()
            .copied()
            .filter(|&(start, interval_end)| segment.start <= start && interval_end <= end)
            .collect();
        validate_witness_coverage(segment.start, segment.length, owned)?;
    }
    Ok(package)
}

/// Every component check of a package, without the terminal option (returned
/// raw) and the physical coverage checks.
pub(super) fn decode_components(
    raw: RawPackage,
    native_application: Option<Arc<PreparedApplication>>,
) -> Result<(LoadedPackage, Vec<Value>), PackageError> {
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
    if schema != sealed::INNER_PACKAGE_SCHEMA {
        return Err(PackageError::Invalid("schema version"));
    }
    validate_profile(profile)?;
    validate_poseidon(poseidon)?;

    let layout = validate_layout(layout)?;
    if let Some(application) = &native_application {
        application.validate(&layout)?;
    }
    let relation = relation::validate(relation)?;
    let permutation = validate_permutation(permutation)?;
    let hash_chains = chains
        .into_iter()
        .map(|chain| validate_chain(chain, &layout, &permutation))
        .collect::<Result<Vec<_>, _>>()?;

    let permutation_invocations = invocations
        .into_iter()
        .map(|invocation| validate_permutation_invocation(invocation, &layout, &permutation))
        .collect::<Result<Vec<_>, _>>()?;
    validate_permutation_invocation_order(&permutation_invocations)?;

    let compact_templates = compact::validate_templates(compact_templates)?;

    let witness_start = witness_segments(&layout)
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

    let package = LoadedPackage {
        layout,
        relation,
        terminal: None,
        permutation,
        hash_chains,
        permutation_invocations,
        compact_templates,
        compact_invocations,
        witness_batches,
        witness_instructions,
        assertion_rows,
        native_application,
    };
    Ok((package, terminal))
}

pub(super) fn witness_segments(layout: &Layout) -> Vec<Segment> {
    layout
        .private_segments
        .iter()
        .copied()
        .filter(|segment| v1_2::is_witness_role(segment.role))
        .collect()
}

/// The columns each producer writes; every interval lies in one witness
/// segment.
pub(super) fn owned_witness_intervals(
    package: &LoadedPackage,
    segments: &[Segment],
) -> Result<Vec<(usize, usize)>, PackageError> {
    let mut witness_intervals = package
        .hash_chains
        .iter()
        .map(|chain| (chain.witness_start, chain.witness_start + chain.witness_length))
        .collect::<Vec<_>>();
    witness_intervals.extend(package.permutation_invocations.iter().map(|invocation| {
        (
            invocation.witness_start,
            invocation.witness_start + package.permutation.local_column_count,
        )
    }));
    witness_intervals.extend(
        package
            .compact_invocations
            .iter()
            .map(|invocation| (invocation.output_column, invocation.output_column + 1)),
    );
    witness_intervals.extend(package.compact_invocations.iter().filter_map(|invocation| {
        let local_count = invocation.local_column_count(&package.compact_templates);
        (local_count != 0).then_some((invocation.local_start, invocation.local_start + local_count))
    }));
    witness_intervals.extend(
        package
            .witness_batches
            .iter()
            .map(|batch| (batch.start, batch.end())),
    );
    witness_intervals.extend(
        package
            .witness_instructions
            .iter()
            .map(|instruction| (instruction.target, instruction.target + 1)),
    );
    if let Some(application) = package
        .native_application
        .as_deref()
        .filter(|app| app.records().recipe_count() != 0)
    {
        let range = application.plan().private_range();
        witness_intervals.push((range.start, range.end));
    }
    if witness_intervals
        .iter()
        .any(|&(start, end)| !interval_owned_by_witness_segment(start, end, segments))
    {
        return Err(PackageError::Invalid("witness interval ownership"));
    }
    Ok(witness_intervals)
}

fn interval_owned_by_witness_segment(start: usize, end: usize, segments: &[Segment]) -> bool {
    end > start
        && segments
            .iter()
            .any(|segment| segment.start <= start && end <= segment.start + segment.length)
}

pub(super) fn validate_profile(raw: RawProfile) -> Result<(), PackageError> {
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
    ) != (
        Goldilocks::ORDER_U64,
        DECOMPOSITION_BASE,
        DECOMPOSITION_K_RHO as u64,
        DECOMPOSITION_NORM_BOUND,
        1,
        PI_DEC_V1_2_CHILD_COUNT as u64,
        PI_CCS_V1_2_SOURCE_COUNT as u64,
        PI_DEC_V1_2_CHILD_COUNT as u64,
        PI_CCS_V1_2_MATRIX_COUNT as u64,
        PI_CCS_V1_2_ROUND_COUNT as u64,
    ) {
        return Err(PackageError::Invalid("fixed production profile"));
    }
    Ok(())
}

pub(super) fn validate_poseidon(raw: RawPoseidonSchedule) -> Result<(), PackageError> {
    let RawPoseidonSchedule(width, rate, digest, initial, partial, terminal, recipes, output_start) = raw;
    if (width, rate, digest, initial, partial, terminal, recipes, output_start) != (16, 12, 4, 4, 22, 4, 1096, 1080) {
        return Err(PackageError::Invalid("Poseidon2 schedule"));
    }
    Ok(())
}

pub(super) fn validate_terminal(
    raw: &[Value],
    relation_row_count: usize,
) -> Result<Option<LoadedTerminalLayout>, PackageError> {
    match raw {
        [Value::Number(tag), Value::Array(layout)] if tag.as_u64() == Some(1) => {
            let [row_start, row_count, running, fresh] = layout.as_slice() else {
                return Err(PackageError::Invalid("pilot terminal option"));
            };
            let decoded = LoadedTerminalLayout {
                row_start: row_start
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(PackageError::Invalid("pilot terminal option"))?,
                row_count: row_count
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(PackageError::Invalid("pilot terminal option"))?,
                running_claim_count: running
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(PackageError::Invalid("pilot terminal option"))?,
                fresh_claim_count: fresh
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(PackageError::Invalid("pilot terminal option"))?,
            };
            if decoded
                != (LoadedTerminalLayout {
                    row_start: 0,
                    row_count: relation_row_count,
                    running_claim_count: 16,
                    fresh_claim_count: 1,
                })
            {
                return Err(PackageError::Invalid("pilot terminal option"));
            }
            Ok(Some(decoded))
        }
        _ => Err(PackageError::Invalid("pilot terminal option")),
    }
}

pub(super) fn validate_layout(raw: RawPhysicalLayout) -> Result<Layout, PackageError> {
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

    let mut expected_private_roles = v1_2::private_segment_roles();
    let prefix_segment_count = expected_private_roles.len();
    expected_private_roles.extend([sealed::APPLICATION_WITNESS_ROLE, sealed::APPLICATION_LOCAL_ROLE]);
    let private_segments = validate_segments(private_segments, 0, private_column_count, &expected_private_roles, 2)?;
    v1_2::validate_private_segments(&private_segments[..prefix_segment_count])?;
    let public_segments = validate_segments(public_segments, constant_column + 1, total_column_count, &[4, 5, 10], 0)?;
    v1_2::validate_public_segments(&public_segments)?;

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
    zero_length_suffix: usize,
) -> Result<Vec<Segment>, PackageError> {
    if raw.len() != expected_roles.len() {
        return Err(PackageError::Invalid("layout segment count"));
    }
    let mut cursor = first;
    let mut segments = Vec::with_capacity(raw.len());
    let zero_length_start = expected_roles.len() - zero_length_suffix;
    for (ordinal, (raw, expected_role)) in raw.into_iter().zip(expected_roles).enumerate() {
        let RawSegment(role, start, length) = raw;
        let start = word_to_usize(start, "segment start")?;
        let length = word_to_usize(length, "segment length")?;
        if role != *expected_role || start != cursor || (length == 0 && ordinal < zero_length_start) {
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
    if (input_count, local_column_count, output_local_start) != (16, 1096, 1080) || rows.len() != local_column_count {
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

pub(super) fn validate_template_combination(
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

    let rate = neo_ccs::crypto::poseidon2_goldilocks::RATE;
    let expected_absorbs = chain
        .input_length
        .checked_add(rate - 1)
        .ok_or(PackageError::Invalid("hash input length overflow"))?
        / rate;
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

pub(super) fn validate_row_coverage(package: &LoadedPackage) -> Result<(), PackageError> {
    let mut intervals = package
        .hash_chains
        .iter()
        .map(|chain| (chain.row_start, chain.row_start + chain.witness_length))
        .collect::<Vec<_>>();
    intervals.extend(package.permutation_invocations.iter().map(|invocation| {
        (
            invocation.row_start,
            invocation.row_start + package.permutation.rows.len(),
        )
    }));
    intervals.extend(package.compact_invocations.iter().map(|invocation| {
        (
            invocation.row_start,
            invocation.row_start + invocation.row_count(&package.compact_templates),
        )
    }));
    intervals.extend(
        package
            .witness_instructions
            .iter()
            .map(|instruction| (instruction.row_index, instruction.row_index + 1)),
    );
    intervals.extend(
        package
            .assertion_rows
            .iter()
            .map(|row| (row.row_index, row.row_index + 1)),
    );
    if let Some(application) = package
        .native_application
        .as_deref()
        .filter(|app| app.records().row_count() != 0)
    {
        let range = application.plan().row_range();
        intervals.push((range.start, range.end));
    }
    intervals.sort_unstable();
    let mut cursor = 0usize;
    for (start, end) in intervals {
        if start != cursor || end <= start {
            return Err(PackageError::Invalid("physical row coverage"));
        }
        cursor = end;
    }
    if cursor != package.layout.row_count {
        return Err(PackageError::Invalid("physical row count coverage"));
    }
    Ok(())
}

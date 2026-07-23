use std::collections::HashMap;

use super::{affine_step, normalized_run_token, slice_run, Census, Resolution, Run, RunToken, Stage};

const SIS_INPUT: &str = "r1cs.sis_accumulator.input_binding";
const SIS_COMPRESSION: &str = "r1cs.sis_accumulator.digest_compression";
const TEMPLATE_ATOMS: usize = 32;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum Atom {
    Ordinary(RunToken),
    SisBatch { batch: usize },
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum OpeningGroupKind {
    Alias { source: usize, source_stride: usize },
    Direct,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) struct OpeningGroup {
    pub opening_start: usize,
    pub length: usize,
    pub direct_before: usize,
    pub kind: OpeningGroupKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Batch {
    pub source_start: usize,
    pub source_end: usize,
    pub input_binding: bool,
    pub commitment_fields: usize,
    pub openings: usize,
    pub direct_openings: usize,
    pub groups: Vec<OpeningGroup>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Summary {
    pub source_columns: usize,
    pub fresh_coordinates: usize,
    pub census: Census,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Template {
    pub atoms: Vec<Atom>,
    pub summary: Summary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Call {
    pub template: usize,
    pub source_start: usize,
    pub final_start: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum Owner {
    Ordinary {
        call: usize,
        atom: usize,
        offset: usize,
    },
    Batch {
        call: usize,
        atom: usize,
        batch: usize,
        offset: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Consumer {
    Ordinary {
        call: usize,
        atom: usize,
        offset: usize,
    },
    Batch {
        call: usize,
        atom: usize,
        batch: usize,
        group: usize,
        offset: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AliasLink {
    pub consumer: Consumer,
    pub length: usize,
    pub target: Owner,
    pub target_offset_stride: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AliasConsumer {
    pub consumer: Consumer,
    pub length: usize,
    pub link_start: usize,
    pub link_stop: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Program {
    pub templates: Vec<Template>,
    pub calls: Vec<Call>,
    pub batches: Vec<Batch>,
    pub alias_links: Vec<AliasLink>,
    pub alias_consumers: Vec<AliasConsumer>,
    pub summary: Summary,
}

fn add_census(left: Census, right: Census) -> Result<Census, String> {
    let add = |a: usize, b: usize| {
        a.checked_add(b)
            .ok_or_else(|| "private decoder census overflows".to_string())
    };
    Ok(Census {
        eliminated: add(left.eliminated, right.eliminated)?,
        unit: add(left.unit, right.unit)?,
        balanced: add(left.balanced, right.balanced)?,
        binary: add(left.binary, right.binary)?,
        decomposition_aliases: add(left.decomposition_aliases, right.decomposition_aliases)?,
        equality_aliases: add(left.equality_aliases, right.equality_aliases)?,
        equality_alias_savings: add(left.equality_alias_savings, right.equality_alias_savings)?,
        retained_coordinates_before_aliases: add(
            left.retained_coordinates_before_aliases,
            right.retained_coordinates_before_aliases,
        )?,
        centered_columns: add(left.centered_columns, right.centered_columns)?,
    })
}

fn add_summary(left: Summary, right: Summary) -> Result<Summary, String> {
    Ok(Summary {
        source_columns: left
            .source_columns
            .checked_add(right.source_columns)
            .ok_or_else(|| "private decoder source summary overflows".to_string())?,
        fresh_coordinates: left
            .fresh_coordinates
            .checked_add(right.fresh_coordinates)
            .ok_or_else(|| "private decoder final summary overflows".to_string())?,
        census: add_census(left.census, right.census)?,
    })
}

fn width_census(length: usize, width: usize, centered: bool) -> Result<Census, String> {
    let retained = length
        .checked_mul(width)
        .ok_or_else(|| "private decoder retained-width census overflows".to_string())?;
    let mut census = Census {
        retained_coordinates_before_aliases: retained,
        centered_columns: usize::from(centered) * length,
        ..Census::default()
    };
    match width {
        1 => census.unit = length,
        41 => census.balanced = length,
        64 => census.binary = length,
        _ => return Err(format!("private decoder atom has unsupported width {width}")),
    }
    Ok(census)
}

fn ordinary_summary(token: RunToken) -> Result<Summary, String> {
    let (source_columns, fresh_coordinates, census) = match token {
        RunToken::Direct {
            length,
            width,
            centered,
            ..
        } => {
            let width = usize::from(width);
            (length, length * width, width_census(length, width, centered)?)
        }
        RunToken::DecompositionAlias { length, centered, .. } => {
            let mut census = width_census(length, 1, centered)?;
            census.decomposition_aliases = length;
            (length, 0, census)
        }
        RunToken::EqualityAlias {
            length,
            width,
            centered,
            ..
        } => {
            let width = usize::from(width);
            let mut census = width_census(length, width, centered)?;
            census.equality_aliases = length;
            census.equality_alias_savings = length * width;
            (length, 0, census)
        }
        RunToken::LinearDefinition { length } | RunToken::TraceEliminated { length } => (
            length,
            0,
            Census {
                eliminated: length,
                ..Census::default()
            },
        ),
    };
    Ok(Summary {
        source_columns,
        fresh_coordinates,
        census,
    })
}

fn batch_summary(batch: &Batch) -> Result<Summary, String> {
    let source_columns = 2usize
        .checked_add(batch.commitment_fields)
        .and_then(|value| value.checked_add(batch.openings.checked_mul(122)?))
        .ok_or_else(|| "private decoder SIS source summary overflows".to_string())?;
    if source_columns != batch.source_end - batch.source_start {
        return Err("private decoder SIS source span differs from its exact geometry".into());
    }
    let fresh_coordinates = batch
        .commitment_fields
        .checked_mul(41)
        .and_then(|value| value.checked_add(batch.openings.checked_mul(40)?))
        .and_then(|value| value.checked_add(batch.direct_openings.checked_mul(41)?))
        .ok_or_else(|| "private decoder SIS final summary overflows".to_string())?;
    let alias_openings = batch.openings - batch.direct_openings;
    Ok(Summary {
        source_columns,
        fresh_coordinates,
        census: Census {
            eliminated: 2 + 41 * batch.openings,
            unit: 81 * batch.openings,
            balanced: batch.commitment_fields,
            binary: 0,
            decomposition_aliases: 41 * alias_openings,
            equality_aliases: 0,
            equality_alias_savings: 0,
            retained_coordinates_before_aliases: 81 * batch.openings + 41 * batch.commitment_fields,
            centered_columns: 41 * batch.openings,
        },
    })
}

fn atom_summary(atom: Atom, batches: &[Batch]) -> Result<Summary, String> {
    match atom {
        Atom::Ordinary(token) => ordinary_summary(token),
        Atom::SisBatch { batch } => batch_summary(
            batches
                .get(batch)
                .ok_or_else(|| "private decoder template names an absent SIS batch".to_string())?,
        ),
    }
}

fn span_runs(runs: &[Run], span: &Stage) -> Result<Vec<Run>, String> {
    let mut out = Vec::new();
    for &run in runs {
        let end = run
            .source_start
            .checked_add(run.length)
            .ok_or_else(|| "private decoder run endpoint overflows".to_string())?;
        let start = run.source_start.max(span.source_start);
        let stop = end.min(span.source_end);
        if start < stop {
            out.push(slice_run(run, start, stop)?);
        }
    }
    if out.iter().map(|run| run.length).sum::<usize>() != span.source_end - span.source_start {
        return Err(format!("private decoder SIS span {} is not exactly covered", span.path));
    }
    Ok(out)
}

fn push_group(groups: &mut Vec<OpeningGroup>, opening: usize, direct_before: usize, kind: OpeningGroupKind) {
    groups.push(OpeningGroup {
        opening_start: opening,
        length: 1,
        direct_before,
        kind,
    });
}

fn extend_group(group: &mut OpeningGroup, next: OpeningGroupKind) -> bool {
    match (group.kind, next) {
        (OpeningGroupKind::Direct, OpeningGroupKind::Direct) => {
            group.length += 1;
            true
        }
        (OpeningGroupKind::Alias { source, source_stride }, OpeningGroupKind::Alias { source: next, .. }) => {
            let Some(stride) = affine_step(source, source_stride, group.length, next) else {
                return false;
            };
            group.kind = OpeningGroupKind::Alias {
                source,
                source_stride: stride,
            };
            group.length += 1;
            true
        }
        _ => false,
    }
}

fn parse_batch(runs: &[Run], family: &Stage) -> Result<Batch, String> {
    let body = span_runs(runs, family)?;
    let input_binding = match family.path {
        SIS_INPUT => true,
        SIS_COMPRESSION => false,
        _ => return Err("non-SIS family passed to the SIS batch parser".into()),
    };
    let commitment_fields = if input_binding { 108 } else { 54 };
    if body.len() < 2
        || body[0]
            != (Run {
                source_start: family.source_start,
                length: 2,
                resolution: Resolution::LinearDefinition,
            })
        || !matches!(
            body[1],
            Run {
                length,
                resolution: Resolution::Direct {
                    start_stride: 41,
                    width: 41,
                    centered: false,
                    ..
                },
                ..
            } if length == commitment_fields
        )
        || !(body.len() - 2).is_multiple_of(3)
    {
        return Err(format!(
            "private decoder SIS batch {} has a malformed prefix",
            family.path
        ));
    }

    let mut groups = Vec::new();
    let mut direct_openings = 0usize;
    for (opening_index, opening) in body[2..].chunks_exact(3).enumerate() {
        let kind = match opening[0] {
            Run {
                length: 41,
                resolution:
                    Resolution::DecompositionAlias {
                        source,
                        source_stride: 0,
                        digit: 0,
                        digit_stride: 1,
                        start_stride: 1,
                        centered: true,
                        ..
                    },
                ..
            } => OpeningGroupKind::Alias {
                source,
                source_stride: 0,
            },
            Run {
                length: 41,
                resolution:
                    Resolution::Direct {
                        start_stride: 1,
                        width: 1,
                        centered: true,
                        ..
                    },
                ..
            } => {
                direct_openings += 1;
                OpeningGroupKind::Direct
            }
            _ => return Err("private decoder SIS digit word is not exact".into()),
        };
        if opening[1].length != 41
            || opening[1].resolution != Resolution::TraceEliminated
            || !matches!(
                opening[2],
                Run {
                    length: 40,
                    resolution: Resolution::Direct {
                        start_stride: 1,
                        width: 1,
                        centered: false,
                        ..
                    },
                    ..
                }
            )
        {
            return Err("private decoder SIS opening suffix is not exact".into());
        }
        if groups
            .last_mut()
            .is_none_or(|group| !extend_group(group, kind))
        {
            push_group(
                &mut groups,
                opening_index,
                direct_openings - usize::from(kind == OpeningGroupKind::Direct),
                kind,
            );
        }
    }
    let openings = (body.len() - 2) / 3;
    let batch = Batch {
        source_start: family.source_start,
        source_end: family.source_end,
        input_binding,
        commitment_fields,
        openings,
        direct_openings,
        groups,
    };
    batch_summary(&batch)?;
    Ok(batch)
}

fn program_atoms(runs: &[Run], batches: &[Batch]) -> Result<Vec<Atom>, String> {
    let mut atoms = Vec::new();
    let mut cursor = super::SOURCE_START;
    let mut run_index = 0usize;
    let emit_ordinary = |start: usize, end: usize, atoms: &mut Vec<Atom>, run_index: &mut usize| {
        while *run_index < runs.len() && runs[*run_index].source_start + runs[*run_index].length <= start {
            *run_index += 1;
        }
        while *run_index < runs.len() && runs[*run_index].source_start < end {
            let run = runs[*run_index];
            let slice_start = run.source_start.max(start);
            let slice_end = (run.source_start + run.length).min(end);
            if slice_start < slice_end {
                atoms.push(Atom::Ordinary(normalized_run_token(slice_run(
                    run,
                    slice_start,
                    slice_end,
                )?)?));
            }
            if run.source_start + run.length <= end {
                *run_index += 1;
            } else {
                break;
            }
        }
        Ok::<_, String>(())
    };
    for (batch, audit) in batches.iter().enumerate() {
        if audit.source_start < cursor || audit.source_end < audit.source_start {
            return Err("private decoder SIS batches overlap or are out of order".into());
        }
        emit_ordinary(cursor, audit.source_start, &mut atoms, &mut run_index)?;
        atoms.push(Atom::SisBatch { batch });
        cursor = audit.source_end;
    }
    emit_ordinary(cursor, super::SOURCE_STOP, &mut atoms, &mut run_index)?;
    Ok(atoms)
}

fn locate_owner(program: &Program, column: usize) -> Result<Owner, String> {
    let call_index = program
        .calls
        .partition_point(|call| call.source_start <= column)
        .checked_sub(1)
        .ok_or_else(|| "private decoder alias target precedes the program".to_string())?;
    let call = program.calls[call_index];
    let template = &program.templates[call.template];
    let mut cursor = call.source_start;
    for (atom_index, &atom) in template.atoms.iter().enumerate() {
        let length = atom_summary(atom, &program.batches)?.source_columns;
        if column < cursor + length {
            let offset = column - cursor;
            return Ok(match atom {
                Atom::Ordinary(_) => Owner::Ordinary {
                    call: call_index,
                    atom: atom_index,
                    offset,
                },
                Atom::SisBatch { batch } => Owner::Batch {
                    call: call_index,
                    atom: atom_index,
                    batch,
                    offset,
                },
            });
        }
        cursor += length;
    }
    Err("private decoder alias target escapes its call".into())
}

fn push_alias_sequence(
    program: &Program,
    consumer: Consumer,
    consumer_columns: impl IntoIterator<Item = usize>,
    sources: impl IntoIterator<Item = usize>,
    links: &mut Vec<AliasLink>,
) -> Result<(), String> {
    let mut current: Option<AliasLink> = None;
    for (index, (consumer_column, source)) in consumer_columns.into_iter().zip(sources).enumerate() {
        if source >= consumer_column {
            return Err("private decoder alias source is not strictly earlier".into());
        }
        let target = locate_owner(program, source)?;
        let offset = match target {
            Owner::Ordinary { offset, .. } | Owner::Batch { offset, .. } => offset,
        };
        match &mut current {
            Some(link) if core::mem::discriminant(&link.target) == core::mem::discriminant(&target) => {
                let same_owner = match (link.target, target) {
                    (
                        Owner::Ordinary { call, atom, .. },
                        Owner::Ordinary {
                            call: next_call,
                            atom: next_atom,
                            ..
                        },
                    ) => call == next_call && atom == next_atom,
                    (
                        Owner::Batch { call, atom, batch, .. },
                        Owner::Batch {
                            call: next_call,
                            atom: next_atom,
                            batch: next_batch,
                            ..
                        },
                    ) => call == next_call && atom == next_atom && batch == next_batch,
                    _ => false,
                };
                if same_owner {
                    let first_offset = match link.target {
                        Owner::Ordinary { offset, .. } | Owner::Batch { offset, .. } => offset,
                    };
                    if let Some(stride) = affine_step(first_offset, link.target_offset_stride, link.length, offset) {
                        link.target_offset_stride = stride;
                        link.length += 1;
                        continue;
                    }
                }
                links.push(current.take().expect("current alias link"));
                current = Some(AliasLink {
                    consumer: advance_consumer(consumer, index),
                    length: 1,
                    target,
                    target_offset_stride: 0,
                });
            }
            Some(_) => {
                links.push(current.take().expect("current alias link"));
                current = Some(AliasLink {
                    consumer: advance_consumer(consumer, index),
                    length: 1,
                    target,
                    target_offset_stride: 0,
                });
            }
            None => {
                current = Some(AliasLink {
                    consumer: advance_consumer(consumer, index),
                    length: 1,
                    target,
                    target_offset_stride: 0,
                });
            }
        }
    }
    if let Some(link) = current {
        links.push(link);
    }
    Ok(())
}

fn advance_consumer(consumer: Consumer, offset: usize) -> Consumer {
    match consumer {
        Consumer::Ordinary {
            call,
            atom,
            offset: first,
        } => Consumer::Ordinary {
            call,
            atom,
            offset: first + offset,
        },
        Consumer::Batch {
            call,
            atom,
            batch,
            group,
            offset: first,
        } => Consumer::Batch {
            call,
            atom,
            batch,
            group,
            offset: first + offset,
        },
    }
}

fn consumer_offset(consumer: Consumer) -> usize {
    match consumer {
        Consumer::Ordinary { offset, .. } | Consumer::Batch { offset, .. } => offset,
    }
}

fn same_consumer(left: Consumer, right: Consumer) -> bool {
    match (left, right) {
        (
            Consumer::Ordinary { call, atom, .. },
            Consumer::Ordinary {
                call: right_call,
                atom: right_atom,
                ..
            },
        ) => call == right_call && atom == right_atom,
        (
            Consumer::Batch {
                call,
                atom,
                batch,
                group,
                ..
            },
            Consumer::Batch {
                call: right_call,
                atom: right_atom,
                batch: right_batch,
                group: right_group,
                ..
            },
        ) => call == right_call && atom == right_atom && batch == right_batch && group == right_group,
        _ => false,
    }
}

fn expected_alias_consumers(program: &Program) -> Result<Vec<(Consumer, usize)>, String> {
    let mut consumers = Vec::new();
    for (call_index, call) in program.calls.iter().copied().enumerate() {
        let template = program
            .templates
            .get(call.template)
            .ok_or_else(|| "private decoder call names an absent template".to_string())?;
        for (atom_index, &atom) in template.atoms.iter().enumerate() {
            match atom {
                Atom::Ordinary(RunToken::DecompositionAlias { length, .. })
                | Atom::Ordinary(RunToken::EqualityAlias { length, .. }) => {
                    consumers.push((
                        Consumer::Ordinary {
                            call: call_index,
                            atom: atom_index,
                            offset: 0,
                        },
                        length,
                    ));
                }
                Atom::SisBatch { batch } => {
                    let batch_data = program
                        .batches
                        .get(batch)
                        .ok_or_else(|| "private decoder atom names an absent batch".to_string())?;
                    for (group, entry) in batch_data.groups.iter().enumerate() {
                        if matches!(entry.kind, OpeningGroupKind::Alias { .. }) {
                            consumers.push((
                                Consumer::Batch {
                                    call: call_index,
                                    atom: atom_index,
                                    batch,
                                    group,
                                    offset: 0,
                                },
                                entry.length,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(consumers)
}

fn alias_consumers(program: &Program) -> Result<Vec<AliasConsumer>, String> {
    let expected = expected_alias_consumers(program)?;
    let mut link = 0usize;
    let mut consumers = Vec::with_capacity(expected.len());
    for (consumer, length) in expected {
        if length == 0 {
            return Err("private decoder alias consumer is empty".into());
        }
        let link_start = link;
        let mut offset = 0usize;
        while offset < length {
            let entry = program
                .alias_links
                .get(link)
                .ok_or_else(|| "private decoder alias consumer is missing links".to_string())?;
            if !same_consumer(entry.consumer, consumer) || consumer_offset(entry.consumer) != offset {
                return Err("private decoder alias links break consumer order".into());
            }
            offset = offset
                .checked_add(entry.length)
                .filter(|&stop| stop <= length)
                .ok_or_else(|| "private decoder alias links overrun their consumer".to_string())?;
            link += 1;
        }
        consumers.push(AliasConsumer {
            consumer,
            length,
            link_start,
            link_stop: link,
        });
    }
    if link != program.alias_links.len() {
        return Err("private decoder alias links contain an extra consumer".into());
    }
    Ok(consumers)
}

fn alias_links(program: &Program) -> Result<Vec<AliasLink>, String> {
    let mut links = Vec::new();
    for (call_index, call) in program.calls.iter().copied().enumerate() {
        let template = &program.templates[call.template];
        let mut cursor = call.source_start;
        for (atom_index, &atom) in template.atoms.iter().enumerate() {
            match atom {
                Atom::Ordinary(RunToken::DecompositionAlias {
                    length,
                    source_delta,
                    source_stride,
                    ..
                })
                | Atom::Ordinary(RunToken::EqualityAlias {
                    length,
                    source_delta,
                    source_stride,
                    ..
                }) => {
                    let source = cursor
                        .checked_sub(source_delta)
                        .ok_or_else(|| "private decoder alias delta is not earlier".to_string())?;
                    push_alias_sequence(
                        program,
                        Consumer::Ordinary {
                            call: call_index,
                            atom: atom_index,
                            offset: 0,
                        },
                        (0..length).map(|offset| cursor + offset),
                        (0..length).map(|offset| source + source_stride * offset),
                        &mut links,
                    )?;
                }
                Atom::SisBatch { batch } => {
                    for (group_index, group) in program.batches[batch].groups.iter().enumerate() {
                        if let OpeningGroupKind::Alias { source, source_stride } = group.kind {
                            let first_column =
                                cursor + 2 + program.batches[batch].commitment_fields + group.opening_start * 122;
                            push_alias_sequence(
                                program,
                                Consumer::Batch {
                                    call: call_index,
                                    atom: atom_index,
                                    batch,
                                    group: group_index,
                                    offset: 0,
                                },
                                (0..group.length).map(|offset| first_column + offset * 122),
                                (0..group.length).map(|offset| source + source_stride * offset),
                                &mut links,
                            )?;
                        }
                    }
                }
                _ => {}
            }
            let length = atom_summary(atom, &program.batches)?.source_columns;
            cursor += length;
        }
    }
    Ok(links)
}

pub(super) fn build(runs: &[Run], families: &[Stage]) -> Result<Program, String> {
    let mut batches = families
        .iter()
        .filter(|family| matches!(family.path, SIS_INPUT | SIS_COMPRESSION))
        .map(|family| parse_batch(runs, family))
        .collect::<Result<Vec<_>, _>>()?;
    if batches.is_empty() {
        return Err("private decoder has no SIS input/compression batches".into());
    }
    batches.sort_unstable_by_key(|batch| batch.source_start);
    let atoms = program_atoms(runs, &batches)?;
    let mut template_ids = HashMap::<Vec<Atom>, usize>::new();
    let mut templates = Vec::new();
    let mut calls = Vec::new();
    let mut summary = Summary::default();
    let mut source_cursor = super::SOURCE_START;
    let mut final_cursor = super::FINAL_START;
    for chunk in atoms.chunks(TEMPLATE_ATOMS) {
        let template = if let Some(&template) = template_ids.get(chunk) {
            template
        } else {
            let mut template_summary = Summary::default();
            for &atom in chunk {
                template_summary = add_summary(template_summary, atom_summary(atom, &batches)?)?;
            }
            let template = templates.len();
            templates.push(Template {
                atoms: chunk.to_vec(),
                summary: template_summary,
            });
            template_ids.insert(chunk.to_vec(), template);
            template
        };
        calls.push(Call {
            template,
            source_start: source_cursor,
            final_start: final_cursor,
        });
        let call_summary = templates[template].summary;
        summary = add_summary(summary, call_summary)?;
        source_cursor += call_summary.source_columns;
        final_cursor += call_summary.fresh_coordinates;
    }
    if source_cursor != super::SOURCE_STOP || final_cursor != super::BRANCH_STOP {
        return Err(format!(
            "private decoder compact program ends at source {source_cursor} and final {final_cursor}"
        ));
    }
    let mut program = Program {
        templates,
        calls,
        batches,
        alias_links: Vec::new(),
        alias_consumers: Vec::new(),
        summary,
    };
    program.alias_links = alias_links(&program)?;
    program.alias_consumers = alias_consumers(&program)?;
    Ok(program)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PointResolution {
    Direct {
        start: usize,
        width: usize,
        centered: bool,
    },
    DecompositionAlias {
        source: usize,
        digit: usize,
        start: usize,
        centered: bool,
    },
    EqualityAlias {
        source: usize,
        start: usize,
        width: usize,
        centered: bool,
    },
    LinearDefinition,
    TraceEliminated,
}

fn run_point(run: Run, column: usize) -> Result<PointResolution, String> {
    let offset = column
        .checked_sub(run.source_start)
        .filter(|&offset| offset < run.length)
        .ok_or_else(|| "private decoder point escapes its run".to_string())?;
    Ok(match run.resolution {
        Resolution::Direct {
            start,
            start_stride,
            width,
            centered,
        } => PointResolution::Direct {
            start: super::affine(start, start_stride, offset)?,
            width,
            centered,
        },
        Resolution::DecompositionAlias {
            source,
            source_stride,
            digit,
            digit_stride,
            start,
            start_stride,
            centered,
        } => PointResolution::DecompositionAlias {
            source: super::affine(source, source_stride, offset)?,
            digit: super::affine(digit, digit_stride, offset)?,
            start: super::affine(start, start_stride, offset)?,
            centered,
        },
        Resolution::EqualityAlias {
            source,
            source_stride,
            start,
            start_stride,
            width,
            centered,
        } => PointResolution::EqualityAlias {
            source: super::affine(source, source_stride, offset)?,
            start: super::affine(start, start_stride, offset)?,
            width,
            centered,
        },
        Resolution::LinearDefinition => PointResolution::LinearDefinition,
        Resolution::TraceEliminated => PointResolution::TraceEliminated,
    })
}

struct Expansion<'a> {
    runs: &'a [Run],
    run: usize,
    source: usize,
    final_cursor: usize,
    starts: Vec<usize>,
    widths: Vec<u8>,
    centered: Vec<bool>,
}

impl<'a> Expansion<'a> {
    fn new(runs: &'a [Run]) -> Self {
        let mut starts = vec![0usize; super::SOURCE_STOP];
        let mut widths = vec![0u8; super::SOURCE_STOP];
        for column in 0..super::SOURCE_START {
            starts[column] = column;
            widths[column] = 1;
        }
        Self {
            runs,
            run: 0,
            source: super::SOURCE_START,
            final_cursor: super::FINAL_START,
            starts,
            widths,
            centered: vec![false; super::SOURCE_STOP],
        }
    }

    fn accept(&mut self, point: PointResolution) -> Result<(), String> {
        while self.run < self.runs.len() && self.runs[self.run].source_start + self.runs[self.run].length <= self.source
        {
            self.run += 1;
        }
        let expected = self
            .runs
            .get(self.run)
            .copied()
            .ok_or_else(|| "compact private decoder emits beyond the source relation".to_string())?;
        if run_point(expected, self.source)? != point {
            return Err(format!(
                "compact private decoder differs at source column {}",
                self.source
            ));
        }
        match point {
            PointResolution::Direct { start, width, centered } => {
                if start != self.final_cursor || !matches!(width, 1 | 41 | 64) {
                    return Err("compact private decoder direct allocation breaks the final cursor".into());
                }
                self.starts[self.source] = start;
                self.widths[self.source] = width as u8;
                self.centered[self.source] = centered;
                self.final_cursor += width;
            }
            PointResolution::DecompositionAlias {
                source,
                digit,
                start,
                centered,
            } => {
                if source >= self.source
                    || digit >= usize::from(self.widths[source])
                    || start != self.starts[source] + digit
                {
                    return Err("compact private decoder decomposition target is invalid".into());
                }
                self.starts[self.source] = start;
                self.widths[self.source] = 1;
                self.centered[self.source] = centered;
            }
            PointResolution::EqualityAlias {
                source,
                start,
                width,
                centered,
            } => {
                if source >= self.source
                    || usize::from(self.widths[source]) != width
                    || self.starts[source] != start
                    || self.centered[source] != centered
                {
                    return Err("compact private decoder equality target is invalid".into());
                }
                self.starts[self.source] = start;
                self.widths[self.source] = width as u8;
                self.centered[self.source] = centered;
            }
            PointResolution::LinearDefinition | PointResolution::TraceEliminated => {}
        }
        self.source += 1;
        Ok(())
    }
}

fn expand_ordinary(token: RunToken, expansion: &mut Expansion<'_>) -> Result<(), String> {
    let atom_source = expansion.source;
    let atom_final = expansion.final_cursor;
    match token {
        RunToken::Direct {
            length,
            start_stride,
            width,
            centered,
        } => {
            for offset in 0..length {
                expansion.accept(PointResolution::Direct {
                    start: super::affine(atom_final, start_stride, offset)?,
                    width: usize::from(width),
                    centered,
                })?;
            }
        }
        RunToken::DecompositionAlias {
            length,
            source_delta,
            source_stride,
            digit,
            digit_stride,
            start_stride,
            centered,
        } => {
            let source = atom_source
                .checked_sub(source_delta)
                .ok_or_else(|| "compact decomposition alias delta is not earlier".to_string())?;
            let first_start = expansion.starts[source] + usize::from(digit);
            for offset in 0..length {
                let source = super::affine(source, source_stride, offset)?;
                let digit = super::affine(usize::from(digit), digit_stride, offset)?;
                expansion.accept(PointResolution::DecompositionAlias {
                    source,
                    digit,
                    start: super::affine(first_start, start_stride, offset)?,
                    centered,
                })?;
            }
        }
        RunToken::EqualityAlias {
            length,
            source_delta,
            source_stride,
            start_stride,
            width,
            centered,
        } => {
            let source = atom_source
                .checked_sub(source_delta)
                .ok_or_else(|| "compact equality alias delta is not earlier".to_string())?;
            let first_start = expansion.starts[source];
            for offset in 0..length {
                expansion.accept(PointResolution::EqualityAlias {
                    source: super::affine(source, source_stride, offset)?,
                    start: super::affine(first_start, start_stride, offset)?,
                    width: usize::from(width),
                    centered,
                })?;
            }
        }
        RunToken::LinearDefinition { length } => {
            for _ in 0..length {
                expansion.accept(PointResolution::LinearDefinition)?;
            }
        }
        RunToken::TraceEliminated { length } => {
            for _ in 0..length {
                expansion.accept(PointResolution::TraceEliminated)?;
            }
        }
    }
    Ok(())
}

fn expand_batch(batch: &Batch, expansion: &mut Expansion<'_>) -> Result<(), String> {
    for _ in 0..2 {
        expansion.accept(PointResolution::LinearDefinition)?;
    }
    for _ in 0..batch.commitment_fields {
        expansion.accept(PointResolution::Direct {
            start: expansion.final_cursor,
            width: 41,
            centered: false,
        })?;
    }
    let mut group_index = 0usize;
    for opening in 0..batch.openings {
        while group_index + 1 < batch.groups.len()
            && batch.groups[group_index].opening_start + batch.groups[group_index].length <= opening
        {
            group_index += 1;
        }
        let group = batch.groups[group_index];
        if opening < group.opening_start || opening >= group.opening_start + group.length {
            return Err("compact SIS opening groups do not exactly cover their batch".into());
        }
        match group.kind {
            OpeningGroupKind::Alias { source, source_stride } => {
                let source = super::affine(source, source_stride, opening - group.opening_start)?;
                for digit in 0..41 {
                    expansion.accept(PointResolution::DecompositionAlias {
                        source,
                        digit,
                        start: expansion.starts[source] + digit,
                        centered: true,
                    })?;
                }
            }
            OpeningGroupKind::Direct => {
                for _ in 0..41 {
                    expansion.accept(PointResolution::Direct {
                        start: expansion.final_cursor,
                        width: 1,
                        centered: true,
                    })?;
                }
            }
        }
        for _ in 0..41 {
            expansion.accept(PointResolution::TraceEliminated)?;
        }
        for _ in 0..40 {
            expansion.accept(PointResolution::Direct {
                start: expansion.final_cursor,
                width: 1,
                centered: false,
            })?;
        }
    }
    Ok(())
}

pub(super) fn validate(program: &Program, runs: &[Run]) -> Result<(), String> {
    let mut expansion = Expansion::new(runs);
    let mut summary = Summary::default();
    for (call_index, &call) in program.calls.iter().enumerate() {
        if call.source_start != expansion.source || call.final_start != expansion.final_cursor {
            return Err(format!(
                "compact private decoder call {call_index} breaks cursor continuity"
            ));
        }
        let template = program
            .templates
            .get(call.template)
            .ok_or_else(|| "compact private decoder call names an absent template".to_string())?;
        let mut checked = Summary::default();
        for &atom in &template.atoms {
            checked = add_summary(checked, atom_summary(atom, &program.batches)?)?;
            match atom {
                Atom::Ordinary(token) => expand_ordinary(token, &mut expansion)?,
                Atom::SisBatch { batch } => expand_batch(
                    program
                        .batches
                        .get(batch)
                        .ok_or_else(|| "compact private decoder atom names an absent batch".to_string())?,
                    &mut expansion,
                )?,
            }
        }
        if checked != template.summary {
            return Err("compact private decoder template summary is stale".into());
        }
        summary = add_summary(summary, checked)?;
    }
    if expansion.source != super::SOURCE_STOP
        || expansion.final_cursor != super::BRANCH_STOP
        || summary != program.summary
    {
        return Err("compact private decoder root summary or endpoint is stale".into());
    }
    let rebuilt_links = alias_links(program)?;
    if rebuilt_links != program.alias_links {
        return Err("compact private decoder alias-link certificate is stale".into());
    }
    let rebuilt_consumers = alias_consumers(program)?;
    if rebuilt_consumers != program.alias_consumers {
        return Err("compact private decoder alias-consumer certificate is stale".into());
    }
    Ok(())
}

pub(super) fn report(runs: &[Run], families: &[Stage]) -> Result<Program, String> {
    let program = build(runs, families)?;
    validate(&program, runs)?;
    let template_atoms = program
        .templates
        .iter()
        .map(|template| template.atoms.len())
        .sum::<usize>();
    let opening_groups = program
        .batches
        .iter()
        .map(|batch| batch.groups.len())
        .sum::<usize>();
    let target_kinds = program
        .alias_links
        .iter()
        .fold([0usize; 6], |mut counts, link| {
            let (call, atom) = match link.target {
                Owner::Ordinary { call, atom, .. } | Owner::Batch { call, atom, .. } => (call, atom),
            };
            let target = program.templates[program.calls[call].template].atoms[atom];
            let index = match target {
                Atom::Ordinary(RunToken::Direct { .. }) => 0,
                Atom::Ordinary(RunToken::DecompositionAlias { .. }) => 1,
                Atom::Ordinary(RunToken::EqualityAlias { .. }) => 2,
                Atom::Ordinary(RunToken::LinearDefinition { .. }) => 3,
                Atom::Ordinary(RunToken::TraceEliminated { .. }) => 4,
                Atom::SisBatch { .. } => 5,
            };
            counts[index] += 1;
            counts
        });
    eprintln!(
        "private decoder compact program: templates={} template_atoms={} calls={} batches={} openings={} opening_groups={} alias_consumers={} alias_links={} target_kinds={target_kinds:?} summary={:?}",
        program.templates.len(),
        template_atoms,
        program.calls.len(),
        program.batches.len(),
        program.batches.iter().map(|batch| batch.openings).sum::<usize>(),
        opening_groups,
        program.alias_consumers.len(),
        program.alias_links.len(),
        program.summary,
    );
    Ok(program)
}

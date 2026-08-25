//! Shared Lean renderer for compact selective source decoders.

use std::fmt::Write as _;
use std::ops::Range;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveProjectedDecoderRunProvenance, SelectiveProjectedSourceDecoderStridedRun,
    SelectiveProjectedSourceDecoderTemplateInstances, SelectiveProjectedSourceResolutionRun,
};

use super::selective_decoder_run_lean::lean_resolution;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidualBatch {
    source_start: usize,
    instance_count: usize,
    instance_stride: usize,
    width: usize,
    resolution: SelectiveProjectedSourceResolutionRun,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidualInterval {
    source_start: usize,
    width: usize,
    resolution: SelectiveProjectedSourceResolutionRun,
}

fn affine(first: usize, stride: usize, offset: usize) -> usize {
    stride
        .checked_mul(offset)
        .and_then(|delta| first.checked_add(delta))
        .expect("decoder affine coordinate must fit usize")
}

fn resolution_at(
    resolution: SelectiveProjectedSourceResolutionRun,
    offset: usize,
) -> SelectiveProjectedSourceResolutionRun {
    match resolution {
        SelectiveProjectedSourceResolutionRun::Direct {
            start,
            start_stride,
            width,
            centered,
        } => SelectiveProjectedSourceResolutionRun::Direct {
            start: affine(start, start_stride, offset),
            start_stride,
            width,
            centered,
        },
        SelectiveProjectedSourceResolutionRun::DecompositionAlias {
            source,
            source_stride,
            digit,
            digit_stride,
            start,
            start_stride,
            centered,
        } => SelectiveProjectedSourceResolutionRun::DecompositionAlias {
            source: affine(source, source_stride, offset),
            source_stride,
            digit: affine(digit, digit_stride, offset),
            digit_stride,
            start: affine(start, start_stride, offset),
            start_stride,
            centered,
        },
        SelectiveProjectedSourceResolutionRun::EqualityAlias {
            source,
            source_stride,
            start,
            start_stride,
            width,
            centered,
        } => SelectiveProjectedSourceResolutionRun::EqualityAlias {
            source: affine(source, source_stride, offset),
            source_stride,
            start: affine(start, start_stride, offset),
            start_stride,
            width,
            centered,
        },
        SelectiveProjectedSourceResolutionRun::LinearDefinition => {
            SelectiveProjectedSourceResolutionRun::LinearDefinition
        }
        SelectiveProjectedSourceResolutionRun::TraceEliminated => {
            SelectiveProjectedSourceResolutionRun::TraceEliminated
        }
    }
}

fn compress_residual_runs(runs: &[SelectiveProjectedSourceDecoderStridedRun]) -> Vec<ResidualBatch> {
    let intervals = runs
        .iter()
        .flat_map(|run| {
            if run.source_stride() == 1 {
                vec![ResidualInterval {
                    source_start: run.source_start(),
                    width: run.count(),
                    resolution: run.resolution(),
                }]
            } else {
                (0..run.count())
                    .map(|offset| ResidualInterval {
                        source_start: affine(run.source_start(), run.source_stride(), offset),
                        width: 1,
                        resolution: resolution_at(run.resolution(), offset),
                    })
                    .collect()
            }
        })
        .collect::<Vec<_>>();
    let mut batches = Vec::new();
    let mut cursor = 0;
    while cursor < intervals.len() {
        let first = intervals[cursor];
        let compatible =
            |interval: ResidualInterval| interval.width == first.width && interval.resolution == first.resolution;
        let instance_stride = intervals
            .get(cursor + 1)
            .copied()
            .filter(|&interval| compatible(interval))
            .and_then(|interval| interval.source_start.checked_sub(first.source_start))
            .filter(|&stride| stride > 0)
            .unwrap_or(0);
        let mut instance_count = 1;
        if instance_stride > 0 {
            while let Some(interval) = intervals
                .get(cursor + instance_count)
                .copied()
                .filter(|&interval| compatible(interval))
            {
                let Some(expected_start) = instance_stride
                    .checked_mul(instance_count)
                    .and_then(|offset| first.source_start.checked_add(offset))
                else {
                    break;
                };
                if interval.source_start != expected_start {
                    break;
                }
                instance_count += 1;
            }
        }
        if instance_count <= 8 && instance_stride > first.width {
            for index in 0..instance_count {
                batches.push(ResidualBatch {
                    source_start: affine(first.source_start, instance_stride, index),
                    instance_count: 1,
                    instance_stride: 0,
                    width: first.width,
                    resolution: first.resolution,
                });
            }
        } else {
            batches.push(ResidualBatch {
                source_start: first.source_start,
                instance_count,
                instance_stride,
                width: first.width,
                resolution: first.resolution,
            });
        }
        cursor += instance_count;
    }
    let mut expanded = batches
        .iter()
        .flat_map(|batch| {
            let batch = *batch;
            (0..batch.instance_count).flat_map(move |index| {
                let source_start = affine(batch.source_start, batch.instance_stride, index);
                (0..batch.width).map(move |offset| (source_start + offset, resolution_at(batch.resolution, offset)))
            })
        })
        .collect::<Vec<_>>();
    let mut exact = runs
        .iter()
        .flat_map(|run| {
            (0..run.count()).map(|offset| {
                (
                    affine(run.source_start(), run.source_stride(), offset),
                    resolution_at(run.resolution(), offset),
                )
            })
        })
        .collect::<Vec<_>>();
    expanded.sort_unstable_by_key(|item| item.0);
    exact.sort_unstable_by_key(|item| item.0);
    assert_eq!(expanded, exact, "residual decoder batches must replay exactly");
    batches
}

fn write_residual_batches(rendered: &mut String, name: &str, batches: &[ResidualBatch]) {
    writeln!(rendered, "def {name} : List RawResidualBatch :=\n  [").expect("render residual decoder batch header");
    for (index, batch) in batches.iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ sourceStart := {}, instanceCount := {}, instanceStride := {}, width := {}, resolution := {} }}",
            batch.source_start,
            batch.instance_count,
            batch.instance_stride,
            batch.width,
            lean_resolution(batch.resolution),
        )
        .expect("render residual decoder batch");
    }
    writeln!(rendered, "  ]\n").expect("render residual decoder batch footer");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TemplateBatch {
    source_start: usize,
    count: usize,
    source_stride: usize,
    final_start: usize,
    final_stride: usize,
    reference_start: usize,
    reference_stride: usize,
    reference_final_start: usize,
    reference_final_stride: usize,
}

fn normalize_template_instances(
    source_width: usize,
    instances: &[SelectiveProjectedSourceDecoderTemplateInstances],
) -> Vec<TemplateBatch> {
    let mut normalized = Vec::new();
    for instance in instances {
        if instance.count() <= 8 && instance.source_stride() > source_width {
            for index in 0..instance.count() {
                normalized.push(TemplateBatch {
                    source_start: affine(instance.source_start(), instance.source_stride(), index),
                    count: 1,
                    source_stride: 0,
                    final_start: affine(instance.final_start(), instance.final_stride(), index),
                    final_stride: 0,
                    reference_start: affine(instance.reference_start(), instance.reference_stride(), index),
                    reference_stride: 0,
                    reference_final_start: affine(
                        instance.reference_final_start(),
                        instance.reference_final_stride(),
                        index,
                    ),
                    reference_final_stride: 0,
                });
            }
        } else {
            normalized.push(TemplateBatch {
                source_start: instance.source_start(),
                count: instance.count(),
                source_stride: instance.source_stride(),
                final_start: instance.final_start(),
                final_stride: instance.final_stride(),
                reference_start: instance.reference_start(),
                reference_stride: instance.reference_stride(),
                reference_final_start: instance.reference_final_start(),
                reference_final_stride: instance.reference_final_stride(),
            });
        }
    }
    let expanded = normalized
        .iter()
        .flat_map(|batch| {
            (0..batch.count).map(|index| {
                (
                    affine(batch.source_start, batch.source_stride, index),
                    affine(batch.final_start, batch.final_stride, index),
                    affine(batch.reference_start, batch.reference_stride, index),
                    affine(batch.reference_final_start, batch.reference_final_stride, index),
                )
            })
        })
        .collect::<Vec<_>>();
    let exact = instances
        .iter()
        .flat_map(|instance| {
            (0..instance.count()).map(|index| {
                (
                    affine(instance.source_start(), instance.source_stride(), index),
                    affine(instance.final_start(), instance.final_stride(), index),
                    affine(instance.reference_start(), instance.reference_stride(), index),
                    affine(
                        instance.reference_final_start(),
                        instance.reference_final_stride(),
                        index,
                    ),
                )
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(expanded, exact, "template batches must replay exactly");
    normalized
}

fn write_template_instances(rendered: &mut String, name: &str, instances: &[TemplateBatch]) {
    writeln!(rendered, "def {name} : List RawTemplateInstances :=\n  [").expect("render template-instance header");
    for (index, instance) in instances.iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ sourceStart := {}, count := {}, sourceStride := {}, finalStart := {}, finalStride := {}, referenceStart := {}, referenceStride := {}, referenceFinalStart := {}, referenceFinalStride := {} }}",
            instance.source_start,
            instance.count,
            instance.source_stride,
            instance.final_start,
            instance.final_stride,
            instance.reference_start,
            instance.reference_stride,
            instance.reference_final_start,
            instance.reference_final_stride,
        )
        .expect("render template instance");
    }
    writeln!(rendered, "  ]\n").expect("render template-instance footer");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnerRef {
    Template {
        template_index: usize,
        batch_index: usize,
    },
    Residual {
        batch_index: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OwnerGeometry {
    owner: OwnerRef,
    source_start: usize,
    count: usize,
    stride: usize,
    width: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CoverGroup {
    source_start: usize,
    count: usize,
    stride: usize,
    owners: Vec<OwnerRef>,
}

fn build_cover_groups(
    source_range: Range<usize>,
    template_widths: &[usize],
    template_instances: &[Vec<TemplateBatch>],
    residual_batches: &[ResidualBatch],
) -> Vec<CoverGroup> {
    assert_eq!(template_widths.len(), template_instances.len());
    let mut geometries = template_instances
        .iter()
        .enumerate()
        .flat_map(|(template_index, batches)| {
            batches
                .iter()
                .enumerate()
                .map(move |(batch_index, batch)| OwnerGeometry {
                    owner: OwnerRef::Template {
                        template_index,
                        batch_index,
                    },
                    source_start: batch.source_start,
                    count: batch.count,
                    stride: batch.source_stride,
                    width: template_widths[template_index],
                })
        })
        .chain(
            residual_batches
                .iter()
                .enumerate()
                .map(|(batch_index, batch)| OwnerGeometry {
                    owner: OwnerRef::Residual { batch_index },
                    source_start: batch.source_start,
                    count: batch.instance_count,
                    stride: batch.instance_stride,
                    width: batch.width,
                }),
        )
        .collect::<Vec<_>>();
    geometries.sort_unstable_by_key(|geometry| geometry.source_start);
    for geometry in &geometries {
        assert!(geometry.count > 0, "decoder owner count must be positive");
        assert!(geometry.width > 0, "decoder owner width must be positive");
        if geometry.count == 1 {
            assert_eq!(geometry.stride, 0, "singleton decoder owner must be normalized");
        } else {
            assert!(
                geometry.width <= geometry.stride,
                "repeated decoder owner intervals must not overlap"
            );
        }
    }

    let mut used = vec![false; geometries.len()];
    let mut groups = Vec::new();
    while let Some(first_index) = used.iter().position(|used| !*used) {
        let first = geometries[first_index];
        if first.count == 1 {
            used[first_index] = true;
            groups.push(CoverGroup {
                source_start: first.source_start,
                count: 1,
                stride: first.width,
                owners: vec![first.owner],
            });
            continue;
        }
        let period_end = first
            .source_start
            .checked_add(first.stride)
            .expect("decoder cover period must fit usize");
        let matching = geometries
            .iter()
            .enumerate()
            .filter(|(index, geometry)| {
                !used[*index]
                    && geometry.count == first.count
                    && geometry.stride == first.stride
                    && geometry.source_start < period_end
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let mut cursor = first.source_start;
        let mut owners = Vec::with_capacity(matching.len());
        for index in matching {
            let geometry = geometries[index];
            assert_eq!(
                geometry.source_start, cursor,
                "decoder cover owners must tile one complete period"
            );
            cursor = cursor
                .checked_add(geometry.width)
                .expect("decoder cover width must fit usize");
            used[index] = true;
            owners.push(geometry.owner);
        }
        assert_eq!(cursor, period_end, "decoder cover period must have no gap");
        groups.push(CoverGroup {
            source_start: first.source_start,
            count: first.count,
            stride: first.stride,
            owners,
        });
    }

    groups.sort_unstable_by_key(|group| group.source_start);
    let mut cursor = source_range.start;
    for group in &groups {
        assert_eq!(
            group.source_start, cursor,
            "decoder cover groups must form the exact source interval"
        );
        cursor = group
            .count
            .checked_mul(group.stride)
            .and_then(|width| group.source_start.checked_add(width))
            .expect("decoder cover group span must fit usize");
    }
    assert_eq!(cursor, source_range.end, "decoder cover must end at the source bound");
    assert!(used.into_iter().all(|used| used), "every decoder owner must occur once");
    groups
}

fn lean_owner(owner: OwnerRef) -> String {
    match owner {
        OwnerRef::Template {
            template_index,
            batch_index,
        } => format!(".template {template_index} {batch_index}"),
        OwnerRef::Residual { batch_index } => format!(".residual {batch_index}"),
    }
}

fn write_cover_groups(rendered: &mut String, name: &str, groups: &[CoverGroup]) {
    writeln!(rendered, "def {name} : List RawCoverGroup :=\n  [").expect("render decoder cover header");
    for (index, group) in groups.iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        let owners = group
            .owners
            .iter()
            .map(|owner| lean_owner(*owner))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            rendered,
            "{separator}{{ sourceStart := {}, count := {}, stride := {}, owners := [{owners}] }}",
            group.source_start, group.count, group.stride,
        )
        .expect("render decoder cover group");
    }
    writeln!(rendered, "  ]\n").expect("render decoder cover footer");
}

pub(super) fn write_decoder_arm(
    rendered: &mut String,
    label: &str,
    decoder: &SelectiveProjectedDecoderRunProvenance,
    template_rule_prefix: &str,
) {
    let normalized_templates = decoder
        .repeated_templates()
        .iter()
        .map(|template| normalize_template_instances(template.source_width(), template.instances()))
        .collect::<Vec<_>>();
    for (index, normalized_instances) in normalized_templates.iter().enumerate() {
        write_template_instances(
            rendered,
            &format!("{label}TemplateInstances{index:02}"),
            normalized_instances,
        );
    }
    writeln!(rendered, "def {label}Templates : List RawTemplate :=\n  [").expect("render template list header");
    for (index, template) in decoder.repeated_templates().iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ sourceWidth := {}, relativeRuns := {template_rule_prefix}{index:02}, instances := {label}TemplateInstances{index:02} }}",
            template.source_width(),
        )
        .expect("render template list item");
    }
    writeln!(rendered, "  ]\n").expect("render template list footer");
    let residual_batches = compress_residual_runs(decoder.residual_strided_runs());
    write_residual_batches(rendered, &format!("{label}ResidualBatches"), &residual_batches);
    let range = decoder.source_range();
    let template_widths = decoder
        .repeated_templates()
        .iter()
        .map(|template| template.source_width())
        .collect::<Vec<_>>();
    let cover_groups = build_cover_groups(
        range.clone(),
        &template_widths,
        &normalized_templates,
        &residual_batches,
    );
    write_cover_groups(rendered, &format!("{label}CoverGroups"), &cover_groups);
    writeln!(
        rendered,
        "def {label}Arm : RawArm where\n  schemaVersion := 1\n  arm := {}\n  sourceStart := {}\n  sourceEnd := {}\n  finalColumns := {}\n  templates := {label}Templates\n  residualBatches := {label}ResidualBatches\n  coverGroups := {label}CoverGroups\n",
        decoder.arm(),
        range.start,
        range.end,
        decoder.final_columns(),
    )
    .expect("render decoder arm");
}

fn lean_template_batch(batch: TemplateBatch) -> String {
    format!(
        "{{ sourceStart := {}, count := {}, sourceStride := {}, finalStart := {}, finalStride := {}, referenceStart := {}, referenceStride := {}, referenceFinalStart := {}, referenceFinalStride := {} }}",
        batch.source_start,
        batch.count,
        batch.source_stride,
        batch.final_start,
        batch.final_stride,
        batch.reference_start,
        batch.reference_stride,
        batch.reference_final_start,
        batch.reference_final_stride,
    )
}

fn lean_residual_batch(batch: ResidualBatch) -> String {
    format!(
        "{{ sourceStart := {}, instanceCount := {}, instanceStride := {}, width := {}, resolution := {} }}",
        batch.source_start,
        batch.instance_count,
        batch.instance_stride,
        batch.width,
        lean_resolution(batch.resolution),
    )
}

fn lean_cover_group(group: &CoverGroup) -> String {
    format!(
        "{{ sourceStart := {}, count := {}, stride := {}, owners := [{}] }}",
        group.source_start,
        group.count,
        group.stride,
        group
            .owners
            .iter()
            .copied()
            .map(lean_owner)
            .collect::<Vec<_>>()
            .join(", "),
    )
}

/// Render one exact decoder arm without separate list declarations. This is
/// the same checked `RawArm` schema as `write_decoder_arm`; only the inert Lean
/// data layout differs. It is used when many arms share one template rule set.
pub(super) fn write_decoder_arm_inline(
    rendered: &mut String,
    name: &str,
    decoder: &SelectiveProjectedDecoderRunProvenance,
    template_rule_prefix: &str,
    source_definition_runs: &str,
    retained_row_runs: &str,
    canonical_opening_runs: &str,
) {
    let normalized_templates = decoder
        .repeated_templates()
        .iter()
        .map(|template| normalize_template_instances(template.source_width(), template.instances()))
        .collect::<Vec<_>>();
    let templates = decoder
        .repeated_templates()
        .iter()
        .zip(&normalized_templates)
        .enumerate()
        .map(|(index, (template, instances))| {
            format!(
                "{{ sourceWidth := {}, relativeRuns := {template_rule_prefix}{index:02}, instances := [{}] }}",
                template.source_width(),
                instances
                    .iter()
                    .copied()
                    .map(lean_template_batch)
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        })
        .collect::<Vec<_>>();
    let residual_batches = compress_residual_runs(decoder.residual_strided_runs());
    let range = decoder.source_range();
    let template_widths = decoder
        .repeated_templates()
        .iter()
        .map(|template| template.source_width())
        .collect::<Vec<_>>();
    let cover_groups = build_cover_groups(
        range.clone(),
        &template_widths,
        &normalized_templates,
        &residual_batches,
    );
    writeln!(
        rendered,
        "def {name} : RawArm where\n  schemaVersion := 1\n  arm := {}\n  sourceStart := {}\n  sourceEnd := {}\n  finalColumns := {}\n  templates := [{}]\n  residualBatches := [{}]\n  coverGroups := [{}]\n  sourceDefinitionRuns := {source_definition_runs}\n  retainedRowRuns := {retained_row_runs}\n  canonicalOpeningRuns := {canonical_opening_runs}\n",
        decoder.arm(),
        range.start,
        range.end,
        decoder.final_columns(),
        templates.join(", "),
        residual_batches
            .iter()
            .copied()
            .map(lean_residual_batch)
            .collect::<Vec<_>>()
            .join(", "),
        cover_groups
            .iter()
            .map(lean_cover_group)
            .collect::<Vec<_>>()
            .join(", "),
    )
    .expect("render inline decoder arm");
}

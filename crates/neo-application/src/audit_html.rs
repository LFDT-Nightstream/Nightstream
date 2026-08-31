//! Self-contained HTML rendering for application column audits.

use core::fmt::Display;

use neo_math::{balanced::to_balanced_i128, F};
use p3_field::PrimeField64;
use serde::Serialize;

use crate::{
    continuity_column_occurrences, memory_column_occurrences, ApplicationRelation, ColumnConstraintIndex,
    ContinuityCatalog, ContinuityColumnRole, GadgetColumnRole, MemoryCatalog, MemoryColumnRole, MemoryKind,
    MemoryPortActivation, MemoryPortKind,
};

const REPORT_MARKER: &str = "/*__COLUMN_AUDIT_REPORT__*/";
const HTML_TEMPLATE: &str = include_str!("audit_template.html");

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    title: String,
    columns: Vec<ColumnRecord>,
    rows: Vec<RowRecord>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ColumnRecord {
    index: usize,
    name: String,
    family: &'static str,
    region: &'static str,
    width: String,
    role: &'static str,
    visibility: &'static str,
    constant_one: bool,
    generated: bool,
    row_indices: Vec<usize>,
    gadgets: Vec<GadgetRecord>,
    memory_ports: Vec<MemoryRecord>,
    continuity: Vec<ContinuityRecord>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RowRecord {
    index: usize,
    label: &'static str,
    scope: String,
    a: Vec<TermRecord>,
    b: Vec<TermRecord>,
    c: Vec<TermRecord>,
}

#[derive(Serialize)]
struct TermRecord {
    column: usize,
    signed: String,
    canonical: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GadgetRecord {
    row_start: usize,
    row_end: usize,
    label: &'static str,
    scope: String,
    role: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MemoryRecord {
    memory: String,
    memory_kind: &'static str,
    memory_index: usize,
    port_index: usize,
    port_kind: &'static str,
    column_role: String,
    address: Vec<usize>,
    value: usize,
    value_before: Option<usize>,
    activation: ActivationRecord,
}

#[derive(Serialize)]
struct ActivationRecord {
    mode: &'static str,
    column: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ContinuityRecord {
    name: &'static str,
    description: &'static str,
    group_index: usize,
    link_index: usize,
    endpoint_role: &'static str,
    previous_step: usize,
    next_step: usize,
}

/// Render a complete, self-contained column audit without writing it to disk.
///
/// `generated_regions` identifies application-owned regions that the browser
/// should hide during initial browsing. Search still includes those columns.
pub fn render_column_audit_html<Owner: Display, Id: Display>(
    title: &str,
    relation: &ApplicationRelation<Owner>,
    memory: &MemoryCatalog<Id>,
    continuity: &ContinuityCatalog,
    generated_regions: &[&str],
) -> String {
    let report = build_report(title, relation, memory, continuity, generated_regions);
    let json = serde_json::to_string(&report).expect("the audit report contains only JSON-serializable values");
    // Prevent report strings from terminating the data script. Dynamic text is
    // also inserted through textContent rather than interpreted as HTML.
    let json = json
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e");
    assert_eq!(
        HTML_TEMPLATE.matches(REPORT_MARKER).count(),
        1,
        "the audit template must contain exactly one report marker"
    );
    HTML_TEMPLATE.replacen(REPORT_MARKER, &json, 1)
}

fn build_report<Owner: Display, Id: Display>(
    title: &str,
    relation: &ApplicationRelation<Owner>,
    memory: &MemoryCatalog<Id>,
    continuity: &ContinuityCatalog,
    generated_regions: &[&str],
) -> Report {
    let index = ColumnConstraintIndex::new(relation);
    let rows = relation
        .r1cs()
        .catalog()
        .rows()
        .iter()
        .enumerate()
        .map(|(row_index, tagged)| RowRecord {
            index: row_index,
            label: tagged.tag().label(),
            scope: tagged.tag().owner().to_string(),
            a: terms(tagged.row().a_terms()),
            b: terms(tagged.row().b_terms()),
            c: terms(tagged.row().c_terms()),
        })
        .collect();

    let columns = (0..relation.columns().column_count())
        .map(|column| {
            let family = relation
                .columns()
                .family_for_column(column)
                .expect("report iterates in-range columns");
            let mut row_indices = Vec::new();
            for occurrence in index
                .r1cs_occurrences(column)
                .expect("report iterates in-range columns")
            {
                if row_indices.last() != Some(&occurrence.row_index()) {
                    row_indices.push(occurrence.row_index());
                }
            }

            let gadgets = index
                .gadget_occurrences(column)
                .expect("report iterates in-range columns")
                .iter()
                .map(|gadget| {
                    let occurrence = gadget.occurrence();
                    GadgetRecord {
                        row_start: occurrence.row_range().start,
                        row_end: occurrence.row_range().end,
                        label: occurrence.tag().label(),
                        scope: occurrence.tag().owner().to_string(),
                        role: format_gadget_role(gadget.role()),
                    }
                })
                .collect();

            let memory_ports = memory_column_occurrences(memory, column)
                .into_iter()
                .map(|occurrence| {
                    let memory = occurrence.memory();
                    let port = occurrence.port();
                    let value_before = match port.kind {
                        MemoryPortKind::Write { value_before_column } => value_before_column,
                        MemoryPortKind::Read => None,
                    };
                    let activation = match port.activation {
                        MemoryPortActivation::Always => ActivationRecord {
                            mode: "always",
                            column: None,
                        },
                        MemoryPortActivation::When(column) => ActivationRecord {
                            mode: "when",
                            column: Some(column),
                        },
                        MemoryPortActivation::Unless(column) => ActivationRecord {
                            mode: "unless",
                            column: Some(column),
                        },
                    };
                    MemoryRecord {
                        memory: memory.id.to_string(),
                        memory_kind: format_memory_kind(memory.kind),
                        memory_index: occurrence.memory_index(),
                        port_index: occurrence.port_index(),
                        port_kind: format_port_kind(port.kind),
                        column_role: format_memory_role(occurrence.role()),
                        address: port.address_columns.clone(),
                        value: port.value_column,
                        value_before,
                        activation,
                    }
                })
                .collect();

            let continuity = continuity_column_occurrences(continuity, column)
                .into_iter()
                .map(|occurrence| ContinuityRecord {
                    name: occurrence.group().name,
                    description: occurrence.group().role,
                    group_index: occurrence.group_index(),
                    link_index: occurrence.link_index(),
                    endpoint_role: match occurrence.role() {
                        ContinuityColumnRole::PreviousStep => "previous-step",
                        ContinuityColumnRole::NextStep => "next-step",
                    },
                    previous_step: occurrence.link().previous_step_column,
                    next_step: occurrence.link().next_step_column,
                })
                .collect();

            ColumnRecord {
                index: column,
                name: column_name(relation, column),
                family: family.name,
                region: family.region,
                width: format!("{:?}", family.width),
                role: family.role,
                visibility: if column < relation.r1cs().public_input_count() {
                    "public input"
                } else {
                    "private witness"
                },
                constant_one: column == relation.r1cs().const_one_column(),
                generated: generated_regions.contains(&family.region),
                row_indices,
                gadgets,
                memory_ports,
                continuity,
            }
        })
        .collect();

    Report {
        title: title.to_owned(),
        columns,
        rows,
    }
}

fn column_name<Owner>(relation: &ApplicationRelation<Owner>, column: usize) -> String {
    let family = relation
        .columns()
        .family_for_column(column)
        .expect("relation terms and catalogs use in-range columns");
    if family.len == 1 {
        family.name.to_owned()
    } else {
        format!("{}[{}]", family.name, column - family.start)
    }
}

fn terms(terms: &[(usize, F)]) -> Vec<TermRecord> {
    terms
        .iter()
        .map(|&(column, coefficient)| TermRecord {
            column,
            signed: to_balanced_i128(coefficient).to_string(),
            canonical: coefficient.as_canonical_u64().to_string(),
        })
        .collect()
}

fn format_gadget_role(role: GadgetColumnRole) -> String {
    match role {
        GadgetColumnRole::ZeroTestExpression {
            term_index,
            coefficient,
        } => format!(
            "zero-test expression term {term_index}, coefficient {}",
            to_balanced_i128(coefficient)
        ),
        GadgetColumnRole::ZeroTestInverse => "zero-test inverse".to_owned(),
        GadgetColumnRole::ZeroTestIsZero => "zero-test result".to_owned(),
        GadgetColumnRole::ConditionalSelectActivation => "conditional-select activation".to_owned(),
        GadgetColumnRole::ConditionalSelectCondition {
            term_index,
            coefficient,
        } => format!(
            "conditional-select condition term {term_index}, coefficient {}",
            to_balanced_i128(coefficient)
        ),
        GadgetColumnRole::ConditionalSelectLhs => "conditional-select lhs".to_owned(),
        GadgetColumnRole::ConditionalSelectRhs => "conditional-select rhs".to_owned(),
        GadgetColumnRole::ConditionalSelectOutput => "conditional-select output".to_owned(),
        GadgetColumnRole::ConditionalSelectDelta => "conditional-select delta".to_owned(),
    }
}

const fn format_memory_kind(kind: MemoryKind) -> &'static str {
    match kind {
        MemoryKind::Rom => "ROM",
        MemoryKind::Ram => "RAM",
    }
}

const fn format_port_kind(kind: MemoryPortKind) -> &'static str {
    match kind {
        MemoryPortKind::Read => "read",
        MemoryPortKind::Write { .. } => "write",
    }
}

fn format_memory_role(role: MemoryColumnRole) -> String {
    match role {
        MemoryColumnRole::Address { position } => format!("address[{position}]"),
        MemoryColumnRole::Value => "value".to_owned(),
        MemoryColumnRole::ValueBefore => "value-before".to_owned(),
        MemoryColumnRole::Activation => "activation".to_owned(),
    }
}

//! Counts constructed WASM rows and the change caused by one more trace row.
//! Fixed memory geometry widths stay unchanged between the two samples.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::num::NonZeroUsize;
use std::path::Path;

use neo_application::ColumnWidth;
use neo_ccs::crypto::poseidon2::{INSTANCE_16, WIDTH};
use neo_wasm::nebula::{WasmApplication, WasmSize};
use neo_wasm::operations::r1cs::operation_circuit;
use neo_wasm::tagged_r1cs_builder::WasmConstraintScope;
use neo_wasm::{build_wasm_relation, build_wasm_relation_layout, plan_ordered_wasm_memory_slots};
use nightstream::nebula::{size_parts, Geometry};
use p3_field::PrimeField64;
use serde_json::json;

struct Sample {
    relation_rows: usize,
    parts: BTreeMap<&'static str, nightstream::ApplicationSize>,
}

fn sample(trace_rows: usize, slots: usize) -> Result<Sample, Box<dyn Error>> {
    let size = WasmSize::Small;
    let mut geometry = size.geometry();
    geometry.b_ops = trace_rows * slots;
    let app = WasmApplication::new(
        &geometry,
        size.limits().regions()?.0,
        NonZeroUsize::new(trace_rows).ok_or("the sample needs a trace row")?,
    )?;
    let relation_rows = app.relation_rows();
    let parts = size_parts(&geometry, &app)?
        .into_iter()
        .map(|part| (part.name, part.size))
        .collect();
    Ok(Sample { relation_rows, parts })
}

fn dump_relation(path: &Path) -> Result<(), Box<dyn Error>> {
    let circuit = operation_circuit()?;
    let relation = circuit.relation();
    let columns = relation
        .columns()
        .families()
        .iter()
        .map(|family| {
            json!({
                "start": family.start, "len": family.len, "name": family.name,
                "region": family.region, "width": format!("{:?}", family.width),
            })
        })
        .collect::<Vec<_>>();
    let rows = relation
        .r1cs()
        .catalog()
        .rows()
        .iter()
        .map(|tagged| {
            let row = tagged.row();
            let terms = [row.a_terms(), row.b_terms(), row.c_terms()].map(|side| {
                side.iter()
                    .map(|&(column, coefficient)| (column, coefficient.as_canonical_u64()))
                    .collect::<Vec<_>>()
            });
            json!({"label": tagged.tag().label(), "owner": tagged.tag().owner().to_string(), "terms": terms})
        })
        .collect::<Vec<_>>();
    let layout = build_wasm_relation_layout();
    let continuity = layout
        .auxiliary
        .continuity
        .groups()
        .iter()
        .flat_map(|group| &group.links)
        .map(|link| (link.previous_step_column, link.next_step_column))
        .collect::<Vec<_>>();
    let value = json!({
        "one": relation.r1cs().const_one_column(),
        "column_count": relation.columns().column_count(),
        "operation_rows": [circuit.operation_rows().start, circuit.operation_rows().end],
        "columns": columns, "rows": rows, "continuity": continuity,
    });
    serde_json::to_writer(std::fs::File::create(path)?, &value)?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() > 1 {
        return Err("usage: constraint_census [relation-json-path]".into());
    }
    let revision = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()?;
    if !revision.status.success() {
        return Err("cannot identify the source revision".into());
    }
    println!("source_revision\t{}", String::from_utf8(revision.stdout)?.trim());
    let base = build_wasm_relation()?;
    let range_labels: BTreeSet<_> = base
        .columns()
        .families()
        .iter()
        .filter(|family| family.region != neo_wasm::RANGE_BITS_REGION && family.width != ColumnWidth::Field)
        .map(|family| family.name.as_ref())
        .collect();
    let circuit = operation_circuit()?;
    let rows = circuit.relation().r1cs().catalog().rows();
    let mut categories = BTreeMap::<&str, usize>::new();
    for (index, row) in rows.iter().enumerate() {
        let tag = row.tag();
        let category = if circuit.operation_rows().contains(&index) {
            "operation_circuits"
        } else if range_labels.contains(tag.label()) {
            "range_checks"
        } else if tag.owner() == &WasmConstraintScope::HostEvent {
            if tag.label() == "host event absorb" {
                "host_event_hash"
            } else {
                "other_host_event"
            }
        } else {
            "other_wasm_logic"
        };
        *categories.entry(category).or_default() += 1;
    }
    if categories.values().sum::<usize>() != rows.len() {
        return Err("row categories do not cover the constructed relation".into());
    }
    println!("work\twasm_relation_rows\t{}", rows.len());
    println!("work\twitness_columns\t{}", circuit.relation().columns().column_count());
    for (name, count) in &categories {
        println!("variable\t{name}\t{count}");
    }
    for group in circuit.groups() {
        println!("operation_group\t{}\t{}", group.name, group.rows.len());
    }
    let layout = build_wasm_relation_layout();
    let slots = plan_ordered_wasm_memory_slots(layout.auxiliary.memory.entries())
        .slots
        .len();
    println!("work\tphysical_slots_per_trace_row\t{slots}");
    let first = sample(1, slots)?;
    let second = sample(2, slots)?;
    let full_rounds = INSTANCE_16.initial.len() + INSTANCE_16.terminal.len();
    let partial_rounds = INSTANCE_16.internal.len();
    let sboxes = full_rounds * WIDTH + partial_rounds;
    let raw_permutation_rows = WIDTH + full_rounds * (5 * WIDTH) + partial_rounds * (4 + WIDTH);
    println!("native_permutation\traw_rows\t{raw_permutation_rows}");
    println!("native_permutation\tcompiled_sbox_rows\t{sboxes}");
    let mut variable = rows.len();
    let links = second.relation_rows - first.relation_rows - rows.len();
    println!("variable\tcontinuity_links\t{links}");
    variable += links;
    for (name, before) in &first.parts {
        let after = second
            .parts
            .get(name)
            .ok_or("phase missing from the second sample")?;
        let delta = after
            .rows
            .checked_sub(before.rows)
            .ok_or("a phase shrank when a row was added")?;
        println!("phase\t{name}\t{}\t{}\t{delta}", before.rows, after.rows);
        match *name {
            "Bit rows" | "Operation slot rows" => {
                variable += delta;
                println!("variable\t{name}\t{delta}");
            }
            "Application rows" => {
                let binding = delta - (second.relation_rows - first.relation_rows);
                variable += binding;
                println!("variable\tmemory_binding\t{binding}");
                println!(
                    "fixed\tapplication_endpoints_and_one\t{}",
                    before.rows - first.relation_rows - binding
                );
            }
            "Operation record chain sponge" => {
                if before.rows % raw_permutation_rows != 0 || after.rows % raw_permutation_rows != 0 {
                    return Err("record hashing contains rows outside its native permutation blocks".into());
                }
                println!(
                    "variable\trecord_hash_for_this_increment\t{}",
                    delta / raw_permutation_rows * sboxes
                );
            }
            _ => {
                if delta != 0 {
                    return Err(format!("uncategorized variable phase {name}").into());
                }
            }
        }
    }
    println!("variable\tlinear_subtotal_before_record_hash\t{variable}");
    println!(
        "geometry\t{:?}",
        Geometry {
            b_ops: slots,
            ..WasmSize::Small.geometry()
        }
    );
    if let Some(path) = args.first() {
        dump_relation(Path::new(path))?;
    }
    println!("errors\t0");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("errors\t1\t{error}");
        std::process::exit(1);
    }
}

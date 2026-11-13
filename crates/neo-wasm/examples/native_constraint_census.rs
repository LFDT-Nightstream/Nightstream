use std::error::Error;
use std::io::{BufReader, Read};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use neo_wasm::nebula::{WasmApplication, WasmSize};
use neo_wasm::{build_wasm_relation_layout, plan_ordered_wasm_memory_slots};
use nightstream::{nebula::Circuit, Goldilocks as F, PrimeCharacteristicRing, PrimeField64};
use nightstream_fprime::{
    ApplicationRecords, Linear, NativeBlock, NativeShape, PlanRows, BOUNDED_DEPTH_PACKAGES_IDENTITY,
};

fn matrix_entries(rows: &PlanRows, layout: &nightstream_fprime::LogicalLayout) -> ([usize; 4], [usize; 4]) {
    let mut source = [0; 4];
    let mut logical = [0; 4];
    for row in 0..rows.row_count() {
        for port in 0..4 {
            for (column, _) in rows.port(row, port) {
                source[port] += 1;
                logical[port] += layout.slot(column).unwrap().1.width();
            }
        }
    }
    assert_eq!(source.iter().sum::<usize>(), rows.entry_count());
    (source, logical)
}

fn word(input: &mut impl Read) -> Result<usize, Box<dyn Error>> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes)?;
    Ok(usize::try_from(u64::from_le_bytes(bytes))?)
}

fn field(input: &mut impl Read) -> Result<F, Box<dyn Error>> {
    let value = word(input)? as u64;
    if value >= F::ORDER_U64 {
        return Err("noncanonical saved field".into());
    }
    Ok(F::from_u64(value))
}

fn saved(path: &Path) -> Result<(usize, Arc<ApplicationRecords>, NativeShape), Box<dyn Error>> {
    let mut input = BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0; 8];
    input.read_exact(&mut magic)?;
    if &magic != b"NSOPTM\0\x01" {
        return Err("expected a saved memory circuit".into());
    }
    let private = word(&mut input)?;
    let mut forms = [0; 4];
    input.read_exact(&mut forms)?;
    for _ in 0..4 {
        word(&mut input)?;
    }
    let rows = word(&mut input)?;
    let recipes = word(&mut input)?;
    let records = Arc::new(ApplicationRecords::read_from(&mut input, rows, recipes)?);
    let mut shape = NativeShape::default();
    for _ in 0..word(&mut input)? {
        let first_local = word(&mut input)?;
        let mut lanes: [Linear; 16] = Default::default();
        for lane in &mut lanes {
            lane.constant = field(&mut input)?;
            for _ in 0..word(&mut input)? {
                lane.terms.push((word(&mut input)?, field(&mut input)?));
            }
        }
        shape.blocks.push(NativeBlock { first_local, lanes });
    }
    if input.read(&mut [0])? != 0 {
        return Err("trailing saved circuit data".into());
    }
    Ok((private, records, shape))
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: native_constraint_census NEW_OUTPUT_DIRECTORY [SAVED_APPLICATION_DIRECTORY|-] [dump-application|-] [TRACE_ROWS_CSV] [unshared-host]")?;
    let output = Path::new(&output);
    std::fs::create_dir(output)?;
    let artifacts = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../formal/nightstream-fprime/artifacts");
    let packages = nightstream_fprime::load_bounded_depth_packages(
        &std::fs::read(artifacts.join("nightstream-fprime-bounded-depth-light-v1.json"))?,
        &std::fs::read(artifacts.join("nightstream-fprime-bounded-depth-refresh-v1.json"))?,
        BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )?;
    let layout = build_wasm_relation_layout();
    let slots = plan_ordered_wasm_memory_slots(layout.auxiliary.memory.entries())
        .slots
        .len();
    let saved_directory = std::env::args().nth(2).filter(|value| value != "-");
    let unshared = std::env::args().nth(5).as_deref() == Some("unshared-host");
    let mut previous = None;
    let mut previous_refresh = None;
    let mut previous_trace_rows = None;
    let trace_counts = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "1,2,3".into())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    for trace_rows in trace_counts {
        let mut geometry = WasmSize::Small.geometry();
        geometry.b_ops = trace_rows * slots;
        let path = if let Some(directory) = &saved_directory {
            Path::new(directory).join(format!("rows-{trace_rows}.application"))
        } else {
            let path = output.join(format!("rows-{trace_rows}.application"));
            let regions = WasmSize::Small.limits().regions()?.0;
            let rows = NonZeroUsize::new(trace_rows).unwrap();
            if unshared {
                use nightstream::nebula::{R1csApplication, RowRelation, Size};
                let source = neo_wasm::batch::build_batched_wasm_ccs(1)?.sparse_r1cs;
                let mut terms = vec![[Vec::new(), Vec::new(), Vec::new()]; source.n];
                for (side, matrix) in [&source.a, &source.b, &source.c].into_iter().enumerate() {
                    let neo_ccs::CcsMatrix::Csc(csc) = matrix else {
                        return Err("expected CSC source".into());
                    };
                    for column in 0..csc.ncols() {
                        for entry in csc.column_range(column) {
                            terms[csc.row_index(entry)][side].push((column, csc.vals()[entry]));
                        }
                    }
                }
                let relation = RowRelation {
                    rows: terms,
                    width: source.m,
                    one: neo_wasm::layout::COL_ONE,
                };
                let size = Size {
                    geometry,
                    rows,
                    regions,
                    slots: neo_wasm::nebula::plan_ordered_slots(rows)?,
                };
                let app = R1csApplication::new(relation, &layout.auxiliary.continuity, &layout.auxiliary.memory, size)?;
                Circuit::compile(geometry, app)?.write(&path)?;
            } else {
                let app = WasmApplication::new(&geometry, regions, rows)?;
                Circuit::compile(geometry, app)?.write(&path)?;
            }
            path
        };
        let (private, records, shape) = saved(&path)?;
        let compiled = packages.with_application(records.clone(), private, &shape)?;
        let package = compiled.light();
        let rows = package.plan_rows()?;
        let source = package.source_rows()?;
        if std::env::args().nth(3).as_deref() == Some("dump-application") {
            use std::io::Write;
            for (label, plan) in [("source", &source), ("plan", &rows)] {
                let path = output.join(format!("rows-{trace_rows}-{label}.jsonl"));
                let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
                for row in plan.piece_ranges()[0].clone() {
                    let sides: Vec<_> = (0..4)
                        .map(|port| {
                            plan.port(row, port)
                                .map(|(column, coefficient)| (column, coefficient.as_canonical_u64()))
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    serde_json::to_writer(&mut out, &sides)?;
                    writeln!(&mut out)?;
                }
            }
        }
        let layout = package.layout()?;
        let (source_entries, logical_entries) = matrix_entries(&rows, &layout);
        let refresh = compiled.refresh();
        let refresh_rows = refresh.plan_rows()?;
        let refresh_layout = refresh.layout()?;
        let (refresh_source_entries, refresh_logical_entries) = matrix_entries(&refresh_rows, &refresh_layout);
        let refresh_count = refresh_rows.row_count();
        let refresh_source_count = refresh.source_rows()?.row_count();
        let companions = layout.spans().widths.len() - (package.sbox_start() + 150 * package.block_count());
        let pairs = package.packed_boolean_count();
        let products = package.packed_product_count();
        let zeros = package.packed_zero_count();
        let zero_maps = package
            .packed_zero_products()
            .map(|map| {
                serde_json::json!({
                        "source_row":map[0],
                        "source_columns":[map[1],map[3],map[5]],
                        "complement_inputs":[map[2]!=0,map[4]!=0],
                "logical_coordinates":([map[1],map[3],map[5]].map(|column|layout.slot(column).unwrap().0)),
                    })
            })
            .collect::<Vec<_>>();
        std::fs::write(
            output.join(format!("rows-{trace_rows}.zero-products.json")),
            serde_json::to_vec(&zero_maps)?,
        )?;
        let maps = package.packed_products().map(|map| {
            serde_json::json!({
                "source_row": map[0],
                "source_columns": &map[1..],
                "logical_coordinates": map[1..].iter().map(|&column| layout.slot(column).unwrap().0).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>();
        std::fs::write(
            output.join(format!("rows-{trace_rows}.products.json")),
            serde_json::to_vec(&maps)?,
        )?;
        let count = rows.row_count();
        let mut circuits = Vec::new();
        for (kind, package) in [("light", compiled.light()), ("refresh", compiled.refresh())] {
            let plan = package.plan_rows()?;
            let source = package.source_rows()?;
            let layout = package.layout()?;
            let spans = layout.spans();
            let compressed: [usize; 4] = std::array::from_fn(|port| {
                (0..plan.row_count())
                    .map(|row| plan.port(row, port).count())
                    .sum()
            });
            let expanded: [usize; 4] = std::array::from_fn(|port| {
                (0..plan.row_count())
                    .flat_map(|row| plan.port(row, port))
                    .map(|(column, _)| usize::from(spans.widths[column]))
                    .sum()
            });
            let native_powers = (0..plan.ordinary_row_count())
                .filter(|&row| plan.port(row, 3).next().is_some())
                .count();
            circuits.push(serde_json::json!({
                "kind": kind,
                "rows": plan.row_count(),
                "original_source_rows": source.row_count(),
                "source_columns": spans.widths.len(),
                "live_source_slots": spans.widths.iter().filter(|&&width| width != 0).count(),
                "field_slots": spans.widths.iter().filter(|&&width| width == 41).count(),
                "unit_slots": spans.widths.iter().filter(|&&width| width == 1).count(),
                "logical_coordinates": layout.width(),
                "compressed_entries_per_matrix": compressed,
                "expanded_entries_per_matrix": expanded,
                "native_power7_rows": native_powers,
                "matrix_kind": "A*B-C+S^7",
                "matrix_count": 4,
                "fresh_norm_bound": 2,
                "k_rho": 16,
            }));
        }
        let adjacent = previous_trace_rows == trace_rows.checked_sub(1);
        let value = serde_json::json!({
            "trace_rows": trace_rows,
            "trace_rows_since_previous_sample": previous_trace_rows.map(|before| trace_rows - before),
            "constraint_difference_from_previous_sample": previous.map(|before| count - before),
            "native_memory_hash_blocks": shape.blocks.len(),
            "shared_host": !unshared && trace_rows > 1,
            "shared_absorb_groups": if !unshared && trace_rows > 1 { trace_rows.div_ceil(8) } else { trace_rows },
            "new_input_gate_rows": if !unshared && trace_rows > 1 { 12 * trace_rows } else { 0 },
            "preserved_output_gate_rows": 4 * trace_rows,
            "shared_source_columns_per_group": if !unshared && trace_rows > 1 { 728 } else { 0 },
            "opcodes":neo_wasm::WasmOpcode::supported().len(),
            "slots_per_trace_row": slots,
            "geometry": {"r":geometry.r,"mu":geometry.mu,"w_ts":geometry.w_ts,"b_scan":geometry.b_scan,"n":geometry.n,"s_max":geometry.s_max},
            "raw_application_rows":records.row_count(),
            "application_witness_words":private,
            "compiled_light_rows":count,
            "unpacked_source_rows":source.row_count(),
            "compiled_source_entries":rows.entry_count(),
            "matrix_source_entries":source_entries,
            "matrix_logical_entries":logical_entries,
            "refresh":{
                "compiled_rows":refresh_count,
                "unpacked_source_rows":refresh_source_count,
                "source_columns":refresh_layout.spans().widths.len(),
                "logical_coordinates":refresh_layout.width(),
                "matrix_source_entries":refresh_source_entries,
                "matrix_logical_entries":refresh_logical_entries,
                "added_trace_row_constraints":if adjacent { previous_refresh.map(|before|refresh_count-before) } else { None },
            },
            "logical_coordinates":layout.width(),
            "source_columns":layout.spans().widths.len(),
            "auxiliary_coordinates":companions,
            "paired_bits":pairs,
            "lowered_products":products,
            "lowered_zero_products":zeros,
            "packed_zero_rows":zeros.div_ceil(31),
            "packed_product_rows":products.div_ceil(24),
            "packed_boolean_rows":pairs.div_ceil(31),
            "source_minus_plan_rows":source.row_count() - count,
            "circuits":circuits,
            "added_trace_row_constraints":if adjacent { previous.map(|before| count-before) } else { None },
            "identity":compiled.identity(),
            "fixed_cost_divided_by_trace_rows":false,
        });
        println!("{value}");
        std::fs::write(
            output.join(format!("rows-{trace_rows}.json")),
            serde_json::to_vec_pretty(&value)?,
        )?;
        previous = Some(count);
        previous_refresh = Some(refresh_count);
        previous_trace_rows = Some(trace_rows);
    }
    Ok(())
}

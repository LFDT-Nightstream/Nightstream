use std::error::Error;
use std::io::{BufReader, Read};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use neo_wasm::nebula::{WasmApplication, WasmSize};
use neo_wasm::{build_wasm_relation_layout, plan_ordered_wasm_memory_slots};
use nightstream::{nebula::Circuit, Goldilocks as F, PrimeCharacteristicRing, PrimeField64};
use nightstream_fprime::{ApplicationRecords, Linear, NativeBlock, NativeShape, BOUNDED_DEPTH_PACKAGES_IDENTITY};

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
        .ok_or("usage: native_constraint_census NEW_OUTPUT_DIRECTORY")?;
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
    let mut previous = None;
    for trace_rows in 1..=3 {
        let mut geometry = WasmSize::Small.geometry();
        geometry.b_ops = trace_rows * slots;
        let app = WasmApplication::new(
            &geometry,
            WasmSize::Small.limits().regions()?.0,
            NonZeroUsize::new(trace_rows).unwrap(),
        )?;
        let path = output.join(format!("rows-{trace_rows}.application"));
        Circuit::compile(geometry, app)?.write(&path)?;
        let (private, records, shape) = saved(&path)?;
        let compiled = packages.with_application(records.clone(), private, &shape)?;
        let package = compiled.light();
        let rows = package.plan_rows()?;
        let source = package.source_rows()?;
        let layout = package.layout()?;
        let companions = layout.spans().widths.len() - (package.sbox_start() + 150 * package.block_count());
        let count = rows.row_count();
        let value = serde_json::json!({
            "trace_rows": trace_rows,
            "slots_per_trace_row": slots,
            "geometry": {"r":geometry.r,"mu":geometry.mu,"w_ts":geometry.w_ts,"b_scan":geometry.b_scan,"n":geometry.n,"s_max":geometry.s_max},
            "raw_application_rows":records.row_count(),
            "compiled_light_rows":count,
            "unpacked_source_rows":source.row_count(),
            "compiled_source_entries":rows.entry_count(),
            "logical_coordinates":layout.width(),
            "paired_bits":companions,
            "packed_boolean_rows":companions.div_ceil(31),
            "removed_boolean_rows":source.row_count() + companions.div_ceil(31) - count,
            "added_trace_row_constraints":previous.map(|before| count-before),
            "identity":compiled.identity(),
            "fixed_cost_divided_by_trace_rows":false,
        });
        println!("{value}");
        std::fs::write(
            output.join(format!("rows-{trace_rows}.json")),
            serde_json::to_vec_pretty(&value)?,
        )?;
        previous = Some(count);
    }
    Ok(())
}

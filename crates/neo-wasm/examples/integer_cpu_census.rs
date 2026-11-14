use std::{collections::BTreeMap, error::Error, path::Path, sync::Arc};

use neo_math::F;
use nightstream_fprime::{
    ApplicationForm, ApplicationRecordsWriter, ApplicationRowHeader, ApplicationTerm, NativeShape,
    BOUNDED_DEPTH_PACKAGES_IDENTITY,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

struct Row {
    terms: [Vec<(usize, u64)>; 3],
}
struct Relation {
    one: usize,
    column_count: usize,
    rows: Vec<Row>,
    continuity: Vec<(usize, usize)>,
}

fn append(records: &mut ApplicationRecordsWriter, sides: [Vec<(usize, F)>; 3]) -> Result<(), Box<dyn Error>> {
    let sides = sides.map(|side| {
        let mut terms = BTreeMap::new();
        for (column, value) in side {
            *terms.entry(column).or_insert(F::ZERO) += value;
        }
        terms
            .into_iter()
            .filter(|(_, value)| *value != F::ZERO)
            .collect::<Vec<_>>()
    });
    if (sides[0].is_empty() || sides[1].is_empty()) && sides[2].is_empty() {
        return Ok(());
    }
    let is_constant = |side: &Vec<(usize, F)>| side.iter().all(|(column, _)| *column == 0);
    if is_constant(&sides[0]) || is_constant(&sides[1]) {
        let (constant, terms) = if is_constant(&sides[0]) {
            (&sides[0], &sides[1])
        } else {
            (&sides[1], &sides[0])
        };
        let multiplier = constant.first().map_or(F::ZERO, |(_, value)| *value);
        let mut difference = BTreeMap::new();
        for &(column, value) in terms {
            *difference.entry(column).or_insert(F::ZERO) += multiplier * value;
        }
        for &(column, value) in &sides[2] {
            *difference.entry(column).or_insert(F::ZERO) -= value;
        }
        if difference.values().all(|value| *value == F::ZERO) {
            return Ok(());
        }
    }
    let mut header = ApplicationRowHeader {
        constants: [0; 3],
        term_counts: [0; 3],
    };
    let mut terms = Vec::new();
    for (side, form) in [ApplicationForm::A, ApplicationForm::B, ApplicationForm::C]
        .into_iter()
        .enumerate()
    {
        for &(column, value) in &sides[side] {
            if column == 0 {
                header.constants[side] = value.as_canonical_u64();
            } else {
                terms.push(Ok(ApplicationTerm {
                    form,
                    variable: column + 3,
                    coefficient: value.as_canonical_u64(),
                }));
                header.term_counts[side] += 1;
            }
        }
    }
    records.append_row(header, terms)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: integer_cpu_census RELATION_JSON NEW_OUTPUT_DIRECTORY".into());
    }
    let value: serde_json::Value = serde_json::from_reader(std::fs::File::open(&args[0])?)?;
    let relation = Relation {
        one: value["one"].as_u64().ok_or("constant column")? as usize,
        column_count: value["column_count"].as_u64().ok_or("column count")? as usize,
        rows: value["rows"]
            .as_array()
            .ok_or("rows")?
            .iter()
            .map(|row| serde_json::from_value(row["terms"].clone()).map(|terms| Row { terms }))
            .collect::<Result<Vec<_>, _>>()?,
        continuity: serde_json::from_value(value["continuity"].clone())?,
    };
    if relation.one != 0 {
        return Err("expected the zero-index constant column".into());
    }
    let output = Path::new(&args[1]);
    std::fs::create_dir(output)?;
    let artifacts = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../formal/nightstream-fprime/artifacts");
    let packages = nightstream_fprime::load_bounded_depth_packages(
        &std::fs::read(artifacts.join("nightstream-fprime-bounded-depth-light-v1.json"))?,
        &std::fs::read(artifacts.join("nightstream-fprime-bounded-depth-refresh-v1.json"))?,
        BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )?;
    let width = relation.column_count - 1;
    let mut previous = None;
    for n in 1..=3 {
        let mut records = ApplicationRecordsWriter::new()?;
        for step in 0..n {
            for row in &relation.rows {
                let sides = row.terms.clone().map(|side| {
                    side.into_iter()
                        .map(|(column, value)| {
                            (if column == 0 { 0 } else { column + step * width }, F::from_u64(value))
                        })
                        .collect()
                });
                append(&mut records, sides)?;
            }
        }
        for step in 1..n {
            for &(previous, next) in &relation.continuity {
                append(
                    &mut records,
                    [
                        vec![(previous + (step - 1) * width, F::ONE), (next + step * width, -F::ONE)],
                        vec![(0, F::ONE)],
                        vec![],
                    ],
                )?;
            }
        }
        let records = Arc::new(records.finish()?);
        let compiled = packages.with_application(records.clone(), n * width, &NativeShape::default())?;
        let light = compiled.clone().expand()?.light;
        let (package, rows, layout) = (light.package(), light.rows(), light.layout());
        let count = rows.row_count();
        let companions = layout.spans().widths.len() - (package.sbox_start() + 150 * package.block_count());
        let value = serde_json::json!({
            "trace_rows":n,"compiled_light_rows":count,
            "added_trace_row_constraints":previous.map(|before|count-before),
            "source_rows":records.row_count(),"source_columns":n*width,
            "compiled_source_entries":rows.entry_count(),"logical_coordinates":layout.width(),
            "paired_bits":companions,"packed_boolean_rows":companions.div_ceil(31),
            "full_path":false,"scope":"complete CPU, host, and continuity equations; memory protocol excluded",
            "fixed_cost_divided_by_trace_rows":false,"identity":compiled.identity(),
        });
        std::fs::write(
            output.join(format!("rows-{n}.json")),
            serde_json::to_vec_pretty(&value)?,
        )?;
        println!("{value}");
        previous = Some(count);
    }
    Ok(())
}

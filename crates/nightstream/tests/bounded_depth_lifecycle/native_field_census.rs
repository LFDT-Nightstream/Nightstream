use std::{
    collections::BTreeMap,
    ops::{ControlFlow, Range},
    path::PathBuf,
};

use neo_ccs::GeometricRowRun;
use neo_math::F;
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows};
use neo_reductions::PiCcsError;
use nightstream_fprime::{LogicalLayout, PlanRows, SlotKind as Kind, SlotSpans};
use p3_field::PrimeCharacteristicRing;

use super::Circuit;
use crate::lifecycle::bounded_depth::Relation;

struct Census<'a> {
    plan: &'a PlanRows,
    slots: &'a SlotSpans,
    next: usize,
    entries: [usize; 4],
    expanded: [usize; 4],
    finished: usize,
}

impl MatrixRowSink for Census<'_> {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        assert_eq!(row * 4 + matrix, self.finished);
        let source = self.plan.columns()[self.next] as usize;
        assert_eq!(run.row(), row);
        assert_eq!(run.column_start(), self.slots.starts[source] as usize);
        assert_eq!(run.len(), self.slots.widths[source] as usize);
        assert_eq!(*run.initial(), self.plan.coefficients()[self.next]);
        assert_eq!(*run.ratio(), self.slots.ratio);
        self.next += 1;
        self.entries[matrix] += 1;
        self.expanded[matrix] += run.len();
        Ok(())
    }

    fn finish_matrix_row(&mut self, row: usize, matrix: usize) -> Result<ControlFlow<()>, PiCcsError> {
        assert_eq!(row * 4 + matrix, self.finished);
        assert_eq!(self.next, self.plan.offsets()[4 * row + matrix + 1]);
        self.finished += 1;
        Ok(ControlFlow::Continue(()))
    }
}

fn form_kind(plan: &PlanRows, layout: &LogicalLayout, row: usize, port: usize) -> &'static str {
    let mut constant = F::ZERO;
    let mut terms = Vec::new();
    for (column, coefficient) in plan.port(row, port) {
        if column == 0 {
            constant += coefficient;
        } else {
            terms.push((column, coefficient));
        }
    }
    if terms.is_empty() {
        return "constant";
    }
    let direct = constant == F::ZERO && terms.len() == 1 && terms[0].1 == F::ONE;
    let bit = |column| layout.slot(column).unwrap().1 == Kind::Bit;
    if direct && bit(terms[0].0) {
        return "direct_checked_bit";
    }
    if constant == F::ONE && terms.len() == 1 && terms[0].1 == -F::ONE && bit(terms[0].0) {
        return "checked_bit_complement";
    }
    if terms.iter().all(|&(column, _)| bit(column)) {
        return "affine_over_checked_bits";
    }
    if terms
        .iter()
        .any(|&(column, _)| layout.slot(column).unwrap().1 == Kind::Field)
    {
        return if direct { "direct_field" } else { "affine_with_field" };
    }
    if direct {
        "direct_unclassified"
    } else {
        "affine_with_unclassified"
    }
}

fn row_categories(plan: &PlanRows, layout: &LogicalLayout, range: Range<usize>) -> serde_json::Value {
    let mut categories = BTreeMap::<String, usize>::new();
    let mut products = BTreeMap::<String, usize>::new();
    for row in range.clone() {
        let kind = if plan.port(row, 3).next().is_some() {
            "native_power7"
        } else {
            let mut sides = [form_kind(plan, layout, row, 0), form_kind(plan, layout, row, 1)];
            if sides.contains(&"constant") {
                "linear"
            } else {
                sides.sort_unstable();
                *products
                    .entry(format!("{} * {}", sides[0], sides[1]))
                    .or_default() += 1;
                "product"
            }
        };
        *categories.entry(kind.to_owned()).or_default() += 1;
    }
    assert_eq!(categories.values().sum::<usize>(), range.len());
    serde_json::json!({"rows":range.len(), "categories":categories, "product_forms":products})
}

fn census(capacity: usize) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = root.join(format!(".audit/baseline/rows-{capacity}.application"));
    let circuit = Circuit::load(&source).unwrap();
    let mut values = Vec::new();
    let mut lowering = false;
    for (kind, package) in [
        ("light", circuit.packages.light()),
        ("refresh", circuit.packages.refresh()),
    ] {
        let relation = Relation::new(package, 109).unwrap();
        let plan = package.plan_rows().unwrap();
        let layout = package.layout().unwrap();
        let slots = layout.spans();
        let shape = MatrixRows::shape(&relation);
        assert_eq!(shape.matrices, 4);
        assert_eq!(shape.rows, plan.row_count());
        assert_eq!(relation.params().b(), 2);
        assert_eq!(relation.params().k_rho(), 16);
        let factored = relation.factored().unwrap().unwrap();
        assert_eq!(factored.shape(), shape);
        assert_eq!(factored.slot_count(), slots.widths.len());
        let mut sink = Census {
            plan: &plan,
            slots: &slots,
            next: 0,
            entries: [0; 4],
            expanded: [0; 4],
            finished: 0,
        };
        relation.visit_rows(0..shape.rows, &mut sink).unwrap();
        assert_eq!(sink.next, plan.entry_count());
        assert_eq!(sink.finished, 4 * shape.rows);
        let powers = (0..plan.ordinary_row_count())
            .filter(|&row| plan.port(row, 3).next().is_some())
            .count();
        lowering |= powers != 0;
        let application = plan.piece_ranges().first().unwrap().clone();
        values.push(serde_json::json!({
            "kind":kind, "rows":shape.rows, "matrix_count":shape.matrices,
            "padded_logical_coordinates":shape.columns, "logical_coordinates":layout.width(),
            "source_columns":factored.slot_count(), "entries_per_matrix":sink.entries,
            "expanded_entries_per_matrix":sink.expanded, "native_power7_rows":powers,
            "minimum_security_bits":109, "fresh_norm_bound":relation.params().b(), "k_rho":relation.params().k_rho(),
            "all_actual_matrix_runs_match_plan_and_slots":true,
            "application_ordinary":row_categories(&plan, &layout, application.clone()),
            "other_ordinary":row_categories(&plan, &layout, application.end..plan.ordinary_row_count()),
            "native_block_sbox_rows":plan.row_count()-plan.ordinary_row_count(),
            "owner_attribution":"not retained in ApplicationRecords",
            "affine_over_checked_bits_is_not_a_boolean_claim":true,
        }));
    }
    let mode = if lowering { "lowered" } else { "baseline" };
    let directory = root.join(format!(".audit/actual-factored-{mode}"));
    std::fs::create_dir_all(&directory).unwrap();
    let value =
        serde_json::json!({"capacity":capacity,"application":source,"identity":circuit.identity(),"circuits":values});
    std::fs::write(
        directory.join(format!("rows-{capacity}.json")),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .unwrap();
    eprintln!("{value}");
}

#[test]
#[ignore = "Requires .audit/baseline from native_constraint_census. Run each capacity under its own 300-second cap."]
fn complete_wasm_capacity_one_factored_census() {
    census(1);
}

#[test]
#[ignore = "Requires .audit/baseline from native_constraint_census. Run each capacity under its own 300-second cap."]
fn complete_wasm_capacity_two_factored_census() {
    census(2);
}

#[test]
#[ignore = "Requires .audit/baseline from native_constraint_census. Run each capacity under its own 300-second cap."]
fn complete_wasm_capacity_three_factored_census() {
    census(3);
}

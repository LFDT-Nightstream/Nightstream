//! Factored rows equal the plan rows read through the slot layout (Lean
//! `SlotLayout.layout`).

use super::super::Relation;
use super::support::packages;
use super::MINIMUM_SECURITY_BITS;
use neo_ccs::Mat;
use neo_math::{D, F};
use neo_reductions::superneo_eval::{MatrixRows, SuperneoZBlocks};
use p3_field::PrimeCharacteristicRing;

#[test]
fn factored_rows_match_the_slot_layout_rows() {
    let packages = packages();
    for package in [packages.light(), packages.refresh()] {
        let relation = Relation::new(package, MINIMUM_SECURITY_BITS).expect("relation");
        let (rows, layout) = (package.plan_rows().expect("rows"), package.layout().expect("layout"));
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let logical: Vec<F> = (0..layout.width())
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                F::from_u64((state >> 62) % 3) - F::ONE
            })
            .collect();
        let blocks = relation.blocks();
        let witness = Mat::from_row_major(
            D,
            blocks,
            (0..D)
                .flat_map(|lane| (0..blocks).map(move |block| block * D + lane))
                .map(|column| logical.get(column).copied().unwrap_or(F::ZERO))
                .collect(),
        );
        let witness = SuperneoZBlocks::from_witness_mat(&witness, relation.structure().m()).expect("witness");
        let factored = relation
            .factored()
            .expect("factored rows")
            .expect("a factored form");
        let values = factored.slot_values(&witness);
        for row in 0..rows.row_count() {
            for port in 0..4 {
                assert_eq!(
                    factored.row_value(port, row, &values),
                    layout.evaluate(&rows, row, port, &logical).expect("row"),
                    "circuit {} row {row} port {port}",
                    package.circuit()
                );
            }
        }
    }
}

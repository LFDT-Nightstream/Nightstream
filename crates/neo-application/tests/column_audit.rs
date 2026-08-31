#[cfg(feature = "audit-html")]
use neo_application::render_column_audit_html;
use neo_application::{
    continuity_column_occurrences, memory_column_occurrences, ApplicationRelation, ColumnConstraintIndex,
    ColumnFamilySpec, ColumnRegistry, ColumnWidth, ConditionalSelect, ConstraintTag, ContinuityCatalog,
    ContinuityColumnRole, ContinuityGroup, ContinuityLink, GadgetColumnRole, MemoryCatalog, MemoryColumnRole,
    MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemorySpec, R1csBuilder, R1csSide, ZeroTest,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn test_columns() -> ColumnRegistry {
    ColumnRegistry::new([ColumnFamilySpec {
        region: "test",
        start: 0,
        len: 8,
        name: "TEST_COLUMNS",
        role: "synthetic audit columns",
        width: ColumnWidth::Boolean,
    }])
    .expect("valid test columns")
}

fn test_relation() -> ApplicationRelation<&'static str> {
    let mut builder = R1csBuilder::new(8, 1, 0).expect("valid builder shape");
    let mut tagged = builder.tagged(ConstraintTag::new("synthetic rows", "test"));
    tagged.push_row([(2, F::ONE), (2, -F::ONE)], [(3, F::from_u64(2))], [(2, -F::ONE)]);
    ZeroTest::column(2, 4, 5).push_constraints(&mut tagged);
    ConditionalSelect {
        activation: 1,
        condition: [(5, F::ONE)],
        lhs: 2,
        rhs: 3,
        output: 6,
        delta: 7,
    }
    .push_constraints(&mut tagged);

    let r1cs = builder.build().expect("valid relation");
    ApplicationRelation::new(r1cs, test_columns()).expect("matching relation and registry")
}

#[test]
fn reverse_index_retains_every_sparse_term_and_gadget_role() {
    let relation = test_relation();
    let index = ColumnConstraintIndex::new(&relation);
    let r1cs = index.r1cs_occurrences(2).expect("in-range column");

    assert_eq!(r1cs[0].row_index(), 0);
    assert_eq!(r1cs[0].side(), R1csSide::A);
    assert_eq!(r1cs[0].coefficient(), F::ONE);
    assert_eq!(r1cs[1].row_index(), 0);
    assert_eq!(r1cs[1].side(), R1csSide::A);
    assert_eq!(r1cs[1].coefficient(), -F::ONE);
    assert_eq!(r1cs[2].row_index(), 0);
    assert_eq!(r1cs[2].side(), R1csSide::C);
    assert_eq!(r1cs[2].tagged_row().tag().label(), "synthetic rows");

    let gadget_roles: Vec<_> = index
        .gadget_occurrences(2)
        .expect("in-range column")
        .iter()
        .map(|occurrence| occurrence.role())
        .collect();
    assert_eq!(
        gadget_roles,
        [
            GadgetColumnRole::ZeroTestExpression {
                term_index: 0,
                coefficient: F::ONE,
            },
            GadgetColumnRole::ConditionalSelectLhs,
        ]
    );

    assert!(index.r1cs_occurrences(8).is_none());
    assert!(index.gadget_occurrences(8).is_none());
}

#[test]
fn memory_and_continuity_queries_report_exact_column_roles() {
    let columns = test_columns();
    let memory = MemoryCatalog::new(
        [MemorySpec {
            id: "test_ram",
            kind: MemoryKind::Ram,
            ports: vec![MemoryPortSpec {
                address_columns: vec![2, 5],
                value_column: 3,
                kind: MemoryPortKind::Write {
                    value_before_column: Some(4),
                },
                activation: MemoryPortActivation::When(5),
            }],
        }],
        &columns,
    )
    .expect("valid memory catalog");
    let memory_roles: Vec<_> = memory_column_occurrences(&memory, 5)
        .iter()
        .map(|occurrence| occurrence.role())
        .collect();
    assert_eq!(
        memory_roles,
        [MemoryColumnRole::Address { position: 1 }, MemoryColumnRole::Activation]
    );

    let continuity = ContinuityCatalog::new(
        [ContinuityGroup {
            name: "state",
            role: "synthetic state",
            links: vec![ContinuityLink {
                previous_step_column: 2,
                next_step_column: 6,
            }],
        }],
        &columns,
    )
    .expect("valid continuity catalog");
    let previous = continuity_column_occurrences(&continuity, 2);
    assert_eq!(previous.len(), 1);
    assert_eq!(previous[0].role(), ContinuityColumnRole::PreviousStep);
    assert_eq!(previous[0].group().name, "state");
    assert_eq!(previous[0].link().next_step_column, 6);
}

#[test]
#[cfg(feature = "audit-html")]
fn html_renderer_uses_application_metadata_without_domain_knowledge() {
    let relation = test_relation();
    let memory = MemoryCatalog::<&str>::new([], relation.columns()).expect("empty memory catalog");
    let continuity = ContinuityCatalog::new([], relation.columns()).expect("empty continuity catalog");
    let html = render_column_audit_html("Synthetic </script> audit", &relation, &memory, &continuity, &["test"]);

    let marker = "<script id=\"report-data\" type=\"application/json\">";
    let json_start = html.find(marker).expect("embedded report data") + marker.len();
    let json_end = html[json_start..]
        .find("</script>")
        .expect("end of embedded report data")
        + json_start;
    let report: serde_json::Value = serde_json::from_str(&html[json_start..json_end]).expect("valid embedded JSON");

    assert_eq!(report["title"], "Synthetic </script> audit");
    assert_eq!(
        report["rows"].as_array().unwrap().len(),
        relation.r1cs().catalog().len()
    );
    assert!(report["columns"]
        .as_array()
        .unwrap()
        .iter()
        .all(|column| column["generated"] == true));
    assert!(html.contains("R1CS rows show pre-coalescing builder terms"));
    assert!(!html.contains("Synthetic </script> audit"));
    assert!(!html.contains("startsWith('col_')"));
}

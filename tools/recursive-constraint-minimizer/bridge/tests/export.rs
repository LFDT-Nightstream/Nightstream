use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder};
use neo_fold_clean::frontends::r1cs_f_prime::lower_field_r1cs;
use neo_math::F;
use nightstream_constraint_exporter::{export_problem, export_sparse_problem, sparse_family_census, ExportRequest};
use p3_field::PrimeCharacteristicRing;
use recursive_constraint_minimizer::Scope;

fn source() -> (neo_fold_clean::engine::r1cs_circuit::R1csSnapshot, Vec<RowFamilyRange>) {
    let mut builder = R1csBuilder::new();
    let x = builder.alloc(F::ZERO);
    let one = Lc::from_const(F::ONE);
    let zero = Lc::zero();

    builder.enforce(&Lc::from_var(x), &one, &zero);
    let inner_start = builder.rows();
    builder.enforce(&Lc::from_var(x), &one, &zero);
    builder.record_row_family("inner", inner_start);
    builder.enforce(&Lc::from_var(x), &one, &zero);
    builder.record_row_family("outer", 0);

    (builder.snapshot(), builder.row_family_ranges().to_vec())
}

fn request(source_rows: Vec<usize>, complete_families: Vec<&str>) -> ExportRequest {
    ExportRequest {
        profile: "export-test".to_owned(),
        scope: Scope::Branch,
        public_input_count: 1,
        source_rows,
        complete_families: complete_families.into_iter().map(str::to_owned).collect(),
    }
}

fn sparse_source() -> neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs {
    let mut builder = R1csBuilder::new();
    builder.begin_encoding_stage("outer");
    let x = builder.alloc(F::ZERO);
    let one = Lc::from_const(F::ONE);
    let zero = Lc::zero();

    builder.enforce(&Lc::from_var(x), &one, &zero);
    let inner_start = builder.rows();
    builder.begin_encoding_stage("inner");
    builder.enforce(&Lc::from_var(x), &one, &zero);
    builder.record_row_family("inner", inner_start);
    builder.begin_encoding_stage("outer");
    builder.enforce(&Lc::from_var(x), &one, &zero);
    builder.record_row_family("outer", 0);

    lower_field_r1cs(builder, &[])
        .expect("lower sparse export fixture")
        .into_parts()
        .0
}

#[test]
fn exports_exact_rows_with_narrowest_family_owner() {
    let (snapshot, ranges) = source();
    let problem =
        export_problem(&snapshot, &ranges, request(vec![0, 1, 2], vec!["inner", "outer"])).expect("valid export");

    assert_eq!(problem.source.total_rows, 3);
    assert_eq!(problem.rows[0].family, "outer");
    assert_eq!(problem.rows[1].family, "inner");
    assert_eq!(problem.rows[2].family, "outer");
    assert_eq!(problem.rows[1].source_index, 1);
    assert_eq!(problem.rows[1].a[0].column, 1);
    assert_eq!(problem.rows[1].a[0].coefficient, "1");
    assert!(problem.source.artifact_digest.starts_with("sha256:"));

    let repeated = export_problem(&snapshot, &ranges, request(vec![1], vec!["inner"])).expect("valid bounded export");
    assert_eq!(problem.source.artifact_digest, repeated.source.artifact_digest);
}

#[test]
fn exports_exact_sparse_rows_with_narrowest_family_owner() {
    let source = sparse_source();
    let census = sparse_family_census(&source).expect("valid sparse family census");
    assert_eq!(
        census
            .iter()
            .map(|family| (family.name(), family.source_rows()))
            .collect::<Vec<_>>(),
        [("inner", [1].as_slice()), ("outer", [0, 2].as_slice())]
    );
    let problem =
        export_sparse_problem(&source, request(vec![0, 1, 2], vec!["inner", "outer"])).expect("valid sparse export");

    assert_eq!(problem.source.total_rows, 3);
    assert_eq!(problem.column_count, source.m);
    assert_eq!(problem.public_input_count, source.m_in);
    assert_eq!(problem.rows[0].family, "outer");
    assert_eq!(problem.rows[1].family, "inner");
    assert_eq!(problem.rows[2].family, "outer");
    assert_eq!(problem.rows[1].a[0].coefficient, "1");
    assert!(problem.source.artifact_digest.starts_with("sha256:"));

    let repeated =
        export_sparse_problem(&source, request(vec![1], vec!["inner"])).expect("valid bounded sparse export");
    assert_eq!(problem.source.artifact_digest, repeated.source.artifact_digest);
}

#[test]
fn rejects_sparse_public_prefix_drift() {
    let source = sparse_source();
    let mut drifted = request(vec![1], vec!["inner"]);
    drifted.public_input_count = source.m_in + 1;
    let error = export_sparse_problem(&source, drifted).expect_err("must reject public-prefix drift");
    assert!(error
        .to_string()
        .contains("differs from sparse source prefix"));
}

#[test]
fn rejects_incomplete_family_claim() {
    let (snapshot, ranges) = source();
    let error = export_problem(&snapshot, &ranges, request(vec![0], vec!["outer"]))
        .expect_err("must reject an incomplete family");
    assert!(error.to_string().contains("1 of 2"));
}

#[test]
fn rejects_partial_range_overlap() {
    let (snapshot, _) = source();
    let ranges = [
        RowFamilyRange {
            name: "left",
            row_start: 0,
            row_end: 2,
        },
        RowFamilyRange {
            name: "right",
            row_start: 1,
            row_end: 3,
        },
    ];
    let error = export_problem(&snapshot, &ranges, request(vec![0], vec![])).expect_err("must reject partial overlap");
    assert!(error.to_string().contains("partially overlap"));
}

#[test]
fn rejects_ambiguous_equal_ranges() {
    let (snapshot, _) = source();
    let ranges = [
        RowFamilyRange {
            name: "left",
            row_start: 0,
            row_end: 3,
        },
        RowFamilyRange {
            name: "right",
            row_start: 0,
            row_end: 3,
        },
    ];
    let error =
        export_problem(&snapshot, &ranges, request(vec![0], vec![])).expect_err("must reject ambiguous ownership");
    assert!(error.to_string().contains("ambiguous"));
}

#[test]
fn rejects_selected_row_without_an_owner() {
    let (snapshot, _) = source();
    let ranges = [RowFamilyRange {
        name: "tail",
        row_start: 1,
        row_end: 3,
    }];
    let error =
        export_problem(&snapshot, &ranges, request(vec![0], vec![])).expect_err("must reject unowned selected row");
    assert!(error.to_string().contains("no row-family owner"));
}

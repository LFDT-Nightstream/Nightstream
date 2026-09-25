use nightstream_fprime::components::{FormulaLibrary, SparseForm};

#[test]
fn component_form_row_ranges_check_bounds_and_match_complete_rows() {
    let library = FormulaLibrary::from_json(include_bytes!("../artifacts/shared-formulas-v1.json")).unwrap();
    let component = library.component("poseidon2-permutation-v1").unwrap();
    let variant = component.variant(0).unwrap();
    let width = component.input_count();
    let inputs: Vec<_> = (0..width).map(SparseForm::variable).collect();
    let all = variant.rows(&inputs, width).unwrap();
    assert_eq!(variant.rows_range(&inputs, width, 0..1).unwrap(), all[..1]);
    let end = all.len();
    assert_eq!(
        variant.rows_range(&inputs, width, end - 1..end).unwrap(),
        all[end - 1..]
    );
    assert!(variant.rows_range(&inputs, width, 0..0).unwrap().is_empty());
    assert!(variant
        .rows_range(&inputs, width, end..end)
        .unwrap()
        .is_empty());
    assert!(variant.rows_range(&inputs, width, 1..0).is_err());
    assert!(variant.rows_range(&inputs, width, 0..end + 1).is_err());
    assert!(variant
        .rows_range(&inputs, width, end + 1..end + 1)
        .is_err());
    assert!(variant
        .rows_range(&inputs[..width - 1], width, 0..0)
        .is_err());
    assert!(variant.rows_range(&inputs, width - 1, 0..0).is_err());
}

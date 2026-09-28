use super::*;
use std::io::Cursor;

#[test]
fn source_comparison_checks_content_and_length_across_read_boundaries() {
    let source = b"source witness bytes";
    // A one-byte buffer is the smallest read unit; the other reader returns
    // the complete slice. Equal files need not have equal read boundaries.
    assert!(equal_contents(BufReader::with_capacity(1, Cursor::new(source)), Cursor::new(source)).unwrap());
    assert!(!equal_contents(Cursor::new(source), Cursor::new(b"source witness byteX")).unwrap());
    assert!(!equal_contents(Cursor::new(source), Cursor::new(&source[..source.len() - 1])).unwrap());
}

#[test]
fn ccs_comparison_includes_output_openings_with_unchanged_sumcheck() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage1_actual_nifs/actual_result.json");
    let proof = super::super::super::proof(&read(path)).pi_ccs;
    let mut changed = proof.clone();
    changed.outputs[0].eval_a[0][0] += K::ONE;
    assert_eq!(changed.sumcheck.canonical_bytes(), proof.sumcheck.canonical_bytes());
    assert_ne!(changed.canonical_bytes(), proof.canonical_bytes());
}

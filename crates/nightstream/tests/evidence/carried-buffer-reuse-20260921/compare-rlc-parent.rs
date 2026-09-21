use std::{env, fs::File, io::BufReader};
use neo_ccs::Mat;
use neo_math::F;
fn main() {
    let paths: Vec<_> = env::args().skip(1).collect();
    assert_eq!(paths.len(), 2);
    let before: Mat<F> = serde_json::from_reader(BufReader::new(File::open(&paths[0]).unwrap())).unwrap();
    eprintln!("loaded baseline: {} rows, {} columns", before.rows(), before.cols());
    let after: Mat<F> = serde_json::from_reader(BufReader::new(File::open(&paths[1]).unwrap())).unwrap();
    eprintln!("loaded candidate: {} rows, {} columns", after.rows(), after.cols());
    assert!(before == after, "complete parent matrices differ");
    println!("equal_field_elements={}", before.rows() * before.cols());
}

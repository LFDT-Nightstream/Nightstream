//! Prepare the actual bounded base opening and matrix-image prefixes for
//! PiCCS fixtures. The package owns the witness IR,
//! matrix rows, and polynomial. This executable is fixture infrastructure.

use std::{
    env, fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

use neo_ajtai::nightstream_fprime_setup::{
    commit_production_signed_units, PRODUCTION_CARRIER_WIDTH, PRODUCTION_MESSAGE_COLUMNS,
};
use neo_ccs::{SparsePoly, Term};
use neo_math::{D, F};
use nightstream_fprime::{load_per_application_package, PackageError};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;

#[path = "generate_pi_ccs_fixture/child_commitment.rs"]
mod child_commitment;
#[path = "generate_pi_ccs_fixture/child_evaluations.rs"]
mod child_evaluations;
#[path = "generate_pi_ccs_fixture/folded_opening.rs"]
mod folded_opening;
#[path = "generate_pi_ccs_fixture/oracle.rs"]
mod oracle;
#[path = "generate_pi_ccs_fixture/ring_output.rs"]
mod ring_output;
#[path = "generate_pi_ccs_fixture/rounds.rs"]
mod rounds;
#[path = "generate_pi_ccs_fixture/running_prefix.rs"]
mod running_prefix;

const MATRIX_COUNT: usize = 14;
const LIVE_MATRICES: usize = MATRIX_COUNT - 1;
const PUBLIC_WORDS: usize = 270;
const ROUND_COUNT: usize = 28;
const MODULUS: u64 = 0xffff_ffff_0000_0001;

#[derive(Deserialize)]
struct BaseResult(
    [u64; 4],
    [u64; 4],
    Vec<u64>,
    [[u64; 2]; ROUND_COUNT],
    [u64; 8],
    [u64; 8],
    Vec<u64>,
);

#[derive(Deserialize)]
struct BaseFixture(u64, [u64; 4], Vec<u64>, Vec<u64>, BaseResult);

fn prepare(candidate: &Path, expected: [u64; 4], fixture: &Path, output: &Path) {
    let started = Instant::now();
    assert!(!output.exists(), "use a fresh external cache directory");
    let bytes = fs::read(candidate).expect("Lean candidate package");
    let package = load_per_application_package(&bytes, expected).expect("selected Lean structural identity");
    drop(bytes);
    let binding = package
        .production_verifier_binding()
        .expect("selected verifier binding");
    let BaseFixture(schema, context, private, public, result) =
        serde_json::from_slice(&fs::read(fixture).expect("Lean base fixture")).expect("base fixture schema");
    assert_eq!(schema, 1);
    assert_eq!(
        context,
        binding.verifier_context().digest(),
        "current base verifier context"
    );
    assert_eq!(private.len(), package.private_input_count());
    assert_eq!(public.len(), package.public_input_count());
    assert!(context
        .iter()
        .chain(&private)
        .chain(&public)
        .chain(&result.0)
        .chain(&result.1)
        .chain(&result.2)
        .chain(result.3.iter().flatten())
        .chain(&result.4)
        .chain(&result.5)
        .chain(&result.6)
        .all(|&word| word < MODULUS));

    let physical = package
        .execute_witness(&private, &public)
        .expect("exported base witness IR");
    let logical = package
        .execute_logical_assignment(&physical)
        .expect("exported logical transport");
    drop(physical);
    assert_eq!(logical.len(), package.logical_column_count());
    assert!(logical
        .balanced_values()
        .iter()
        .all(|value| (-1..=1).contains(value)));
    assert_eq!(logical.value(0).expect("constant coordinate"), 1);
    assert_eq!(result.2.len(), PUBLIC_WORDS);
    for (column, &expected) in result.2.iter().enumerate() {
        assert_eq!(logical.value(column).expect("fresh public projection"), expected);
    }
    let carrier_width = logical.len().div_ceil(D) * D;
    assert_eq!(carrier_width, PRODUCTION_CARRIER_WIDTH);
    assert_eq!(carrier_width / D, PRODUCTION_MESSAGE_COLUMNS as usize);
    let header = package.ccs_relation();
    assert_eq!(header.cube_variables(), ROUND_COUNT);
    assert_eq!(header.matrix_sources().len(), MATRIX_COUNT);
    assert_eq!(header.degree_bound(), 9);
    let polynomial = SparsePoly::new(
        MATRIX_COUNT,
        header
            .terms()
            .iter()
            .map(|term| Term {
                coeff: F::from_u64(term.coefficient()),
                exps: term
                    .exponents()
                    .iter()
                    .map(|&exponent| u32::try_from(exponent).expect("exponent"))
                    .collect(),
            })
            .collect(),
    );
    let encoded_terms = header
        .terms()
        .iter()
        .map(|term| json!([term.coefficient(), term.exponents()]))
        .collect::<Vec<_>>();
    println!("base source and selected binding: {:?}", started.elapsed());

    let mut images = vec![[0u64; LIVE_MATRICES]; package.row_count()];
    let workers = rayon::current_num_threads();
    let rows_per_worker = images.len().div_ceil(workers);
    images
        .par_chunks_mut(rows_per_worker)
        .enumerate()
        .try_for_each(|(worker, chunk)| {
            let start = worker * rows_per_worker;
            package.visit_matrix_rows(start..start + chunk.len(), |ordinal, row| {
                let mut values = [F::ZERO; MATRIX_COUNT];
                for (matrix, value) in values[..LIVE_MATRICES].iter_mut().enumerate() {
                    for entry in row
                        .matrix(matrix)
                        .ok_or(PackageError::Invalid("matrix family"))?
                    {
                        let coefficient = F::from_u64(entry.coefficient());
                        match logical.balanced_values()[entry.column()] {
                            -1 => *value -= coefficient,
                            0 => {}
                            1 => *value += coefficient,
                            _ => return Err(PackageError::Invalid("bounded base carrier")),
                        }
                    }
                }
                if !row
                    .matrix(LIVE_MATRICES)
                    .ok_or(PackageError::Invalid("zero matrix family"))?
                    .is_empty()
                {
                    return Err(PackageError::Invalid("matrix 13 is not zero"));
                }
                if polynomial.eval(&values) != F::ZERO {
                    eprintln!("base CCS polynomial failed at row {ordinal}");
                    return Err(PackageError::Invalid("base CCS row"));
                }
                chunk[ordinal - start] = std::array::from_fn(|matrix| values[matrix].as_canonical_u64());
                Ok(())
            })
        })
        .expect("all canonical matrix images and CCS rows");
    println!(
        "all {} scalar matrix rows checked: {:?}",
        images.len(),
        started.elapsed()
    );

    let mut carrier = logical.balanced_values().to_vec();
    carrier.resize(carrier_width, 0);
    let commitment = commit_production_signed_units(&carrier).expect("actual selected-key base commitment");
    assert_eq!((commitment.d, commitment.kappa), (D, 22));
    let commitment = commitment
        .data
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect::<Vec<_>>();
    assert_eq!(commitment.len(), 1_188);
    assert!(commitment.iter().any(|&word| word != 0));
    println!("actual bounded carrier commitment: {:?}", started.elapsed());

    fs::create_dir(output).expect("fresh external cache directory");
    let mut carrier_file = BufWriter::new(fs::File::create(output.join("carrier.i8")).expect("carrier sink"));
    for block in carrier.chunks_exact(D) {
        let bytes: [u8; D] = std::array::from_fn(|lane| block[lane] as u8);
        carrier_file
            .write_all(&bytes)
            .expect("bounded carrier block");
    }
    carrier_file.flush().expect("complete carrier sink");
    let mut image_file = BufWriter::new(fs::File::create(output.join("matrix-images.u64")).expect("matrix-image sink"));
    for row in &images {
        let mut encoded = [0u8; LIVE_MATRICES * 8];
        for (matrix, value) in row.iter().enumerate() {
            encoded[matrix * 8..matrix * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
        image_file
            .write_all(&encoded)
            .expect("canonical matrix-image row");
    }
    image_file.flush().expect("complete matrix-image sink");
    let metadata = json!([
        1,
        expected,
        context,
        logical.len(),
        carrier_width,
        images.len(),
        MATRIX_COUNT,
        ROUND_COUNT,
        encoded_terms,
        result.2,
        commitment
    ]);
    let mut encoded = serde_json::to_vec(&metadata).expect("numeric opening metadata");
    encoded.push(b'\n');
    fs::write(output.join("opening.json"), encoded).expect("opening metadata sink");
    println!(
        "prepared_pi_ccs_opening={} elapsed={:?}",
        output.display(),
        started.elapsed()
    );
    println!("honest_round_messages_and_full_ring_outputs=still_required");
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .first()
        .is_some_and(|mode| mode == "child-family" || mode == "child-family-at-rounds")
    {
        let at_rounds = arguments[0] == "child-family-at-rounds";
        assert_eq!(arguments.len(), if at_rounds { 11 } else { 10 }, "usage: generate_pi_ccs_fixture <child-family|child-family-at-rounds> <candidate> <id0> <id1> <id2> <id3> <folded-cache> <child-commitments-dir> <comma-separated-families> [round-result] <external-file-or-batch-directory>");
        let identity = std::array::from_fn(|lane| arguments[lane + 2].parse().expect("identity word"));
        child_evaluations::generate(
            Path::new(&arguments[1]),
            identity,
            Path::new(&arguments[6]),
            Path::new(&arguments[7]),
            &arguments[8],
            at_rounds.then(|| Path::new(&arguments[9])),
            Path::new(arguments.last().expect("child family output")),
        );
        return;
    }
    if arguments
        .first()
        .is_some_and(|mode| mode == "zero-child-commitments")
    {
        assert_eq!(arguments.len(), 8, "usage: generate_pi_ccs_fixture zero-child-commitments <candidate> <id0> <id1> <id2> <id3> <folded-cache> <children-dir>");
        let identity = std::array::from_fn(|lane| arguments[lane + 2].parse().expect("identity word"));
        child_commitment::generate_zero_children(
            Path::new(&arguments[1]),
            identity,
            Path::new(&arguments[6]),
            Path::new(&arguments[7]),
        );
        return;
    }
    if arguments
        .first()
        .is_some_and(|mode| mode == "child-commitment")
    {
        assert_eq!(arguments.len(), 9, "usage: generate_pi_ccs_fixture child-commitment <candidate> <id0> <id1> <id2> <id3> <folded-cache> <child> <external-output>");
        let identity = std::array::from_fn(|lane| arguments[lane + 2].parse().expect("identity word"));
        child_commitment::generate(
            Path::new(&arguments[1]),
            identity,
            Path::new(&arguments[6]),
            arguments[7].parse().expect("child index"),
            Path::new(&arguments[8]),
        );
        return;
    }
    if arguments
        .first()
        .is_some_and(|mode| mode == "check-child-commitments")
    {
        assert_eq!(arguments.len(), 10, "usage: generate_pi_ccs_fixture check-child-commitments <candidate> <id0> <id1> <id2> <id3> <base-cache> <folded-cache> <children-dir> <external-output>");
        let identity = std::array::from_fn(|lane| arguments[lane + 2].parse().expect("identity word"));
        child_commitment::check(
            Path::new(&arguments[1]),
            identity,
            Path::new(&arguments[6]),
            Path::new(&arguments[7]),
            Path::new(&arguments[8]),
            Path::new(&arguments[9]),
        );
        return;
    }
    if arguments.first().is_some_and(|mode| mode == "fold-base") {
        assert_eq!(arguments.len(), 9, "usage: generate_pi_ccs_fixture fold-base <candidate> <id0> <id1> <id2> <id3> <opening-cache> <Lean-PiCCS-result> <external-output>");
        let identity = std::array::from_fn(|lane| arguments[lane + 2].parse().expect("identity word"));
        folded_opening::generate(
            Path::new(&arguments[1]),
            identity,
            Path::new(&arguments[6]),
            Path::new(&arguments[7]),
            Path::new(&arguments[8]),
        );
        return;
    }
    if arguments
        .first()
        .is_some_and(|mode| mode == "running-prefix" || mode == "child-prefix")
    {
        let with_children = arguments[0] == "child-prefix";
        assert_eq!(arguments.len(), if with_children { 10 } else { 9 }, "usage: generate_pi_ccs_fixture <running-prefix|child-prefix> <candidate> <id0> <id1> <id2> <id3> <opening-cache> <Lean-prelude-result> [folded-child-cache] <external-output>");
        let identity = std::array::from_fn(|lane| arguments[lane + 2].parse().expect("identity word"));
        running_prefix::generate(
            Path::new(&arguments[1]),
            identity,
            Path::new(&arguments[6]),
            Path::new(&arguments[7]),
            with_children.then(|| Path::new(&arguments[8])),
            Path::new(arguments.last().expect("running prefix output")),
        );
        return;
    }
    if arguments.first().is_some_and(|mode| mode == "family") {
        assert_eq!(arguments.len(), 10, "usage: generate_pi_ccs_fixture family <candidate> <id0> <id1> <id2> <id3> <opening-cache> <rounds> <K|A0..A13|all> <external-output>");
        let identity = std::array::from_fn(|lane| arguments[lane + 2].parse().expect("identity word"));
        ring_output::generate(
            Path::new(&arguments[1]),
            identity,
            Path::new(&arguments[6]),
            Path::new(&arguments[7]),
            &arguments[8],
            Path::new(&arguments[9]),
        );
        return;
    }
    if arguments
        .first()
        .is_some_and(|mode| matches!(mode.as_str(), "rounds" | "running-rounds" | "child-rounds"))
    {
        let with_running = arguments[0] != "rounds";
        let with_children = arguments[0] == "child-rounds";
        assert_eq!(
            arguments.len(),
            if with_children { 6 } else if with_running { 5 } else { 4 },
            "usage: generate_pi_ccs_fixture <rounds|running-rounds|child-rounds> <opening-cache> <Lean-prelude-result> [running-prefix] [folded-child-cache] <external-output>"
        );
        rounds::generate(
            Path::new(&arguments[1]),
            Path::new(&arguments[2]),
            with_running.then(|| Path::new(&arguments[3])),
            with_children.then(|| Path::new(&arguments[4])),
            Path::new(arguments.last().expect("round output path")),
        );
        return;
    }
    assert_eq!(
        arguments.len(), 8,
        "usage: generate_pi_ccs_fixture prepare <candidate> <id0> <id1> <id2> <id3> <base-fixture> <external-cache-dir>"
    );
    assert_eq!(arguments[0], "prepare");
    let expected = std::array::from_fn(|lane| arguments[lane + 2].parse().expect("identity word"));
    prepare(
        Path::new(&arguments[1]),
        expected,
        Path::new(&arguments[6]),
        Path::new(&arguments[7]),
    );
}

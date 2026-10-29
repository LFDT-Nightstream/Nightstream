//! Writes the runs of the production layer-1 key for the Lean `setupCheck` and `runsCheck`.
//!
//! The file is little endian: the rows, the columns, the matrices and the run
//! count as `u64`, then each run in visit order as its row and start column
//! (`u32`), its matrix and length (`u8`), and its initial value and ratio
//! (canonical `u64`).

use std::ops::{ControlFlow, Range};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use neo_ccs::GeometricRowRun;
use neo_math::F;
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows, MatrixShape};
use neo_reductions::PiCcsError;
use neo_spartan::Key;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::staged::prepare;

const HEADER: usize = 32;
const RECORD: usize = 26;

/// Appends each pushed run to the file bytes.
struct Writer(Vec<u8>);

impl MatrixRowSink for Writer {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        let out = &mut self.0;
        out.extend_from_slice(
            &u32::try_from(row)
                .expect("a row fits 32 bits")
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &u32::try_from(run.column_start())
                .expect("a start fits 32 bits")
                .to_le_bytes(),
        );
        out.push(u8::try_from(matrix).expect("a matrix fits 8 bits"));
        out.push(u8::try_from(run.len()).expect("a length fits 8 bits"));
        out.extend_from_slice(&run.initial().as_canonical_u64().to_le_bytes());
        out.extend_from_slice(&run.ratio().as_canonical_u64().to_le_bytes());
        Ok(())
    }

    fn finish_matrix_row(&mut self, _row: usize, _matrix: usize) -> Result<ControlFlow<()>, PiCcsError> {
        Ok(ControlFlow::Continue(()))
    }
}

/// The runs of a file, pushed in file order for each row and matrix.
struct RunsFile {
    shape: MatrixShape,
    bytes: Vec<u8>,
}

impl RunsFile {
    fn read(path: &Path) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let word = |i: usize| u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap()) as usize;
        let shape = MatrixShape {
            rows: word(0),
            columns: word(1),
            matrices: word(2),
        };
        assert_eq!(bytes.len(), HEADER + RECORD * word(3), "the file holds its run count");
        Self { shape, bytes }
    }

    fn count(&self) -> usize {
        (self.bytes.len() - HEADER) / RECORD
    }

    fn record(&self, k: usize) -> &[u8] {
        &self.bytes[HEADER + RECORD * k..HEADER + RECORD * (k + 1)]
    }

    fn row(&self, k: usize) -> usize {
        u32::from_le_bytes(self.record(k)[0..4].try_into().unwrap()) as usize
    }

    fn matrix(&self, k: usize) -> usize {
        self.record(k)[8] as usize
    }

    fn run(&self, k: usize) -> GeometricRowRun<F> {
        let record = self.record(k);
        let start = u32::from_le_bytes(record[4..8].try_into().unwrap()) as usize;
        let word = |at: usize| {
            let value = u64::from_le_bytes(record[at..at + 8].try_into().unwrap());
            assert!(value < F::ORDER_U64, "a canonical word");
            F::from_u64(value)
        };
        GeometricRowRun::new(self.row(k), start, record[9] as usize, word(10), word(18))
    }
}

impl MatrixRows for RunsFile {
    fn shape(&self) -> MatrixShape {
        self.shape
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        let mut k = (0..self.count())
            .find(|&k| self.row(k) >= rows.start)
            .unwrap_or(self.count());
        for row in rows {
            for matrix in 0..self.shape.matrices {
                while k < self.count() && self.row(k) == row && self.matrix(k) == matrix {
                    sink.push_run(row, matrix, self.run(k))?;
                    k += 1;
                }
                if sink.finish_matrix_row(row, matrix)?.is_break() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The layer-1 key of the committed production key, read as `neo-spartan`'s
/// `production_key` reads it.
fn committed_layer1() -> Key {
    let bytes = std::fs::read(workspace().join("crates/nightstream/tests/fixtures/compression/key.bin")).unwrap();
    let (key, ..): (Key, [u64; 4], u32, f64, neo_spartan::shrink::Shape) = bincode::deserialize(&bytes).unwrap();
    key
}

#[test]
#[ignore = "Writes about 1 GB under target/lean-lowering; run this test separately under the 300-second cap."]
fn production_runs_rebuild_the_layer1_key() {
    let dir = workspace().join("target/lean-lowering");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("production_runs.bin");
    {
        let package = prepare();
        let rows = package.matrix_rows();
        let shape = rows.shape();
        let started = Instant::now();
        let mut writer = Writer(vec![0; HEADER]);
        rows.visit_rows(0..shape.rows, &mut writer).unwrap();
        let count = (writer.0.len() - HEADER) / RECORD;
        for (i, word) in [shape.rows, shape.columns, shape.matrices, count]
            .into_iter()
            .enumerate()
        {
            writer.0[8 * i..8 * i + 8].copy_from_slice(&(word as u64).to_le_bytes());
        }
        std::fs::write(&path, &writer.0).unwrap();
        eprintln!(
            "{count} runs, {} bytes, written in {:?}",
            writer.0.len(),
            started.elapsed()
        );
    }
    let started = Instant::now();
    let key = Key::derive(&RunsFile::read(&path)).unwrap();
    eprintln!("key derived from the file in {:?}", started.elapsed());
    assert!(
        key == committed_layer1(),
        "the runs in the file rebuild the committed production layer-1 key"
    );
}

#[test]
#[ignore = "needs production_runs.bin of production_runs_rebuild_the_layer1_key, the production_layer1.json of \
            neo-spartan's production_layer1_matches_lean, the production_setup_columns.bin of neo-spartan's \
            production_setup_columns, and lake build setupCheck in formal/compression-lean"]
fn production_setup_matches_lean() {
    let dir = workspace().join("target/lean-lowering");
    let layer1 = dir.join("production_layer1.json");
    let runs = dir.join("production_runs.bin");
    let columns = dir.join("production_setup_columns.bin");
    assert!(
        layer1.exists(),
        "run neo-spartan's production_layer1_matches_lean first"
    );
    assert!(runs.exists(), "run production_runs_rebuild_the_layer1_key first");
    assert!(columns.exists(), "run neo-spartan's production_setup_columns first");
    let check = workspace().join("formal/compression-lean/.lake/build/bin/setupCheck");
    assert!(
        check.exists(),
        "build the check with `lake build setupCheck` in formal/compression-lean"
    );
    let started = Instant::now();
    let output = Command::new(&check)
        .arg(&layer1)
        .arg(&runs)
        .arg(&columns)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("setupCheck: {} in {:?}", stdout.trim_end(), started.elapsed());
    assert!(
        output.status.success() && stdout.contains("setup facts hold") && stdout.contains("setup columns match"),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "needs production_runs.bin of production_runs_rebuild_the_layer1_key, and lake build runsCheck in \
            formal/compression-lean"]
fn production_runs_match_lean() {
    let runs = workspace().join("target/lean-lowering/production_runs.bin");
    assert!(runs.exists(), "run production_runs_rebuild_the_layer1_key first");
    let check = workspace().join("formal/compression-lean/.lake/build/bin/runsCheck");
    assert!(
        check.exists(),
        "build the check with `lake build runsCheck` in formal/compression-lean"
    );
    let started = Instant::now();
    let output = Command::new(&check).arg(&runs).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("runsCheck: {} in {:?}", stdout.trim_end(), started.elapsed());
    assert!(
        output.status.success() && stdout.contains("runs give the matrices"),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

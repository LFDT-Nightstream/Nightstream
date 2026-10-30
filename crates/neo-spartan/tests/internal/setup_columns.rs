//! Writes the run oracles of the production layer-1 key for the Lean `setupCheck`.
//!
//! The file is little endian: the run count and the slot count (`u64`), then
//! per run its row, per run its slot (`u32`), per run its coefficient
//! (canonical `u64`), and per slot its block (`u32`). The test requires each
//! oracle to be zero past these.

use std::ops::{ControlFlow, Range};
use std::path::{Path, PathBuf};

use neo_ccs::GeometricRowRun;
use neo_math::F;
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows, MatrixShape};
use neo_reductions::PiCcsError;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_field_v08::{PrimeCharacteristicRing as _, PrimeField64 as _};

use super::lowering::production_key;
use crate::field::Gl;
use crate::matrix::Tables;
use crate::{kappa, key, oracle};

/// The runs of `production_runs.bin` (`nightstream`'s `setup_runs.rs`) in file
/// order, which is visit order: by row, then by matrix.
struct RunsFile {
    shape: MatrixShape,
    runs: Vec<(usize, usize, GeometricRowRun<F>)>,
}

impl RunsFile {
    fn read(path: &Path) -> Self {
        let bytes = std::fs::read(path).unwrap_or_else(|error| {
            panic!(
                "{}: {error}; run nightstream's production_runs_rebuild_the_layer1_key first",
                path.display()
            )
        });
        let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
        let shape = MatrixShape {
            rows: word(0),
            columns: word(8),
            matrices: word(16),
        };
        let count = word(24);
        assert_eq!(bytes.len(), 32 + 26 * count, "the file holds its run count");
        let runs = bytes[32..]
            .chunks_exact(26)
            .map(|record| {
                let half = |at: usize| u32::from_le_bytes(record[at..at + 4].try_into().unwrap()) as usize;
                let canonical = |at: usize| {
                    let value = u64::from_le_bytes(record[at..at + 8].try_into().unwrap());
                    assert!(value < F::ORDER_U64, "a canonical word");
                    F::from_u64(value)
                };
                let run = GeometricRowRun::new(half(0), half(4), record[9] as usize, canonical(10), canonical(18));
                (half(0), record[8] as usize, run)
            })
            .collect();
        Self { shape, runs }
    }
}

impl MatrixRows for RunsFile {
    fn shape(&self) -> MatrixShape {
        self.shape
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        let mut k = self.runs.partition_point(|run| run.0 < rows.start);
        for row in rows {
            for matrix in 0..self.shape.matrices {
                while let Some((_, _, run)) = self
                    .runs
                    .get(k)
                    .filter(|run| run.0 == row && run.1 == matrix)
                {
                    sink.push_run(row, matrix, run.clone())?;
                    k += 1;
                }
                if let ControlFlow::Break(()) = sink.finish_matrix_row(row, matrix)? {
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

#[test]
#[ignore = "needs production_runs.bin of nightstream's production_runs_rebuild_the_layer1_key; writes about \
            700 MB under target/lean-lowering"]
fn production_setup_columns() {
    let dir = workspace().join("target/lean-lowering");
    let runs = RunsFile::read(&dir.join("production_runs.bin"));
    let tables = Tables::from_rows(&runs).unwrap();
    let derived = key(&runs, &tables, None).unwrap();
    assert!(
        derived == production_key().0,
        "the runs derive the committed production key"
    );
    let structure = tables.structure();
    let run_count = *structure.run_bounds.last().unwrap() as usize;
    let slot_count = structure.slots as usize;
    assert_eq!(
        run_count,
        runs.runs.len(),
        "every run of the file is a run of the tables"
    );
    let variables = derived.setup_variables();
    let counts = [run_count, run_count, run_count, slot_count];
    let columns: Vec<Vec<Gl>> = (0..4)
        .map(|j| oracle(&tables, derived.blocks, kappa() + j, variables))
        .collect();
    for (j, (column, &count)) in columns.iter().zip(&counts).enumerate() {
        assert!(
            column[count..].iter().all(|&x| x == Gl::ZERO),
            "run oracle {j} is zero past its entries"
        );
    }
    let small = |x: &Gl| {
        u32::try_from(x.as_canonical_u64())
            .expect("a row, slot or block fits 32 bits")
            .to_le_bytes()
    };
    let mut out = Vec::with_capacity(16 + 16 * run_count + 4 * slot_count);
    out.extend_from_slice(&(run_count as u64).to_le_bytes());
    out.extend_from_slice(&(slot_count as u64).to_le_bytes());
    columns[0][..run_count]
        .iter()
        .for_each(|x| out.extend_from_slice(&small(x)));
    columns[1][..run_count]
        .iter()
        .for_each(|x| out.extend_from_slice(&small(x)));
    columns[2][..run_count]
        .iter()
        .for_each(|x| out.extend_from_slice(&x.as_canonical_u64().to_le_bytes()));
    columns[3][..slot_count]
        .iter()
        .for_each(|x| out.extend_from_slice(&small(x)));
    std::fs::write(dir.join("production_setup_columns.bin"), &out).unwrap();
    eprintln!("{run_count} runs, {slot_count} slots, {} bytes", out.len());
}

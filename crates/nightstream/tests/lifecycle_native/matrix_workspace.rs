//! Count fixed verifier rows without retaining their matrix coefficients.

use super::{artifact, fs, load_poseidon2_hash_chain_v1_package, PreparedLifecycle};
use neo_ajtai::nightstream_fprime_setup::PRODUCTION_CARRIER_WIDTH;
use neo_ccs::GeometricRowRun;
use neo_math::F;
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows, MatrixShape, MatrixWindow};
use neo_reductions::PiCcsError;
use p3_field::PrimeCharacteristicRing;
use serde::Deserialize;
use std::{mem::size_of, ops::ControlFlow, ops::Range, time::Instant};

#[derive(Deserialize)]
struct Geometry {
    domain: usize,
    source_rows: [usize; 4],
    logical_rows: [usize; 4],
}

#[derive(Deserialize)]
struct Child {
    id: String,
    row_start: [usize; 4],
    row_count: [usize; 4],
    replaceable: bool,
}

#[derive(Deserialize)]
struct Manifest {
    geometry: Geometry,
    reference: [usize; 3],
    children: Vec<Child>,
}

fn dimension(coefficients: [usize; 4], counts: [usize; 3]) -> usize {
    coefficients
        .into_iter()
        .zip([1, counts[0], counts[1], counts[2]])
        .try_fold(0usize, |sum, (coefficient, count)| {
            sum.checked_add(coefficient.checked_mul(count)?)
        })
        .expect("manifest dimension fits the address space")
}

struct RowCounts {
    shape: MatrixShape,
    rows: Range<usize>,
    row: usize,
    matrix: usize,
    runs: usize,
    payload: usize,
    explicit: bool,
    geometric: bool,
    maximum: (usize, usize, usize),
    maximum_runs: (usize, usize),
    stop_after_first: bool,
    stopped: bool,
}

impl RowCounts {
    fn new(shape: MatrixShape, rows: Range<usize>) -> Self {
        Self {
            shape,
            row: rows.start,
            maximum: (rows.start, 0, 0),
            maximum_runs: (rows.start, 0),
            rows,
            matrix: 0,
            runs: 0,
            payload: 0,
            explicit: false,
            geometric: false,
            stop_after_first: false,
            stopped: false,
        }
    }

    fn slot(&self, row: usize, matrix: usize) {
        assert!(!self.stopped, "source called the sink after Break");
        assert_eq!((row, matrix), (self.row, self.matrix), "ordered matrix slots");
        assert!(self.rows.contains(&row));
    }
}

impl MatrixRowSink for RowCounts {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        self.slot(row, matrix);
        assert_eq!(run.row(), row);
        self.runs = self.runs.checked_add(1).expect("row run count");
        // Only the variable part orders one-row costs. Descriptor, dense
        // sentinel and counting bytes are constant across this source. The
        // full cost is obtained from MatrixWindow::required_workspace below.
        let bytes = if run.len() == 1 {
            self.explicit = true;
            size_of::<u32>()
                + if *run.initial() == F::ONE || *run.initial() == -F::ONE {
                    0
                } else {
                    size_of::<[u32; 2]>() + size_of::<u32>() + size_of::<u8>() + size_of::<F>()
                }
        } else {
            self.geometric = true;
            size_of::<[u64; 3]>()
        };
        self.payload = self.payload.checked_add(bytes).unwrap();
        Ok(())
    }

    fn finish_matrix_row(&mut self, row: usize, matrix: usize) -> Result<ControlFlow<()>, PiCcsError> {
        self.slot(row, matrix);
        // Each nonempty family owns start/end u32 offsets for one row.
        self.payload += (usize::from(self.explicit) + usize::from(self.geometric)) * 2 * size_of::<u32>();
        self.explicit = false;
        self.geometric = false;
        self.matrix += 1;
        if self.matrix == self.shape.matrices {
            if self.payload > self.maximum.2 {
                self.maximum = (row, self.runs, self.payload);
            }
            if self.runs > self.maximum_runs.1 {
                self.maximum_runs = (row, self.runs);
            }
            self.runs = 0;
            self.payload = 0;
            self.matrix = 0;
            self.row += 1;
            if self.stop_after_first {
                self.stopped = true;
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }
}

#[test]
#[ignore = "Count all stored fixed verifier rows; use the 300-second test cap. No proof or device work."]
fn fixed_prefix_rows_fit_the_minimum_matrix_workspace() {
    let started = Instant::now();
    // Deserialize only geometry and child ranges; the fixed matrix programs
    // come from the same prepared-package fixture as the lifecycle tests.
    let manifest: Manifest = serde_json::from_slice(include_bytes!("../../artifacts/shared-verifier-v1.json")).unwrap();
    let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let loaded = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
    drop(bytes);
    let binding = loaded.production_verifier_binding().unwrap();
    let mut package =
        PreparedLifecycle::from_package(loaded.into(), binding, crate::engine::Backend::Optimized, 114).unwrap();
    assert_eq!(
        package.structure.n,
        dimension(manifest.geometry.logical_rows, manifest.reference),
        "fixture uses the selected manifest row geometry"
    );

    // These are the existing source-domain and commitment-key bounds. Only
    // the header is changed, long enough to use the production reserve formula.
    assert_eq!(manifest.geometry.source_rows[1..], [0, 0, 1]);
    assert_eq!(manifest.geometry.logical_rows[1..], [0, 0, 1]);
    let maximum_application_rows = manifest.geometry.domain - manifest.geometry.source_rows[0];
    let maximum_rows = dimension(manifest.geometry.logical_rows, [0, 0, maximum_application_rows]);
    assert!(maximum_rows <= manifest.geometry.domain);
    let actual_shape = (package.structure.n, package.structure.m);
    package.structure.n = maximum_rows;
    package.structure.m = PRODUCTION_CARRIER_WIDTH;
    let minimum_workspace = package.matrix_workspace_bytes().unwrap();
    (package.structure.n, package.structure.m) = actual_shape;
    assert!(package.matrix_workspace_bytes().unwrap() >= minimum_workspace);

    let source = package.matrix_rows();
    let shape = source.shape();
    let mut maximum = (0usize, 0usize, 0usize);
    let mut maximum_runs = (0usize, 0usize);
    let mut fixed_rows = 0usize;
    let mut expected_start = 0usize;
    let mut application_rows = 0usize;
    let mut application_range = None;
    for child in &manifest.children {
        let start = dimension(child.row_start, manifest.reference);
        let count = dimension(child.row_count, manifest.reference);
        let end = start.checked_add(count).unwrap();
        assert_eq!(start, expected_start, "complete child row coverage");
        expected_start = end;
        if child.replaceable {
            assert_eq!(child.id, "application");
            application_rows += count;
            assert!(application_range.replace(start..end).is_none());
            continue;
        }
        assert_eq!(child.row_count[1..], [0, 0, 0], "fixed child row count");
        let mut counts = RowCounts::new(shape, start..end);
        source.visit_rows(start..end, &mut counts).unwrap();
        assert_eq!((counts.row, counts.matrix, counts.runs), (end, 0, 0));
        if counts.maximum.2 > maximum.2 {
            maximum = counts.maximum;
        }
        if counts.maximum_runs.1 > maximum_runs.1 {
            maximum_runs = counts.maximum_runs;
        }
        fixed_rows += count;
    }
    assert_eq!(expected_start, shape.rows);
    assert_eq!(application_rows, manifest.reference[2]);
    assert_eq!(fixed_rows, manifest.geometry.logical_rows[0]);

    // The caller can stop only after a complete row. A callback after Break
    // fails in slot(), including another matrix callback for that same row.
    let application_range = application_range.expect("one application child");
    let maximum_end = if maximum.0 < application_range.start {
        application_range.start
    } else {
        assert!(maximum.0 >= application_range.end);
        shape.rows
    };
    assert!(application_range.start > 1, "fixed prefix has a following row");
    for requested in [0..application_range.start, maximum.0..maximum_end] {
        if requested.len() <= 1 {
            continue;
        }
        let start = requested.start;
        let mut counts = RowCounts::new(shape, requested.clone());
        counts.stop_after_first = true;
        source.visit_rows(requested, &mut counts).unwrap();
        assert!(counts.stopped, "source reached the first complete row");
        assert_eq!((counts.row, counts.matrix, counts.runs), (start + 1, 0, 0));
    }

    // Count the maximum encoded-cost row again through the production
    // workspace API, with no MatrixWindow/cache allocation. The maximum
    // run-count row can differ now that scalar and geometric costs differ.
    let maximum_workspace = MatrixWindow::required_workspace(&source, maximum.0..maximum.0 + 1, 0).unwrap();
    eprintln!(
        "fixed_prefix_matrix_workspace rows={fixed_rows} max_row={} max_row_runs={} max_run_count_row={} max_runs={} encoded_workspace_bytes={maximum_workspace} minimum_workspace_bytes={minimum_workspace} elapsed={:?}",
        maximum.0, maximum.1, maximum_runs.0, maximum_runs.1, started.elapsed(),
    );
    assert!(
        maximum_workspace <= minimum_workspace,
        "one fixed verifier row must fit every accepted shape's matrix allowance"
    );
    // Variable application-row costs have a separate affine-run derivation;
    // this test supplies evidence only for the stored verifier prefix and tail.
}

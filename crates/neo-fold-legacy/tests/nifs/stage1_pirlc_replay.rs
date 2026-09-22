//! PiRLC replay transport. Source capture never reads the combined witness.
//! Missing source blocks encode zero; comparison checks the complete carrier.

use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

use neo_ajtai::nightstream_fprime_setup::PRODUCTION_MESSAGE_COLUMNS;
use neo_ccs::Mat;
use neo_fold_legacy::paper::pi_ccs;
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::de::DeserializeOwned;
use serde_json::json;

use super::{parent, stage1_actual, stage1_values};

fn read<T: DeserializeOwned>(path: &Path) -> T {
    serde_json::from_reader(BufReader::new(File::open(path).expect("replay input"))).expect("typed replay input")
}

fn create(path: &Path) -> BufWriter<File> {
    BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("fresh replay output"),
    )
}

pub(super) fn masks(witness: &Mat<F>, block: usize) -> (u64, u64) {
    if let Some((positive, negative)) = witness.packed_signed_unit_column_masks() {
        let pair = (positive[block], negative[block]);
        assert_eq!(pair.0 & pair.1, 0);
        assert_eq!((pair.0 | pair.1) >> D, 0);
        return pair;
    }
    if witness.virtual_constant_value() == Some(&F::ZERO) {
        return (0, 0);
    }
    let mut pair = (0, 0);
    for lane in 0..D {
        match witness[(lane, block)].as_canonical_u64() {
            0 => {}
            1 => pair.0 |= 1u64 << lane,
            value if value == F::ORDER_U64 - 1 => pair.1 |= 1u64 << lane,
            _ => panic!("source coefficient exceeds signed-unit norm"),
        }
    }
    pair
}

/// Capture all source coordinates after checking the saved C proof against
/// the original claims. The parent witness is deliberately absent from this API.
pub fn export_sources(package: &Path, sources: &Path, ccs: &Path, output: &Path) {
    let started = Instant::now();
    assert!(!output.exists(), "fresh source capture directory");
    let actual = stage1_actual::load_path(package, sources);
    let saved: parent::SavedCcs = read(ccs);
    assert_eq!(saved.schema, 1);
    assert_eq!(saved.structural_identifier, actual.package.structural_identifier());
    assert_eq!(saved.package_identity, actual.package.package_identity());
    assert_eq!(saved.verification_key_digest, actual.package.verification_key_digest());
    let proof = pi_ccs::Proof {
        sumcheck: saved.sumcheck,
        outputs: saved.outputs,
    };
    parent::replay_ccs(
        &actual.params,
        actual.package.structure(),
        &actual.fresh.claim,
        &actual.running,
        &proof,
    );
    let witnesses = std::iter::once(&actual.fresh.witness.Z)
        .chain(actual.running.witnesses.iter())
        .collect::<Vec<_>>();
    assert_eq!(witnesses.len(), 17);
    let blocks = PRODUCTION_MESSAGE_COLUMNS as usize;
    for witness in &witnesses {
        assert_eq!((witness.rows(), witness.cols()), (D, blocks));
    }
    fs::create_dir(output).unwrap();
    let mut ccs_out = create(&output.join("ccs-input.json"));
    serde_json::to_writer(
        &mut ccs_out,
        &stage1_values::ccs_input(&actual.fresh.claim, &actual.running.claims, &proof),
    )
    .unwrap();
    ccs_out.flush().unwrap();
    let mut out = create(&output.join("sources.jsonl"));
    writeln!(out, "{}", json!([1, D, witnesses.len(), blocks])).unwrap();
    let mut active_blocks = 0;
    let mut source_blocks = 0;
    for block in 0..blocks {
        let mut values = Vec::new();
        for (source, witness) in witnesses.iter().enumerate() {
            let (positive, negative) = masks(witness, block);
            if positive | negative != 0 {
                values.push([source as u64, positive, negative]);
            }
        }
        if !values.is_empty() {
            active_blocks += 1;
            source_blocks += values.len();
            writeln!(out, "{}", json!([block, values])).unwrap();
        }
    }
    writeln!(out, "[]").unwrap();
    out.flush().unwrap();
    println!(
        "pirlc_source_capture=passed blocks={blocks} active_blocks={active_blocks} source_blocks={source_blocks} coefficients={} elapsed={:?}",
        blocks * D,
        started.elapsed()
    );
}

/// Compare a consecutive Lean range with every corresponding native field value.
/// A missing emitted block means an independently computed zero block.
fn compare_range(witness: &Mat<F>, path: &Path) -> Result<(usize, usize), String> {
    let mut lines = BufReader::new(File::open(path).map_err(|e| e.to_string())?).lines();
    let header: [usize; 4] = serde_json::from_str(
        &lines
            .next()
            .ok_or("missing replay header")?
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let [schema, blocks, start, end] = header;
    if schema != 1 || blocks != witness.cols() || start >= end || end > blocks {
        return Err("invalid complete replay range".into());
    }
    let mut cursor = start;
    let check = |block: usize, values: Option<&[u64]>| -> Result<(), String> {
        if values.is_some_and(|v| v.len() != D || v.iter().any(|&x| x >= F::ORDER_U64)) {
            return Err("invalid canonical replay block".into());
        }
        for lane in 0..D {
            let expected = values.map_or(0, |v| v[lane]);
            if witness[(lane, block)].as_canonical_u64() != expected {
                return Err(format!("coefficient mismatch at block {block}, lane {lane}"));
            }
        }
        Ok(())
    };
    let mut complete = false;
    for line in lines {
        let line = line.map_err(|e| e.to_string())?;
        if complete {
            return Err("extra data after replay terminator".into());
        }
        if line == "[]" {
            complete = true;
            continue;
        }
        let (block, values): (usize, Vec<u64>) = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if block < cursor || block >= end {
            return Err("duplicate or out-of-range replay block".into());
        }
        for zero in cursor..block {
            check(zero, None)?;
        }
        check(block, Some(&values))?;
        cursor = block + 1;
    }
    if !complete {
        return Err("missing replay terminator".into());
    }
    for zero in cursor..end {
        check(zero, None)?;
    }
    Ok((start, end))
}

/// Export an actual parent range as canonical fields, with no Lean result input.
pub fn export_target(parent_witness: &Path, start: usize, end: usize, output: &Path) {
    let witness: Mat<F> = read(parent_witness);
    assert_eq!(
        (witness.rows(), witness.cols()),
        (D, PRODUCTION_MESSAGE_COLUMNS as usize)
    );
    assert!(start < end && end <= witness.cols());
    let mut out = create(output);
    writeln!(out, "{}", json!([1, witness.cols(), start, end])).unwrap();
    for block in start..end {
        let values = (0..D)
            .map(|lane| witness[(lane, block)].as_canonical_u64())
            .collect::<Vec<_>>();
        writeln!(out, "{}", json!([block, values])).unwrap();
    }
    writeln!(out, "[]").unwrap();
    out.flush().unwrap();
    println!("pirlc_target_range=exported start={start} end={end}");
}

/// Check contiguous coverage of the full carrier, then alter its last tail
/// coefficient and require the same comparison routine to reject it.
pub fn compare(parent_witness: &Path, ranges: &[String]) {
    let started = Instant::now();
    let mut witness: Mat<F> = read(parent_witness);
    assert_eq!(
        (witness.rows(), witness.cols()),
        (D, PRODUCTION_MESSAGE_COLUMNS as usize)
    );
    assert!(!ranges.is_empty());
    let mut cursor = 0;
    let mut last_start = 0;
    for range in ranges {
        let (start, end) =
            compare_range(&witness, Path::new(range)).unwrap_or_else(|error| panic!("replay range {range}: {error}"));
        assert_eq!(start, cursor, "no gap or overlap in replay coverage");
        last_start = start;
        cursor = end;
    }
    assert_eq!(cursor, witness.cols(), "complete carrier including tail");
    let last = witness.cols() - 1;
    witness[(D - 1, last)] += F::ONE;
    let error = compare_range(&witness, Path::new(ranges.last().unwrap())).unwrap_err();
    assert_eq!(error, format!("coefficient mismatch at block {last}, lane {}", D - 1));
    println!(
        "pirlc_witness_replay=passed coefficients={} tail_mutation=rejected block={last} lane={} range={last_start}..{} elapsed={:?}",
        witness.rows() * witness.cols(),
        D - 1,
        witness.cols(),
        started.elapsed()
    );
}

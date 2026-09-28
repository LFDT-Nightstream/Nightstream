//! Exact private-witness comparisons for independent Lean C/R/D generation.
//! Adapted comparison kernels from tests/nifs/stage1_{pirlc,pidec}_replay.rs
//! at 7f51e101; no producer input is obtained from these native comparison targets.

use super::super::super::{load, Mat, Path, PathBuf, D, F};
use neo_ajtai::nightstream_fprime_setup::PRODUCTION_MESSAGE_COLUMNS;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    ops::Range,
    time::Instant,
};
const CHILDREN: usize = 16;

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

fn compare_parent_range(witness: &Mat<F>, path: &Path) -> Result<(usize, usize), String> {
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

/// Check contiguous coverage of the full carrier, then alter its last tail
/// coefficient and require the same comparison routine to reject it.
pub(super) fn parent(parent_witness: &Path, ranges: &[PathBuf]) {
    let started = Instant::now();
    let mut witness: Mat<F> = load(parent_witness);
    assert_eq!(
        (witness.rows(), witness.cols()),
        (D, PRODUCTION_MESSAGE_COLUMNS as usize)
    );
    assert!(!ranges.is_empty());
    let mut cursor = 0;
    let mut last_start = 0;
    for range in ranges {
        let (start, end) =
            compare_parent_range(&witness, range).unwrap_or_else(|error| panic!("replay range {range:?}: {error}"));
        assert_eq!(start, cursor, "no gap or overlap in replay coverage");
        last_start = start;
        cursor = end;
    }
    assert_eq!(cursor, witness.cols(), "complete carrier including tail");
    let last = witness.cols() - 1;
    witness[(D - 1, last)] += F::ONE;
    let error = compare_parent_range(&witness, ranges.last().unwrap()).unwrap_err();
    assert_eq!(error, format!("coefficient mismatch at block {last}, lane {}", D - 1));
    println!(
        "pirlc_witness_replay=passed coefficients={} tail_mutation=rejected block={last} lane={} range={last_start}..{} elapsed={:?}",
        witness.rows() * witness.cols(),
        D - 1,
        witness.cols(),
        started.elapsed()
    );
}
type ChildMasks = [[u64; 2]; CHILDREN];
const ZERO_MASKS: ChildMasks = [[0; 2]; CHILDREN];

fn actual_masks(children: &[Mat<F>; CHILDREN], block: usize) -> ChildMasks {
    std::array::from_fn(|child| {
        let (positive, negative) = masks(&children[child], block);
        [positive, negative]
    })
}

fn decode_masks(entries: &[[u64; 3]]) -> Result<ChildMasks, String> {
    if entries.is_empty() {
        return Err("zero child blocks must be omitted".into());
    }
    let mut output = ZERO_MASKS;
    let mut next = 0;
    for &[child, positive, negative] in entries {
        if child < next || child >= CHILDREN as u64 {
            return Err("child indices must be unique and increasing".into());
        }
        if (positive | negative) >> D != 0 || positive & negative != 0 || positive | negative == 0 {
            return Err("invalid signed-unit child masks".into());
        }
        output[child as usize] = [positive, negative];
        next = child + 1;
    }
    Ok(output)
}

/// Equality of both validated masks checks every lane, including its sign.
/// A mismatch reports the first changed coefficient rather than a digest.
fn compare_block(actual: &ChildMasks, expected: &ChildMasks, block: usize, range: &Range<usize>) -> Result<(), String> {
    for child in 0..CHILDREN {
        let difference = (actual[child][0] ^ expected[child][0]) | (actual[child][1] ^ expected[child][1]);
        if difference != 0 {
            let lane = difference.trailing_zeros();
            return Err(format!(
                "coefficient mismatch at child {child}, block {block}, lane {lane}, range {}..{}",
                range.start, range.end
            ));
        }
    }
    Ok(())
}

/// Header: [1,54,16,blocks,start,end]. Rows: [block,[[child,pos,neg],...]].
/// Every omitted child or block is compared as zero. A final [] is required.
fn compare_range(children: &[Mat<F>; CHILDREN], path: &Path) -> Result<(Range<usize>, ChildMasks), String> {
    let mut lines = BufReader::new(File::open(path).map_err(|error| error.to_string())?).lines();
    let header: [usize; 6] = serde_json::from_str(
        &lines
            .next()
            .ok_or("missing PiDEC replay header")?
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let [schema, degree, count, blocks, start, end] = header;
    if schema != 1
        || degree != D
        || count != CHILDREN
        || blocks != PRODUCTION_MESSAGE_COLUMNS as usize
        || start >= end
        || end > blocks
    {
        return Err("invalid selected PiDEC replay range".into());
    }
    let range = start..end;
    let mut cursor = start;
    let mut complete = false;
    let mut last_masks = ZERO_MASKS;
    for line in lines {
        let line = line.map_err(|error| error.to_string())?;
        if complete {
            return Err("extra data after PiDEC replay terminator".into());
        }
        if line == "[]" {
            complete = true;
            continue;
        }
        let (block, entries): (usize, Vec<[u64; 3]>) =
            serde_json::from_str(&line).map_err(|error| error.to_string())?;
        if block < cursor || block >= end {
            return Err("duplicate or out-of-range PiDEC replay block".into());
        }
        let expected = decode_masks(&entries)?;
        for zero in cursor..block {
            compare_block(&actual_masks(children, zero), &ZERO_MASKS, zero, &range)?;
        }
        compare_block(&actual_masks(children, block), &expected, block, &range)?;
        if block == end - 1 {
            last_masks = expected;
        }
        cursor = block + 1;
    }
    if !complete {
        return Err("missing PiDEC replay terminator".into());
    }
    for zero in cursor..end {
        compare_block(&actual_masks(children, zero), &ZERO_MASKS, zero, &range)?;
    }
    Ok((range, last_masks))
}

/// Compare consecutive Lean ranges against all 16 saved child matrices, then
/// change exactly one final target coefficient and require localized rejection.
pub(super) fn children(child_directory: &Path, ranges: &[PathBuf]) {
    let started = Instant::now();
    assert!(!ranges.is_empty(), "at least one complete replay range");
    let blocks = PRODUCTION_MESSAGE_COLUMNS as usize;
    let children: [Mat<F>; CHILDREN] = std::array::from_fn(|child| {
        let path = child_directory.join(format!("digit-{child}.json"));
        let bytes = fs::read(&path).unwrap_or_else(|error| panic!("child {child} file: {error}"));
        let witness: Mat<F> =
            serde_json::from_slice(&bytes).unwrap_or_else(|error| panic!("child {child} matrix: {error}"));
        assert_eq!((witness.rows(), witness.cols()), (D, blocks), "child {child} shape");
        witness
    });
    println!("pidec_child_targets_loaded={CHILDREN} elapsed={:?}", started.elapsed());
    let mut cursor = 0;
    let mut tail = None;
    for path in ranges {
        let (range, last_masks) =
            compare_range(&children, path).unwrap_or_else(|error| panic!("PiDEC replay range {path:?}: {error}"));
        assert_eq!(range.start, cursor, "no gap or overlap in PiDEC replay coverage");
        cursor = range.end;
        println!(
            "pidec_child_range=matched start={} end={} coefficients={} elapsed={:?}",
            range.start,
            range.end,
            (range.end - range.start) * D * CHILDREN,
            started.elapsed()
        );
        tail = Some((range, last_masks));
    }
    assert_eq!(cursor, blocks, "complete PiDEC carrier including every child tail");
    let (last_range, last_expected) = tail.expect("nonempty replay ranges");
    let last = blocks - 1;
    let child = CHILDREN - 1;
    let lane = D - 1;
    let mut changed_target = actual_masks(&children, last);
    let bit = 1u64 << lane;
    // Change +1 to zero, or change zero/-1 to +1. Other coordinates stay exact.
    if changed_target[child][0] & bit != 0 {
        changed_target[child][0] &= !bit;
    } else {
        changed_target[child][0] |= bit;
        changed_target[child][1] &= !bit;
    }
    let error = compare_block(&changed_target, &last_expected, last, &last_range).unwrap_err();
    assert_eq!(
        error,
        format!(
            "coefficient mismatch at child {child}, block {last}, lane {lane}, range {}..{}",
            last_range.start, last_range.end
        )
    );
    println!(
        "pidec_private_witness_replay=passed children={CHILDREN} blocks={blocks} coefficients={} tail_mutation=rejected child={child} block={last} lane={lane} range={}..{} elapsed={:?}",
        blocks * D * CHILDREN,
        last_range.start,
        last_range.end,
        started.elapsed()
    );
}

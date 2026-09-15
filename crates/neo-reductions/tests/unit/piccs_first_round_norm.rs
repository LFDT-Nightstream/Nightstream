//! Compare the actual CPU norm kernel with a Lean inner-cubic range result.
//! Source masks and public coins are inputs; Lean coefficients are targets only.
//! This file is a private cfg(test) child of cpu_oracle.rs, not an integration crate.

use std::{
    fs::File,
    io::{BufRead, BufReader, Cursor},
    path::PathBuf,
    time::Instant,
};

use neo_ajtai::nightstream_fprime_setup::PRODUCTION_MESSAGE_COLUMNS;
use neo_ccs::Mat;
use neo_math::{from_complex, KExtensions, D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;

use super::{norm_coefficients, prefix, Assignment, EqualityWeights};

// The selected Stage 1 profile has one fresh source and k_rho running sources.
const SOURCES: usize = 1 + neo_params::nightstream_goldilocks_k16::K_RHO as usize;
const ROUNDS: usize = 28;
const BLOCKS: usize = PRODUCTION_MESSAGE_COLUMNS as usize;

#[derive(Deserialize)]
struct LeanNorm(u64, usize, usize, [[u64; 2]; ROUNDS], [u64; 2], [[u64; 2]; 4]);

fn field(words: [u64; 2]) -> Result<K, String> {
    if words.iter().any(|word| *word >= F::ORDER_U64) {
        return Err("noncanonical extension-field word".into());
    }
    Ok(from_complex(F::from_u64(words[0]), F::from_u64(words[1])))
}

fn words(value: K) -> [u64; 2] {
    value.to_limbs_u64().into()
}

fn next_line(reader: &mut impl BufRead, line: &mut String) -> Result<bool, String> {
    line.clear();
    reader
        .read_line(line)
        .map(|size| size != 0)
        .map_err(|error| error.to_string())
}

/// Validate the complete ordered capture. Store only the requested interval;
/// omitted blocks/sources remain zero in all seventeen compact matrices.
fn source_range(mut reader: impl BufRead, first: usize, end: usize) -> Result<Vec<Mat<F>>, String> {
    if first >= end || end > BLOCKS {
        return Err("invalid selected carrier block range".into());
    }
    let mut line = String::new();
    if !next_line(&mut reader, &mut line)? {
        return Err("missing source header".into());
    }
    let header: [usize; 4] = serde_json::from_str(&line).map_err(|error| error.to_string())?;
    if header != [1, D, SOURCES, BLOCKS] {
        return Err("source header does not match the selected carrier".into());
    }
    let count = end - first;
    let mut masks: Vec<(Vec<u64>, Vec<u64>)> = (0..SOURCES)
        .map(|_| (vec![0; count], vec![0; count]))
        .collect();
    let mut next_block = 0;
    loop {
        if !next_line(&mut reader, &mut line)? {
            return Err("missing source terminator".into());
        }
        if line.trim_matches(|character: char| character.is_ascii_whitespace()) == "[]" {
            if next_line(&mut reader, &mut line)? {
                return Err("extra data after source terminator".into());
            }
            break;
        }
        let (block, entries): (usize, Vec<[u64; 3]>) =
            serde_json::from_str(&line).map_err(|error| error.to_string())?;
        if block < next_block || block >= BLOCKS {
            return Err("duplicate or out-of-range source block".into());
        }
        next_block = block + 1;
        if entries.is_empty() {
            return Err("zero source blocks must be omitted".into());
        }
        let mut next_source = 0;
        for [source, positive, negative] in entries {
            let source = usize::try_from(source).map_err(|error| error.to_string())?;
            if source < next_source || source >= SOURCES {
                return Err("source indices must be unique and increasing".into());
            }
            next_source = source + 1;
            if (positive | negative) >> D != 0 || (positive & negative) != 0 || (positive | negative) == 0 {
                return Err("invalid signed-unit source masks".into());
            }
            if first <= block && block < end {
                masks[source].0[block - first] = positive;
                masks[source].1[block - first] = negative;
            }
        }
    }
    masks
        .into_iter()
        .map(|(positive, negative)| {
            Mat::compact_signed_unit_from_column_masks(D, count, &positive, &negative)
                .map_err(|error| format!("source matrix: {error:?}"))
        })
        .collect()
}

fn compare(actual: &[K; 4], target: &[K; 4]) -> Result<(), String> {
    for (coefficient, (actual, target)) in actual.iter().zip(target).enumerate() {
        if actual != target {
            return Err(format!(
                "inner norm coefficient {coefficient}: CPU {actual:?}, Lean {target:?}"
            ));
        }
    }
    Ok(())
}

#[test]
#[ignore = "Requires original sources.jsonl and lean-norm.json in the local fixture directory; run under the 300-second cap."]
fn original_first_round_inner_norm_matches_lean() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/piccs_first_round_norm");
    let started = Instant::now();
    let LeanNorm(schema, first, end, alpha_words, gamma_words, target_words) = serde_json::from_reader(BufReader::new(
        File::open(fixture.join("lean-norm.json")).expect("Lean norm result"),
    ))
    .expect("exact six-field Lean norm schema");
    assert_eq!(schema, 1);
    assert_eq!(
        (first, end),
        (0, BLOCKS),
        "the completion check requires the complete carrier"
    );
    assert_eq!(D % 2, 0, "adjacent pairs stay within complete carrier blocks");
    assert!(BLOCKS * D <= 1usize << ROUNDS, "selected carrier fits the Boolean cube");
    let alpha = alpha_words
        .into_iter()
        .map(field)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let gamma = field(gamma_words).unwrap();
    let target = target_words.map(|value| field(value).unwrap());
    let witnesses = source_range(
        BufReader::new(File::open(fixture.join("sources.jsonl")).expect("original source capture")),
        first,
        end,
    )
    .expect("complete valid original source capture");
    assert_eq!(witnesses.len(), SOURCES);
    let load_seconds = started.elapsed().as_secs_f64();
    let assignments = witnesses
        .iter()
        .map(|witness| Assignment::new(witness, (end - first) * D))
        .collect::<Vec<_>>();
    let weights = EqualityWeights::new(&alpha[1..]);
    let compute_started = Instant::now();
    // Only original masks, public coins and the declared block bounds reach
    // the same kernel used by the production CPU oracle. Targets do not.
    let actual = norm_coefficients(&assignments, gamma, &weights, first * (D / 2));
    let compute_seconds = compute_started.elapsed().as_secs_f64();
    compare(&actual, &target).unwrap();
    let mut changed = target;
    changed[3] += K::ONE;
    assert!(
        compare(&actual, &changed).is_err(),
        "a changed cubic coefficient must reject"
    );
    println!(
        "{}",
        serde_json::json!({
            "event": "pi_ccs_first_round_inner_norm_passed",
            "first_block": first,
            "end_block": end,
            "sources": SOURCES,
            "coefficients": actual.map(words),
            "load_seconds": load_seconds,
            "compute_seconds": compute_seconds,
            "total_seconds": started.elapsed().as_secs_f64(),
            "changed_coefficient_rejected": true
        })
    );
}

fn capture(records: &str) -> String {
    format!("[1,{D},{SOURCES},{BLOCKS}]\n{records}[]\n")
}

#[test]
fn range_uses_absolute_weights_and_keeps_the_last_running_lane() {
    let last_source = SOURCES - 1;
    let text = capture(&format!("[1,[[0,1,0],[{last_source},0,{}]]]\n", 1u64 << (D - 1)));
    let witnesses = source_range(Cursor::new(text), 1, 3).unwrap();
    assert_eq!(witnesses.len(), SOURCES);
    for witness in &witnesses {
        assert_eq!((witness.rows(), witness.cols()), (D, 2));
        assert_eq!(witness[(D - 1, 1)], F::ZERO, "omitted second block is zero");
    }
    assert_eq!(witnesses[last_source][(D - 1, 0)], -F::ONE);
    let assignments = witnesses
        .iter()
        .map(|witness| Assignment::new(witness, 2 * D))
        .collect::<Vec<_>>();
    let alpha: Vec<_> = (0..ROUNDS)
        .map(|index| K::from(F::from_u64(index as u64 + 2)))
        .collect();
    let gamma = K::from(F::from_u64(3));
    let weights = EqualityWeights::new(&alpha[1..]);
    let first_pair = D / 2;
    let actual = norm_coefficients(&assignments, gamma, &weights, first_pair);
    let fresh = prefix::norm_pair(K::ONE, K::ZERO);
    let running = prefix::norm_pair(K::ZERO, -K::ONE);
    let running_power = super::gamma_power(gamma, last_source);
    let expected = std::array::from_fn(|index| {
        fresh[index] * weights.at(first_pair) + running_power * running[index] * weights.at(first_pair + D / 2 - 1)
    });
    assert_eq!(actual, expected);
    assert_ne!(actual, norm_coefficients(&assignments, gamma, &weights, 0));
}

#[test]
fn source_capture_rejects_malformed_records_even_outside_the_range() {
    let bad_records = [
        "[0,[]]\n".to_owned(),
        "[0,[[0,0,0]]]\n".to_owned(),
        "[0,[[0,1,1]]]\n".to_owned(),
        format!("[0,[[0,{},0]]]\n", 1u64 << D),
        format!("[0,[[{SOURCES},1,0]]]\n"),
        "[0,[[0,1,0],[0,2,0]]]\n".to_owned(),
        "[0,[[1,1,0],[0,2,0]]]\n".to_owned(),
        "[0,[[0,1,0]]]\n[0,[[0,2,0]]]\n".to_owned(),
        format!("[{BLOCKS},[[0,1,0]]]\n"),
        "[0,[[0,1,0,0]]]\n".to_owned(),
    ];
    for records in bad_records {
        assert!(source_range(Cursor::new(capture(&records)), 1, 2).is_err(), "{records}");
    }
    let header = format!("[1,{D},{SOURCES},{BLOCKS}]\n");
    assert!(source_range(Cursor::new(header), 1, 2).is_err());
    assert!(source_range(Cursor::new(format!("{}[]\n", capture(""))), 1, 2).is_err());
    assert!(source_range(Cursor::new(capture("")), 2, 2).is_err());
    assert!(source_range(Cursor::new(capture("")), 0, BLOCKS + 1).is_err());
    assert!(source_range(Cursor::new(format!("[1,{D},1,{BLOCKS}]\n[]\n")), 1, 2).is_err());
    assert!(field([F::ORDER_U64, 0]).is_err());
    assert!(field([0, F::ORDER_U64]).is_err());
}

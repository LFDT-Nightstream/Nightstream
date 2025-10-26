//! Transcript parity (prover vs verifier) for Poseidon2Transcript users in neo-fold.
//!
//! This uses the same child-process harness pattern as the bridge test,
//! but exercises neo-transcript directly so enabling `neo-fold/debug-logs`
//! will activate dumps on stable.

#![allow(deprecated)]

use std::process::{Command, Stdio};

use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

/// Event parsed from neo-transcript dump line
#[derive(Debug, Clone, PartialEq, Eq)]
struct EventLine { op: String, label: String, len: usize, st_prefix: [u64; 4] }

fn parse_event_line(line: &str) -> Option<EventLine> {
    if !line.starts_with('[') || !line.contains(" op=") { return None; }
    let op_start = line.find("op=")? + 3;
    let label_key = " label=\"";
    let label_pos = line[op_start..].find(label_key)? + op_start;
    let op = line[op_start..label_pos].trim().to_string();

    let label_start = label_pos + label_key.len();
    let label_end = line[label_start..].find('"')? + label_start;
    let label = line[label_start..label_end].to_string();

    let len_key = " len=";
    let len_pos = line[label_end..].find(len_key)? + label_end + len_key.len();
    let st_key = " st0..3=[";
    let st_pos = line[len_pos..].find(st_key)? + len_pos;
    let len_str = line[len_pos..st_pos].trim();
    let len: usize = len_str.parse().ok()?;

    let st_start = st_pos + st_key.len();
    let st_end = line[st_start..].find(']')? + st_start;
    let mut it = line[st_start..st_end].split(',').map(|s| s.trim());
    let mut arr = [0u64; 4];
    for i in 0..4 { arr[i] = it.next()?.parse().ok()?; }
    Some(EventLine { op, label, len, st_prefix: arr })
}

fn parse_events_grouped_by_tag(output: &str) -> (Vec<EventLine>, Vec<EventLine>) {
    let mut cur_tag: Option<String> = None;
    let mut prover: Vec<EventLine> = Vec::new();
    let mut verifier: Vec<EventLine> = Vec::new();
    for line in output.lines() {
        if let Some(hdr_pos) = line.find("--- ") {
            if let Some(open) = line[hdr_pos + 4..].find('[') {
                let open_idx = hdr_pos + 4 + open + 1;
                if let Some(close_rel) = line[open_idx..].find(']') {
                    let tag = &line[open_idx..open_idx + close_rel];
                    cur_tag = Some(tag.to_string());
                    continue;
                }
            }
        }
        if let Some(ev) = parse_event_line(line) {
            match cur_tag.as_deref() {
                Some("PROVER") => prover.push(ev),
                Some("VERIFIER") => verifier.push(ev),
                _ => { /* ignore */ }
            }
        }
    }
    (prover, verifier)
}

fn run_child_test_capture(test_name: &str) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.arg("--nocapture")
        .arg("--test-threads=1")
        .arg("--exact")
        .arg(test_name)
        .env("NEO_TRANSCRIPT_DUMP", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let out = cmd.output().expect("child run");
    let mut s = String::new();
    s.push_str(&String::from_utf8_lossy(&out.stdout));
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

#[test]
fn transcript_parity_prover_vs_verifier_fold() {
    let out = run_child_test_capture("__transcript_parity_fold_subprocess");
    let (prover_events, verifier_events) = parse_events_grouped_by_tag(&out);

    if prover_events.is_empty() && verifier_events.is_empty() {
        eprintln!("[SKIP] neo-transcript debug-log not enabled; no dumps captured");
        return;
    }

    let n = prover_events.len().min(verifier_events.len());
    for i in 0..n {
        let a = &prover_events[i];
        let b = &verifier_events[i];
        let same_core = a.op == b.op && a.label == b.label && a.len == b.len;
        let same_state = a.st_prefix == b.st_prefix;
        if !same_core || !same_state {
            panic!(
                "Transcript diverged at event #{i}:\n\
                 PROVER:   op={} label=\"{}\" len={} st0..3={:?}\n\
                 VERIFIER: op={} label=\"{}\" len={} st0..3={:?}",
                a.op, a.label, a.len, a.st_prefix,
                b.op, b.label, b.len, b.st_prefix
            );
        }
    }
    assert_eq!(prover_events.len(), verifier_events.len(),
        "Transcript length mismatch: prover {} vs verifier {}",
        prover_events.len(), verifier_events.len());
}

#[test]
fn __transcript_parity_fold_subprocess() {
    // Prover stream
    std::env::set_var("NEO_TRANSCRIPT_TAG", "PROVER");
    let mut tp = Poseidon2Transcript::new(b"neo/fold");
    tp.append_message(b"CCS_HEADER", b"");
    tp.append_fields(b"CCS_DIMS", &[F::from_u64(8), F::from_u64(2), F::from_u64(3)]);
    tp.append_fields(b"COMMIT_COORDS", &[F::from_u64(1), F::from_u64(2)]);
    let _cp = tp.challenge_field(b"RHO");
    let _dp = tp.digest32();

    // Verifier stream (mirror)
    std::env::set_var("NEO_TRANSCRIPT_TAG", "VERIFIER");
    let mut tv = Poseidon2Transcript::new(b"neo/fold");
    tv.append_message(b"CCS_HEADER", b"");
    tv.append_fields(b"CCS_DIMS", &[F::from_u64(8), F::from_u64(2), F::from_u64(3)]);
    tv.append_fields(b"COMMIT_COORDS", &[F::from_u64(1), F::from_u64(2)]);
    let _cv = tv.challenge_field(b"RHO");
    let _dv = tv.digest32();
}


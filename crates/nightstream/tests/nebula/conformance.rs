//! Spec §14 rejection cases that need an invalid witness. Each case changes
//! one record or word of an honest run and keeps every other value
//! consistent. The faulty invocation is proved last without the prover's
//! assertion check. Two results are required: the verifier rejects the proof
//! at the fresh CCS relation, and the first assertion row that the witness
//! fails is in the named check. The names come from Lean
//! (`tests/fixtures/nebula-memory-*-rows.json`).

use std::ops::Range;

use super::*;
use crate::lifecycle::{VerifyError, FAILED_ASSERTION_ROW, UNCHECKED_WITNESS};
use crate::nebula::{N1_PACKAGE_STRUCTURAL_IDENTIFIER, PACKAGE_STRUCTURAL_IDENTIFIER};
use crate::{Circuit, Engine, Error, Proof, Prover, Verifier};

/// `loadi 5; store 0`, then `load 0; halt`.
const EXECUTE: [Step; 2] = [Step::Execute, Step::Execute];

/// A saved memory package, its pinned identifier, its plan, and its row names.
struct Package {
    artifact: &'static str,
    identifier: [u64; 4],
    plan: fn() -> Plan,
    rows: &'static [u8],
}

const FIRST: Package = Package {
    artifact: "nightstream-fprime-stage2-nebula-memory-v1.json",
    identifier: PACKAGE_STRUCTURAL_IDENTIFIER,
    plan: Plan::first,
    rows: include_bytes!("../fixtures/nebula-memory-v1-rows.json"),
};

const N1: Package = Package {
    artifact: "nightstream-fprime-stage2-nebula-memory-n1.json",
    identifier: N1_PACKAGE_STRUCTURAL_IDENTIFIER,
    plan: Plan::second,
    rows: include_bytes!("../fixtures/nebula-memory-n1-rows.json"),
};

struct Harness {
    context: Context,
    prover: Prover,
    verifier: Verifier,
    checks: Vec<(String, Range<usize>)>,
}

impl Harness {
    fn new(engine: Engine, package: &Package) -> Self {
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("artifacts")
                .join(package.artifact),
        )
        .expect("saved Lean memory package");
        let circuit = Circuit::load_package(&bytes, package.identifier).unwrap();
        let rows = circuit.application_rows();
        let named: Vec<(String, usize, usize)> = serde_json::from_slice(package.rows).unwrap();
        assert_eq!(
            named.last().map(|check| check.2),
            Some(rows.len()),
            "names cover the rows"
        );
        Self {
            context: Context::new((package.plan)()).unwrap(),
            prover: circuit.prover(engine, 114).unwrap(),
            verifier: Verifier::from_package(&circuit, engine, 114).unwrap(),
            checks: named
                .into_iter()
                .map(|(name, first, end)| (name, rows.start + first..rows.start + end))
                .collect(),
        }
    }

    fn step(&self, proof: Option<&Proof>, invocation: &Invocation) -> Result<Proof, Error> {
        match proof {
            None => {
                let z0 = self
                    .context
                    .state(MachineState::default(), &self.context.start_carry());
                self.prover
                    .prove_with_output(z0, &invocation.words, invocation.output)
            }
            Some(proof) => self
                .prover
                .extend_with_output(proof, &invocation.words, invocation.output),
        }
    }

    /// Prove `invocation` after `proof`, or as the first step, without the
    /// assertion check. Require the rejection at the fresh relation, and
    /// require that the first failing row is in `check`.
    fn reject(&self, proof: Option<&Proof>, invocation: &Invocation, check: &str) {
        UNCHECKED_WITNESS.set(true);
        FAILED_ASSERTION_ROW.set(None);
        let proved = self.step(proof, invocation);
        UNCHECKED_WITNESS.set(false);
        let bad = proved.unwrap();
        let row = FAILED_ASSERTION_ROW
            .take()
            .expect("the faulty witness fails an assertion row");
        let named = self
            .checks
            .iter()
            .find(|(_, rows)| rows.contains(&row))
            .map(|(name, _)| name.as_str());
        assert_eq!(named, Some(check), "first failing row {row}");
        eprintln!("rejected at row {row} ({check})");
        assert!(
            matches!(
                self.verifier.verify(bad.state(), &bad),
                Err(Error::Verify(VerifyError::FreshRelation(_)))
            ),
            "{check}"
        );
    }

    /// The first segment's invocations after `fault` edits its execution
    /// memory, its records, and the memories that the scans and the FS
    /// proposal read. No host check runs.
    fn first_segment(&self, fault: impl Fn(&mut FirstSegment)) -> Vec<Invocation> {
        let plan = self.context.plan();
        let mut segment = FirstSegment {
            execution: initial_memory(plan),
            start: initial_memory(plan),
            executed: Vec::new(),
            proposal_records: None,
            final_: Vec::new(),
        };
        fault(&mut segment);
        let steps = vec![Step::Execute; plan.n];
        let (executed, final_memory, _) =
            execute_steps(plan, MachineState::default(), &segment.execution, 0, &steps).unwrap();
        segment.executed = executed;
        segment.final_ = final_memory;
        fault(&mut segment);
        let proposal = self.context.proposal(
            segment
                .proposal_records
                .as_deref()
                .unwrap_or(&segment.executed),
            &segment.final_,
        );
        self.context.invocations(
            self.context.start_carry(),
            proposal,
            &segment.executed,
            &segment.start,
            &segment.final_,
        )
    }
}

/// The parts of the first segment that a fault edits. `fault` runs twice:
/// before execution (`executed` is empty) and after it.
struct FirstSegment {
    execution: Vec<Cell>,
    start: Vec<Cell>,
    executed: Vec<Executed>,
    proposal_records: Option<Vec<Executed>>,
    final_: Vec<Cell>,
}

fn conformance(engine: Engine) {
    let h = Harness::new(engine, &FIRST);
    let plan = h.context.plan();
    let ram = plan.rom_size();
    let memory = initial_memory(plan);
    let honest = h
        .context
        .run_segment(h.context.start_carry(), MachineState::default(), &memory, &EXECUTE)
        .unwrap();
    let base = h.step(None, &honest.invocations[0]).unwrap();

    // An input carry or application words that do not open the input state.
    for word in [words::CARRY_IN + 2, words::APP_IN + 1] {
        let mut changed = honest.invocations[0].clone();
        changed.words[word] += F::ONE;
        h.reject(None, &changed, "state_in");
    }

    // A base arm that reads the `D_seen` headers from the witness, not from
    // the package constants.
    let mut changed = honest.invocations[0].clone();
    changed.words[words::SEEN_PREV] += F::ONE;
    h.reject(None, &changed, "chain_ops");

    // `rt ≥ wt`: O4.
    let invocations = h.first_segment(|s| {
        if let Some(step) = s.executed.first_mut() {
            step.ops[0].rt = step.write_stamps[0];
        }
    });
    h.reject(None, &invocations[0], "O4 slot 0");

    // A nonzero word in a pad slot: O7.
    let invocations = h.first_segment(|s| {
        if let Some(step) = s.executed.first_mut() {
            step.ops[1].vr = 1;
            step.ops[1].vw = 1;
        }
    });
    h.reject(None, &invocations[0], "O7 slot 1");

    // An auxiliary bit equal to −1, with the same diff word: O1.
    let mut changed = honest.invocations[1].clone();
    let layout = &h.context.layout;
    let (bit0, bit1) = (layout.op(0, layout.op_diff(0)), layout.op(0, layout.op_diff(1)));
    assert_eq!((changed.words[bit0], changed.words[bit1]), (F::ONE, F::ZERO));
    changed.words[bit0] = -F::ONE;
    changed.words[bit1] = F::ONE;
    h.reject(Some(&base), &changed, "O1 diff bits");

    // The second step's faults. Each faulty segment proves its own opening
    // step, whose proposals differ from the honest ones.
    let second = |fault: &dyn Fn(&mut FirstSegment), check: &str| {
        let invocations = h.first_segment(fault);
        let opened = h.step(None, &invocations[0]).unwrap();
        h.reject(Some(&opened), &invocations[1], check);
    };
    // A write to ROM: O5.
    second(
        &|s| {
            if let Some(step) = s.executed.get_mut(1) {
                step.ops[1].is_ram = false;
            }
        },
        "O5 slot 1",
    );
    // A record changed after its segment opened.
    second(
        &|s| {
            if s.executed.len() == 2 && s.proposal_records.is_none() {
                s.proposal_records = Some(s.executed.clone());
                s.executed[1].ops[1].vw = 7;
            }
        },
        "close D_seen.ops = D_pre.ops",
    );
    // Segment-0 IS records that differ from the plan images.
    second(
        &|s| {
            if s.executed.is_empty() {
                s.start[ram + 1].value = 9;
                s.execution[ram + 1].value = 9;
            }
        },
        "close D_seen.is = D_mem",
    );
    // The IS and FS records of the second step swapped.
    second(
        &|s| {
            if s.executed.len() == 2 {
                let cells = ram..ram + plan.b_scan;
                let start = s.start[cells.clone()].to_vec();
                s.start[cells.clone()].copy_from_slice(&s.final_[cells.clone()]);
                s.final_[cells].copy_from_slice(&start);
            }
        },
        "close D_seen.is = D_mem",
    );
    // A fetch that reads another value than the last write: the product
    // equation.
    second(
        &|s| {
            if s.executed.is_empty() {
                s.execution[1].value = 6;
            }
        },
        "close product equation",
    );

    // Fresh memory at a segment boundary.
    let opened = h.step(Some(&base), &honest.invocations[1]).unwrap();
    let fresh = initial_memory(plan);
    let (executed, final_memory, _) = execute_steps(plan, honest.state, &fresh, honest.carry.ts, &EXECUTE).unwrap();
    let invocations = h.context.invocations(
        honest.carry,
        h.context.proposal(&executed, &final_memory),
        &executed,
        &fresh,
        &final_memory,
    );
    let opened = h.step(Some(&opened), &invocations[0]).unwrap();
    h.reject(Some(&opened), &invocations[1], "close D_seen.is = D_mem");
}

#[test]
#[ignore = "Production-profile memory proofs; run separately under the 300-second cap."]
fn memory_rejections_on_cpu() {
    conformance(Engine::Optimized);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Production-profile memory proofs on Metal; run separately under the 300-second cap."]
fn memory_rejections_on_metal() {
    conformance(Engine::Metal);
}

/// Spec §9.1: the native packing that the chains read puts at most 63 bits
/// into one element.
#[test]
fn packing_puts_at_most_63_bits_in_one_element() {
    use p3_field::PrimeField64;
    for length in [1, 62, 63, 64, 126, 127] {
        let packed = pack(&vec![true; length]);
        assert_eq!(packed.len(), length.div_ceil(63));
        assert!(packed.iter().all(|word| word.as_canonical_u64() < 1 << 63));
    }
}

/// The case that the first plan cannot reach: its ROM fills its address
/// space, so it has no O6 rows. The second plan has `r < μ`.
fn n1_conformance(engine: Engine) {
    let h = Harness::new(engine, &N1);
    // A ROM address with a bit at or above `r`: O6.
    let invocations = h.first_segment(|s| {
        if let Some(step) = s.executed.first_mut() {
            step.ops[0].addr = 1 << h.context.plan().r;
        }
    });
    h.reject(None, &invocations[0], "O6 slot 0");
}

#[test]
#[ignore = "Production-profile memory proofs; run separately under the 300-second cap."]
fn n1_rejections_on_cpu() {
    n1_conformance(Engine::Optimized);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Production-profile memory proofs on Metal; run separately under the 300-second cap."]
fn n1_rejections_on_metal() {
    n1_conformance(Engine::Metal);
}

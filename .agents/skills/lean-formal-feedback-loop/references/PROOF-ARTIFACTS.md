# Proof-Carrying Artifacts

> Every proof emits an artifact; every artifact is testable. No artifact, no claim.

## The Chain

```
Lean Proof (offline, mathematically certain)
  ↓ extract theorem name, pre/postconditions, hash
Proof Certificate (Rust const in src/formal/witnesses.rs)
  ↓ embed invariant ID + hash
Conformance Test (mirrors Lean pre/postconditions as runtime checks)
  ↓ if Rust drifts
Drift Alert → re-verify
```

## ProofWitness Struct

```rust
#[derive(Debug, Clone)]
pub struct ProofWitness {
    pub theorem: &'static str,
    pub proof_hash: [u8; 32],        // SHA-256 of Lean proof term
    pub lean_line: u32,
    pub invariant: &'static str,     // e.g. "inv.region_close.quiescence"
    pub tier: &'static str,          // "D" or "E"
    pub preconditions: &'static [&'static str],
    pub postconditions: &'static [&'static str],
    pub verified_date: &'static str, // ISO-8601
    pub rust_files: &'static [&'static str],
}
```

### Example

```rust
pub const PROOF_CLOSE_IMPLIES_QUIESCENT: ProofWitness = ProofWitness {
    theorem: "close_implies_quiescent",
    proof_hash: [0xa1, 0xb2, /* ... 30 more bytes */],
    lean_line: 794,
    invariant: "inv.region_close.quiescence",
    tier: "E",
    preconditions: &["Quiescent(state, region_id)", "region_state == closing"],
    postconditions: &["region_state' == closed", "region_ledger' == []"],
    verified_date: "2026-02-14",
    rust_files: &["src/runtime/state.rs", "src/record/region.rs"],
};
```

## Hash Extraction

```bash
cd formal/lean

# Content hash of theorem source block (proxy for proof term hash):
theorem_name="close_implies_quiescent"
sed -n "/^theorem ${theorem_name}/,/^theorem\|^end\|^def\|^lemma/p" \
    Asupersync.lean | head -n -1 | sha256sum | cut -d' ' -f1
```

Verify no `sorry`:
```lean
#print axioms close_implies_quiescent
-- Must print only: propext, Quot.sound, Classical.choice
-- If sorry appears → proof incomplete, DO NOT emit artifact
```

## Staleness Check

```bash
lean_mod=$(git log -1 --format=%ct -- formal/lean/Asupersync.lean)
rust_mod=$(git log -1 --format=%ct -- src/runtime/state.rs)
[ "$rust_mod" -gt "$lean_mod" ] && echo "DRIFT: re-verify!"
```

## Conformance Test Template

```rust
/// Conformance: close_implies_quiescent
/// Witness: PROOF_CLOSE_IMPLIES_QUIESCENT (sha256:a1b2c3d4...)
#[test]
fn conformance_close_implies_quiescent() {
    let lab = LabRuntime::new(LabConfig::default().seed(42));
    lab.run(|cx| async move {
        cx.region(|scope| async {
            scope.spawn(|cx| async { cx.checkpoint()?; Outcome::ok(()) });
        }).await;
        assert!(lab.quiescence_oracle().is_ok());
        assert!(lab.obligation_leak_oracle().is_ok());
    });
}
```

Naming: `conformance_{theorem}`, `conformance_{theorem}_neg`, `conformance_{theorem}_fuzz`

## Emission Workflow

1. `#print axioms` — verify no sorry
2. Extract SHA-256 hash of theorem source
3. Run 7-check conformance (see CONFORMANCE-PROCEDURE.md)
4. Add `ProofWitness` const to `src/formal/witnesses.rs`
5. Add conformance test to `tests/refinement_conformance.rs`
6. Generate proptest variant in `tests/lean_generated/`
7. Update `invariant_theorem_test_link_map.json`
8. Append to `formal/lean/coverage/evidence_ledger.jsonl`
9. `lake build && cargo test --test refinement_conformance`

## Evidence Ledger Entry

```json
{
  "cycle_id": "lfl-2026-02-14-001",
  "theorem": "close_implies_quiescent",
  "hash": "sha256:a1b2c3d4...",
  "invariant": "inv.region_close.quiescence",
  "tier": "E",
  "conformance_result": "aligned",
  "route": "code-first",
  "rust_files": ["src/runtime/state.rs"]
}
```

## Lean Feedback Loop Artifact Contract

Use this compact closure record for each completed frog loop:

```json
{
  "frog_id": "inv.race.losers_drained",
  "theorem_ids": ["race_loser_drained_before_return"],
  "lean_commit": "<sha>",
  "rust_commit": "<sha>",
  "route": "code-first",
  "counterexample_witness": "lab_seed=42 schedule=...",
  "regression_tests": ["tests/race_loser_drain.rs::drains_all_losers"],
  "conformance_result": "aligned",
  "artifact_hash": "sha256:..."
}
```

## File Organization

```
src/formal/witnesses.rs    ← ProofWitness struct + all const witnesses
tests/lean_generated/      ← Proptest variants from theorems
tests/refinement_conformance.rs ← Conformance tests referencing witnesses
formal/lean/coverage/evidence_ledger.jsonl ← Append-only cycle records
```

## Anti-Patterns

| Don't | Why | Do |
|---|---|---|
| Emit without `#print axioms` | May contain sorry | Always verify axioms |
| Hash by file mtime | Breaks on reformat | Hash proof term source |
| Artifact without test | No drift detection | Every artifact gets a conformance test |
| Skip conformance check | Stale model reference | Full 7-check before emission |

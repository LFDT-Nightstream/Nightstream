# AGENTS.md

## General
- **Required communication standard.** Use ASD-STE100 Simplified Technical English for all user-facing answers and progress updates.
- We don't care about backwards compatibility because we are still in development. Keep the code simple and lean.
- Avoid adding new Rust features or ENVs unless it is explicitly approved.
- Never modify this file without explicit approval.
- When creating commits, always include a DCO sign-off (`git commit -s` or an equivalent `Signed-off-by:` trailer).
- No single file should ever exceed 1,500 lines of code unless explicitly confirmed by the user.
- Heavily avoid bloat. We want to maintain a compact and lean codebase.
- Proofs must remain compatible with on-chain verification targets. In proof/transcript/public-digest paths, use Poseidon2-only hashing unless explicitly approved otherwise.
- Do not introduce mixed hash families (e.g., Blake3/SHA prehashes) in protocol-binding paths without explicit user approval.
- You can find the SuperNeo paper which is what the main protocol is based upon in ./docs/superneo-paper
- **Fixed SuperNeo Goldilocks decomposition policy (hard rule).** Keep `b = 2`. Use `k_rho = 16` and `B = 2^16` by default for a Nightstream production profile. Use `k_rho = 18` and `B = 2^18` only when an explicit requirement or a measured result shows that `k_rho = 16` is insufficient. In both cases, `B = b^k_rho`.
- The SuperNeo Appendix B.2 values `k_rho = 14` and `B = 2^14` are reference values only. Do not use them for a Nightstream production proof artifact.
- Each frozen or generated profile must select and protocol-bind one exact allowed pair. Do not mix `k_rho = 16` and `k_rho = 18` within one profile or its artifacts.
- Do not use `b = 4`, `k_rho = 7`, radix-four decomposition, or a `k_rho` value other than 16 or 18 unless the user explicitly approves those exact values in the current task.
- A domain-size target, including `2^24`, a performance result, a cvc5 result, generated-artifact size, or implementation convenience does not authorize a change outside this policy.
- Do not describe a `k_rho = 16` or `k_rho = 18` profile as SuperNeo Appendix B.2 or paper exact. Describe it as a Nightstream Goldilocks profile and state the selected `k_rho` value.
- **5-minute non-Lean test cap (hard).** Every `cargo test` and every other non-Lean test-binary invocation MUST be launched with a timeout of **at most 300 000 ms (5 minutes)**. Pass `timeout: 300000` to the Bash tool — do not omit it, do not raise it. If a test is still running at the cap, kill it and treat the test as failing this slice; either reduce its work (smaller `n`, shared cache) or mark it `#[ignore]` with a clear comment. The 5-minute cap is unconditional; the only way to exceed it is the user explicitly approving a longer run for a specific invocation in the same turn — there is no standing exception.
- **25-minute Lean cap (hard).** Every Lean-related command, including `lake build`, `lake test`, `lake exe`, `lake env lean`, and direct Lean test or executable invocations, MUST be launched with a timeout of **at most 1 500 000 ms (25 minutes)**. Pass `timeout: 1500000` to the Bash tool — do not omit it, do not raise it. If a Lean command is still running at the cap, kill it and treat it as failing this slice. The only way to exceed the cap is the user explicitly approving a longer run for a specific invocation in the same turn — there is no standing exception.

## Operating Discipline
- Before implementing, state the assumptions that matter for the task. If multiple interpretations are plausible and the wrong one would be costly, ask instead of guessing.
- Prefer the smallest code change that solves the stated problem. Do not add speculative features, flexibility, abstractions, flags, or helper systems for a single use case.
- Keep edits surgical. Touch only files and lines that directly support the request, and do not refactor adjacent code, comments, or formatting unless that cleanup is required by your change.
- If your change creates unused imports, variables, functions, or orphaned code, remove that newly-created dead surface. Do not delete pre-existing unrelated dead code unless explicitly asked.
- Define success criteria for non-trivial work before coding. For bug fixes, add or update a test that would fail on the bug; for refactors, identify the compile/test checks that prove behavior was preserved.
- Surface uncertainty and tradeoffs directly. If a simpler approach exists or the requested direction risks extra complexity, say so before implementing.

## MSW — the kernel

Remember to follow the MSW deletion rule for all claims—no exceptions.

### program — complete

```r
contract ← the requested outcome + the smallest criteria that prove it

while ∃ claim c : deleting c leaves contract unmet ∨ unproven
      do c ; prove c

halt ; report
```

### definitions — no behavior lives here, only meaning

**contract** — the requested outcome and the smallest set of acceptance criteria that would prove it, stated before any work. The sole source of necessity; a ceiling as much as a floor. If the request is ambiguous: attended → ask; unattended → bind the smallest reading consistent with stated intent and record the assumption.

**claim** — anything petitioning to become work: a plan step, a change, a test, a reviewer's P1, a discovered edge case, your own instinct that one more pass would help. Everything enters as this type. Nothing enters as a verdict.

**deleting c leaves contract unmet ∨ unproven** — the only test. A claim passes solely by breaking the contract — reproducibly, within the task's actual inputs and environment. Severity is derived from the contract, never inherited from whoever raised the claim. *Useful*, *thorough*, and *possible* are not aliases for *necessary*. A claim that fails receives one line in the report — never a fix, an investigation, or a deferred follow-up.

**do ; prove** — the smallest reliable act that closes the gap, and evidence sized to the claim it settles. An unproven act keeps its claim alive; a proven one closes it — and re-proving a closed claim is itself an inadmissible claim.

**halt** — the fixed point: contract proven, no remaining claim passes. Not reviewer silence; not exhausted imagination. Halting before the fixed point and looping past it are the same bug, mirrored.

**report** — the outcome against the contract; the proof; rejected claims worth the user's attention, one line each. Nothing else.

### fuses — outside the program, for when its evaluator fails

```python
rounds = 3            → halt anyway ; report open items, do not chase them
claim born in round n+1, visible in round n   → rejected
```

### No unauthoritative limits

Never invent a limit. A cap, threshold, quota, budget, timeout, retry or round count, file or line count, acceptance-criterion count, agent count, or similar constraint is admissible only when its exact value is:

- explicitly required by the requester;
- imposed by an applicable technical or platform contract;
- defined by authoritative project policy; or
- derived from measured evidence necessary to meet or prove the task contract.

State the authority or derivation whenever proposing or applying a limit. If no authority exists, omit the limit and use the MSW necessity test. Metrics may be reported as evidence, but they must not become gates, defaults, targets, or recommendations through agent intuition. Examples and representative proportions never become defaults. If a necessary limit is an unresolved owner choice, ask; do not manufacture a value.

## Security
- Digests are fine as compression, but never as authority.
- Across trust boundaries, every carried digest must be either recomputed from authoritative inputs, replayed into a verifier-driven transcript or proof, or explicitly treated as non-authoritative structure.
- Do not rely on self-consistent digest chains as evidence of soundness. If an attacker can mutate data and re-digest upward, the verifier must still fail.

## Design & Architecture
- When evaluating design or architectural decisions, think from first principles: reduce the problem to its irreducible truths—axioms, physical laws, hard constraints—and derive every conclusion strictly from those, rejecting inherited conventions and unstated assumptions.
- Before proposing any architectural change: (1) list every assumption you are making, (2) challenge each by asking "is this a necessity or just a convention?", (3) discard any that fails. Only then derive your answer from what remains.
- Code philosophy north star:
  - John Ousterhout: prefer deep modules with small, stable interfaces and unambiguous ownership.
  - Rich Hickey: prefer simplicity over flexibility theater; do not introduce abstractions, layers, or helper systems until a real repeated need exists.
  - Casey Muratori: prefer explicit data flow, explicit control flow, and mechanically obvious code over cleverness that hides what the machine or proof system is doing.
- Use those principles as a practical test:
  - If ownership is blurry, the design is not done.
  - If a new abstraction mostly moves complexity around instead of removing it, reject it.
  - If understanding a hot path requires chasing wrappers or indirection, simplify it.
  - If a module grows by absorbing unrelated responsibilities, split it by responsibility instead of adding more flags or configuration.
- Rust design quality:
  - Before writing or changing Rust code, identify the shape of the design before choosing syntax.
  - Ask what core concept the code expresses: data, behavior, construction, validation, state transition, or orchestration flow.
  - Design from the simplest correct call site first. If the call site is ugly, the design is probably wrong.
  - Prefer names that make sense to a reader who has not seen the internals.
  - Use the type system to make invalid states hard to express, but do not add ceremony that only moves complexity around.
  - Prefer small public surfaces, domain types, private fields by default, narrow accessors, associated constructors for pure value construction, and free functions for obvious actions and flows.
  - Use traits only for real shared capabilities or conformance contracts, not to make one implementation look abstract.
  - Use step-down functions only when each step names a real phase or hides real complexity.
  - Prefer explicit data flow over clever callbacks, hidden mutation, or compiler-shaped APIs.
  - Reject exposed closure plumbing, noisy bounds, deep paths, tuple soup, state-machine internals, pass-through wrappers, single-use helpers that do not name a real concept, premature generic flexibility, giant mixed-domain structs, borrow-checker clones, and public fields that are not a deliberate ABI/proof boundary.
- Public protocol APIs must use lifecycle names and hide implementation state-machine constructors. A client should call entrypoints such as `prove`, `extend`, `finish_with_spartan`, and `verify`, not deep paths like `direct_ccs::DirectCcsRecursiveIvcState::new_with_canonical_zero_carry`. If such constructors are needed internally, isolate them behind a short, well-named private helper and never put them in crate-root examples, public docs, or expected client flow.
- Rust file/module documentation should optimize for ownership clarity and auditability, not ceremony.
- Do not add top-level file docs to trivial files whose purpose is obvious from the code.
- For normal files, prefer a short `//!` ownership header that states what the file owns and what it does not own.
- For protocol-critical or ABI-critical files, prefer a short contract header that states ownership, inputs/outputs, and invariants.
- Do not use top-level docs for implementation history, migration progress, aspirations, or Jolt/SuperNeo name-dropping without explaining the local ownership boundary.
- Do not write large tutorial-style or paper-recap headers in implementation files; keep top-level docs compact and architectural.

## Testing
- Never add tests in the same implementation file, always prefer to add them to a file inside tests/ (current or new)
- If you add a test to catch a problem, the test should fail if aims to confirm a problem.
- Always use `FoldingMode::Optimized` in tests. Never use `FoldingMode::PaperExact` unless the user explicitly approves it. PaperExact is an O(2^ell) brute-force reference engine meant only for correctness cross-checking, not general test usage.

## Build & Test Commands
- After modifying Rust code, always run `cargo fmt --all` before finishing unless the user explicitly says not to.
- When running tests use --release eg cargo test --workspace --release
- For extra debugs use debug-logs eg --features paper-exact,debug-logs

## Formal Lean Subprojects
- Lean-specific instructions live in subdirectory `AGENTS.md` files so they apply only to the matching formal project.
- For the active Nightstream Lean project, read `formal/nightstream-lean/AGENTS.md`.
- For the SuperNeo Lean project, read `formal/superneo-lean/AGENTS.md`.


## Perf & Constraint Debugging

Use these commands based on what you are measuring. All perf snapshots are `--ignored` by default.

| Question | Command |
|---|---|
| How expensive is lifecycle fold/IVC append work for an F′ chain? | `cargo test -p neo-fold-clean --release --test perf_fibonacci_bits -- --ignored --nocapture fibonacci_bits_perf_snapshot` |
| What R1CS shape does the full-history audit circuit hand to the decider? | `cargo test -p neo-fold-clean --release --test perf_fibonacci_bits -- --ignored --nocapture fibonacci_decider_r1cs_shape_snapshot` (chain length via `NEO_FOLD_FIB_DECIDER_VALUES`) |
| How do low-norm ring-action encodings compare in committed width/rows? | `cargo test -p neo-fold-clean --release --test perf_ring_action_low_norm_prototype -- --nocapture` |

## Profiling

| Tool | Use Case | Output |
|------|----------|--------|
| `profile_for_ai.sh` | Quick CPU profiling, filters system calls | `profile-output.txt` |
| `profile_xctrace.sh` | Full detail + Instruments GUI (supports `--template`) | `profile-xctrace.txt` + `.trace` |
| `profile_memory_deep.sh` | Memory allocation debugging | Text with allocation sites |

Usage: `./scripts/<tool> <package> <test_file> <test_function> [--ignored]`

For xctrace, add `--template <name>` (Allocations, Leaks, File Activity, System Trace, etc.)

Examples:
```bash
./scripts/profile_for_ai.sh neo-fold test_sha256_single_step test_sha256_preimage_4k --ignored
./scripts/profile_xctrace.sh neo-fold test_sha256_single_step test_sha256_preimage_4k --ignored
./scripts/profile_xctrace.sh neo-fold test_sha256_single_step test_sha256_preimage_4k --ignored --template Allocations
./scripts/profile_memory_deep.sh neo-fold test_sha256_single_step test_sha256_preimage_4k --ignored
```

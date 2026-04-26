# AGENTS.md

## General
- We don't care about backwards compatibility because we are still in development. Keep the code simple and lean.
- Avoid adding new Rust features or ENVs unless it is explicitly approved.
- Never modify this file without explicit approval.
- When creating commits, always include a DCO sign-off (`git commit -s` or an equivalent `Signed-off-by:` trailer).
- No single file should ever exceed 1,500 lines of code unless explicitly confirmed by the user.
- Heavily avoid bloat. We want to maintain a compact and lean codebase.
- Proofs must remain compatible with on-chain verification targets. In proof/transcript/public-digest paths, use Poseidon2-only hashing unless explicitly approved otherwise.
- Do not introduce mixed hash families (e.g., Blake3/SHA prehashes) in protocol-binding paths without explicit user approval.
- For difficult questions, hard design/review tradeoffs, or high-confidence soundness checks, you may use the project-local multi-AI council skill at `./.codex/skills/multi-ai-council/SKILL.md` (it may take between 5 - 25 min to answer).
- You can find the SuperNeo paper which is what the main protocol is based upon in ./docs/superneo-paper
- If any tests take longer than 60 seconds cancel it, unless the user explicitly approves a longer run.

## Operating Discipline
- Before implementing, state the assumptions that matter for the task. If multiple interpretations are plausible and the wrong one would be costly, ask instead of guessing.
- Prefer the smallest code change that solves the stated problem. Do not add speculative features, flexibility, abstractions, flags, or helper systems for a single use case.
- Keep edits surgical. Touch only files and lines that directly support the request, and do not refactor adjacent code, comments, or formatting unless that cleanup is required by your change.
- If your change creates unused imports, variables, functions, or orphaned code, remove that newly-created dead surface. Do not delete pre-existing unrelated dead code unless explicitly asked.
- Define success criteria for non-trivial work before coding. For bug fixes, add or update a test that would fail on the bug; for refactors, identify the compile/test checks that prove behavior was preserved.
- Surface uncertainty and tradeoffs directly. If a simpler approach exists or the requested direction risks extra complexity, say so before implementing.

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

## Formal Lean Subproject (`formal/superneo-lean`)
- Use this 3-layer layout for each formalized component:
  - Human spec: `formal/superneo-lean/specs/<Name>.spec.md`
  - Typed Lean interface: `formal/superneo-lean/SuperNeo/<Name>Interface.lean`
  - Lean implementation: `formal/superneo-lean/SuperNeo/<Name>.lean`
- Lean build discipline:
  - During iteration, build only the target module(s) you changed and their dependencies, not the whole package.
  - Prefer narrow commands such as `lake build SuperNeo.<Name>` while working.
  - If several Lean modules changed, build the narrowest affected theorem-facing targets that cover those edits.
  - Only once the Lean work is complete, run a full `lake build` to catch package-wide breakage before finishing.
- Closure standard (mandatory): **Paper-faithful proof-complete**.
  - A module is only considered complete when the exact mathematical construction/claim from
    `./formal/superneo-lean/SuperNeo.pdf.md` is proved in Lean at quantified theorem level.
  - Regression checks (`lake exe check`, generated vectors, booleans) are required but are never
    sufficient evidence for completion.
  - Interface-level or assumption-level closure (`Done (Boundary)`) is intermediate only.
  - Do not claim proof completion by redefining theorem-facing surfaces to be definitionally equal
    to the target expression while leaving the executable/paper construction unproved; prove the
    bridge theorem explicitly.
  - Any trusted assumption/axiom that remains must be explicit, minimal, and accompanied by a
    concrete closure plan in the module spec and README.
- Project-local skill for this workflow:
  - Path: `./.codex/skills/superneo-lean-interface-spec/SKILL.md`
  - Purpose: create/update per-module Lean contract pairs
    (`SuperNeo/<Name>Interface.lean` + `specs/<Name>.spec.md`).
  - Use when: standardizing specs, adding missing interface/spec files, or
    auditing assumptions/consumers against `./formal/superneo-lean/SuperNeo.pdf.md`.
- Keep interface files colocated with implementations (Objective-C style), not in a separate top-level folder.
- `*.spec.md` is the external/human-facing specification; `*Interface.lean` is the machine-checked boundary.
- Specs must be **stateless**: they describe the timeless mathematical target (what the module must achieve), never the current implementation progress. Do not use language like "currently proved", "not yet implemented", "scaffold", "pending", or "in progress" in specs. A spec should read identically whether the module is 0% or 100% complete.
- Avoid naming Lean boundary files as `*Spec.lean` or `*Contract.lean` to prevent confusion with prose specs and crypto terminology.
- Interfaces should expose theorem/definition shapes and boundary assumptions clearly; implementations should satisfy or instantiate those interfaces.
- Prefer thin/stable interfaces and keep implementation details out of `*Interface.lean`.

## Perf & Constraint Debugging

Perf tests live in `crates/neo-fold/tests/suites/perf/single_addi_metrics_nightstream.rs`. All use `--ignored`.

Full constraint architecture report (main CCS, bus, Route-A claims, openings, timing):
```bash
NS_DEBUG_N=10000 cargo test -p neo-fold-next --release --test perf_rv64im_with_spartan -- --ignored --nocapture rv64im_mixed_opcode_perf_snapshot
```
N: number of riscv instructions + 1 (halt).

Main-recursion shape probe (constraint count / aux count / synth-time before compression):
```bash
NS_DEBUG_N=10 cargo run -p neo-fold-next --bin rv64im_main_recursion_shape_probe
```
Use this when the question is "how many constraints are we handing to Spartan?" or when optimizing the recursive circuit shape without paying for `compress()`.

RV64IM no-Spartan IVC perf/debug snapshot (append/build only; stops before `compress()`):
```bash
NS_DEBUG_N=5 cargo test -p neo-fold-next --release --test perf_rv64im_no_spartan -- --ignored --nocapture rv64im_mixed_opcode_no_spartan_ivc_perf_snapshot
```
Use this test for RV64IM no-Spartan IVC append/build performance work and stage-by-stage IVC timing breakdowns.

RV64IM with-Spartan product-surface IVC compression snapshot (live state build, then `compress()` + compressed verify):
```bash
NS_DEBUG_N=2 cargo test -p neo-fold-next --release --test perf_rv64im_with_spartan rv64im_ivc_product_surface_with_spartan_compress_and_verify_snapshot -- --ignored --exact --nocapture
```
This test now builds the live IVC state from `NS_DEBUG_N`; it is no longer fixture-backed.

Optional chunk sizing for the live product-surface snapshot uses an extra libtest delimiter and defaults to `1` when omitted:
```bash
NS_DEBUG_N=5 cargo test -p neo-fold-next --release --test perf_rv64im_with_spartan rv64im_ivc_product_surface_with_spartan_compress_and_verify_snapshot -- --ignored --exact --nocapture -- --chunk-size 2
```
Alias:
```bash
NS_DEBUG_N=5 cargo test -p neo-fold-next --release --test perf_rv64im_with_spartan rv64im_ivc_product_surface_with_spartan_compress_and_verify_snapshot -- --ignored --exact --nocapture -- --rows-per-chunk 2
```

Which perf command to use:
- Use `rv64im_main_recursion_shape_probe` when optimizing recursive constraint count, aux count, or pre-compression synth cost. This is the default tool for "reduce what Spartan has to compress".
- Use `rv64im_mixed_opcode_no_spartan_ivc_perf_snapshot` when optimizing append/build time before Spartan. This is the right tool for `F'`, Construction-2, Ajtai commit, and other no-compression hotspots.
- Use `rv64im_ivc_product_surface_with_spartan_compress_and_verify_snapshot` when measuring the final compressed RV64IM IVC artifact: live state build, `compress()`, compressed verify, and serialized proof size.
- Use `rv64im_mixed_opcode_perf_snapshot` under `--test perf_rv64im_with_spartan` when you need the larger theorem-facing public-path architecture report across main CCS, bus, Route-A claims, openings, and timing, not just the recursive lane.

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

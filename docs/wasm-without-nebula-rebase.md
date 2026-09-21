# WASM rebase experiment

`codex/test-wasm-without-nebula` rebases the committed frontend onto main.
The working branch and its uncommitted changes were not modified.

- Source: `00df48f596dcbcb094ba768bc1f5526f0b9747d5`
- Target: `4696e82167218bdbca33bb1a3f93634907c62cab` (`origin/main`)
- Original history above target: 159 commits; retained historical commits: 81,
  followed by one cleanup/documentation commit.
- Fixed recovery reference: `codex/wasm-nebula-rebase-source`.
- Original authors and author dates retained; GPG signing and update-refs disabled;
  DCO sign-offs retained.

## History selection

Replay the 91 commits after `e4950c79b`, excluding the 11 commits below.
Also retain the frontend portion of `683f96ab1`: lookup synthesis, terminal-state
constraints, and the test-only relation audit harness. Its Nebula wrapper and
backend tests are excluded. The other 67 inherited backend/formal commits are
omitted. The proving crates remain identical to the target main versions.

| Deferred commit | Work |
| --- | --- |
| `60637be46` | Nebula host-event proof integration |
| `acfd753a5` | Nebula test import cleanup |
| `8194ad6d0` | Backend logical-port multiplexing |
| `0b431f075` | WASM opcode-gated slot routing |
| `b9898069c` | Memory geometry based on physical slots |
| `6307cd48d` | Memory relation cost census |
| `bba2f419f` | Slot-compaction amplification profiling |
| `052338800` | Slot coloring from exclusive activations |
| `8395ac83f` | Opcode-support packing |
| `41df0d60a` | Activation-support packing |
| `051cfe131` | Removal of legacy amplification harness |

Mixed commits retain frontend edits while omitting changes to the removed
Nebula and routing files. The padding-output rejection test bundled with
`41df0d60a` is retained during replay of `f1bb65285`, together with its
output-capture support constraint. The final cleanup replaces that single
constraint with the complete activation-support block, avoiding a duplicate.
This repairs the known failing history window; other intermediate commits
have not been exhaustively tested. The core circuit retains all source constraints. Its activation-support
definitions move to `memory_activation.rs` without the Nebula-dependent packing.
Removing these rows initially made the existing padding-output rejection test
fail, so preserving them is required for this rebase.

## Proof boundary

Tracing, host-event binding, instruction constraints, witnesses, batching,
shared application utilities, native memory/lookup checks, compact lookup
synthesis, and the test-only full-history relation audit harness remain.
Three compact lookup regression tests move from the omitted Nebula red-team
file to `wasm_lookup_semantics.rs`.

Production WASM `preprocess/prove/verify`, the Nebula memory argument, slot
packing and backend-specific tests/profiling are deferred. Activation supports
and their circuit constraints remain in the frontend.
Bare relation audit proofs do not establish program-ROM or RAM consistency;
operation-table checks remain separate from that proof path.

Cleanup removes orphaned Nebula preprocessing and lookup-to-backend assembly
helpers. The lookup constraints and witness-audit implementation remain.
Matrix access uses main's CSC API. No replacement memory prover is added.

## Recovering port packing

Restore from the fixed source commit above, rather than the moving working
branch, and adapt the Nebula-specific types:

- `crates/neo-wasm/src/memory_routing.rs`: support definitions, deterministic
  first-fit packing, batching offsets.
- The support definitions and `memory activation support` constraint block
  already survive here; reuse `memory_activation.rs` when restoring packing.
- `crates/neo-wasm/tests/nebula/memory_routing.rs`: completeness, disjointness,
  declaration order, 77 logical ports / 21 slots, and offset checks.
- `crates/neo-fold-clean/src/frontends/nebula/application.rs`: slot types,
  execution checks, and `enforce_memory_ports` selected-operation constraints.
- `crates/neo-fold-clean/tests/nebula/application{,_r1cs}.rs`: multiplexing and
  collision/binding rejection tests.

## Validation

- `cargo fmt --all` completed; stable rustfmt reports the existing unsupported
  `imports_granularity` option.
- `cargo check -p neo-wasm -p neo-application --release --all-targets
  --features neo-wasm/audit-html` passed.
- `cargo test -p neo-application --release --features audit-html`: 26 passed.
- Release WASM tests: `wasm_row_ccs`, `wasm_memory_semantics`,
  `wasm_host_event_bindings`, `wasm_lookup_semantics`, `wasm_batch`:
  110 passed, one pre-existing ignored folding test not run.
- Every test invocation used a process-group timeout of 300 seconds. The final
  WASM Cargo invocation, including recompilation, completed in 91.73 seconds;
  the application invocation completed in 175.68 seconds including a build-lock wait.
- Activation definitions and support helpers were compared byte-for-byte with
  the original source; they match. Core constraint changes only relocate that
  dependency and correct its misleading redundancy comment.
- Proving crates and `AGENTS.md` are unchanged from target main.

This is a frontend/rebase validation, not validation of a complete memory proof
or of compatibility with Nico's new `nightstream` crate.

## Review follow-up

- The pre-review test branch is preserved as
  `codex/test-wasm-without-nebula-before-review`.
- Activation constraints remain frontend-owned even if a memory backend returns.
  Any removal requires establishing redundancy for the particular constraint;
  the previous blanket redundancy claim is contradicted by the padding test.
- Lookup semantic tests no longer pin the incidental auxiliary count. The
  mutation audit must cover every auxiliary returned by honest synthesis.
- Host-event layout documentation now states the address-bound requirement on
  the eventual memory argument, rather than attributing it to the bare circuit.
- Review validation: `cargo fmt --all` and `git diff --check` passed.
  Release `wasm_row_ccs`, `wasm_lookup_semantics`, and
  `wasm_host_event_bindings` tests passed with `audit-html` enabled (77 tests),
  under the 300-second invocation cap.
- The historical padding-test invocation at the amended immediates commit
  could not compile because orphaned lookup/preprocessing helpers trigger
  existing dead-code errors; those helpers are removed in the final cleanup.
  Source inspection confirms every commit in the 62-commit replay window
  contains the output-capture bound. This is not a claim of full bisectability.

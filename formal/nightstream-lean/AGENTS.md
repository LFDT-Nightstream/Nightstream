# Nightstream Lean instructions

These instructions apply to the active `formal/nightstream-lean` package.

## Claims and trust

- State the assurance tier with every claim: model-level, artifact-checked,
  Rust-conformant, or security-reduced. Do not shorten any of those to
  "verified" without naming the property ID, supported profile, and evidence.
- Generated rows, manifests, digests, and prover-carried conclusions are not
  authority. Recompute them, connect them to verifier-owned inputs, or label
  them as non-authoritative structure.
- Active Lean sources and tests must not contain `sorry`, `admit`, new `axiom`
  declarations, `postulate`, or `unsafe`. Keep exported theorem axiom reports
  fail-closed in `tests/Axioms.lean`.

## Layer direction

Imports flow in one direction:

```text
SuperNeo   HyperNova
    \       /
      Protocol
          |
   Implementation
          |
     Assurance
```

- `SuperNeo` and `HyperNova` are sibling foundations and do not import each
  other.
- `Protocol` may consume either foundation.
- `Implementation` may consume foundations and protocol semantics.
- `Assurance` composes all lower layers.
- `Checks` is an executable harness. Core layers must never import it.
- Run `scripts/check-layer-imports.sh` after changing imports.

See `docs/architecture.md` for ownership boundaries.

## Generated artifacts

- Never hand-edit a generated Lean module. Change the Rust generator, run its
  drift test, inspect the emitted `*.expected` file, and deliberately replace
  the committed artifact.
- Generated modules live below `Nightstream/Implementation/R1CS/Artifacts/**/Generated`.
  Handwritten public consumers must import an artifact facade, not an
  individual shard.
- Run `scripts/check-generated-layout.sh` after moving or regenerating artifacts.
- Do not commit `*.expected`, `.lake/`, build output, or other ephemeral files.
- Generator ownership and commands are recorded in `docs/generated-files.md`.

## Acceptance-criterion gate

- Before each command or edit, state one active acceptance criterion and the
  exact evidence that will close it.
- Give the action a thumbs-up only when deleting that action would leave the
  active criterion unmet or unproven. Ask: **Will this close the current
  claim?** If the answer is not clearly yes, do not take the action.
- Keep only one claim active. Do not inspect downstream drift, unrelated
  consumers, or later contract phases until the active claim is closed or they
  are direct evidence for a blocker to that claim.
- Use the smallest focused check that proves the claim. Do not repeat a green
  check unless a relevant input, implementation, proof, or acceptance criterion
  changed after that result.
- After a claim closes, record its evidence and return to the next unmet
  criterion in the governing contract.
- If the claim is still open after three action rounds, stop work on that claim
  and report the exact open item, the evidence obtained, and the evidence still
  required. Do not replace it with a new claim to avoid this report.

## Editing and validation

- Keep every source file below 1,500 lines. Split by proof responsibility, not
  by arbitrary line count.
- Put executable regressions and axiom guards in `tests/`, not in implementation
  modules.
- Every Lean-related command has a hard 1,500-second limit. Use the bounded
  wrapper; do not invoke an uncapped `lake` or `lean` command:

```bash
./scripts/validate.sh static
./scripts/validate.sh build
./scripts/validate.sh axioms
./scripts/validate.sh check
./scripts/validate.sh all
```

`all` runs each Lean phase under its own cap. A timeout is a failed gate, not a
passing result.

## Performance and resource discipline

- Run at most one Lean build/check or Rust build/test process at a time across
  the parent agent and all subagents. Read-only inspection may run alongside it
  only when it cannot trigger compilation.
- After changing process or memory supervision in `scripts/validate.sh`, its
  bounded monitor self-test must pass before any Lean build. Do not bypass a
  monitor that cannot account for every descendant process. Thread limits are
  not substitutes for aggregate descendant memory accounting.
- Treat RAM ceilings as hard safety limits, not performance targets. Do not
  raise a limit merely to make an unexpectedly large certificate finish.
- Before applying `native_decide` to a generated or concrete collection,
  establish its exact input length. Any partitioned certificate must check its
  maximum shard length, exact coverage, absence of overlap, and final remainder
  size so an unbounded tail cannot masquerade as a shard.
- Use `native_decide` only for small closed facts whose cost does not grow with
  generated artifact size. Do not use it for complete artifact validity, row
  sets, seed schedules, sampler coefficients, witness data, or trust-boundary
  equality. Prove these claims with structural theorems and reusable leaf
  certificates. If a `native_decide` proof reaches the Lean timeout, do not run
  it again unchanged.
- Separate artifact geometry from exact generated-data identity. Prove geometry
  with symbolic arithmetic and structural theorems; do not evaluate the full
  generated collection to prove its shape.
- Prove each exact Rust-emitted schedule or row identity once in the smallest
  Lean leaf module that owns it. Import and reuse that theorem. If Rust emits
  equal schedules for multiple arms, prove the equality once at the
  Rust-to-Lean boundary and transport the leaf certificate; do not recompute
  the complete schedules for each arm.
- Build complete artifact validity by composing leaf theorems. Never unfold or
  evaluate the complete generated artifact again only to reconstruct a validity
  theorem that those leaves already imply.
- Keep exact Rust-emitted data unchanged when replacing a slow proof. A digest,
  cvc5 result, test result, or matching count is not a Lean proof of schedule or
  row identity.
- Avoid closed computation over large proof-carrying structures. Project to the
  smallest compact decidable data that expresses the artifact fact, then use a
  generic kernel theorem to derive the semantic result.
- Never rerun a resource-failing target unchanged. After two memory-,
  heartbeat-, or time-capped attempts at the same obligation, stop compiling
  and isolate the responsible expression in a smaller focused module.
- Validate bottom-up: build the changed leaf first, then its immediate parent.
  Do not run a full build until the focused path is green.
- For long focused runs, report the exact target, elapsed time, peak aggregate
  descendant RSS, and outcome. Distinguish compilation time from editing,
  reasoning, blocking, and waiting.

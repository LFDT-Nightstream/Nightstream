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

## Editing and validation

- Keep every source file below 1,500 lines. Split by proof responsibility, not
  by arbitrary line count.
- Put executable regressions and axiom guards in `tests/`, not in implementation
  modules.
- Every Lean-related command has a hard 900-second limit. Use the bounded
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

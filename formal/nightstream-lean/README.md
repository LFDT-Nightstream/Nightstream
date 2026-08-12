# Nightstream Lean

This is the active assurance-first formalization for Nightstream. The sibling
Lean packages under `formal/` are legacy reference material, not dependencies,
and no theorem is inherited from them.

## Project map

- [`../../protocol-contract/README.md`](../../protocol-contract/README.md) owns
  the normative protocol, threat model, supported profile, requirement rules,
  and completion gates. Lean source is evidence against that contract; it is
  not a second protocol specification.
- [`docs/architecture.md`](docs/architecture.md) defines layer ownership and
  import direction.
- [`docs/generated-files.md`](docs/generated-files.md) records generated-artifact
  ownership and regeneration commands.
- [`assurance/evidence-ledger.jsonl`](assurance/evidence-ledger.jsonl) records
  mutable evidence. Manifests beside it pin exact artifact identities.
- [`AGENTS.md`](AGENTS.md) contains project-local editing and validation rules.

## Milestone status

| Milestone | Current claim |
|---|---|
| M0 / M0.5 | Project specification, threat model, evidence discipline, and concrete relation foundations are complete for their named model profiles. |
| M1 | Concrete CCS/CE relations, global parameters, encoding, SumCheck, and executable checker equivalences are model-proved. |
| M2 | PiCCS/PiRLC/PiDEC folding shapes and their composed reduction expose explicit sampling, binding, and collision events. Probability bounds remain M6 work. |
| M3 | F' base/recursive/terminal semantics and exact trace induction establish `ValidExecution` for the advertised semantic interface. |
| M4 | `CIR-SOUND` and `CIR-COMPLETE` are artifact-checked for the exact plain, stateless, `[1,1]`, one-recursive-step, terminal-fold, direct-terminal-CE, minimal-bit-carrier profile. |
| M5 | The supported Rust-shaped F' and terminal verifier surfaces are proved equivalent to their Lean predicates and pinned by conformance manifests. |
| M6 | Open: compact-decider soundness, final verifier reduction, and probability bounds for recursive/terminal bad-root events. |

These are scoped claims, not an assertion that every production profile is
verified. M4 does not yet cover stateful mode, Nebula, other schedules, multiple
recursive invocations, alternate carriers, or a parameterized circuit family.
The public compact decider remains fail-closed with `Unsupported`.

## Independent Nebula V2 model

`Nightstream/Protocol/NebulaV2/` is a model-level review of
`PaddedRowIdentityMemoryV2`. It is separate from M4 and does not extend the M4
artifact claim. It covers exact memory execution, complete snapshots, segment
composition, fixed physical port positions, fingerprint algebra, product
commitments, delayed-claim indexes, deterministic F-prime close, completed
execution, and the top-level named-bad-event theorem shape. The independent
ideal verifier also ties the actual records to the prechallenge lane
sequences, ties both fingerprint pairs to the verifier-derived F-prime
challenge, and uses one verifier-owned prior-claim predicate. Its direct
soundness theorem derives a completed sequential execution or a named ideal
failure without an `ExecutionWitness` or `AcceptanceReduction` premise. A
separate constructive theorem gives conditional ideal completeness from valid
semantic segments and explicit honest primitive artifacts.

`Nightstream/Implementation/NebulaV2/` adds an exponent-indexed SuperNeo and
HyperNova F-prime schema. It derives the exact prior-claim verification and
consumption order, exact produced successors, proof forwarding, closed terminal
consumption, and one extra augmented invocation. Its constructive lifetime has
no open-tail case. The top implementation-model theorem derives completed
application execution and sequential memory execution over the same receipts,
or a named failure. It does not assume that conclusion.

The model does not prove final generated-circuit or Rust conformance, complete WASM port
coverage, Poseidon2 or Fiat-Shamir security, the late-preimage reduction,
Module-SIS binding, the compact terminal backend, or a 96-bit end-to-end
bound. It also does not prove that generated application rows imply the
independent application-semantics predicate or that deployed acceptance gives
the ideal acceptance object. The tests include countermodels for the
assumptions that must remain in the release theorem.

Consult the protocol contract and evidence ledger for theorem names, hashes,
assumptions, and current gates; do not treat this summary as normative status.

## Validation

Every Lean process is capped at 25 minutes. Run the structural checks, build,
fail-closed axiom report, and executable probes with:

```bash
./scripts/validate.sh all
```

Individual phases are `static`, `build`, `axioms`, and `check`. CI runs the same
bounded wrapper.

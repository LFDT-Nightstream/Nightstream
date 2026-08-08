# Lifecycle

`crates/neo-fold-clean/src/lifecycle/` owns the public direct-CCS lifecycle.

```text
preprocess
  -> prove
  -> extend
  -> finish_uncompressed
  -> verify_uncompressed
```

`UncompressedAudit` retains per-step data.
`verify_uncompressed_audit` replays that data and is linear in the number of
steps. `Uncompressed` drops the audit trail and checks the terminal state.

The recursive R1CS and Nebula frontends compile the authoritative F' induction
and can use terminal-only verification across recursive steps. Direct CCS does
not compile that induction, so its multi-chunk path uses audit replay.

## Terminal proof

`build_decider_statement` creates the public image and witness for the
terminal relation. The recursive R1CS terminal path compiles that relation and
calls `wip-spartan` through `finish_with_spartan`; `verify_spartan`
checks the result.

## Verifier checks

Terminal verification recomputes or checks:

- the final NIFS fold;
- the final accumulator commitments;
- public-input projections;
- low-norm witness entries;
- committed-evaluation relations;
- recursive F' links when the frontend supplies them; and
- Nebula lane openings when configured.

A digest is never used as proof authority without these checks.

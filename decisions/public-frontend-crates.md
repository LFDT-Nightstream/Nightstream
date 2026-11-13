# Public Frontend Crates

**Status:** Accepted

## Problem

Starstream builds its interleaving relation itself and records each WASM run
with its own `wasmtime` store. With only `nightstream` public, it cannot reach
the relation builders or the trace capture, so it uses internal crates and a
relation-only audit with host-checked memory.

## SuperNeo

SuperNeo v1.2 fixes no crate boundary or API. It leaves open how an
application states its relation and what its claim is.

## Decision

The owner approved this on 2026-11-02.

1. **Two public crates.** `nightstream` and `neo-wasm` are public.
   `nightstream::relation` re-exports the relation builders of
   `neo-application`. `neo-wasm` depends on `nightstream`, and `nightstream`
   never depends on `neo-wasm` or `wasmtime`. A finding counts if input
   through either public crate reaches it (`AGENTS.md`, Security).
2. **One application for R1CS relations.** A frontend proves the relation of
   one row with `nebula::R1csApplication`, with its continuity links, its
   memory catalog and a size (`SPEC.md` section 7.6).
3. **The caller owns the claim.** The verifier checks the initial and last
   application states. An event list is not part of it: Starstream compares
   the event chain of each last WASM state with the roots of its interleaving
   statement, and another caller folds its own events.
4. **A verifier compiles its own circuit.** A saved circuit file serves only
   a prover (`SPEC.md` Algorithm 8, Source).

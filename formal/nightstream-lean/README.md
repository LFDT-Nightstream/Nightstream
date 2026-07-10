# Nightstream Lean

This is a proposed assurance-first formalization. The existing Lean projects
(`../superneo-lean`, `../direct-ccs-fprime-lean`, `../twist-shout-lean`) remain
in place and are not dependencies; whether they are superseded by this project
is an open decision.

The normative project structure, threat model, property matrix, evidence states,
and completion gates are defined in
[`specs/formal-verification.md`](specs/formal-verification.md).

## Current status

The project currently provides:

- explicit paper-level CCS and CE membership shapes;
- a Construction-2 state matching the authority-bearing core of the Rust state;
- an executable F' envelope checker;
- a proved `check_sound` theorem for branch, counter, fixed-`pc`, immutable
  boundary, and public-trace coherence;
- the exact end-to-end verifier reduction target the project is working toward.

This is not yet verification of SuperNeo, the F' circuit, or the Rust verifier.
The first theorem is deliberately narrow and labeled model-level until the full
conformance procedure is satisfied.

## Ownership and dependency direction

```text
SuperNeo model       HyperNova model
          \           /
        Protocol composition
                |
      Implementation model
                |
       Assurance theorems
```

SuperNeo and HyperNova are sibling foundations. Concrete composition owns their
integration; implementation and assurance layers consume it. There are no
per-file interface façades. A boundary is introduced only when a real consumer
exists.

## Verification

```bash
lake build
lake exe check
```

The next assurance milestone is a concrete F' step relation plus a theorem that
satisfaction of the generated recursive R1CS implies that relation.

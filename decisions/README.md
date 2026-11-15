# Design Decisions

This directory records why Nightstream selects project-specific designs that
the SuperNeo paper does not fix. These records are tracked rationale. They do
not define accepted proofs or verifier behavior.

Authority is separate. A record does not override these sources. Where a
record and a source differ, the source is correct and the record has a defect.

- Lean in `formal/nightstream-fprime/` is the source of truth for the
  circuits and the proofs.
- [SPEC.md](../SPEC.md) is the only protocol specification. It restates the
  Lean.
- `formal/nightstream-fprime/SECURITY_MODEL.md` gives each adversary and each
  premise.
- Rust implements the protocol.
- `docs/superneo-paper-v1_2/` contains the reviewed paper. It is a local
  folder and is not in git, so a record cites the paper by its v1.2 numbers
  and does not link to it.

Each decision must use one kebab-case file with these sections:

1. **Status**: `Proposed`, `Accepted`, or `Superseded`.
2. **Problem**: the local issue that requires a choice.
3. **SuperNeo**: what the paper fixes and what it leaves open.
4. **Decision**: the selected rule and its required assumptions.

Keep each record short. Link to the normative rule and its evidence. A Lean
name in backticks must exist: `validate.sh axioms` fails when a record cites a name that is not a declaration or a module. Do not
put implementation history, migration plans, generated row counts, benchmark
results, or status diaries here. A decision that changes accepted inputs,
proof messages, transcript order, or proof bytes must also update
[SPEC.md](../SPEC.md), which describes the current protocol.

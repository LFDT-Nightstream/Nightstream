# Design Decisions

This directory records why Nightstream selects project-specific designs that
the SuperNeo paper does not fix. These records are tracked rationale. They do
not define accepted proofs or verifier behavior.

Authority is separate:

- `docs/superneo-paper/` contains the reviewed paper.
- `protocol-contract/` defines the normative Nightstream protocol.
- Rust and Lean provide implementation and proof evidence.

Each decision must use one kebab-case file with these sections:

1. **Status**: `Proposed`, `Accepted`, or `Superseded`.
2. **Problem**: the local issue that requires a choice.
3. **SuperNeo**: what the paper fixes and what it leaves open.
4. **Decision**: the selected rule and its required assumptions.

Keep each record short. Link to the normative rule and its evidence. Do not
put implementation history, migration plans, generated row counts, benchmark
results, or status diaries here. A decision that changes accepted inputs,
proof messages, transcript order, or proof bytes must also update the protocol
contract.

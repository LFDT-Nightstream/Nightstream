# AGENTS.md

Nightstream is a lattice-based folding prover over Goldilocks. Users call only
the `nightstream` crate.

## Design

- SuperNeo v1.2 (ePrint 2026/242) is the folding scheme: PiCCS, PiRLC, and
  PiDEC over Ajtai commitments.
- HyperNova is the recursion. The F′ circuit (R1CS) checks the previous fold
  inside each step.
- Nebula is the memory layer. It is Stage 2 work on
  `nico/nebula-superneo-spec`.
- Spartan over WHIR compresses the final proof (`neo-spartan`).
- The papers are in `docs/` (local only, not in git): `superneo-paper-v1_2`,
  `hypernova-paper`, and `nebula-paper`.

## Lean

Lean in `formal/` is the source of truth for the circuits. Lean emits the
circuit artifacts, and Rust embeds them and pins their identifiers.

- `formal/nightstream-fprime` models SuperNeo v1.2, defines and proves the F′
  circuit, and emits the F′ artifacts. Read its `AGENTS.md` before work there.
- `formal/compression-lean` builds the compression terminal circuit from the
  F′ circuits. It proves the layer-1 verifier sound. That proof assumes WHIR
  and treats Poseidon2 as a random oracle.

## Rules

- Do not change this file without explicit approval.
- Write user-facing answers in ASD-STE100 Simplified Technical English.
- Keep the code small. The project is in development, so do not keep
  backwards compatibility.
- Do not add Rust features or environment variables without approval.
- Sign off every commit (`git commit -s`).
- Keep each file under 1,500 lines unless the user approves more.

## Tests

- Put tests in `tests/`, not in the implementation file.
- Use `FoldingMode::Optimized`. `PaperExact` needs approval.
- Run Rust tests with `--release` and a 5-minute timeout. Lean commands get
  25 minutes. A run that reaches its timeout fails. A longer run needs
  approval for that run.

## Protocol

- Cite SuperNeo only by its v1.2 numbers.
- Use only Poseidon2 in proof, transcript, and public-digest paths. Do not add
  other hash families there without approval.
- Decomposition uses `b = 2` and `B = 2^k_rho`. Use `k_rho = 16` by default.
  Use `k_rho = 18` only when a requirement or a measurement shows that 16 is
  not sufficient. Each profile binds exactly one of these two values. Any
  other value, including the paper's `k_rho = 14`, needs approval in the
  current task.
- Call such a profile "Nightstream Goldilocks, k_rho = N", never "paper exact".

## Security

- Only `nightstream` is public. The `neo-*` crates, `nightstream-fprime`, and
  the prover backends are internal. A finding counts only if input through
  `nightstream` can reach it.
- A digest is compression, never authority. Across a trust boundary, recompute
  each carried digest from its inputs, bind it into the verifier transcript, or
  treat it as unchecked. If an attacker changes data and re-digests upward, the
  verifier must still reject.

# AGENTS.md

Nightstream is a lattice-based folding prover over the prime field
`q = 2^64 − 59`. Users call only the `nightstream` crate and its WASM
frontend, `neo-wasm`.

## Design

- SuperNeo v1.2 (ePrint 2026/242) is the folding scheme: PiCCS, PiRLC, and
  PiDEC over Ajtai commitments.
- HyperNova is the recursion. The F′ circuit (R1CS) checks the previous fold
  inside each step.
- Nebula is the memory layer (`nightstream::nebula`, `SPEC.md` section 7).
- Akita (ePrint 2026/1983) is the polynomial commitment scheme for the
  compression of the final proof. The Spartan, WHIR, and shrink path was
  removed. The compression layer over Akita is not written yet.
- The papers are in `docs/` (local only, not in git): `superneo-paper-v1_2`,
  `hypernova-paper`, and `nebula-paper`.

## Lean

Lean in `formal/` is the source of truth for the circuits. Lean emits the
circuit packages, and Rust checks each package's identity against a pinned
value.

- `formal/nightstream-fprime` models SuperNeo v1.2, defines and proves the F′
  circuit, and emits the F′ artifacts. Read its `AGENTS.md` before work there.

## Specs

- `SPEC.md` is the only protocol specification. It restates the Lean, and
  where the two differ, the Lean is correct. Add a new protocol part as a
  section of `SPEC.md`, in its format (Input, Output, Steps, Rejects,
  Source), with a census pin for each theorem that it cites.
- Put each adversary and each premise in
  `formal/nightstream-fprime/SECURITY_MODEL.md`. Put each owner design
  choice in `decisions/`, in the format of `decisions/README.md`.
- Do not add a separate spec file, a `specs/` folder, prose proofs, or
  status lists. Put progress and review history in the PR description.
- In a comment, cite a `SPEC.md` section or algorithm, or a Lean name.

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
- Run Rust tests with `--release`. Each test gets 5 minutes. The limit is
  for one test, not for a test binary or a suite, so do not stop a suite
  after 5 minutes. A test that reaches its limit fails. A test that needs
  more time needs approval for that run.
- Each Lean command gets 25 minutes. A command that reaches its limit
  fails. A longer command needs approval for that run.

## Protocol

- Cite SuperNeo only by its v1.2 numbers.
- The field is `F_q` with `q = 2^64 − 59`. The ring is `F_q[X]/(X^64 + 1)`.
  The main commitment rank is `κ = 18`
  (`decisions/field-2p64-59-ring-x64.md`). Do not add code for Goldilocks or
  for a second field or ring.
- Use only Poseidon2 and the Ajtai commitment of the scheme as hashes in
  proof, transcript, and public-digest paths of the folding scheme and the
  step circuits. Poseidon2 makes each of their Fiat–Shamir challenges. The
  Ajtai commitment can be a digest where collision resistance is sufficient.
- SHAKE128 expands the Ajtai setup. The Akita commitment scheme can use SHAKE
  for its challenges. Do not add other hash families, for example BLAKE3,
  without approval.
- Decomposition uses `b = 2` and `B = 2^k_rho`. Use `k_rho = 16` by default.
  Use `k_rho = 18` only when a requirement or a measurement shows that 16 is
  not sufficient. Each profile binds exactly one of these two values. Any
  other value, including a value from the paper, needs approval in the
  current task.
- Call such a profile "Nightstream P64, k_rho = N", never "paper exact".

## Security

- Only `nightstream` and `neo-wasm` are public. `nightstream::relation`
  re-exports the relation builders of `neo-application`. The other `neo-*`
  crates, `nightstream-fprime`, and the prover backends are internal. A
  finding counts only if input through `nightstream` or `neo-wasm` can reach
  it.
- A digest is compression, never authority. Across a trust boundary, recompute
  each carried digest from its inputs, bind it into the verifier transcript, or
  treat it as unchecked. If an attacker changes data and re-digests upward, the
  verifier must still reject.

# NIFS proof and consumer links

Base source: `0ff1ef1d96d88020c3d7e6673684cf213f0ffd6c`.
Work branch: `nico/nifs-proof-links`.

The task is to finish the remaining NIFS proof obligations and their consuming
links. Existing proved results remain inputs. Closure needs checked evidence
for the actual selected boundary. An assumption, definition, or recorded test
must keep its correct status.

The profile is Goldilocks, `b = 2`, `k_rho = 16`, `B = 65536`, one fresh
source, 16 running sources, 17 PiRLC inputs, 16 PiDEC children, 14 matrices,
28 PiCCS rounds, and Poseidon2 protocol binding.

## Completion evidence

| Requirement | Evidence needed | Status |
|---|---|---|
| `N.binding.prior_authority` | The checked recursive boundary supplies the exact prior preimage used by the NIFS transcript, including all running claims and the selected context. | Local proof and consumer checked |
| `N.binding.context` | Arbitrary accepted opening and checked public input identify the selected context, or a named collision. | Local proof and consumer checked |
| `N.local.actual_step` | The same accepted opening reaches the NIFS consumer with the actual proof, prior claims, and advertised output. | Local proof and consumer checked |
| `N.security.binding` | The actual extraction collision reaches the selected public-seed MSIS assumption with the required norm and execution scope. | Open |
| `N.security.fiat_shamir` | The selected Poseidon2 transcript and bounded sampler have a justified security connection under the authorized model. | Open |
| `N.conformance.chain` | One nonzero selected-key input and proof have matching Lean and optimized Rust phase values and final output, with required mutations. | Open |
| `N.conformance.executed` | Retained commands, inputs, outcomes, and source identities establish the stated execution scope. | Open |
| `N.conformance.owners` | The checked chain consumes the existing semantic, transcript, assignment, and caller owners. | Open |

The existing conditional interactive proofs do not establish Fiat–Shamir
transfer. The fixed-seed MSIS premise is the exact premise recorded in
`PUBLIC_SEED_MSIS_ASSUMPTION.md`; it supplies no numerical success bound.
Concrete execution-correctness gaps cannot become hardness assumptions.

## Checked terminal connection

Code commit: `42a8c110023d63233358840594db7fca077fd001`.

`ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision` connects the
exact prior-state public input and advertised running output to the NIFS
consumer. It uses the existing
`ActualContextSecurity.terminal_implies_matchingStepOrCollision` result.
The terminal predicate supplies the opening and public check. There is no
extra canonical-assignment or context-equality premise.

`terminal_implies_parentOrBaseOrCollision` consumes this result for the
actual PiDEC parent opening. `terminal_implies_securityOrCollision` consumes
it for the selected NIFS security outcome, with the existing low-norm
invertibility premise. This is a deterministic theorem with named failure
alternatives. It does not supply a probability bound or a history extractor.

The affected module build passed in 17 seconds. Its explicit Stage 1 security
axiom audit passed in 12 seconds and used only `propext`, `Classical.choice`,
and `Quot.sound`. The boundary gate passed. The dependency build took 390
seconds, using a separate project cache and matching dependency cache.
Every Lean command had the project-required 1,500-second hard cap.

This proof connection changes no circuit rows, transcript schedule, package
identity, or profile. The commands, logs, and source identities are retained
in `NIFS_PROOF_LINKS_EVIDENCE.zip`.
The requirements export and all seven existing export tests passed. The
JavaScript syntax check passed. The local map changes only the three NIFS
records above; other requirement records are preserved.

## Active criterion

Connect the actual NIFS extraction collision to the selected fixed-seed MSIS
boundary. The exact response differences and strict norm bounds must survive
the connection. No numerical hardness bound is supplied by the approved
assumption.

The older `protocol-contract/security-reduction.md` uses a different
transcript, sampler, and profile. Its numerical query limits and security
terms are not evidence for this selected NIFS instance.

The primary checkout contains the local normative paper files. The isolated
worktree uses those files for reading. The frozen package instructions are
absent from both checkouts; no frozen proof files are used.

Full HyperNova history extraction, Stage 2, proof-backend execution, and site
publication remain outside this task. Pending independent approvals must not
be replaced by local diagnostic results.

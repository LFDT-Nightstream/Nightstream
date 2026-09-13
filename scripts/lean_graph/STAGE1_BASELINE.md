# Current Stage 1 baseline closure

Owner: the user's September 13 baseline goal. Work only on
`nico/f-prime-constraints-cuda-formal`, starting from `f714497c`.

The final result is the existing selected Goldilocks lifecycle, with
`b = 2`, `k_rho = 16`, its current package, and the approved FS/MSIS
boundaries. The public flow is `Stage1Envelope::initial(z0)`,
`package.extend(envelope, message)`, then
`package.verify(&expected_state, &envelope)`.

## Exact obligations

| Obligation | Required result | Closing evidence |
| --- | --- | --- |
| `stage1-assignment` | Arbitrary selected rows and their actual public input imply the full typed step at the decoded context. | Literal `LeanGraph.Targets.Stage1Assignment`, witnessed by `stage1Assignment` through `ActualPiDECOutput.selectedRowsAndPublic_imply_step`; exact acceptance and axiom audit. |
| Existing terminal and security targets | Bind the decoded step to the verifier-owned context or a named collision; retain the checked linear history bound. | `Stage1TerminalAssignment`, `Stage1TerminalParent`, and `HyperNovaLinearSecurity`, with their existing complete premises and required gates. |
| Public lifecycle | Construct the base assignment from actual state/advice and extend an active envelope through the existing native NIFS, caller packet and witness completion. | Public API execution, full Lean packet/assignment comparison, exact openings and terminal acceptance, with sampler/counter failures explicit. |
| Nonzero running input | Execute a further fold with actual nonzero running witnesses, then construct and verify its successor. | Complete C/R/D values and bytes, transcript/public/counter equality, exact next assignment and required mutation rejection against independent Lean checks. |
| Baseline delivery | Correct stale records and retain exact evidence for every technical link in scope. | Signed checkpoint, ordered static/build/axiom gates, affected identity/conformance checks, independent review and current requirements links. |

`stage1-baseline` tracks the combined result in the existing obligation map.
Its open requirements remain until their actual tests and review exist.
This registration does not change lean-graph's schema or acceptance rules.

## Premises and limits

The assignment target uses the actual public projection, a four-word digest,
and all selected rows. It assumes no honest encoder, NIFS acceptance,
sampler success or child-output match. Its context is decoded; selected
verifier authority remains the separate terminal/hash contract.

Reuse the approved cryptographic premises, mathematical invertibility,
declared clock and query conditions, conditional sampler success, and
canonical counter bounds. No new security model or numerical budget is
selected. The previous executed trace stops at iteration 2; it does not
prove the new nonzero-running execution requirement. Capped stages may
supply their exact evidence, but cannot be reported as an uninterrupted run.

One active obligation and one Lean/Rust build queue apply. The standing
owner override is ten rounds per obligation for this session; previous
attempts count. Native invocations retain the 300-second cap and Lean
commands the 1,500-second cap. Do not repeat an unchanged failed run.

This work excludes constraint reduction, new profiles, protocol changes,
new proof backends, Stage 2 and general cryptographic research. Required
correctness repairs must retain the appropriate preservation and package
checks. External production approval remains separate from local evidence.

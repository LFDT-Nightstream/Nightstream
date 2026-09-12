# R entry proof draft — not validated

`R_ENTRY_DRAFT.lean` preserves the stopped R permutation consumer as review
material on `nico/f-prime-constraints-cuda-formal`. It is outside the production
Lean tree and has no audit or library import. The earlier draft history was
merged here at the owner's instruction; further work uses this branch only.

The exact full draft has SHA-256
`a1ac935e8a6488526967112557cbd98e032552cdfd7bdf710378fcbe5866ba9f`.
Private `entry_sboxes` failed its tenth conservative attempt with kernel deep
recursion after 41.91 seconds. `R_ENTRY_FAILURE.log` records the exact failure
of the isolated candidate, SHA-256
`c12166cf1c636d5861f2b0a7e152b2edec2852030bafba6230bffbdef68cf20e`.

The generic S-box converter, source row proof, input-value proof and initial
C-to-R value proof passed isolated checks. A diagnostic with temporary proof
parameters also passed. Their actual composition did not pass. No temporary
parameter was added to the public consumer as an assumption. The complete
source, including its final adapted consumer, has not passed a file check.

The owner limit is ten attempts per isolated declaration. No further R entry
attempt is authorized without a new owner decision. No recursion/heartbeat
option override, protocol change or cryptographic assumption is proposed here.

# Remaining work

The maintained path is the `nightstream` application compiler, SuperNeo
folding with a light and a refresh step circuit, and the exported
HyperNova-style recursive verifier. Steps prove only on the CPU path.

Keep the active Lean proofs, exact exported rows, Rust witness execution,
and saved conformance fixtures aligned. Complete the concrete Fiat–Shamir
applicability argument and obtain independent review. The shape estimator
alone does not establish total system security.

The compressed proof needs these items
([decision record](../decisions/akita-compression.md)): the owner's review of
the proposed design, a Lean statement of the argument, a succinct verifier
that reads commitments to the Ajtai key and to the matrices, one batched
Akita opening, a run with slot depths near the cap 64, a WASM build of the
verifier, and an audit of Akita.

Device engines for the steps (Metal and CUDA) require separate implementation
and validation. Removed legacy consumers are not
alternate supported paths. Add application interfaces only when their
semantics are bound to the maintained verifier package.

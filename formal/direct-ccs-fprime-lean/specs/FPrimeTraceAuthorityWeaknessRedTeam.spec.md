# F Prime Trace Authority Weakness Red Team

This component records misuse cases for trace-carrying folded `F'` authority.

## Weak Transition

If the transition relation accepts every adjacent image pair, trace authority
can authorize arbitrary one-step images. Therefore the production transition
must include the real step obligations: public-image counters, boundary
preservation, parent-only accumulator update, private DEC correctness, child
membership, and exact child reuse.

## Missing Same-Proof Functionality

Trace soundness proves reachability of each accepted `(steps, image)` pair. It
does not by itself prove that one opaque proof cannot verify for two different
public images. Surfaces that reuse opaque prior proofs across trust boundaries
must also expose same-proof functionality or an equivalent fixed-opener
binding.

## Non-Binding Parent Hash

A constant parent-hash toy model cannot bind two different parent handles. The
Poseidon2 parent hash object is therefore a real binding assumption and must be
instantiated with a binding hash surface.

## Quarantined Probes

The Lean module includes commented break probes for these weak models. They
are intentionally outside the build while the weak models are used only as
negative examples.

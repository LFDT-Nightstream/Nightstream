# HyperNova recursion

HyperNova Construction 2 supplies the recursive compiler structure: F′ runs
the application step, verifies the previous fold, and binds the resulting
state. Nightstream instantiates folding with SuperNeo PiCCS, PiRLC, and
PiDEC. PiDEC returns sixteen low-norm claims rather than one elliptic-curve
accumulator.

The Rust owners are `nightstream/src/folding` and `nightstream/src/lifecycle`.
The Lean owners are `NightstreamFPrime/Lifecycle/Stage1` and
`NightstreamFPrime/Export/Stage1`. The selected package includes the actual
application rows and the verifier rows; separate state digests do not prove
that those computations occurred.

`HyperNovaCompleteness` and `HyperNovaAcceptedNext` cover honest NIFS and
accepted-successor construction for the current key. The wide sampler is
total, so these results no longer require sampler success. Valid application
advice, prior openings, and the iteration bound remain necessary premises.

Soundness and completeness are distinct. The security results, their
adversaries and their premises are in
[the security model](../../formal/nightstream-fprime/SECURITY_MODEL.md).

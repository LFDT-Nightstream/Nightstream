# Formal proofs

`formal/nightstream-fprime` is the maintained Lean authority for this
production line. Its layers connect the protocol relation, circuit, physical
layout, exported package, and Rust consumer. Use its bounded `validate.sh`
wrapper and read its `AGENTS.md` before editing.

Kernel-checked theorems, source-bound conformance executions, and independent
review are separate evidence. A matching artifact digest identifies bytes;
it does not prove Rust semantics or discharge a security assumption. See
[the security model](../../formal/nightstream-fprime/SECURITY_MODEL.md)
and [fresh conformance](../../scripts/GOLDEN_CONFORMANCE.md).

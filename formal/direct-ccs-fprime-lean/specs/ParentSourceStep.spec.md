# Parent Source Step

`ParentSourceStep` specifies the reduced parent-source derivation used by the
direct CCS `F'` accumulator update.

The parent source is the compact source for the single parent `CE(B)` claim
produced by the latest SuperNeo fold. It is derived in two protocol stages:

```text
Pi_CCS:
  prior CE(b)^k accumulator + fresh CCS^K
  -> CE(b)^(K+k) output claims

Pi_RLC:
  CE(b)^(K+k) output claims
  -> one parent CE(B) source
```

The theorem-facing parent-source step is:

```text
exists pi_ccs_output:
  PiCCS(i, prior, pi_ccs_output)
  PiRLC(i, pi_ccs_output, parent_source)
```

The required functionality property is:

```text
PiCCS is functional for a fixed step and prior accumulator
PiRLC is functional for a fixed step and Pi_CCS output
=>
ParentSourceStep is functional for a fixed step and prior accumulator
```

This is the precise obligation needed by the reduced `CE(B)^1` handle
strategy. The private `Pi_DEC` theorem proves that a fixed parent source
authorizes one child table. This module proves that the parent source itself is
fixed when the accepted `Pi_CCS` and `Pi_RLC` stage relations are fixed.

When applied to the direct Construction-2 transition, the composed theorem says
that two accepted latest `F'` transitions from the same prior image cannot
produce different reduced accumulator parent sources or different authorized
next `Pi_CCS` child inputs, assuming:

```text
Pi_CCS stage functionality
Pi_RLC stage functionality
encoded parent digest binding
deterministic statement commitment encoding
Ajtai no-collision
CE-opening adapter
```

Function-computed variants model deterministic implementations directly:

```text
out = computePiCCS(i, prior)
source = computePiRLC(i, out)
```

Such relations are functional by construction. A paper-faithful implementation
must instantiate those functions or accepted-stage relations with the actual
`Pi_CCS` transcript output and `Pi_RLC` parent `CE(B)` computation.

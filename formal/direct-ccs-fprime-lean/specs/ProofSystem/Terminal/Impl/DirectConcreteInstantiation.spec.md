# Direct Concrete Instantiation

`DirectConcreteInstantiation` states the concrete theorem boundary for the
direct CCS terminal `F'` proof with the reduced parent `CE(B)` handle.

The CE relation used by the terminal theorem must be built from the canonical
Ajtai commitment map:

```text
commitMap(assignment) = M || M * toOpening(assignment).witness
```

The concrete CE data must provide:

```text
fixed public Ajtai matrix M
assignment -> Ajtai opening
assignment -> input projection
CCS/CE shape
opening well-formedness
opening norm soundness
opening bound below the Ajtai binding bound
residue projection equality for private Pi_DEC
```

From this data, the module constructs:

```text
ConcreteCEData.ce
ConcreteCEData.ajtaiBackedCommitMap
```

The terminal theorem target is:

```text
encoded parent CE(B) digest binding
MSIS-to-Ajtai reductions
MSIS hardness
accepted proof-carrying folded F' prior authority
accepted latest direct F' step
alternate latest direct F' step from the same prior image
=>
the final image is reachable
and the latest accumulator parent source and next Pi_CCS inputs are functional
```

This module does not instantiate the actual direct application boundary,
`Pi_CCS`, `Pi_RLC`, or folded F' proof system. Those remain concrete protocol
objects supplied to the theorem. The purpose of this module is to remove the
abstract CE commitment-map premise from the final theorem boundary.

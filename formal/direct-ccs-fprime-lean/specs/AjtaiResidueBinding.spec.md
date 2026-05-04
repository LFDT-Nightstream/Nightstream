# Ajtai Residue Binding

`AjtaiResidueBinding` connects the reduced parent-handle proof obligation to
the concrete Ajtai opening relation.

The target statement is:

```text
NoAjtaiBindingCollision(params)
AssignmentOpeningAdapter(params, commitMap)
commitMap(assignmentA) = commitMap(assignmentB)
=>
residues(assignmentA) = residues(assignmentB)
```

The implementation-facing local statement is:

```text
NoAjtaiBindingCollision(params)
CEOpeningAdapter(params, ce)
same statement commitment
CE.Holds(ce, statementA, witnessA)
CE.Holds(ce, statementB, witnessB)
=>
residues(witnessA) = residues(witnessB)
```

`NoAjtaiBindingCollision(params)` is the Prop-level form of the Ajtai binding
assumption for the concrete parameters. It says there is no pair of distinct
bounded openings for the same Ajtai commitment.

`AssignmentOpeningAdapter` supplies the implementation-specific bridge from
the abstract CE commitment map to the concrete Ajtai opening relation:

```text
opensTo(params, commitMap(assignment), toOpening(assignment))
normBound(toOpening(assignment)) < params.bindingNormBound
projectWitnessResidues(toOpening(assignment).witness) = residues(assignment)
```

The canonical implementation-facing bridge is an Ajtai-backed commitment map:

```text
commitmentOfOpening(params, M, opening).payload = M || (M * opening.witness)

ajtaiCommitMap(params, M, toOpening)(assignment)
  = commitmentOfOpening(params, M, toOpening(assignment))

AjtaiBackedCommitMap(params, commitMap):
  fixed public matrix M
  commitMap(assignment) = commitmentOfOpening(params, M, toOpening(assignment))
  toOpening(assignment) is well formed and norm sound
  normBound(toOpening(assignment)) < params.bindingNormBound
  projectWitnessResidues(toOpening(assignment).witness) = residues(assignment)
```

This bridge must induce the assignment-opening adapter:

```text
AjtaiBackedCommitMap(params, commitMap)
=>
AssignmentOpeningAdapter(params, commitMap)
```

The concrete canonical commitment map must also induce that backing directly:

```text
matrix has the Ajtai parameter shape
toOpening is well formed, norm sound, and bounded
projectWitnessResidues(toOpening(assignment).witness) = residues(assignment)
=>
AjtaiBackedCommitMap(params, ajtaiCommitMap(params, M, toOpening))
```

`CEOpeningAdapter` is the narrower adapter used by the protocol theorem. It
only applies to openings accepted by `CE.Holds`, which is the exact terminal
proof condition needed by private `Pi_DEC` authorization.

An assignment-level adapter for the fixed CE commitment map induces the
CE-local adapter:

```text
AssignmentOpeningAdapter(params, ce.commitMap)
=>
CEOpeningAdapter(params, ce)
```

The bridge uses the commitment equality already contained in `CE.Holds`:

```text
statement.commitment = ce.commitMap(witness.assignment)
```

The theorem-facing Ajtai binding assumption implies the Prop-level
no-collision fact needed by the local residue-binding theorem:

```text
AjtaiBindingAssumption(params)
=>
NoAjtaiBindingCollision(params)
```

The SuperNeo lattice reduction boundary also implies the same no-collision
fact from MSIS hardness:

```text
MSISToAjtaiReductions(params)
MSISHardnessAssumption(params)
=>
NoAjtaiBindingCollision(params)
```

The module does not prove MSIS hardness, the Ajtai advantage bound, or
concrete hash security. It proves that, once the theorem-facing Ajtai binding
assumption is supplied, or once the SuperNeo MSIS-to-Ajtai reduction surface
and MSIS hardness assumption are supplied, the abstract
`CommitMapResiduesFunctional` obligation used by parent opening authorization is
discharged.

This is the theorem-facing bridge needed for the reduced `CE(B)^1` source:
private `Pi_DEC` may use parent residues only when those residues are forced by
an accepted Ajtai opening of the same parent commitment.

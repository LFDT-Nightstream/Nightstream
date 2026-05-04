# Direct Parent Only Production Child Membership

`DirectParentOnlyProductionChildMembership` specifies the fixed-CE membership
facts for the private DEC children extracted by the production parent-only
direct CCS `F'` terminal theorem.

The theorem consumes the same production context, accepted latest transition,
alternate latest transition, and sound prior verifier boundary as
`DirectParentOnlyProductionSoundness`. Its conclusion strengthens the extracted
private child table and exposes its non-aggregate audit trail:

```text
PointwisePrivateDecRequirements(parentSource, priorInputs)
PointwiseChildAuditTrail(parentSource, priorInputs)
FixedCEChildMembership(params, ce, priorInputs)
nextImage = ComputedNextImage(priorImage, priorInputs)
altNext = ComputedNextImage(priorImage, priorInputs)
any other pointwise-valid table for parentSource equals priorInputs
```

`FixedCEChildMembership` means:

```text
child bundle CE relation equals the context-fixed CE relation
child bundle Ajtai parameters equal the context-fixed Ajtai parameters
every child satisfies CE.Holds for the context-fixed CE relation
every child opening opens under the context-fixed Ajtai parameters
next Pi_CCS input wires equal the CE witness-derived child table
```

`PointwiseChildAuditTrail` means:

```text
private Pi_DEC accepts over the opened parent CE(B) source
the accepted child bundle uses the context-fixed CE relation
the accepted child bundle uses the context-fixed Ajtai parameters
each child column is binary
each child column has fixed length 14
each column recomposes modulo Goldilocks.q to the opened parent residue
the child digit table equals the CE witness-derived child table
next Pi_CCS input wires equal that child digit table
the same table satisfies FixedCEChildMembership
```

`TerminalChildAuditTrail` packages those pointwise facts with the terminal
public-image equations and uniqueness of the private child table for the parent
source.

The module exposes named projections from `PointwiseChildAuditTrail` so callers
can consume the exact private DEC acceptance, fixed-CE membership, non-aggregate
DEC facts, and next-`Pi_CCS` wire identity without relying on the internal
shape of the existential package.

The proof-carrying folded-prior entry point has the same conclusion. It is the
reference theorem path where prior `F'` reachability is carried directly by the
prior authority object.

The raw compressed-prior entry point has the same conclusion after the caller
supplies the verifier-opening theorem from accepted compressed proofs to
proof-carrying folded `F'` authority.

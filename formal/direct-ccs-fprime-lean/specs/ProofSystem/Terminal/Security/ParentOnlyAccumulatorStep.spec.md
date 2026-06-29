# Parent Only Accumulator Step

`ParentOnlyAccumulatorStep` specifies the optimized direct CCS `F'` accumulator
update whose public accumulator handle carries only a compact parent `CE(B)`
source.

The accumulator handle contains:

```text
parentSource
```

The post-DEC `CE(b)^k` children are private proof advice. They may affect the
latest parent-source computation only through an authorization relation:

```text
AuthorizedPrior(prior.parentSource, priorInputs)
ParentSourceStep(i, prior, priorInputs, next.parentSource)
```

`AuthorizedPrior` owns the private `Pi_DEC` proof that `priorInputs` are the
canonical children of the prior parent source. `ParentSourceStep` owns the
latest `Pi_CCS -> Pi_RLC` parent `CE(B)` computation using exactly those
authorized child inputs.

The functionality target is:

```text
AuthorizedPrior is functional
ParentSourceStep is functional for fixed priorInputs
Step(i, prior, next_a)
Step(i, prior, next_b)
=>
next_a.parentSource = next_b.parentSource
```

The canonical authorization relation is not an aggregate check. Every accepted
authorization exposes pointwise private-DEC requirements:

```text
child bundle CE relation equals the fixed stage CE relation
child bundle Ajtai parameters equal the fixed stage Ajtai parameters
binary digits for every column
exact DEC length 14 for every column
Goldilocks recomposition for every column
child table equals the CE witness-derived digit table
next Pi_CCS input wires equal that same child table
```

Those requirements expose fixed-CE child membership:

```text
child bundle CE relation = fixed stage CE relation
child bundle Ajtai parameters = fixed stage Ajtai parameters
for every child: CE.Holds(fixed stage CE relation, statement, witness)
for every child: opensTo(fixed Ajtai parameters, statement.commitment, opening)
next Pi_CCS input wires equal the CE witness-derived child table
```

Those pointwise requirements are themselves functional for one parent source
under parent-handle binding, Ajtai-backed opening binding, and canonical DEC:

```text
PointwisePrivateDecRequirements(source, next_a)
PointwisePrivateDecRequirements(source, next_b)
=>
next_a = next_b
```

The no-swap audit for that theorem exposes the equalities behind the final
input equality:

```text
opened parent residues agree
private DEC digit tables agree pointwise
CE witness-derived digit tables agree pointwise
child-bundle next Pi_CCS wires agree
requested next Pi_CCS inputs agree
```

For existing stage interfaces, `ParentSourceFromPiStages` adapts a private
authorized child table into the child-carrying handle expected by `Pi_CCS`, then
feeds the resulting `Pi_CCS` output to `Pi_RLC`. Functional `Pi_CCS` and
functional `Pi_RLC` therefore induce a functional parent-source computation for
the parent-only public handle.

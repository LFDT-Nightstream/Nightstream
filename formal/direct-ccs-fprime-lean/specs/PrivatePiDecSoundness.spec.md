# Private Pi_DEC Soundness

`PrivatePiDecSoundness` states the proof-critical verifier boundary for the
reduced `CE(B)^1` handle strategy.

The verifier target is over the actual SuperNeo child bundle, not a standalone
digit table:

```text
Verify(source, parent_residues, child_bundle, proof)
```

An accepted verifier proof must imply:

```text
binary digits for child_bundle.digitTable
fixed column length k_dec = 14
Goldilocks modular recomposition to parent_residues
```

The child bundle itself carries the remaining required links:

```text
CE.Holds for every child CE
Ajtai opensTo for every child commitment
child_bundle.digitTable = digits extracted from child CE witnesses
next Pi_CCS inputs = child_bundle.digitTable
```

The theorem target is:

```text
same reduced source
+ source binds parent residues functionally
+ private Pi_DEC verifier soundness
+ accepted child bundles
=>
same next Pi_CCS child inputs
```

For the full reduced-handle path, the source-binding hypothesis is discharged
from canonical parent `CE(B)` digest binding plus the local Ajtai CE-opening
adapter.

The concrete sumcheck protocol is outside this component's ownership. This
component owns the minimal verifier-soundness facts that any private `Pi_DEC`
verifier must establish for the reduced-handle strategy to be sound.

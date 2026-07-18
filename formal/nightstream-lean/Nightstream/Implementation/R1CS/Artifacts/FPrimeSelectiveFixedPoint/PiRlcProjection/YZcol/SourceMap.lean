/-!
Independent producer-side coordinates for the bounded tiny-fixture `y_zcol`
projection boundary.

Owns: raw serializer field indices and the source-R1CS columns allocated for
one ordered coefficient vector.

Does not own: consumer columns, serializer semantics, source authority,
projection equations, generated values, or a proof that any binding is valid.

Emits constraints: no.

| Field | Mathematical role | Authority class |
|---|---|---|
| `serializerFieldIndex` | coordinate in the typed PiCCS-output serializer layout | untrusted artifact coordinate |
| `sourceColumn` | R1CS column allocated by that serializer coordinate | untrusted artifact coordinate |
| `ProducerVector.entries` | ordered 54-lane producer view for one source and limb | untrusted artifact structure |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

/-- One producer-side serializer-to-column coordinate. The consumer column is
deliberately absent: it is reconstructed independently from the projection
trace in `Schema.lean`. -/
structure ProducerEntry where
  serializerFieldIndex : Nat
  sourceColumn : Nat
deriving DecidableEq, Repr

/-- One ordered producer vector for a source claim and one coefficient limb. -/
structure ProducerVector where
  sourceIndex : Nat
  limb : Nat
  entries : List ProducerEntry
deriving DecidableEq, Repr

namespace ProducerVector

def serializerFieldIndices (owner : ProducerVector) : List Nat :=
  owner.entries.map ProducerEntry.serializerFieldIndex

def sourceColumns (owner : ProducerVector) : List Nat :=
  owner.entries.map ProducerEntry.sourceColumn

/-- Local shape only. The independent serializer formula and consumer equality
belong to correspondence, not to artifact data. -/
def HasShape (owner : ProducerVector) (laneCount : Nat) : Prop :=
  owner.entries.length = laneCount ∧
    owner.serializerFieldIndices.Nodup ∧
    owner.sourceColumns.Nodup

end ProducerVector

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

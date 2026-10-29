import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalEncodes
import NightstreamFPrime.Export.Stage1.RecursivePublicOutputPlan

namespace NightstreamFPrime.Export.Stage1.CanonicalPublicOutput

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open PerApplicationCanonicalAssignment

private theorem publicInput_eq {application : Program} (raw : RawValues application) :
    RecursivePublicOutputPlan.publicInput (PerApplicationFixedPoint.geometry application)
      raw.assignment = encHash raw.outputDigest := by
  funext column
  exact CanonicalBlockAssignment.assignment_publicColumn
    (encodedHashCells raw.outputDigest) raw.schedule
    (RecursivePublicOutputPlan.publicFits (PerApplicationFixedPoint.geometry application)) column

private theorem outputDigest_word {application : Program} (raw : RawValues application)
    (word : Fin 4) :
    raw.outputDigest.getD word.val 0 =
      PilotOrdinaryDirectPlan.pilotEnv application raw.base
        (PilotSpartan.sourceToSpartan (PilotProduction.outputDigestStart + word.val)) := by
  exact PriorStateHash.ofFn_getD _ word (0 : F)

theorem rowsZero {application : Program} (raw : RawValues application) :
    (RecursivePublicOutputPlan.plan (PerApplicationFixedPoint.geometry application)).RowsZero
      raw.assignment := by
  apply (RecursivePublicOutputPlan.rowsZero_iff_matches _ raw.assignment).mpr
  intro word
  rw [publicInput_eq raw, decodeHashWord_encHash]
  have held := PilotOrdinaryDirectPlan.Location.form_eval
    (RecursivePublicOutputPlan.pilotOrdinaryGeometry (PerApplicationFixedPoint.geometry application))
    raw.assignment raw.base raw.groupValue
    (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).prior.pilotOrdinary
    (.outputDigest word)
  exact held.trans (outputDigest_word raw word).symm

end NightstreamFPrime.Export.Stage1.CanonicalPublicOutput

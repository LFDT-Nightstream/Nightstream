import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateLane
import Nightstream.Implementation.R1CS.Core.Relabel
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows0
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows1
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows2
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows3
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows4
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows5

/-! Reusable exact checked-row templates for one alphabet-sampler lane and tail. -/

namespace Nightstream.Implementation.R1CS.AlphabetSamplingResidualTemplate

open Nightstream.Implementation.R1CS

def laneTemplateRows : List Row := Generated.laneRows

def tailTemplateRows : List Row :=
  Generated.tailRows0 ++
    Generated.tailRows1 ++
    Generated.tailRows2 ++
    Generated.tailRows3 ++
    Generated.tailRows4 ++
    Generated.tailRows5

def laneColumnMap (bitStart cumPrev : Nat) : List Nat :=
  [0] ++ (List.range 64).map (fun index => bitStart + index) ++
  [cumPrev] ++ (List.range 92).map (fun index => bitStart + 66 + index)

def chunkBases (bitStarts : List Nat) : List Nat :=
  bitStarts.flatMap fun bitStart =>
    (List.range 4).map fun chunk => bitStart + 66 + 23 * chunk

def tailInputColumns (bitStarts : List Nat) : List Nat :=
  let bases := chunkBases bitStarts
  [0] ++ bases ++ bases.map (fun base => base + 21) ++
    bases.map (fun base => base + 22)

def tailColumnMap (bitStarts : List Nat) (firstAllocated : Nat) : List Nat :=
  tailInputColumns bitStarts ++
    (List.range 3516).map (fun index => firstAllocated + index)

def laneRows (bitStart cumPrev : Nat) : List Row :=
  laneTemplateRows.map (Relabel.row (laneColumnMap bitStart cumPrev))

def tailRows (bitStarts : List Nat) (firstAllocated : Nat) : List Row :=
  tailTemplateRows.map (Relabel.row (tailColumnMap bitStarts firstAllocated))

theorem laneTemplateRows_length : laneTemplateRows.length = 104 := by native_decide

theorem tailTemplateRows_length : tailTemplateRows.length = 2599 := by native_decide

theorem laneRows_length (bitStart cumPrev : Nat) :
  (laneRows bitStart cumPrev).length = 104 := by
  simp [laneRows, laneTemplateRows_length]

theorem tailRows_length (bitStarts : List Nat) (firstAllocated : Nat) :
  (tailRows bitStarts firstAllocated).length = 2599 := by
  simp [tailRows, tailTemplateRows_length]

end Nightstream.Implementation.R1CS.AlphabetSamplingResidualTemplate

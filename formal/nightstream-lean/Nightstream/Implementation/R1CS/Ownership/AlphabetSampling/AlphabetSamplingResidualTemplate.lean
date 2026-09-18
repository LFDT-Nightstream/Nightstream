import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateLane
import Nightstream.Implementation.R1CS.Core.Relabel
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows0
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows1
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows2
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows3
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows4
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows5
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows6
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows7
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows8
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows9
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows10
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows11
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows12
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows13
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows14
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows15
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows16
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows17
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows18
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows19
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows20
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows21
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows22
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows23
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows24
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows25
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows26
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows27
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows28
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows29
import Nightstream.Implementation.R1CS.Artifacts.AlphabetSampling.Generated.AlphabetSamplingResidualTemplateRows30

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
    Generated.tailRows5 ++
    Generated.tailRows6 ++
    Generated.tailRows7 ++
    Generated.tailRows8 ++
    Generated.tailRows9 ++
    Generated.tailRows10 ++
    Generated.tailRows11 ++
    Generated.tailRows12 ++
    Generated.tailRows13 ++
    Generated.tailRows14 ++
    Generated.tailRows15 ++
    Generated.tailRows16 ++
    Generated.tailRows17 ++
    Generated.tailRows18 ++
    Generated.tailRows19 ++
    Generated.tailRows20 ++
    Generated.tailRows21 ++
    Generated.tailRows22 ++
    Generated.tailRows23 ++
    Generated.tailRows24 ++
    Generated.tailRows25 ++
    Generated.tailRows26 ++
    Generated.tailRows27 ++
    Generated.tailRows28 ++
    Generated.tailRows29 ++
    Generated.tailRows30

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

theorem laneTemplateRows_length : laneTemplateRows.length = 100 := by native_decide

theorem tailTemplateRows_length : tailTemplateRows.length = 13885 := by native_decide

theorem laneRows_length (bitStart cumPrev : Nat) :
  (laneRows bitStart cumPrev).length = 100 := by
  simp [laneRows, laneTemplateRows_length]

theorem tailRows_length (bitStarts : List Nat) (firstAllocated : Nat) :
  (tailRows bitStarts firstAllocated).length = 13885 := by
  simp [tailRows, tailTemplateRows_length]

end Nightstream.Implementation.R1CS.AlphabetSamplingResidualTemplate

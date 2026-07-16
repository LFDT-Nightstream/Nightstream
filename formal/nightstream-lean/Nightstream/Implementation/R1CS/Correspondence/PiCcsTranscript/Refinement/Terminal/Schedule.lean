import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Rows

/-!
Exact terminal-profile call tree for the `Pi_CCS` instance digest, authority
binding, and pre-SumCheck challenge schedule.

Assurance tier: implementation/R1CS structural correspondence. The four call
families and every descriptor are named independently, then checked against
the exact generated owner. Row order alone assigns no transcript semantics.

Owns: `instance digest -> binding -> main challenges -> beta_m`; all 25 exact
Poseidon2 call descriptors; family cardinalities; owner-piece addresses; and
closed equality between this tree and the generated terminal owner.

Does not own: input-pin values, Poseidon2 call acceptance, inter-call state
connectivity, instance-digest correctness, semantic challenge partitioning,
SumCheck, Rust conformance, cost totals, or row removal.

Emits constraints: no.

Authority boundary: a call descriptor is structural evidence only. Later
modules must independently replay each accepted call and derive every input
from verifier-owned values.

| Protocol | Phase | Constraint family | Calls | Exact structural obligation |
|---|---|---|---:|---|
| `Pi_CCS` | instance authority | instance-digest Poseidon2 | 7 | hash preimage path preceding transcript absorption |
| `Pi_CCS` | authority binding | raw-message boundaries | 6 | header, instance, running count, and checked-parent handle |
| `Pi_CCS` | main challenges | raw squeeze Poseidon2 | 7 | `alpha`, `beta_a`, `beta_r`, and `gamma` response stream |
| `Pi_CCS` | `beta_m` | raw squeeze Poseidon2 | 5 | NC-column response stream |
| `Pi_CCS` | complete tree | all call families | 25 | every current Poseidon2 call has one semantic owner |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Schedule

open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1000000

def emptyCall : Poseidon2Call.Call :=
  { rowStart := 0, rowEnd := 0, inputColumns := [], firstAllocatedColumn := 0 }

def instanceDigestCalls : List Poseidon2Call.Call :=
  [ { rowStart := 18, rowEnd := 618
      inputColumns := [1619061, 1619062, 1619063, 1619064,
        1619060, 1619060, 1619060, 1619060]
      firstAllocatedColumn := 1619065 },
    { rowStart := 622, rowEnd := 1222
      inputColumns := [1619665, 1619666, 1619667, 1619668,
        1619661, 1619662, 1619663, 1619664]
      firstAllocatedColumn := 1619669 },
    { rowStart := 1226, rowEnd := 1826
      inputColumns := [1620269, 1620270, 1620271, 1620272,
        1620265, 1620266, 1620267, 1620268]
      firstAllocatedColumn := 1620273 },
    { rowStart := 1830, rowEnd := 2430
      inputColumns := [1620873, 1620874, 1620875, 1620876,
        1620869, 1620870, 1620871, 1620872]
      firstAllocatedColumn := 1620877 },
    { rowStart := 2434, rowEnd := 3034
      inputColumns := [1621477, 1621478, 1621479, 1621480,
        1621473, 1621474, 1621475, 1621476]
      firstAllocatedColumn := 1621481 },
    { rowStart := 3035, rowEnd := 3635
      inputColumns := [1622081, 1622074, 1622075, 1622076,
        1622077, 1622078, 1622079, 1622080]
      firstAllocatedColumn := 1622082 },
    { rowStart := 3636, rowEnd := 4236
      inputColumns := [1622682, 1622675, 1622676, 1622677,
        1622678, 1622679, 1622680, 1622681]
      firstAllocatedColumn := 1622683 } ]

def bindingCalls : List Poseidon2Call.Call :=
  [ { rowStart := 4242, rowEnd := 4842
      inputColumns := [1132300, 1132301, 1623288, 1623287,
        1132304, 1132305, 1132306, 1132307]
      firstAllocatedColumn := 1623289 },
    { rowStart := 4842, rowEnd := 5442
      inputColumns := [1623283, 1623284, 1623285, 1623286,
        1623885, 1623886, 1623887, 1623888]
      firstAllocatedColumn := 1623889 },
    { rowStart := 5444, rowEnd := 6044
      inputColumns := [1624490, 1624489, 1623275, 1623276,
        1624485, 1624486, 1624487, 1624488]
      firstAllocatedColumn := 1624491 },
    { rowStart := 6046, rowEnd := 6646
      inputColumns := [1623277, 1623278, 1625092, 1625091,
        1625087, 1625088, 1625089, 1625090]
      firstAllocatedColumn := 1625093 },
    { rowStart := 6651, rowEnd := 7251
      inputColumns := [1625695, 1625693, 1625694, 1625697,
        1625689, 1625690, 1625691, 1625692]
      firstAllocatedColumn := 1625698 },
    { rowStart := 7251, rowEnd := 7851
      inputColumns := [1625696, 1619039, 1619040, 1619041,
        1626294, 1626295, 1626296, 1626297]
      firstAllocatedColumn := 1626298 } ]

def mainChallengeCalls : List Poseidon2Call.Call :=
  [ { rowStart := 7854, rowEnd := 8454
      inputColumns := [1619042, 1626898, 1626899, 1626900,
        1626894, 1626895, 1626896, 1626897]
      firstAllocatedColumn := 1626901 },
    { rowStart := 8455, rowEnd := 9055
      inputColumns := [1627501, 1627494, 1627495, 1627496,
        1627497, 1627498, 1627499, 1627500]
      firstAllocatedColumn := 1627502 },
    { rowStart := 9056, rowEnd := 9656
      inputColumns := [1628102, 1628095, 1628096, 1628097,
        1628098, 1628099, 1628100, 1628101]
      firstAllocatedColumn := 1628103 },
    { rowStart := 9657, rowEnd := 10257
      inputColumns := [1628703, 1628696, 1628697, 1628698,
        1628699, 1628700, 1628701, 1628702]
      firstAllocatedColumn := 1628704 },
    { rowStart := 10258, rowEnd := 10858
      inputColumns := [1629304, 1629297, 1629298, 1629299,
        1629300, 1629301, 1629302, 1629303]
      firstAllocatedColumn := 1629305 },
    { rowStart := 10859, rowEnd := 11459
      inputColumns := [1629905, 1629898, 1629899, 1629900,
        1629901, 1629902, 1629903, 1629904]
      firstAllocatedColumn := 1629906 },
    { rowStart := 11460, rowEnd := 12060
      inputColumns := [1630506, 1630499, 1630500, 1630501,
        1630502, 1630503, 1630504, 1630505]
      firstAllocatedColumn := 1630507 } ]

def betaMCalls : List Poseidon2Call.Call :=
  [ { rowStart := 12063, rowEnd := 12663
      inputColumns := [1631107, 1631108, 1631109, 1631102,
        1631103, 1631104, 1631105, 1631106]
      firstAllocatedColumn := 1631110 },
    { rowStart := 12664, rowEnd := 13264
      inputColumns := [1631710, 1631703, 1631704, 1631705,
        1631706, 1631707, 1631708, 1631709]
      firstAllocatedColumn := 1631711 },
    { rowStart := 13265, rowEnd := 13865
      inputColumns := [1632311, 1632304, 1632305, 1632306,
        1632307, 1632308, 1632309, 1632310]
      firstAllocatedColumn := 1632312 },
    { rowStart := 13866, rowEnd := 14466
      inputColumns := [1632912, 1632905, 1632906, 1632907,
        1632908, 1632909, 1632910, 1632911]
      firstAllocatedColumn := 1632913 },
    { rowStart := 14467, rowEnd := 15067
      inputColumns := [1633513, 1633506, 1633507, 1633508,
        1633509, 1633510, 1633511, 1633512]
      firstAllocatedColumn := 1633514 } ]

def instanceCount : Nat := 7
def bindingCount : Nat := 6
def mainChallengeCount : Nat := 7
def betaMCount : Nat := 5

theorem familyLengths :
    instanceDigestCalls.length = instanceCount /\
    bindingCalls.length = bindingCount /\
    mainChallengeCalls.length = mainChallengeCount /\
    betaMCalls.length = betaMCount /\
    instanceCount + bindingCount + mainChallengeCount + betaMCount = 25 := by
  decide

def instanceCall (index : Fin instanceCount) : Poseidon2Call.Call :=
  instanceDigestCalls.getD index.val emptyCall

def bindingCall (index : Fin bindingCount) : Poseidon2Call.Call :=
  bindingCalls.getD index.val emptyCall

def mainChallengeCall (index : Fin mainChallengeCount) : Poseidon2Call.Call :=
  mainChallengeCalls.getD index.val emptyCall

def betaMCall (index : Fin betaMCount) : Poseidon2Call.Call :=
  betaMCalls.getD index.val emptyCall

/-- Affine descriptor for the seven engine-response calls. The first input is
the only special boundary; all successors consume one squeeze pin followed by
the preceding call's final seven lanes. -/
def mainChallengeCallFormula
    (index : Fin mainChallengeCount) : Poseidon2Call.Call :=
  { rowStart := 7854 + 601 * index.val
    rowEnd := 8454 + 601 * index.val
    inputColumns :=
      if index.val = 0 then
        [1619042, 1626898, 1626899, 1626900,
         1626894, 1626895, 1626896, 1626897]
      else
        [1626900 + 601 * index.val,
         1626893 + 601 * index.val,
         1626894 + 601 * index.val,
         1626895 + 601 * index.val,
         1626896 + 601 * index.val,
         1626897 + 601 * index.val,
         1626898 + 601 * index.val,
         1626899 + 601 * index.val]
    firstAllocatedColumn := 1626901 + 601 * index.val }

/-- Affine descriptor for the five `beta_m` calls. The first input includes
the `[3]` domain boundary; successors have the uniform squeeze form. -/
def betaMCallFormula (index : Fin betaMCount) : Poseidon2Call.Call :=
  { rowStart := 12063 + 601 * index.val
    rowEnd := 12663 + 601 * index.val
    inputColumns :=
      if index.val = 0 then
        [1631107, 1631108, 1631109, 1631102,
         1631103, 1631104, 1631105, 1631106]
      else
        [1631109 + 601 * index.val,
         1631102 + 601 * index.val,
         1631103 + 601 * index.val,
         1631104 + 601 * index.val,
         1631105 + 601 * index.val,
         1631106 + 601 * index.val,
         1631107 + 601 * index.val,
         1631108 + 601 * index.val]
    firstAllocatedColumn := 1631110 + 601 * index.val }

theorem mainChallengeCall_formula : forall index : Fin mainChallengeCount,
    mainChallengeCall index = mainChallengeCallFormula index := by
  decide

theorem betaMCall_formula : forall index : Fin betaMCount,
    betaMCall index = betaMCallFormula index := by
  decide

def instancePieceIndex (index : Fin instanceCount) : Fin Rows.pieceCount :=
  ⟨1 + 2 * index.val, by
    have indexLt := index.isLt
    simp only [instanceCount, Rows.pieceCount] at indexLt ⊢
    omega⟩

def bindingPieceIndex (index : Fin bindingCount) : Fin Rows.pieceCount :=
  ⟨15 + 2 * index.val -
      (if index.val = 0 then 0 else if index.val = 5 then 2 else 1), by
    have indexLt := index.isLt
    simp only [bindingCount, Rows.pieceCount] at indexLt ⊢
    by_cases first : index.val = 0
    · simp [first]
    by_cases last : index.val = 5
    · simp [last]
    · simp [last]
      omega⟩

def mainChallengePieceIndex
    (index : Fin mainChallengeCount) : Fin Rows.pieceCount :=
  ⟨25 + 2 * index.val, by
    have indexLt := index.isLt
    simp only [mainChallengeCount, Rows.pieceCount] at indexLt ⊢
    omega⟩

def betaMPieceIndex (index : Fin betaMCount) : Fin Rows.pieceCount :=
  ⟨39 + 2 * index.val, by
    have indexLt := index.isLt
    simp only [betaMCount, Rows.pieceCount] at indexLt ⊢
    omega⟩

def instancePiece (index : Fin instanceCount) : Piece :=
  Rows.pieceAt (instancePieceIndex index)

def bindingPiece (index : Fin bindingCount) : Piece :=
  Rows.pieceAt (bindingPieceIndex index)

def mainChallengePiece (index : Fin mainChallengeCount) : Piece :=
  Rows.pieceAt (mainChallengePieceIndex index)

def betaMPiece (index : Fin betaMCount) : Piece :=
  Rows.pieceAt (betaMPieceIndex index)

def instanceOwnerStarts : List Nat :=
  [1568735, 1569339, 1569943, 1570547, 1571151, 1571752, 1572353]
def instanceOwnerEnds : List Nat :=
  [1569335, 1569939, 1570543, 1571147, 1571751, 1572352, 1572953]
def bindingOwnerStarts : List Nat :=
  [1572959, 1573559, 1574161, 1574763, 1575368, 1575968]
def bindingOwnerEnds : List Nat :=
  [1573559, 1574159, 1574761, 1575363, 1575968, 1576568]
def mainChallengeOwnerStarts : List Nat :=
  [1576571, 1577172, 1577773, 1578374,
   1578975, 1579576, 1580177]
def mainChallengeOwnerEnds : List Nat :=
  [1577171, 1577772, 1578373, 1578974,
   1579575, 1580176, 1580777]
def betaMOwnerStarts : List Nat :=
  [1580780, 1581381, 1581982, 1582583, 1583184]
def betaMOwnerEnds : List Nat :=
  [1581380, 1581981, 1582582, 1583183, 1583784]

def expectedInstancePiece (index : Fin instanceCount) : Piece :=
  { rowStart := instanceOwnerStarts.getD index.val 0
    rowEnd := instanceOwnerEnds.getD index.val 0
    payload := .poseidon (instanceCall index) }

def expectedBindingPiece (index : Fin bindingCount) : Piece :=
  { rowStart := bindingOwnerStarts.getD index.val 0
    rowEnd := bindingOwnerEnds.getD index.val 0
    payload := .poseidon (bindingCall index) }

def expectedMainChallengePiece
    (index : Fin mainChallengeCount) : Piece :=
  { rowStart := mainChallengeOwnerStarts.getD index.val 0
    rowEnd := mainChallengeOwnerEnds.getD index.val 0
    payload := .poseidon (mainChallengeCall index) }

def expectedBetaMPiece (index : Fin betaMCount) : Piece :=
  { rowStart := betaMOwnerStarts.getD index.val 0
    rowEnd := betaMOwnerEnds.getD index.val 0
    payload := .poseidon (betaMCall index) }

/-- Closed protocol/phase/call tree over all 25 current terminal calls. -/
theorem phaseTree_eq :
    (forall index : Fin instanceCount,
      instancePiece index = expectedInstancePiece index) /\
    (forall index : Fin bindingCount,
      bindingPiece index = expectedBindingPiece index) /\
    (forall index : Fin mainChallengeCount,
      mainChallengePiece index = expectedMainChallengePiece index) /\
    (forall index : Fin betaMCount,
      betaMPiece index = expectedBetaMPiece index) := by
  decide

theorem instancePiece_eq (index : Fin instanceCount) :
    instancePiece index = expectedInstancePiece index :=
  phaseTree_eq.1 index

theorem bindingPiece_eq (index : Fin bindingCount) :
    bindingPiece index = expectedBindingPiece index :=
  phaseTree_eq.2.1 index

theorem mainChallengePiece_eq (index : Fin mainChallengeCount) :
    mainChallengePiece index = expectedMainChallengePiece index :=
  phaseTree_eq.2.2.1 index

theorem betaMPiece_eq (index : Fin betaMCount) :
    betaMPiece index = expectedBetaMPiece index :=
  phaseTree_eq.2.2.2 index

theorem instancePiece_mem (index : Fin instanceCount) :
    instancePiece index ∈
      FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces :=
  Rows.pieceAt_mem _

theorem bindingPiece_mem (index : Fin bindingCount) :
    bindingPiece index ∈
      FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces :=
  Rows.pieceAt_mem _

theorem mainChallengePiece_mem (index : Fin mainChallengeCount) :
    mainChallengePiece index ∈
      FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces :=
  Rows.pieceAt_mem _

theorem betaMPiece_mem (index : Fin betaMCount) :
    betaMPiece index ∈
      FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces :=
  Rows.pieceAt_mem _

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Schedule

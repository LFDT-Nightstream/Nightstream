import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Profile

/-!
Contract: expose the three authoritative operand bundles and single output
bundle of one typed `nifsVerify` call frame.

The accessors preserve the exact reference order declared by
`Vocabulary.callInputs`.  The decoding and encoding theorems merely unpack
`RefBundles`; they do not introduce a second operand view or any semantic
acceptance premise.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

universe u

def runningOperand
    {parameters : Parameters}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (bundles :
      RefBundles
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    ColumnBundle running.port.layout :=
  match bundles with
  | .cons runningBundle (.cons _ (.cons _ .nil)) => runningBundle

def freshOperand
    {parameters : Parameters}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (bundles :
      RefBundles
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    ColumnBundle fresh.port.layout :=
  match bundles with
  | .cons _ (.cons freshBundle (.cons _ .nil)) => freshBundle

def proofOperand
    {parameters : Parameters}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (bundles :
      RefBundles
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    ColumnBundle proof.port.layout :=
  match bundles with
  | .cons _ (.cons _ (.cons proofBundle .nil)) => proofBundle

@[simp] theorem operand_ids
    {parameters : Parameters}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (bundles :
      RefBundles
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    (runningOperand bundles).ids ++
        ((freshOperand bundles).ids ++ (proofOperand bundles).ids) =
      bundles.ids := by
  cases bundles with
  | cons runningBundle tail =>
      cases tail with
      | cons freshBundle tail =>
          cases tail with
          | cons proofBundle tail =>
              cases tail
              simp [runningOperand, freshOperand, proofOperand,
                RefBundles.ids, RefBundles.columns,
                RefBundles.portColumns, ColumnBundle.ids]

theorem decodes_iff
    {parameters : Parameters}
    (family : Family (typeSystem parameters))
    (assignment : ColumnId → Field)
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (bundles :
      RefBundles
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (running : parameters.Running)
    (fresh : parameters.Fresh)
    (proof : parameters.NifsProof) :
    bundles.Decodes family assignment
        (.cons running (.cons fresh (.cons proof .nil))) ↔
      (runningOperand bundles).Decodes family (.data .running)
          assignment running ∧
        (freshOperand bundles).Decodes family (.data .fresh)
            assignment fresh ∧
          (proofOperand bundles).Decodes family (.data .nifsProof)
            assignment proof := by
  cases bundles with
  | cons runningBundle tail =>
      cases tail with
      | cons freshBundle tail =>
          cases tail with
          | cons proofBundle tail =>
              cases tail
              simp [RefBundles.Decodes, runningOperand, freshOperand,
                proofOperand]

theorem encodes_iff
    {parameters : Parameters}
    (family : Family (typeSystem parameters))
    (assignment : ColumnId → Field)
    {context : Schema (typeSystem parameters)}
    {runningRef :
      Ref (typeSystem parameters) context (.data .running)}
    {freshRef :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (bundles :
      RefBundles
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (running : parameters.Running)
    (fresh : parameters.Fresh)
    (proof : parameters.NifsProof) :
    bundles.Encodes family assignment
        (.cons running (.cons fresh (.cons proof .nil))) ↔
      (runningOperand bundles).Encodes family (.data .running)
          assignment running ∧
        (freshOperand bundles).Encodes family (.data .fresh)
            assignment fresh ∧
          (proofOperand bundles).Encodes family (.data .nifsProof)
            assignment proof := by
  cases bundles with
  | cons runningBundle tail =>
      cases tail with
      | cons freshBundle tail =>
          cases tail with
          | cons proofBundle tail =>
              cases tail
              simp [RefBundles.Encodes, runningOperand, freshOperand,
                proofOperand]

theorem running_widthsAgree
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    (family.codecFor (.data .running)).width =
      running.port.layout.owners.length :=
  frame.operandWidthsAgree.1

theorem fresh_widthsAgree
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    (family.codecFor (.data .fresh)).width =
      fresh.port.layout.owners.length :=
  frame.operandWidthsAgree.2.1

theorem proof_widthsAgree
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    (family.codecFor (.data .nifsProof)).width =
      proof.port.layout.owners.length :=
  frame.operandWidthsAgree.2.2.1

/-- Exact physical width of the running operand. -/
theorem running_operand_ids_length
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    (runningOperand frame.operands).ids.length =
      (family.codecFor (.data .running)).width := by
  simp only [ColumnBundle.ids, List.length_map]
  rw [(runningOperand frame.operands).length_eq,
    ← running_widthsAgree frame]

/-- Exact physical width of the fresh operand. -/
theorem fresh_operand_ids_length
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    (freshOperand frame.operands).ids.length =
      (family.codecFor (.data .fresh)).width := by
  simp only [ColumnBundle.ids, List.length_map]
  rw [(freshOperand frame.operands).length_eq,
    ← fresh_widthsAgree frame]

/-- Exact physical width of the proof operand. -/
theorem proof_operand_ids_length
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    (proofOperand frame.operands).ids.length =
      (family.codecFor (.data .nifsProof)).width := by
  simp only [ColumnBundle.ids, List.length_map]
  rw [(proofOperand frame.operands).length_eq,
    ← proof_widthsAgree frame]

/-- The operand namespace has exactly the sum of the three selected codec
widths. -/
theorem operand_ids_length
    {parameters : Parameters}
    {family : Family (typeSystem parameters)}
    {context : Schema (typeSystem parameters)}
    {running :
      Ref (typeSystem parameters) context (.data .running)}
    {fresh :
      Ref (typeSystem parameters) context (.data .fresh)}
    {proof :
      Ref (typeSystem parameters) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature parameters) family Call.nifsVerify
        (Refs.cons running (Refs.cons fresh (Refs.cons proof .nil)))) :
    frame.operands.ids.length =
      (family.codecFor (.data .running)).width +
        (family.codecFor (.data .fresh)).width +
        (family.codecFor (.data .nifsProof)).width := by
  have lengths :=
    congrArg List.length (operand_ids frame.operands)
  simp only [List.length_append, ColumnBundle.ids,
    List.length_map] at lengths
  rw [(runningOperand frame.operands).length_eq,
    (freshOperand frame.operands).length_eq,
    (proofOperand frame.operands).length_eq] at lengths
  rw [← running_widthsAgree frame, ← fresh_widthsAgree frame,
    ← proof_widthsAgree frame] at lengths
  simpa only [Nat.add_assoc] using lengths.symm

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame

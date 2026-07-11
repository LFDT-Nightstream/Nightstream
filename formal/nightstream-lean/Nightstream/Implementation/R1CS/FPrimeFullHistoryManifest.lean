import Nightstream.Implementation.R1CS.FPrimeFullHistoryManifestData

/-!
Contract: kernel-checked ownership of the smallest complete production
full-history F' audit profile.

The generated data comes from a two-batch plain/stateless execution: one base
step, one recursive step, their state link, a terminal fold and continuity
link, public-image pins, and direct terminal CE closure.  Coverage proves that
no row is outside those owners.  It is an ownership theorem, not yet a claim
that each owner implements its protocol predicate.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest

open Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

theorem topLevel_covers_program :
    covers 0 totalRows topLevelFamilies = true := by
  decide

theorem topLevel_row_count :
    (topLevelFamilies.map RowRange.rowCount).sum = totalRows := by
  decide

theorem exact_owner_schedule :
    topLevelFamilies.map RowRange.name =
      [ "decider.step.base"
      , "decider.step.recursive"
      , "decider.state_link"
      , "decider.terminal_fold"
      , "decider.terminal_continuity"
      , "decider.public_pins"
      , "decider.terminal_ce" ] := by
  decide

theorem recursive_covers_top_owner :
    covers (topLevelFamilies[1]!).rowStart
      (topLevelFamilies[1]!).rowEnd recursiveFamilies = true := by
  decide

theorem recursiveNifs_covers_recursive_owner :
    covers (recursiveFamilies[2]!).rowStart
      (recursiveFamilies[2]!).rowEnd recursiveNifsFamilies = true := by
  decide

theorem terminal_covers_top_owner :
    covers (topLevelFamilies[3]!).rowStart
      (topLevelFamilies[3]!).rowEnd terminalFamilies = true := by
  decide

theorem terminalNifs_covers_terminal_owner :
    covers (terminalFamilies[0]!).rowStart
      (terminalFamilies[0]!).rowEnd terminalNifsFamilies = true := by
  decide

theorem recursivePiCcs_covers_owner :
    covers (recursiveNifsFamilies[0]!).rowStart
      (recursiveNifsFamilies[0]!).rowEnd recursivePiCcsFamilies = true := by
  decide

theorem recursivePiRlc_covers_owner :
    covers (recursiveNifsFamilies[1]!).rowStart
      (recursiveNifsFamilies[1]!).rowEnd recursivePiRlcFamilies = true := by
  decide

theorem terminalPiCcs_covers_owner :
    covers (terminalNifsFamilies[1]!).rowStart
      (terminalNifsFamilies[1]!).rowEnd terminalPiCcsFamilies = true := by
  decide

theorem terminalPiRlc_covers_owner :
    covers (terminalNifsFamilies[2]!).rowStart
      (terminalNifsFamilies[2]!).rowEnd terminalPiRlcFamilies = true := by
  decide

theorem exact_recursive_schedule :
    recursiveFamilies.map RowRange.name =
      [ "fprime.recursive.prelude"
      , "fprime.recursive.transcript"
      , "fprime.recursive.nifs"
      , "fprime.recursive.prior_link"
      , "fprime.recursive.nebula"
      , "fprime.recursive.accumulator"
      , "fprime.recursive.counter"
      , "fprime.recursive.output" ] := by
  decide

theorem exact_recursiveNifs_schedule :
    recursiveNifsFamilies.map RowRange.name =
      [ "nifs.pi_ccs", "nifs.pi_rlc", "nifs.pi_dec", "nifs.point_binding" ] := by
  decide

theorem exact_terminal_schedule :
    terminalFamilies.map RowRange.name =
      [ "terminal.nifs"
      , "terminal.running_link"
      , "terminal.parent_link"
      , "terminal.latest_link"
      , "terminal.accumulator" ] := by
  decide

theorem exact_terminalNifs_schedule :
    terminalNifsFamilies.map RowRange.name =
      [ "terminal.transcript"
      , "nifs.pi_ccs"
      , "nifs.pi_rlc"
      , "nifs.pi_dec"
      , "nifs.point_binding" ] := by
  decide

theorem exact_piCcs_schedule :
    recursivePiCcsFamilies.map RowRange.name =
        terminalPiCcsFamilies.map RowRange.name ∧
      recursivePiCcsFamilies.map RowRange.name =
        [ "nifs.pi_ccs.allocation"
        , "nifs.pi_ccs.authority"
        , "nifs.pi_ccs.fresh_digests"
        , "nifs.pi_ccs.running_authority"
        , "nifs.pi_ccs.transcript"
        , "nifs.pi_ccs.fe_initial"
        , "nifs.pi_ccs.fe_sumcheck"
        , "nifs.pi_ccs.nc_sumcheck"
        , "nifs.pi_ccs.output_binding"
        , "nifs.pi_ccs.fe_terminal"
        , "nifs.pi_ccs.nc_terminal"
        , "nifs.pi_ccs.catchup" ] := by
  decide

theorem exact_piRlc_schedule :
    recursivePiRlcFamilies.map RowRange.name =
        terminalPiRlcFamilies.map RowRange.name ∧
      recursivePiRlcFamilies.map RowRange.name =
        [ "nifs.pi_rlc.transcript_rhos"
        , "nifs.pi_rlc.shape"
        , "nifs.pi_rlc.linear_folds"
        , "nifs.pi_rlc.projection_binding"
        , "nifs.pi_rlc.projection_shared"
        , "nifs.pi_rlc.projection_identities" ] := by
  decide

theorem nested_row_counts_match_parents :
    (recursiveFamilies.map RowRange.rowCount).sum =
        (topLevelFamilies[1]!).rowCount ∧
    (recursiveNifsFamilies.map RowRange.rowCount).sum =
        (recursiveFamilies[2]!).rowCount ∧
    (terminalFamilies.map RowRange.rowCount).sum =
        (topLevelFamilies[3]!).rowCount ∧
    (terminalNifsFamilies.map RowRange.rowCount).sum =
        (terminalFamilies[0]!).rowCount ∧
    (recursivePiCcsFamilies.map RowRange.rowCount).sum =
        (recursiveNifsFamilies[0]!).rowCount ∧
    (recursivePiRlcFamilies.map RowRange.rowCount).sum =
        (recursiveNifsFamilies[1]!).rowCount ∧
    (terminalPiCcsFamilies.map RowRange.rowCount).sum =
        (terminalNifsFamilies[1]!).rowCount ∧
    (terminalPiRlcFamilies.map RowRange.rowCount).sum =
        (terminalNifsFamilies[2]!).rowCount := by
  decide

theorem stateless_nebula_owner_is_empty :
    (recursiveFamilies[4]!).name = "fprime.recursive.nebula" ∧
    (recursiveFamilies[4]!).rowCount = 0 := by
  decide

theorem every_owner_nonempty :
    ∀ owner ∈ topLevelFamilies, owner.rowStart < owner.rowEnd := by
  decide

theorem profile_shape :
    schemaVersion = 2 ∧ totalRows = 4076614 ∧ totalColumns = 3298653 := by
  decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest

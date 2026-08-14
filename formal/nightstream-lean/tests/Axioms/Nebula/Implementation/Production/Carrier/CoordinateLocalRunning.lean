import Nightstream.Implementation.Nebula.Production.Carrier.CoordinateLocalRunning
import tests.Axioms.Support

/-! Dependency audit for the coordinate-local running-claim view. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.toRunning_ofRunning' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.toRunning_ofRunning

/-- info: 'Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.ofRunning_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.ofRunning_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.coordinateLocalCodec_width' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.coordinateLocalCodec_width

/-- info: 'Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.decodeRunning_encodeRunning' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.decodeRunning_encodeRunning

/-- info: 'Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.encodeRunning_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.encodeRunning_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.totalFieldCount_eq_runningFieldCountFor' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning.totalFieldCount_eq_runningFieldCountFor

/-! `totalFieldCount_r26` and the 810-field window are closed arithmetic. -/

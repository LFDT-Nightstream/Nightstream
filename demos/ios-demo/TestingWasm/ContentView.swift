import SwiftUI
import UniformTypeIdentifiers

#if canImport(UIKit)
import UIKit
#endif

struct ContentView: View {
    private enum DetailSection: String, CaseIterable, Identifiable {
        case circuit
        case output
        case metal

        var id: String { rawValue }

        var title: String {
            switch self {
            case .circuit:
                return "Circuit"
            case .output:
                return "Output"
            case .metal:
                return "Metal"
            }
        }
    }

    @StateObject private var vm = WasmViewModel()
    @StateObject private var metalBenchmark = MetalBenchmarkService()
    @State private var isFileImporterPresented = false
    @State private var isJsonEditorPresented = false
    @State private var selectedSection: DetailSection = .circuit

    private var availableBackends: [WasmViewModel.ProverBackend] {
        NeoFoldNativeService.isAvailable ? WasmViewModel.ProverBackend.allCases : [.wasm]
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                    statusRow
                    controlsPanel
                    sectionPicker
                    sectionContent
                }
                .padding()
            }
            .navigationTitle("Nightstream wasm demo")
            .navigationBarTitleDisplayMode(.inline)
        }
        .background(
            NeoFoldWasmWebViewHost()
                .frame(width: 1, height: 1)
                .opacity(0.01)
                .allowsHitTesting(false)
        )
        .task { await vm.onAppear() }
        .fileImporter(
            isPresented: $isFileImporterPresented,
            allowedContentTypes: [UTType.json],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                vm.importJsonFile(from: url)
                selectedSection = .circuit
            case .failure(let error):
                vm.appendLog("ERROR: \(error.localizedDescription)")
            }
        }
        .sheet(isPresented: $isJsonEditorPresented) {
            JsonEditorSheet(
                title: vm.circuitSource.isEmpty ? "Circuit JSON" : vm.circuitSource,
                initialText: vm.circuitJson,
                onSave: { newJson in
                    vm.setCircuitJsonFromEditor(newJson)
                }
            )
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Nightstream wasm demo")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Lattice-based folding (neo-fold) with Twist/Shout memory, running via WebAssembly.")
                .foregroundStyle(.secondary)

            Text(
                "Paste or load a circuit JSON (R1CS matrices + per-step witness) and run folding prove+verify."
            )
            .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                Link("Learn more", destination: URL(string: "https://github.com/nicarq/nightstream/blob/main/README.md")!)
                Link("GitHub repo", destination: URL(string: "https://github.com/LFDT-Nightstream/Nightstream/")!)
            }
            .font(.subheadline)
        }
    }

    private var statusRow: some View {
        HStack(spacing: 10) {
            Badge(text: vm.wasmStatus)
            Badge(text: vm.threadsStatus, kind: .warning)
            if vm.isRunning {
                Badge(text: "Run: in progress", kind: .warning)
            }
            if metalBenchmark.isRunning {
                Badge(text: "Metal: profiling", kind: .warning)
            }
        }
    }

    private var controlsPanel: some View {
        Panel {
            VStack(alignment: .leading, spacing: 12) {
                let columns = [GridItem(.adaptive(minimum: 180), spacing: 10, alignment: .leading)]
                LazyVGrid(columns: columns, alignment: .leading, spacing: 10) {
                    exampleButton(.toySquare)
                    exampleButton(.toySquareFolding8Steps)
                    exampleButton(.poseidon2Batch1)
                    Button("Load JSON file") { isFileImporterPresented = true }
                    Button("Prove + Verify") {
                        selectedSection = .output
                        vm.runProveVerify()
                    }
                        .buttonStyle(.borderedProminent)
                        .disabled(vm.isRunning)
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("Backend")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    Picker("Backend", selection: $vm.proverBackend) {
                        ForEach(availableBackends) { backend in
                            Text(backend.title).tag(backend)
                        }
                    }
                    .pickerStyle(.segmented)

                    if !NeoFoldNativeService.isAvailable {
                        Text("Native prover backend is not linked in the Metal benchmark target.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Toggle("Compress with Spartan2 (experimental)", isOn: $vm.compressSpartan)
                if vm.proverBackend.includesNative {
                    Text("Native Spartan2 requires `NeoFoldFFI.xcframework` built with `--features spartan`.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                HStack(spacing: 10) {
                    if let url = vm.spartanShareURL {
                        ShareLink(item: url) {
                            Text("Share Spartan SNARK")
                        }
                        .buttonStyle(.bordered)
                    } else {
                        Button("Share Spartan SNARK") {}
                            .buttonStyle(.bordered)
                            .disabled(true)
                    }
                }
            }
        }
    }

    private func exampleButton(_ example: WasmViewModel.Example) -> some View {
        let isSelected = vm.selectedExample == example

        return Button {
            vm.loadExample(example)
            selectedSection = .circuit
        } label: {
            HStack(spacing: 8) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
                Text(example.title)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .buttonStyle(.bordered)
    }

    private var sectionPicker: some View {
        Picker("Section", selection: $selectedSection) {
            ForEach(DetailSection.allCases) { section in
                Text(section.title).tag(section)
            }
        }
        .pickerStyle(.segmented)
    }

    @ViewBuilder
    private var sectionContent: some View {
        switch selectedSection {
        case .circuit:
            circuitPanel
        case .output:
            outputPanel
        case .metal:
            metalBenchmarkPanel
        }
    }

    private var metalBenchmarkPanel: some View {
        Panel {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    Button {
                        metalBenchmark.run(.quick)
                    } label: {
                        Label("Quick profile", systemImage: "bolt")
                    }
                    .buttonStyle(.bordered)
                    .disabled(!MetalBenchmarkService.isAvailable || metalBenchmark.isRunning)

                    Button {
                        metalBenchmark.run(.full)
                    } label: {
                        Label("M0 - M5", systemImage: "gauge.with.dots.needle.67percent")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!MetalBenchmarkService.isAvailable || metalBenchmark.isRunning)

                    Button {
                        metalBenchmark.run(.m6)
                    } label: {
                        Label("M6 soak", systemImage: "timer")
                    }
                    .buttonStyle(.bordered)
                    .disabled(!MetalBenchmarkService.isAvailable || metalBenchmark.isRunning)

                    if metalBenchmark.isRunning {
                        ProgressView()
                    }
                }

                if !MetalBenchmarkService.isAvailable {
                    Text("NeoMetalBench.xcframework unavailable")
                        .foregroundStyle(.secondary)
                } else if let error = metalBenchmark.errorMessage {
                    Text(error)
                        .font(.system(.footnote, design: .monospaced))
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                } else if let report = metalBenchmark.report {
                    metalReport(report)
                } else {
                    Text("No Metal profile")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func metalReport(_ report: MetalBenchmarkReport) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(report.device.gpuName)
                    .font(.headline)
                Spacer()
                Badge(text: report.m1ParityPassed ? "Parity: pass" : "Parity: fail", kind: report.m1ParityPassed ? .normal : .error)
                Badge(text: report.m2LifecyclePassed ? "Proof: pass" : "Proof: fail", kind: report.m2LifecyclePassed ? .normal : .error)
                Badge(text: report.m3ResidencyPassed ? "Resident: pass" : "Resident: fail", kind: report.m3ResidencyPassed ? .normal : .error)
                Badge(text: report.m4ProjectionPassed ? "DEC: Metal" : "DEC: open", kind: report.m4ProjectionPassed ? .normal : .warning)
                Badge(text: report.m5AdapterPassed ? "Adapter: pass" : "Adapter: open", kind: report.m5AdapterPassed ? .normal : .warning)
                Badge(text: report.m6Passed ? "M6: pass" : "M6: open", kind: report.m6Passed ? .normal : .warning)
            }

            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 7) {
                GridRow {
                    Text("Primitive")
                    Text("CPU")
                    Text("Metal")
                    Text("Speedup")
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                ForEach(report.primitives) { primitive in
                    let selected = primitive.candidates.first { $0.name == primitive.selectedCandidate }
                    GridRow {
                        Text(primitive.name)
                        Text(metalMs(primitive.cpu.medianMs))
                        Text(metalMs(selected?.timing.medianMs))
                        Text(String(format: "%.2fx", primitive.selectedSpeedupOverCpu))
                    }
                }
            }
            .font(.system(.caption, design: .monospaced))

            if !report.lifecycle.isEmpty {
                Divider()
                ForEach(report.lifecycle) { lifecycle in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("\(lifecycle.name) [\(lifecycle.backend)]")
                            Spacer()
                            Text("prove \(metalMs(lifecycle.online.medianMs))")
                            Text("verify \(metalMs(lifecycle.verifyMs.medianMs))")
                        }
                        if let pipeline = lifecycle.pipeline {
                            Text("synthesis \(metalMs(pipeline.synthesisWork.medianMs))  fold \(metalMs(pipeline.foldWork.medianMs))  final \(metalMs(pipeline.finalMaterialization.medianMs))  overlap \(metalMs(pipeline.overlapSaved.medianMs))")
                        }
                        if let profile = lifecycle.nifsProfile {
                            Text("NIFS \(metalMs(profile.total.medianMs))  CCS \(metalMs(profile.piCcs.medianMs))  RLC \(metalMs(profile.piRlc.medianMs))  DEC \(metalMs(profile.piDec.medianMs))")
                            Text("DEC forms \(metalMs(profile.decFormBuild.medianMs))  projection \(metalMs(profile.decProjection.medianMs))  host \(metalMs(profile.decHostMaterialization.medianMs))")
                            Text("route FE:\(route(profile.feOnMetal)) NC:\(route(profile.ncOnMetal)) RLC:\(route(profile.rlcWitnessOnMetal)) rho-i8:\(route(profile.rlcRhoSmallCoefficients)) resident-only:\(route(profile.rlcWitnessResidentOnly)) split:\(route(profile.decSplitOnMetal)) recompose:\(route(profile.decRecompositionOnMetal)) forms:\(route(profile.decFormsOnMetal)) y:\(route(profile.decYOnMetal)) commit:\(route(profile.decCommitOnMetal))")
                            Text("resident \(profile.residentInputFolds)/\(profile.residentOutputFolds)  deferred \(profile.deferredProofFolds)/\(profile.deferredRunningFolds)  replay:\(profile.recursiveCompileReverifyRequired ? "required" : "skipped")")
                            Text("commands \(profile.activityPerSample.commandBuffers)  dispatches \(profile.activityPerSample.dispatches)  waits \(profile.activityPerSample.hostWaits)")
                                .foregroundStyle(.secondary)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                }
            }

            ForEach(report.lifecycleCrossover) { crossover in
                Text("\(crossover.name) crossover median \(String(format: "%.2fx", crossover.medianSpeedupOverCpu))  p95 \(String(format: "%.2fx", crossover.p95SpeedupOverCpu))  parity:\(crossover.proofParityOk ? "pass" : "fail")")
                    .font(.system(.caption, design: .monospaced))
            }

            if let sustained = report.sustained {
                Text("sustained \(sustained.secondsPerBackend)s/backend  CPU \(String(format: "%.3f", sustained.cpuProofsPerSecond)) proof/s  Metal \(String(format: "%.3f", sustained.metalProofsPerSecond)) proof/s  \(String(format: "%.2fx", sustained.speedupOverCpu))")
                    .font(.system(.caption, design: .monospaced))
            }

            DisclosureGroup("Raw report") {
                ScrollView {
                    Text(metalBenchmark.rawJSON)
                        .font(.system(.caption2, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 180)
            }
        }
    }

    private func metalMs(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.3f ms", value)
    }

    private func route(_ onMetal: Bool) -> String {
        onMetal ? "GPU" : "CPU"
    }

    private var circuitPanel: some View {
        Panel {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Circuit JSON")
                        .font(.headline)
                    Spacer()
                    if vm.circuitSizeBytes > 0 {
                        Text("\(vm.circuitSizeBytes) bytes")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                if !vm.circuitSource.isEmpty {
                    Text(vm.circuitSource)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                if vm.isCircuitLarge {
                    HStack(spacing: 10) {
                        Button("Edit JSON") { isJsonEditorPresented = true }
                        Spacer()
                    }

                    ScrollView {
                        Text(vm.circuitPreview.isEmpty ? "No JSON provided." : vm.circuitPreview)
                            .font(.system(.footnote, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(minHeight: 240)
                } else {
                    ZStack(alignment: .topLeading) {
                        if vm.circuitJson.isEmpty {
                            Text("Paste circuit JSON here…")
                                .foregroundStyle(.secondary)
                                .padding(.top, 10)
                                .padding(.leading, 6)
                        }

                        TextEditor(text: $vm.circuitJson)
                            .font(.system(.body, design: .monospaced))
                            .frame(minHeight: 240)
                            .onChange(of: vm.circuitJson) { _, newValue in
                                vm.circuitSizeBytes = newValue.utf8.count
                                vm.circuitPreview = String(newValue.prefix(6_000))
                                if vm.circuitSource.isEmpty {
                                    vm.circuitSource = "Custom JSON"
                                }
                            }
                    }
                }
            }
        }
    }

    private var outputPanel: some View {
        Panel {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Button("Clear output") { vm.clearLog() }
                    Button("Copy output") { copyToPasteboard(vm.logText) }
                        .disabled(vm.logText.isEmpty)
                    Spacer()
                }

                if vm.lastWasmTimings != nil || vm.lastNativeTimings != nil {
                    timingsSummary
                    Divider()
                }

                ScrollView {
                    Text(vm.logText.isEmpty ? "Output will appear here." : vm.logText)
                        .font(.system(.footnote, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 240)
            }
        }
    }

    private var timingsSummary: some View {
        let wasm = vm.lastWasmTimings
        let native = vm.lastNativeTimings
        let showSpartan = (wasm?.spartanProveMs != nil) || (native?.spartanProveMs != nil)

        var rows: [(String, String, String)] = [
            ("Session ready", fmtMs(wasm?.sessionCreateMs), fmtMs(native?.sessionCreateMs)),
            ("Ajtai setup", fmtMs(wasm?.ajtaiSetupMs), fmtMs(native?.ajtaiSetupMs)),
            ("Build CCS", fmtMs(wasm?.buildCcsMs), fmtMs(native?.buildCcsMs)),
            ("Session init", fmtMs(wasm?.sessionInitMs), fmtMs(native?.sessionInitMs)),
            ("Add steps", fmtMs(wasm?.addStepsMs), fmtMs(native?.addStepsMs)),
            ("Fold+Prove", fmtMs(wasm?.foldAndProveMs), fmtMs(native?.foldAndProveMs)),
            ("Verify", fmtMs(wasm?.verifyMs), fmtMs(native?.verifyMs)),
            ("Total", fmtMs(wasm?.totalMs), fmtMs(native?.totalMs)),
        ]

        if showSpartan {
            rows.append(("Spartan prove", fmtMs(wasm?.spartanProveMs), fmtMs(native?.spartanProveMs)))
            rows.append(("Spartan verify", fmtMs(wasm?.spartanVerifyMs), fmtMs(native?.spartanVerifyMs)))
            rows.append(("Spartan ok", fmtBool(wasm?.spartanVerifyOk), fmtBool(native?.spartanVerifyOk)))
            rows.append(("Spartan snark", fmtBytes(wasm?.spartanSnarkBytes), fmtBytes(native?.spartanSnarkBytes)))
            rows.append(("Spartan vk", fmtBytes(wasm?.spartanVkBytes), fmtBytes(native?.spartanVkBytes)))
            rows.append(("Spartan total", fmtBytes(wasm?.spartanVkAndSnarkBytes), fmtBytes(native?.spartanVkAndSnarkBytes)))
        }

        rows.append(("Verify ok", fmtBool(wasm?.verifyOk), fmtBool(vm.lastNativeVerifyOk)))
        rows.append(("Steps", fmtInt(wasm?.steps), fmtInt(vm.lastNativeSteps)))

        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("Summary")
                    .font(.headline)
                Spacer()
                if vm.lastRunInputBytes > 0 {
                    Text("\(vm.lastRunInputBytes) bytes")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                GridRow {
                    Text("Phase")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("WASM")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Native")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                ForEach(rows, id: \.0) { row in
                    GridRow {
                        Text(row.0)
                        Text(row.1)
                            .monospacedDigit()
                        Text(row.2)
                            .monospacedDigit()
                    }
                }
            }
            .font(.system(.footnote, design: .monospaced))
        }
        .textSelection(.enabled)
    }

    private func fmtMs(_ ms: Double?) -> String {
        guard let ms, ms.isFinite else { return "—" }
        return String(format: "%.1f ms", ms)
    }

    private func fmtBool(_ value: Bool?) -> String {
        guard let value else { return "—" }
        return value ? "true" : "false"
    }

    private func fmtInt(_ value: Int?) -> String {
        guard let value else { return "—" }
        return String(value)
    }

    private func fmtBytes(_ bytes: Int?) -> String {
        guard let bytes else { return "—" }
        if bytes < 1024 {
            return "\(bytes) B"
        }
        let kb = Double(bytes) / 1024.0
        if kb < 1024.0 {
            return String(format: "%.2f KB", kb)
        }
        let mb = kb / 1024.0
        return String(format: "%.2f MB", mb)
    }

    private func copyToPasteboard(_ text: String) {
        #if canImport(UIKit)
        UIPasteboard.general.string = text
        #endif
    }
}

#Preview {
    ContentView()
}

private struct Panel<Content: View>: View {
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .padding(12)
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(.quaternary, lineWidth: 1)
            )
    }
}

private struct Badge: View {
    enum Kind {
        case normal
        case warning
        case error
    }

    let text: String
    var kind: Kind = .normal

    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(foreground)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(
                Capsule(style: .continuous)
                    .fill(.clear)
                    .overlay(Capsule(style: .continuous).stroke(border, lineWidth: 1))
            )
    }

    private var foreground: Color {
        switch kind {
        case .normal:
            return .secondary
        case .warning:
            return .orange
        case .error:
            return .red
        }
    }

    private var border: Color {
        switch kind {
        case .normal:
            return .secondary.opacity(0.35)
        case .warning:
            return .orange.opacity(0.65)
        case .error:
            return .red.opacity(0.65)
        }
    }
}

private struct JsonEditorSheet: View {
    let title: String
    let initialText: String
    let onSave: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var text: String

    init(title: String, initialText: String, onSave: @escaping (String) -> Void) {
        self.title = title
        self.initialText = initialText
        self.onSave = onSave
        self._text = State(initialValue: initialText)
    }

    var body: some View {
        NavigationStack {
            TextEditor(text: $text)
                .font(.system(.body, design: .monospaced))
                .padding()
                .navigationTitle(title)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Cancel") { dismiss() }
                    }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Save") {
                            onSave(text)
                            dismiss()
                        }
                    }
                }
        }
    }
}

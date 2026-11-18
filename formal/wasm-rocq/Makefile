.PHONY: all audit check-exports traces

all: check-exports traces Makefile.coq
	$(MAKE) -f Makefile.coq

audit: all
	python3 scripts/check_audit.py

Makefile.coq: _CoqProject
	coq_makefile -f _CoqProject -o Makefile.coq

check-exports:
	cargo run --release --manifest-path ../../Cargo.toml -p neo-wasm --bin export_wasm_rocq -- --check
	cargo run --release --manifest-path ../../Cargo.toml -p neo-wasm --bin export_wasm_rocq -- --binding ../../target/wasm-binding-export
	cd ../../target/wasm-binding-export && sha256sum -c "$(CURDIR)/scripts/binding-exports.sha256"
	python3 -B scripts/binding_catalog_trace.py ../../target/wasm-binding-export tests/BindingCatalogTrace.v

# The witness traces are build outputs. The exporter rewrites only changed files.
traces:
	cargo test --release --manifest-path ../../Cargo.toml -p neo-wasm --test wasm_rocq_witness_export -- --ignored
	python3 -B scripts/lean_parity_trace.py ../nightstream-fprime/artifacts/nightstream-fprime-memory-trace-parity-v1.json tests/LeanParityTrace.v

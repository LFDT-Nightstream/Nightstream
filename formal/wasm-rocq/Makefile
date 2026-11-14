.PHONY: all audit check-exports traces

all: check-exports traces Makefile.coq
	$(MAKE) -f Makefile.coq

audit: all
	python3 scripts/check_audit.py

Makefile.coq: _CoqProject
	coq_makefile -f _CoqProject -o Makefile.coq

check-exports:
	cargo run --release --manifest-path ../../Cargo.toml -p neo-wasm --bin export_wasm_rocq -- --check

# The witness traces are build outputs. The exporter rewrites only changed files.
traces:
	cargo test --release --manifest-path ../../Cargo.toml -p neo-wasm --test wasm_rocq_witness_export -- --ignored

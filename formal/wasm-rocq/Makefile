.PHONY: all audit

all: Makefile.coq
	$(MAKE) -f Makefile.coq

audit: all
	python3 scripts/check_audit.py

Makefile.coq: _CoqProject
	coq_makefile -f _CoqProject -o Makefile.coq

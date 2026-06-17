PYTHON ?= python
VENV_SCRIPT ?= source .venv/bin/activate
PORT ?= 8000
VECTORS := data/faiss_index/active

.PHONY: help install install-upgrade run run-dev ask-status reset-vectors clean

help:
	@echo.
	@echo "Chat with PDF - Quick Makefile"
	@echo "Usage:"
	@echo "  make install      Install requirements"
	@echo "  make run          Start app on http://localhost:$(PORT)"
	@echo "  make run-dev      Start app with reload"
	@echo "  make ask-status   Show vector store status"
	@echo "  make reset-vectors Remove persisted vectors from disk"
	@echo "  make clean        Remove temporary cache folders"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-upgrade:
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m uvicorn main:app --host 0.0.0.0 --port $(PORT)

run-dev:
	$(PYTHON) -m uvicorn main:app --reload --host 0.0.0.0 --port $(PORT)

ask-status:
	@if exist "$(VECTORS)" (echo Vector store exists.) else (echo No vector store found.)

reset-vectors:
	@if exist "$(VECTORS)" (rmdir /S /Q "$(VECTORS)") else (echo No vector store found)
	@echo Done.

clean:
	@if exist .pytest_cache (rmdir /S /Q .pytest_cache)
	@if exist __pycache__ (rmdir /S /Q __pycache__)
	@if exist static\__pycache__ (rmdir /S /Q static\__pycache__)

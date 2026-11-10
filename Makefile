.DEFAULT_GOAL := help

PYTHON ?= python
CONFIG ?= configs/default.yaml
IMAGE ?=
LABEL ?= Cardiomegaly

.PHONY: help install lint format test smoke train eval gradcam serve download fixture

help:
	@echo "install   CPU PyTorch, then this package and dev tools"
	@echo "lint      ruff check and format check"
	@echo "format    apply ruff format and autofixes"
	@echo "test      pytest"
	@echo "fixture   rewrite the synthetic images under tests/fixtures"
	@echo "smoke     one CPU epoch on the synthetic fixture"
	@echo "download  NIH ChestX-ray14 images into ./data (about 45 GB)"
	@echo "train     train from $(CONFIG)"
	@echo "eval      evaluate checkpoints/best.pt"
	@echo "serve     FastAPI on port 8000"
	@echo "gradcam   IMAGE=path LABEL=Finding OUTPUT optional"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

fixture:
	$(PYTHON) -m cxr_triage.data.synthetic

smoke:
	$(PYTHON) -m cxr_triage.training.train --config configs/smoke.yaml

download:
	$(PYTHON) -m cxr_triage.data.download --output-dir data

train:
	$(PYTHON) -m cxr_triage.training.train --config $(CONFIG)

eval:
	$(PYTHON) -m cxr_triage.evaluation.evaluate --config $(CONFIG) --checkpoint checkpoints/best.pt

gradcam:
	@test -n "$(IMAGE)" || (echo "Set IMAGE=path/to/radiograph.png" && exit 1)
	$(PYTHON) -m cxr_triage.explainability.gradcam \
		--checkpoint checkpoints/best.pt \
		--image "$(IMAGE)" \
		--label "$(LABEL)" \
		--output outputs/gradcam.png

serve:
	uvicorn cxr_triage.api.app:app --host 0.0.0.0 --port 8000

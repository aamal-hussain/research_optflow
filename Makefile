.PHONY: env-conda env-pip rmenv-conda rmenv-pip install install-dev lint-check lint pre-commit pre-commit-all clean debug

env:
	@conda create -y -n optflow python=3.11

rmenv:
	@echo "Removing conda environment..."
	@conda remove --all -y -n optflow
	@echo "Conda environment removed."


install:
	@python3 -m pip install uv
	@python3 -m uv pip install -e src/

lint-check:
	@ruff check ./

lint:
	@ruff format ./
	@ruff check --fix ./

pre-commit:
	@pre-commit run

pre-commit-all:
	@pre-commit run --all-files

clean:
	@echo "Cleaning..."
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf **/.ruff_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf **/*.egg-info
	@rm -rf .ipynb_checkpoints
	@rm -rf **/.ipynb_checkpoints
	@rm -rf htmlcov
	@rm -rf **/__pycache__

debug:
	@echo "Credentials exists check: $(wildcard .pxs_credentials)"
	@echo "USER_NAME: '$(ARTIFACTORY_USER_NAME)'"
	@echo "ACCESS_TOKEN: '$(ARTIFACTORY_ACCESS_TOKEN)'"

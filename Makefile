.PHONY: env-conda env-pip rmenv-conda rmenv-pip install install-dev clean debug

include .pxs_credentials

env:
	@conda create -y -n optflow python=3.11

rmenv:
	@echo "Removing conda environment..."
	@conda remove --all -y -n optflow
	@echo "Conda environment removed."


install:
	@python3 -m pip install uv
	@python3 -m uv pip install -e src/ --prerelease=allow --index-strategy unsafe-best-match --extra-index-url https://$(ARTIFACTORY_USER_NAME):$(ARTIFACTORY_ACCESS_TOKEN)@physicsx.jfrog.io/artifactory/api/pypi/px-pypi-release/simple

install-dev:
	@cd ~/product/libraries/pxs && pip install -e ".[gpu,experimental-acheron,opora]" --extra-index-url https://$(ARTIFACTORY_USER_NAME):$(ARTIFACTORY_ACCESS_TOKEN)@physicsx.jfrog.io/artifactory/api/pypi/px-pypi-release/simple

clean:
	@echo "Cleaning..."
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf **/*.egg-info
	@rm -rf .ipynb_checkpoints
	@rm -rf **/.ipynb_checkpoints
	@rm -rf htmlcov
	@rm -rf **/__pycache__

debug:
	@echo "Credentials exists check: $(wildcard ./conf/local/pxs_credentials)"
	@echo "USER_NAME: '$(ARTIFACTORY_USER_NAME)'"
	@echo "ACCESS_TOKEN: '$(ARTIFACTORY_ACCESS_TOKEN)'"

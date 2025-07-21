# Research Optflow


## Getting started

1. Clone the repository:

```bash
git clone git@github.com:aamal-hussain/research_optflow.git
```

2. (If on ml1/ml2) Create a .pxs_credentials (DO NOT COMMIT IT!):
```bash
  touch .pxs_credentials
  echo "ARTIFACTORY_USER_NAME = <user_name>" >> .pxs_credentials
  echo "ARTIFACTORY_ACCESS_TOKEN = <access_token>" >> .pxs_credentials
```
3. Create and install environment:

```bash
make env
conda activate optflow
make install
```
## Example usage

```bash
python -m train.diffusion
```

The above loads the diffusion transformer, with the config defined in
`conf/config.yaml` and runs the training pipeline (c.f. `train/diffusion.py`).

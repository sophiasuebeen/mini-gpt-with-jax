# Mini GPT with JAX

Local environment for the DeepLearning.AI mini GPT with JAX tutorial.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name mini-gpt-jax --display-name "Python (.venv: mini-gpt-jax)"
```


## Quick Check

```bash
python - <<'PY'
import jax
import flax.nnx as nnx
print(jax.__version__)
print(jax.devices())
print(nnx.Module)
PY
```

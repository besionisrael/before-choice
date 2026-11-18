# Before the Choice — starter environment

> **IMPORTANT:** Read the scope rules before any action. Only CPU phases D0/D1 are pre-authorized; cloud GPU runs require case-by-case written approval. The research objective is prospective path representation (H1), not AI welfare or J-space experiments.

A preliminary software scaffold **only**. No LLM is run or evaluated by this package. No claims about forward sweep are implied.

## 1. Local Python environment (Windows PowerShell)

Install Python 3.11+ and open a terminal in this directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python generate_worlds.py --pairs 25 --seed 42 --output data/worlds.jsonl
```

The generator and independent oracle use Python's standard library only. The JSONL output contains a `prompt` (shown to the model in future work) and an `oracle` (private ground truth for evaluation). Keep these separate when implementing inference.

**Counterfactual pairs:** `*-base` and `*-cf` share initial actions, state labels, and terminal rewards. The two final destination mappings are swapped. The oracle calculates and verifies the optimal action independently. Start with this controlled toy setting, then add more complex graph topologies in the full protocol.

## 2. Optional Modal GPU test

Modal is for the later LLM/activation/patching stages, **not required** to generate the worlds. Modal GPU access requires an account, payment method, and spending limits set on the dashboard. The Starter plan includes monthly compute credits; actual availability and prices can change.

```powershell
python -m pip install modal
modal setup
modal token info
modal run modal_smoke.py
```

This starts an L4 GPU Function briefly, prints the GPU name and memory, and exits. It does not download a model or save activations. This call uses billable compute; set a Workspace spend limit before running it. The Modal Function runs in an isolated cloud container automatically. You do not need the special `modal.Sandbox` API for this trusted research code.

See: https://modal.com/docs/guide/getting-started and https://modal.com/docs/guide/budgets

## 3. Planned modules (not yet implemented)

- `run_llm.py`: load a specified open-weight checkpoint, answer once per world, record logits and decisions.
- `capture_activations.py`: run model forward passes, capture selected positions and layers.
- `train_probes.py`: train/test held-out probes for both trajectory branches.
- `patch_activations.py`: conduct controlled causal interventions.
- `analyze.py`: compare to baselines and report uncertainties.

## 4. Methodological limits

- The environment has two deterministic linear branches. It is a controlled starting point, not sufficient by itself to demonstrate a true sequential forward sweep.
- Input tokens explicitly mention state labels; basic probes may merely recover textual information. The main study needs prompt-only baselines, counterbalanced labels, structured relational probes, and causal controls.
- Counterfactuals swap two terminal destinations; this cleanly changes route values while preserving labels, but may allow shortcut strategies. Later experiments should diversify graph topology and control terminal information carefully.
- A representation decoded across transformer layers is not automatically a neuronal time sweep.
- Model selection, generation decoding, sample size, and exact intervention sites remain to be defined in the full protocol.

## 5. Other services

- Neuronpedia is optional for visual exploration; a particular Qwen checkpoint might not have a fitted lens.
- Hugging Face hosts open-weight models such as `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0). Downloading the weights does not require a paid inference API.


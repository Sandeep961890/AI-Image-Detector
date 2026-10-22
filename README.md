# Local AI Image Detector

A local Flask website that classifies an uploaded image with the
[`delpot/steganograph-ia-detector`](https://huggingface.co/delpot/steganograph-ia-detector)
image classifier. Images are validated and analyzed in memory; the application
does not save uploads, use a database, require authentication, or call a hosted
inference API.

The model can make mistakes, particularly on edited, compressed, unusual, or
out-of-distribution images. Its scores are normalized model scores, **not
calibrated probabilities**. This project has not measured accuracy on an
independent, representative test set, so it makes no independent accuracy
claim.

## Requirements

- Windows 10/11 and 64-bit Python 3.11.
- Internet for the first model download. Inference can run offline after the
  model and processor have been cached.
- Optional NVIDIA GPU. CPU inference is supported but can be slow.
- Enough disk space for the Python dependencies and approximately 343 MB of
  model files, plus cache overhead.

## Windows PowerShell setup

Open PowerShell in the project directory (`S:\AI-Image-Detector`).

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, allow scripts for the current PowerShell
process and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Install PyTorch for your hardware

Select the PyTorch install command at the official
[PyTorch Start Locally selector](https://pytorch.org/get-started/locally/).
Choose Windows, Pip, Python, and the CUDA option compatible with your NVIDIA
driver and GPU. The selector's CUDA runtime package does not normally require
installing the full CUDA Toolkit separately. If you do not have a compatible
NVIDIA setup, select the CPU build.

Run the selected PyTorch command inside the activated `.venv`. Do not copy a
CUDA version from another machine or blindly pin a CUDA package: driver
compatibility is specific to your system. Then install the remaining project
dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the environment and whether CUDA is available to this PyTorch build:

```powershell
python -c "import sys, flask, PIL, torch, transformers; print('Python:', sys.version); print('Flask:', flask.__version__); print('Pillow:', PIL.__version__); print('Torch:', torch.__version__); print('Transformers:', transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` means inference will use the CPU. On an NVIDIA system,
confirm that the installed PyTorch command came from the official selector,
the selected CUDA runtime is compatible with the installed driver, and
`nvidia-smi` can see the GPU.

## Run the application

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

The development server listens on loopback only. Open
<http://127.0.0.1:5000>. The first analysis may download the pinned model and
processor from Hugging Face and can take longer than later analyses. Keep the
server process running while using the page; press `Ctrl+C` to stop it.

The model is loaded once per server process and kept in the Hugging Face cache.
After the required files have downloaded, inference works without a network
connection. To explicitly prevent Hub network access, set offline mode before
starting the server:

```powershell
$env:HF_HUB_OFFLINE = "1"
python app.py
```

Unset it in the current PowerShell session when online access is needed again:

```powershell
Remove-Item Env:HF_HUB_OFFLINE
```

## Tests

Run the focused Flask and detector-contract tests with Python's standard
library test runner:

```powershell
python -m unittest discover -s tests -v
```

The suite exercises real image validation and endpoint behavior. Its real model
integration test calls the configured detector and checks the response schema;
if the model is unavailable, it reports a skip rather than inventing a
prediction. Test doubles are confined to negative-path or label-mapping tests
and are never returned to the website as predictions.

## Model and hardware notes

- Model: `delpot/steganograph-ia-detector`, pinned to a specific Hugging Face
  revision in `detector.py`.
- Classes are read from the checkpoint configuration; the current checkpoint
  maps `real` and `ai_generated`.
- The processor and normalization follow the model's documented image
  preprocessing.
- CUDA is selected only when `torch.cuda.is_available()` is true; otherwise
  inference falls back to CPU.
- The checkpoint is roughly 343 MB. GPU memory needs depend on the GPU, its
  display/workload use, PyTorch build, and other allocations. A GPU with about
  4 GB or more available is a practical target, not a guaranteed minimum.
  CUDA out-of-memory errors are reported; retrying with a CPU-only PyTorch
  environment is a fallback if GPU memory is insufficient.
- Model license and source documentation are available on the linked model
  card. Review its terms and limitations for your intended use.

## Troubleshooting

**`py -3.11` is not found**  
Install 64-bit Python 3.11 and ensure the Python Launcher is available. Reopen
PowerShell and check `py -0p` for installed interpreters.

**Activation is blocked**  
Use the process-scoped execution policy command above. It does not change the
machine-wide policy.

**Torch reports no CUDA device**  
Check `torch.cuda.is_available()`, `nvidia-smi`, the NVIDIA driver, and the
exact Windows/CUDA/Pip command chosen in the official PyTorch selector. A CPU
build or an incompatible driver will report CUDA unavailable; the app will use
CPU in that case.

**The model is unavailable or cannot download**  
The first run needs internet access to fetch the pinned checkpoint. Check the
connection, available disk space, and Hugging Face cache permissions. If
`HF_HUB_OFFLINE=1` is set, all required model files must already be cached;
unset it and restart to allow the initial download.

**Inference is slow or fails from memory pressure**  
CPU inference can be substantially slower than GPU inference. Close other GPU
workloads or use a compatible CPU-only PyTorch environment if CUDA runs out of
memory. The application reports inference failures without exposing internal
tracebacks to the browser.

**An image is rejected**  
Only JPEG, PNG, and WebP are accepted. The server checks the decoded file
contents (not just its extension or browser MIME type), limits uploads to
10 MB, and rejects malformed or excessively large images.

## Privacy and limitations

Uploaded image bytes are held in memory for validation and inference and are
not written to an application upload directory or exposed as public files.
The first model download contacts Hugging Face; image inference itself is local
and does not send images to an external service. The application has no
database or user accounts. The Hugging Face cache contains model files, not
user uploads.

This classifier is not a forensic guarantee. It can misclassify real and
AI-generated images, and its output should not be used as the sole basis for
high-impact decisions. No independent accuracy result is claimed; evaluating
accuracy requires a suitable, representative, independent labeled test set.

During a local diagnostic, the pinned checkpoint classified [NASA's Apollo 8
Earthrise photograph](https://commons.wikimedia.org/wiki/File:NASA-Apollo8-Dec24-Earthrise.jpg)
(PD NASA) as `ai_generated` with a 0.874 softmax score. This is a demonstrated
false positive on a known real photograph, despite the model card's reported
benchmark performance. The app does not change the model's threshold or
override its output. The detector logs the checkpoint label mappings,
processor configuration, raw logits, and per-class softmax scores to the
Python terminal to support further evaluation.

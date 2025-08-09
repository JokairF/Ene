# app/llm_loader.py
import sys
import os
from pathlib import Path

# Dossiers CUDA
for p in [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\libnvvp",
]:
    if os.path.isdir(p) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(p)

# Dossier des DLLs llama-cpp du venv courant
venv_lib = Path(os.__file__).resolve().parents[1] / "site-packages" / "llama_cpp" / "lib"
if venv_lib.is_dir() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(venv_lib))


# Ajout dossier DLL llama_cpp
_DLL_DIR = Path(sys.prefix) / "Lib" / "site-packages" / "llama_cpp" / "lib"
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_DLL_DIR))
else:
    os.environ["PATH"] += os.pathsep + str(_DLL_DIR)

# Ajout dossier CUDA
for vv in ("12.4","12.3","12.2","12.1","12.0"):
    cuda_bin = fr"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{vv}\bin"
    if os.path.isdir(cuda_bin):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(cuda_bin)
        else:
            os.environ["PATH"] += os.pathsep + cuda_bin
        break

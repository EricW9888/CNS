# Verified software environment

Independent experiment reruns and repository checks completed on Windows with
Python 3.11.9.
Core versions were:

```text
numpy==2.0.2
matplotlib==3.11.1
pandas==3.0.5
psutil==7.2.2
pyarrow==25.0.1
scipy==1.17.1
openpyxl==3.1.5
pytest==9.1.1
networkx==3.6.1
flygym==1.2.1
mujoco==3.2.7
brian2==2.9.0
```

`requirements.txt` defines supported ranges for the neural-only core.
`requirements-exp003-flygym.txt` is optional historical embodiment support.
`requirements-verified-windows.txt` records the exact verified environment;
it is a reproducibility snapshot, not a promise that every pinned binary is
portable to every platform.

Brian2 is not used by the active experiment runners. Its compatibility
diagnostic is isolated in `requirements-brian2-diagnostic.txt`.

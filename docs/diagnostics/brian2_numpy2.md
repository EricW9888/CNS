# Brian2 / NumPy 2 diagnostic

Recorded 2026-09-10 in the project virtual environment before changing the
model backend.

## Environment

| Package | Version |
|---|---|
| Python | 3.11.9 (`C:\Users\Eric\Dev\CNS\.venv\Scripts\python.exe`) |
| NumPy | 2.4.6 |
| Brian2 | 2.9.0 |

The requirements file keeps NumPy in the 2.x line. No NumPy `<2` constraint is
used to work around this failure.

## Complete import traceback

Command:

```powershell
.venv\Scripts\python.exe -c "import brian2"
```

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\Eric\Dev\CNS\.venv\Lib\site-packages\brian2\__init__.py", line 58, in <module>
    import brian2.numpy_ as numpy
  File "C:\Users\Eric\Dev\CNS\.venv\Lib\site-packages\brian2\numpy_.py", line 12, in <module>
    from brian2.units.unitsafefunctions import *
  File "C:\Users\Eric\Dev\CNS\.venv\Lib\site-packages\brian2\units\__init__.py", line 7, in <module>
    from .allunits import (
  File "C:\Users\Eric\Dev\CNS\.venv\Lib\site-packages\brian2\units\allunits.py", line 14, in <module>
    from .fundamentalunits import (
  File "C:\Users\Eric\Dev\CNS\.venv\Lib\site-packages\brian2\units\fundamentalunits.py", line 984, in <module>
    class Quantity(np.ndarray):
  File "C:\Users\Eric\Dev\CNS\.venv\Lib\site-packages\brian2\units\fundamentalunits.py", line 1661, in Quantity
    ptp = wrap_function_keep_dimensions(np.ndarray.ptp)
                                        ^^^^^^^^^^^^^^
AttributeError: type object 'numpy.ndarray' has no attribute 'ptp'
```

## Diagnosis

The failing code is in the installed Brian2 package itself:

`brian2/units/fundamentalunits.py`, class `Quantity`, line 1661.

It evaluates `np.ndarray.ptp` while defining the class. NumPy 2 removed the
`ndarray.ptp` method; the supported function form is `np.ptp`. The failure
happens during Brian2 import, before this repository constructs a neuron,
synapse, or graph and before SciPy or Pandas can be involved. This is not a
MaleCNS data or connectome error.

Brian2 2.7 release notes document NumPy-2 compatibility, and current upstream
development source guards this optional attribute with `hasattr`. The released
2.9.0 wheel installed here still contains the unconditional access, so the
documented compatibility claim and this wheel’s import behavior disagree. The
project therefore records the failure and uses an equivalent transparent
NumPy/SciPy sparse LIF integrator for the first experiment. It does not modify
the installed third-party package and does not downgrade NumPy.

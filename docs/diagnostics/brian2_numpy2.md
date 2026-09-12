# Brian2 / NumPy 2 diagnostic

Recorded 2026-09-10 in the project virtual environment before changing the
model backend.

## Environment

| Package | Version |
|---|---|
| Python | 3.11.9 (`<repo>\.venv\Scripts\python.exe`) |
| NumPy | 2.4.6 |
| Brian2 | 2.9.0 |

No NumPy `<2` constraint is used to work around this failure.

## Complete import traceback

Command:

```powershell
.venv\Scripts\python.exe -c "import brian2"
```

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "<repo>\.venv\Lib\site-packages\brian2\__init__.py", line 58, in <module>
    import brian2.numpy_ as numpy
  File "<repo>\.venv\Lib\site-packages\brian2\numpy_.py", line 12, in <module>
    from brian2.units.unitsafefunctions import *
  File "<repo>\.venv\Lib\site-packages\brian2\units\__init__.py", line 7, in <module>
    from .allunits import (
  File "<repo>\.venv\Lib\site-packages\brian2\units\allunits.py", line 14, in <module>
    from .fundamentalunits import (
  File "<repo>\.venv\Lib\site-packages\brian2\units\fundamentalunits.py", line 984, in <module>
    class Quantity(np.ndarray):
  File "<repo>\.venv\Lib\site-packages\brian2\units\fundamentalunits.py", line 1661, in Quantity
    ptp = wrap_function_keep_dimensions(np.ndarray.ptp)
                                        ^^^^^^^^^^^^^^
AttributeError: type object 'numpy.ndarray' has no attribute 'ptp'
```

## Diagnosis

The failing code is in the installed Brian2 package itself:

`brian2/units/fundamentalunits.py`, class `Quantity`, line 1661.

It evaluates `np.ndarray.ptp` while defining the class. NumPy 2.4.6 no longer
exposes that class attribute; the supported function form is `np.ptp`. The failure
happens during Brian2 import, before this repository constructs a neuron,
synapse, or graph and before SciPy or Pandas can be involved. This is not a
MaleCNS data or connectome error.

Brian2 2.7 release notes document compatibility with NumPy 2.0. That statement
does not establish compatibility with every later NumPy 2.x release. As a
control, the same Brian2 2.9.0 wheel imports successfully in this repository's
verified NumPy 2.0.2 environment, where `np.ndarray.ptp` is still exposed. The
failure is therefore specifically reproduced for NumPy 2.4.6; it must not be
reported as a generic Brian2/NumPy-2 incompatibility. The runnable baseline uses
its transparent NumPy/SciPy sparse LIF integrator and does not modify either
third-party package.

## Verification control — 2026-09-12

Fresh isolated environment:

```text
Python 3.11.9
NumPy 2.4.6
Brian2 2.9.0
import brian2 -> AttributeError at fundamentalunits.py:1661
```

Current project environment:

```text
Python 3.11.9
NumPy 2.0.2
Brian2 2.9.0
import brian2 -> OK
```

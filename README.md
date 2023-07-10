# decompy

`decompy` is a Python package containing several robust algorithms for matrix decomposition and analysis. The types of algorithms includes
* Robust PCA or SVD based methods
* Matrix completion methods
* Robust matrix or tensor factorization methods.


## Features

- Data decomposition using various methods
- Support for sparse decomposition, low-rank approximation, and more
- User-friendly API for easy integration into your projects
- Extensive documentation and examples

## Installation

You can install `decompy` using pip:

```bash
pip install decompy
```

## Usage

Here's a simple example demonstrating how to use decompy for data decomposition:

```python
import numpy as np
from decompy.robust_svd import DensityPowerDivergence

# Load your data
data = np.arange(100).reshape(20,5).astype(np.float64)

# Perform data decomposition
algo = DensityPowerDivergence(alpha = 0.5)
result = algo.decompose(data)

# Access the decomposed components
U, V = result.singular_vectors(type = "both")
S = result.singular_values()
low_rank_component = U @ S @ V.T
sparse_component = data - low_rank_component

print(low_rank_component)
print(sparse_component)
```

While the singular values are about 573 and 7.11 for this case (check the `S` variable), it can get highly affected if you use the simple SVD and change a single entry of the `data` matrix.

```
s2 = np.linalg.svd(data, compute_uv = False)
print(np.round(s2, 2))    # estimated by usual SVD
print(np.diag(np.round(S, 2)))    # estimated by robust SVD


data[1, 1] = 10000  # just change a single entry
s3 = np.linalg.svd(data, compute_uv = False)
print(np.round(s3, 2))   # usual SVD shoots up
s4 = algo.decompose(data).singular_values()
print(np.diag(np.round(s4, 2)))
```

You can find more example notebooks in [examples]() folder. For more detailed usage instructions, please refer to the [documentation]().


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please create an issue or submit a pull request on the GitHub repository. For contributing developers, please refer to [Contributing.md](Contributing.md) file.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).



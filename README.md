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
import decompy

# Load your data
data = np.arange(100).reshape(20,5)

# Perform data decomposition
algo = decompy.robust_svd.DensityPowerDivergence(alpha = 0.5)
result = algo.decompose(data, method='sparse')

# Access the decomposed components
U, V = result.get_singular_vectors(type = "both")
S = result.get_singular_values()
low_rank_component = U @ S @ V.T
sparse_component = data - low_rank_component
```

You can find more example notebooks in [examples]() folder. For more detailed usage instructions, please refer to the [documentation]().

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please create an issue or submit a pull request on the GitHub repository. For contributing developers, please refer to [Contributing.md](Contributing.md) file.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).



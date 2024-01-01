# decompy

`decompy` is a Python package containing several robust algorithms for matrix decomposition and analysis. The types of algorithms includes
* Robust PCA or SVD based methods
* Matrix completion methods
* Robust matrix or tensor factorization methods.
* Matrix rank estimation methods.


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


## List of Algorithms available in the `decompy` library

### Robust PCA

1. Alternating Direction Method [(Yuan and Yang, 2009)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.400.8797) - `robust_pca/adm.py`

2. Augmented Lagrangian Method [(Tang and Nehorai)](https://ieeexplore.ieee.org/document/5766144) - `robust_pca/alm.py`

3. Exact Augmented Lagrangian Method [(Lin, Chen and Ma, 2010)](https://arxiv.org/abs/1009.5055) - `robust_pca/ealm.py`

4. Inexact Augmented Lagrangian Method - `robust_pca/ialm.py`

5. Principal Component Pursuit (PCP) Method [(Candes et al. 2009)](https://arxiv.org/abs/0912.3599) - `robust_pca/pcp.py`

6. Robust PCA by M-estimation [(De la Torre and Black, 2001)](https://ieeexplore.ieee.org/document/937541) - `robust_pca/rpca.py`

7. Robust PCA using Variational Bayes method [(Babacan et al 2012)](https://ieeexplore.ieee.org/document/6194350) - `robust_pca/vbrpca.py`


## Robust SVD

1. Robust SVD using Density Power Divergence (rSVDdpd) Algorithm [(Roy et al, 2023)](https://arxiv.org/abs/2109.10680) - `robust_svd/rsvddpd.py`



## Matrix Completion 

1. Outlier Pursuit [Xu et al, 2011](https://guppy.mpe.nus.edu.sg/~mpexuh/papers/OutlierPursuit-TIT.pdf) - `matrix_completion/op.py`


## Rank Estimation Methods

### Penalization Criterion (`rankmethods/penalized.py`)

1. Elbow method

2. Akaike's Information Criterion (AIC) - https://link.springer.com/chapter/10.1007/978-1-4612-1694-0_15

3. Bayesian Information Criterion (BIC) - https://doi.org/10.1214/aos/1176344136

4. Bai and Ng's Information Criterion for spatiotemporal decomposition (PC1, PC2, PC3, IC1, IC2, IC3) - https://doi.org/10.1111/1468-0262.00273

5. Divergence Information Criterion (DIC) - https://doi.org/10.1080/03610926.2017.1307405

### Cross Validation Approaches (`rankmethods/cvrank.py`)

1. Gabriel style Cross validation - http://www.numdam.org/item/JSFS_2002__143_3-4_5_0/

2. Wold style cross validation separate row and column deletion - https://www.jstor.org/stable/1267581

3. Bi-cross validation (Owen and Perry) - https://doi.org/10.1214/08-AOAS227

### Bayesian Approaches (`rankmethods/bayes.py`)

1. Bayesian rank estimation method by Hoffman - https://www.jstor.org/stable/27639896


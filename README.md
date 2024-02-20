# decompy

`decompy` is a Python package containing several robust algorithms for matrix decomposition and analysis. The types of algorithms include
* Robust PCA or SVD-based methods
* Matrix completion methods
* Robust matrix or tensor factorization methods.
* Matrix rank estimation methods.

The latest version of `decompy` is **1.0.0**.

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
from decompy.matrix_factorization import RobustSVDDensityPowerDivergence

# Load your data
data = np.arange(100).reshape(20,5).astype(np.float64)

# Perform data decomposition
algo = RobustSVDDensityPowerDivergence(alpha = 0.5)
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

You can find more example notebooks in the **examples** folder. For more detailed usage instructions, please refer to the **documentation**.


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please create an issue or submit a pull request on the GitHub repository. For contributing developers, please refer to [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).


## List of Algorithms available in the `decompy` library

### Matrix Factorization Methods

Currently, there are **17** matrix factorization methods available in `decompy`, as follows:

* Alternating Direction Method [(Yuan and Yang, 2009)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.400.8797) - `matrix_factorization/adm.py`

* Augmented Lagrangian Method [(Tang and Nehorai)](https://ieeexplore.ieee.org/document/5766144) - `matrix_factorization/alm.py`

* AS-RPCA: Active Subspace: Towards Scalable Low-Rank Learning [(Liu and Yan, 2012)](http://dl.acm.org/citation.cfm?id=2421487) - `matrix_factorization/asrpca.py`

* Dual RPCA [(Lin et al. 2009)](http://arxiv.org/abs/1009.5055) - `matrix_factorization/dual.py`

* Exact Augmented Lagrangian Method [(Lin, Chen and Ma, 2010)](https://arxiv.org/abs/1009.5055) - `matrix_factorization/ealm.py`

* Robust PCA using Fast PCP Method [(Rodriguez and Wohlberg, 2013)](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6738015) - `matrix_factorization/fpcp.py`

* Grassmann Average [(Hauberg et al. 2014)](http://files.is.tue.mpg.de/black/papers/RGA2014.pdf) [website](http://ps.is.tuebingen.mpg.de/project/Robust_PCA) - `matrix_factorization/ga.py`

* Trimmed Grassmann Average [(Hauberg et al. 2014)](http://files.is.tue.mpg.de/black/papers/RGA2014.pdf) [website](http://ps.is.tuebingen.mpg.de/project/Robust_PCA) - `matrix_factorization/ga.py` (with `trim_percent` value more than 0)

* Inexact Augmented Lagrangian Method [(Lin et al. 2009)](http://arxiv.org/abs/1009.5055)  [website](http://perception.csl.illinois.edu/matrix-rank/sample_code.html) - `matrix_factorization/ialm.py`

* L1 Filtering [(Liu et al. 2011)](http://arxiv.org/abs/1108.5359) - `matrix_factorization/l1f.py`

* Linearized ADM with Adaptive Penalty [(Lin et al. 2011)](http://arxiv.org/abs/1109.0367) - `matrix_factorization/ladmap.py`

* Robust PCA using M-estimation [(De la Torre and Black, 2001)](https://ieeexplore.ieee.org/document/937541) [website](http://users.salleurl.edu/~ftorre/papers/rpca2.html) - `matrix_factorization/mest.py`

* Outlier Pursuit [Xu et al, 2011](https://guppy.mpe.nus.edu.sg/~mpexuh/papers/OutlierPursuit-TIT.pdf) - `matrix_factorization/op.py`

* Principal Component Pursuit (PCP) Method [(Candes et al. 2009)](https://arxiv.org/abs/0912.3599) - `matrix_factorization/pcp.py`

* RegL1-ALM: Robust low-rank matrix approximation with missing data and outliers [(Zheng et al. 2012)](https://sites.google.com/site/yinqiangzheng/home/zheng_CVPR12_robust%20L1-norm%20low-rank%20matrix%20factorization.pdf) [website](https://sites.google.com/site/yinqiangzheng/) - `matrix_factorization/regl1alm.py`

* Robust SVD using Density Power Divergence (rSVDdpd) Algorithm [(Roy et al, 2023)](https://arxiv.org/abs/2109.10680) - `matrix_factorization/rsvddpd.py`

* SVT: Singular Value Thresholding [(Cai et al. 2008)](http://arxiv.org/abs/0810.3286) [website](http://perception.csl.illinois.edu/matrix-rank/sample_code.html) - `matrix_factorization/svt.py`

* Symmetric Alternating Direction Augmented Lagrangian Method (SADAL) [(Goldfarb et al. 2010)](http://arxiv.org/abs/0912.4571) - `matrix_factorization/sadal.py`

* Robust PCA using Variational Bayes method [(Babacan et al 2012)](https://ieeexplore.ieee.org/document/6194350) - `matrix_factorization/vbrpca.py`


---

* MoG-RPCA: Mixture of Gaussians RPCA [(Zhao et al. 2014)](http://jmlr.org/proceedings/papers/v32/zhao14.pdf) [website](http://www.cs.cmu.edu/~deyum/index.htm)


### Methods to be added (Coming soon)

* R2PCP: Riemannian Robust Principal Component Pursuit [(Hintermüller and Wu, 2014)](http://link.springer.com/article/10.1007/s10851-014-0527-y)

* DECOLOR: Contiguous Outliers in the Low-Rank Representation [(Zhou et al. 2011)](http://arxiv.org/abs/1109.0882) [website1](https://sites.google.com/site/eeyangc/software/decolor) [website2](https://fling.seas.upenn.edu/~xiaowz/dynamic/wordpress/?p=144)






## Rank Estimation Methods

In `rankmethods/penalized.py` -

* Elbow method

* Akaike's Information Criterion (AIC) - https://link.springer.com/chapter/10.1007/978-1-4612-1694-0_15

* Bayesian Information Criterion (BIC) - https://doi.org/10.1214/aos/1176344136

* Bai and Ng's Information Criterion for spatiotemporal decomposition (PC1, PC2, PC3, IC1, IC2, IC3) - https://doi.org/10.1111/1468-0262.00273

* Divergence Information Criterion (DIC) - https://doi.org/10.1080/03610926.2017.1307405

In `rankmethods/cvrank.py` -

* Gabriel style Cross validation - http://www.numdam.org/item/JSFS_2002__143_3-4_5_0/

* Wold style cross validation separate row and column deletion - https://www.jstor.org/stable/1267581

* Bi-cross validation (Owen and Perry) - https://doi.org/10.1214/08-AOAS227

In `rankmethods/bayes.py` -

* Bayesian rank estimation method by Hoffman - https://www.jstor.org/stable/27639896



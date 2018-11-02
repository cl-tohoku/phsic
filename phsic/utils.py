import numpy as np


def sample_from_unit_hypersphere(dim):
    # 半径dimの単位超球面から乱択
    # via. [球面上に一様分布する乱数の生成 - Qiita](https://qiita.com/mh-northlander/items/a2e643cf62317f129541)
    x = np.random.standard_normal(dim)
    r = np.linalg.norm(x)
    return x / r

| Model | Train RMSE ($100k) | Test RMSE ($100k) | Test R² |
| --- | --- | --- | --- |
| Baseline: always predict the train mean | 1.1562 | 1.1449 | -0.0002 |
| **Gradient descent (mine)**, lr=0.1, 1000 iters | 0.7197 | 0.7457 | 0.5757 |
| scikit-learn `LinearRegression` | 0.7197 | 0.7456 | 0.5758 |

Max |weight difference| vs scikit-learn: 2.74e-03 · bias difference: 2.66e-15

**Learning-rate sweep** (final training cost after 1,000 iterations)

| Learning rate | Final training cost J |
| --- | --- |
| 0.001 | 0.65568 |
| 0.01 | 0.27386 |
| 0.1 | 0.25897 |
| 0.5 | 0.25897 |

**Why scaling matters** (curvature of the cost surface; stable only if lr < 2/λmax)

| Features | λmax | λmin | Condition number | Largest stable lr |
| --- | --- | --- | --- | --- |
| Raw | 3.34e+06 | 5.92e-05 | 5.64e+10 | 5.99e-07 |
| Standardized | 2.03e+00 | 4.73e-02 | 4.28e+01 | 9.88e-01 |

**Does gradient descent reach the exact solution?** (lr = 0.1)

| Iterations | Test RMSE | Max abs weight diff vs sklearn | Final cost J |
| --- | --- | --- | --- |
| 100 | 0.75304 | 4.17e-01 | 0.273987 |
| 300 | 0.74489 | 1.14e-01 | 0.259797 |
| 1,000 | 0.74565 | 2.74e-03 | 0.258967 |
| 3,000 | 0.74558 | 2.00e-07 | 0.258967 |
| 10,000 | 0.74558 | 1.03e-14 | 0.258967 |

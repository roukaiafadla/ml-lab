| Model | Test RMSE ($100k) | Test R² |
|---|---|---|
| Gradient descent (mine, lr=0.1, 1000 iters) | 0.7457 | 0.5757 |
| scikit-learn `LinearRegression` | 0.7456 | 0.5758 |

Max |weight difference| vs scikit-learn: 2.74e-03  
Bias difference: 2.66e-15

| Learning rate | Final training cost J |
|---|---|
| 0.001 | 0.65568 |
| 0.01 | 0.27386 |
| 0.1 | 0.25897 |
| 0.5 | 0.25897 |

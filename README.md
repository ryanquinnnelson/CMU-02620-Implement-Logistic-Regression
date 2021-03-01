# CMU-02620-Implementations
Spring 2021 Machine Learning for Scientists summary - algorithm implementations




## Supervised Learning

### Regression

#### Univariate Regression
##### Model Definition
The model defines a linear relationship between a single feature and a real-valued label.

$$y=\beta_0+\beta x+\epsilon$$

where 
- $\beta_0$ is the intercept
- $\beta$ are regression coefficients
- $\epsilon \sim N(0,\sigma^2)$

##### Maximum Conditional Likelihood Estimate (MCLE) - Mean-Centered
We fit the model without the intercept, using mean-centered data.

$$\hat{ \beta}_{MCLE}=\underset{\beta}{argmin}\sum_{i=1}^N(y_i - \hat{y}_i)^2$$

where $\hat{y}_i=\beta x_i$

$argmin$ can be solved in closed-form:

$$\hat{ \beta}_{MCLE}=\frac{\sum_{i=1}^N x_iy_i}{\sum_{i=1}^N (x_i)^2}$$


#### Multivariate Regression

#### Ridge Regression

#### Sparse Regression

### Classification

### Unsupervised Learning
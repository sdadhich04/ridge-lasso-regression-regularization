# Ridge & LASSO Regression: Regularization on House Sale Prices

A comparison of L2 (Ridge) and L1 (LASSO) regularization for predicting
house sale prices from a real housing dataset — sweeping regularization
strength for each, selecting the best model by validation RMSE, and
examining what each penalty actually does to the learned coefficients
(shrinkage vs. sparsity).

Built from coursework for **CSE 416 (Introduction to Machine Learning)**,
University of Washington. See [Results](#results) for the actual numbers
this notebook produced — nothing here is a target or textbook value, it's
what running this code returned.

## What this does

1. Engineers polynomial features from the base housing features (squared
   and square-root transforms of key predictors) to let a linear model
   capture non-linear relationships.
2. Standardizes features with `StandardScaler` and splits into
   train/validation/test sets.
3. Sweeps 11 Ridge regularization strengths (`np.logspace(-5, 5, 11)`),
   training a model at each and recording validation RMSE.
4. Repeats the same sweep for LASSO across 7 regularization strengths.
5. Selects the best model of each type by minimum validation RMSE, then
   evaluates the selected models on the held-out test set.
6. Compares what the two penalties do differently: Ridge shrinks all
   coefficients toward zero without eliminating them, while LASSO drives
   many coefficients exactly to zero, performing implicit feature
   selection.

## Results

- Baseline (unregularized) linear regression: **test RMSE = 384,955.80**
- Best Ridge model (λ = 10.0): **test RMSE = 354,624.85**, all coefficients
  non-zero (pure shrinkage, no feature elimination)
- Best LASSO model (λ = 10,000.0): **test RMSE = 344,436.85**, with
  **28 of 39** coefficients set exactly to zero

LASSO both fits better on this data *and* produces a much sparser,
more interpretable model — a concrete illustration of why L1 regularization
is often preferred when feature selection matters, not just predictive
accuracy. Two plots (RMSE vs. log-regularization-strength for each penalty
type) are saved as outputs in the notebook itself.

## What's original vs. course-provided

The course notebook provided data loading, the train/validation/test split
setup, and the plotting helper function. Written for this assignment: the
polynomial feature-engineering loop, the `StandardScaler` fit/transform
pipeline, both regularization sweeps (building the RMSE-vs-λ tables from
scratch), and the `idxmin()`-based best-model selection logic for each
penalty type.

## Known limitations

- **Data not included.** The notebook loads `home_data.csv`, a King County,
  WA house-sales dataset used in this course; the raw file isn't available
  to include here. To rerun this notebook, supply a `home_data.csv` with
  matching columns (square footage, bedrooms, bathrooms, etc.) in the same
  directory.
- This dataset and general assignment structure is the kind widely used in
  UW's ML curriculum (in the spirit of the King County housing dataset from
  the Coursera Machine Learning Specialization); the analysis and code
  here are original work built on top of that provided dataset.

## Running this code

Requires Python with `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.
Open `ridge_lasso_regression.ipynb` in Jupyter, supply `home_data.csv` in
the same directory, and run all cells top to bottom.

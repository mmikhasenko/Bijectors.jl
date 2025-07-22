### **Project: Refactoring `Bijectors.RationalQuadraticSpline`**

#### **1. Objective**

The primary goal is to perform a complete rewrite of the `RationalQuadraticSpline` bijector in `src/bijectors/rational_quadratic_spline.jl`. The new implementation must be robust, correct, and align with the conventions of the `Bijectors.jl` library, particularly regarding the handling of multi-dimensional inputs for features and batches.

#### **2. Context & Motivation**

The current implementation has several issues:
*   **Incorrect Dimensionality Handling:** It makes flawed assumptions about input shapes, especially for batched data. The convention in similar libraries is that the feature dimension is the last dimension, which the current code does not consistently follow.
*   **Unclear Parameterization:** The constructors are confusing, mixing parameter processing with raw parameter acceptance (`widths`, `heights`, `-B`, `B`), which leads to inflexibility.
*   **Instability and Bugs:** The underlying mathematical implementation has had issues, leading to incorrect results.

To address this, the refactoring will be based on a trusted reference implementation and best practices from within the `Bijectors.jl` library itself.

*   **Primary Reference (`RQS.jnp.py`):** This Python/JAX implementation from Distrax is considered a correct and numerically stable reference for the core spline logic. We will follow its methods for parameter normalization and the forward/inverse transformations.
*   **Style and Interface Reference (`4B-power.jl`):** This file provides a good, albeit simple, example of how a bijector should handle multi-dimensional inputs (scalars, vectors, matrices) within `Bijectors.jl`. We will adapt its approach for dispatching on different input types.

#### **3. Implementation Plan**

The refactoring will be done from scratch, replacing the existing file content.

**Step 1: Define the `RationalQuadraticSpline` Struct**
The new struct should store the *processed* parameters, not the raw inputs.

```julia
struct RationalQuadraticSpline{T,N} <: Bijector
    x_pos::T           # Knot positions on the x-axis
    y_pos::T           # Knot positions on the y-axis
    knot_slopes::T     # Derivatives at each knot
    range_min::Float64
    range_max::Float64
    # ... other metadata if needed
end
```
*   `T` will be an `AbstractArray`.
*   `N` will be the number of feature dimensions (0 for a single spline, 1 for multiple).

**Step 2: Implement the Main Constructor**
This constructor will be the single, user-facing entry point. It will take raw, unconstrained parameters and process them into the valid spline parameters stored in the struct.

*   **Signature:**
    ```julia
    function RationalQuadraticSpline(
        params::AbstractArray,
        range_min::Real,
        range_max::Real;
        boundary_slopes::Symbol = :unconstrained,
        min_bin_size::Real = 1e-4,
        min_knot_slope::Real = 1e-4,
    )
    ```
*   **Logic:**
    1.  **Input Validation:** Assert that `range_min < range_max`, `min_bin_size > 0`, etc.
    2.  **Parameter Extraction:** Given `params` with size `(3K+1, D)`, where `K` is the number of bins and `D` is the number of feature dimensions, split it into `unnormalized_bin_widths`, `unnormalized_bin_heights`, and `unnormalized_knot_slopes`.
    3.  **Bin Size Normalization:** Use `LogExpFunctions.softmax` on the unnormalized widths and heights along the bin dimension (`dims=1`). Scale the result so that the bins sum to `(range_max - range_min)` while respecting `min_bin_size`.
    4.  **Knot Position Calculation:** Compute `x_pos` and `y_pos` by taking the cumulative sum of the normalized bin widths and heights. Prepend `range_min` and append `range_max` to each.
    5.  **Slope Normalization:** Use `LogExpFunctions.softplus` on the unnormalized slopes to ensure they are positive and greater than `min_knot_slope`, following the numerically stable formula from the Python reference.
    6.  **Boundary Conditions:** Apply the `boundary_slopes` logic (e.g., for `:identity`, set the first and last slopes to 1.0).

**Step 3: Implement Core Spline Logic (Internal Functions)**
Create two internal, scalar-focused functions that perform the core mathematics.

*   `_rational_quadratic_spline_fwd(x::Real, x_pos, y_pos, knot_slopes)`
*   `_rational_quadratic_spline_inv(y::Real, x_pos, y_pos, knot_slopes)`

These functions will handle a single point `x` or `y` with the parameters for a single spline (`x_pos`, etc., will be vectors). They should return a tuple: `(transformed_value, logabsdetjac)`.

**Step 4: Implement the Public API with Multiple Dispatch**
Define the methods that `Bijectors.jl` uses, dispatching to the core logic.

*   **Scalar Input:**
    *   `with_logabsdet_jacobian(b::RationalQuadraticSpline{<:AbstractVector}, x::Real)`: Directly calls `_rational_quadratic_spline_fwd`.
*   **Multi-dimensional Input:**
    *   `with_logabsdet_jacobian(b::RationalQuadraticSpline{<:AbstractMatrix}, x::AbstractVecOrMat)`:
        *   This is the key method for vector and matrix inputs.
        *   It should loop or broadcast over `x`, applying `_rational_quadratic_spline_fwd` to each element.
        *   For each element `x[i]`, it must use the corresponding feature's spline parameters (e.g., `view(b.x_pos, :, feature_idx)`).
        *   The log-determinants should be summed correctly:
            *   For a vector input (one data point), sum the log-dets over all features to produce a scalar.
            *   For a matrix input (a batch of data points), sum the log-dets over the features for *each* data point, returning a vector of log-dets (one for each item in the batch).
*   Implement `transform` and `logabsdetjac` by calling `with_logabsdet_jacobian` and returning the appropriate part of the tuple.
*   Implement methods for `Inverse{<:RationalQuadraticSpline}` similarly.

**Step 5: Implement `bijector_broadcast_shape`**
To ensure compatibility with broadcasting operations in `Bijectors.jl`, implement this method.
*   **Issue:** The function `NNlib.broadcast_shapes` does not exist.
*   **Solution:** Use the internal Julia function `Base.Broadcast.broadcast_shape`.
    ```julia
    Bijectors.bijector_broadcast_shape(b::RationalQuadraticSpline{<:AbstractMatrix}, x::AbstractVecOrMat) =
        Base.Broadcast.broadcast_shape(size(x), (1, size(b.x_pos, 2)))
    ```

#### **4. Testing Plan**

A new test file, `test/simple_test.jl`, will be used.

*   **Constructor Tests:**
    *   Verify that a bijector created with vector `params` has vector-shaped parameters.
    *   Verify that a bijector created with matrix `params` has matrix-shaped parameters.
*   **Scalar Transform Tests:**
    *   Check invertibility: `inverse(b)(transform(b, x)) ≈ x`.
    *   Check the log-determinant property: `logabsdetjac(b, x) + logabsdetjac(inverse(b), transform(b, x)) ≈ 0`.
    *   Check boundary conditions (e.g., with `boundary_slopes=:identity`, transforming a value outside the range should be the identity function).
*   **Vector and Matrix Transform Tests:**
    *   Repeat the invertibility and log-determinant checks for vector and matrix inputs, ensuring the dimensions of the outputs are correct.
*   **Distribution Integration Test:**
    *   Create a `transformed` distribution with `MvNormal` and the `RationalQuadraticSpline`.
    *   Verify that sampling produces values of the correct shape.
    *   Verify that `logpdf` of the transformed distribution is correct: `logpdf(td, y) ≈ logpdf(d, x) - logabsdetjac(b, x)`.

--- 
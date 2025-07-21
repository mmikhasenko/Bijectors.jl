
using LogExpFunctions
using NNlib
#=
Implementation of the Rational Quadratic Spline flow [1].
Reference implementation: https://github.com/deepmind/distrax/blob/master/distrax/_src/bijectors/rational_quadratic_spline.py
This implementation is a rewrite of the original Julia version, adhering more closely to the reference
and addressing issues with multi-dimensional inputs and parameterization.
- Feature dimensions are expected to be the last dimension of the input array.
- The bijector can be parameterized by a vector (for a single spline) or a matrix (for multiple splines, one for each feature).
[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G., Neural Spline Flows, CoRR, arXiv:1906.04032 [stat.ML], (2019).
=#

struct RationalQuadraticSpline{T, S} <: Bijector
    x_pos::T
    y_pos::T
    knot_slopes::T
    range_min::S
    range_max::S
    boundary_slopes::Symbol

    function RationalQuadraticSpline(
        params::AbstractArray,
        range_min::Real = zero(eltype(params)),
        range_max::Real = one(eltype(params));
        boundary_slopes::Symbol = :unconstrained,
        min_bin_size::Real = 1e-4,
        min_knot_slope::Real = 1e-4,
    )
        @assert range_min < range_max
        @assert min_bin_size > 0
        @assert min_knot_slope > 0

        # AD COMPATIBILITY: Ensure all parameters have compatible types
        T = eltype(params)
        range_min_T = T(range_min)
        range_max_T = T(range_max)
        min_bin_size_T = T(min_bin_size)
        min_knot_slope_T = T(min_knot_slope)

        # Determine number of bins and features
        P = size(params, 1)
        num_bins = (P - 1) ÷ 3
        @assert 3 * num_bins + 1 == P "First dimension of `params` must have size `3 * num_bins + 1`"

        # Extract unnormalized parameters along first dimension
        unnormalized_bin_widths = params[1:num_bins, ntuple(i -> :, ndims(params) - 1)...]
        unnormalized_bin_heights = params[num_bins+1:2*num_bins, ntuple(i -> :, ndims(params) - 1)...]
        unnormalized_knot_slopes = params[2*num_bins+1:end, ntuple(i -> :, ndims(params) - 1)...]

        # Normalize bin sizes
        range_size = range_max_T - range_min_T

        # _normalize_bin_sizes from distrax
        bin_widths = LogExpFunctions.softmax(unnormalized_bin_widths; dims = 1)
        bin_widths = bin_widths .* (range_size - convert(T, num_bins) * min_bin_size_T) .+ min_bin_size_T

        bin_heights = LogExpFunctions.softmax(unnormalized_bin_heights; dims = 1)
        bin_heights = bin_heights .* (range_size - convert(T, num_bins) * min_bin_size_T) .+ min_bin_size_T

        # Compute bin positions
        x_pos_inter = cumsum(bin_widths; dims = 1)
        y_pos_inter = cumsum(bin_heights; dims = 1)

        # Add boundaries - handle arbitrary dimensions with AD compatibility
        pad_dims = size(params)[2:end]
        if ndims(params) == 1
            # Scalar parameter case - use typed versions for AD compatibility
            x_pos = vcat([range_min_T], range_min_T .+ x_pos_inter[1:end-1], [range_max_T])
            y_pos = vcat([range_min_T], range_min_T .+ y_pos_inter[1:end-1], [range_max_T])
        else
            # Multi-dimensional parameter case - create compatible arrays
            pad_below = fill(range_min_T, 1, pad_dims...)
            pad_above = fill(range_max_T, 1, pad_dims...)
            x_pos = vcat(pad_below, range_min_T .+ x_pos_inter[1:end-1, ntuple(i -> :, ndims(params) - 1)...], pad_above)
            y_pos = vcat(pad_below, range_min_T .+ y_pos_inter[1:end-1, ntuple(i -> :, ndims(params) - 1)...], pad_above)
        end

        # _normalize_knot_slopes from distrax
        # The offset is such that the normalized knot slope will be equal to 1
        # whenever the unnormalized knot slope is equal to 0.
        if min_knot_slope_T >= one(T)
            throw(ArgumentError("The minimum knot slope must be less than 1; got $(min_knot_slope_T)."))
        end
        offset = log(exp(one(T) - min_knot_slope_T) - one(T))
        knot_slopes_ = LogExpFunctions.softplus.(unnormalized_knot_slopes .+ offset) .+ min_knot_slope_T

        if boundary_slopes === :unconstrained
            knot_slopes = knot_slopes_
        elseif boundary_slopes === :identity
            if ndims(params) == 1
                # Scalar parameter case - use typed one()
                knot_slopes = vcat([one(T)], knot_slopes_[2:end-1], [one(T)])
            else
                # Multi-dimensional parameter case - create compatible arrays
                ones_pad = fill(one(T), 1, pad_dims...)
                knot_slopes = vcat(ones_pad, knot_slopes_[2:end-1, ntuple(i -> :, ndims(params) - 1)...], ones_pad)
            end
        else
            throw(ArgumentError("Unknown boundary_slopes: $boundary_slopes"))
        end

        # Create struct with flexible type handling
        T_array = typeof(x_pos)
        S_scalar = typeof(range_min_T)
        new{T_array, S_scalar}(x_pos, y_pos, knot_slopes, range_min_T, range_max_T, boundary_slopes)
    end
end



# Helper function to validate broadcasting compatibility
function _validate_broadcasting(params, x)
    param_dims = size(params)[2:end]  # Skip first dimension (3*num_bins + 1)
    x_dims = size(x)

    if length(param_dims) > length(x_dims)
        throw(DimensionMismatch("Parameter dimensions $(param_dims) exceed input dimensions $(x_dims). Cannot broadcast to preserve output shape."))
    end

    # Check if the trailing dimensions are compatible for broadcasting
    for i in 1:length(param_dims)
        if param_dims[i] != 1 && param_dims[i] != x_dims[i]
            throw(DimensionMismatch("Parameter dimension $(param_dims[i]) incompatible with input dimension $(x_dims[i]) at axis $i"))
        end
    end

    return true
end

# Helper function to get parameter slice for a given index
function _get_param_slice(params, indices...)
    if ndims(params) == 1
        # Single spline for all elements
        return view(params, :)
    else
        # Get appropriate slice based on broadcasting rules
        param_indices = ntuple(ndims(params) - 1) do i
            if i <= length(indices)
                if size(params, i + 1) == 1
                    1  # Broadcast dimension
                else
                    indices[i]  # Index into parameter dimension
                end
            else
                1  # Default to 1 for missing dimensions
            end
        end
        return view(params, :, param_indices...)
    end
end



# Core forward transformation for a scalar
function _rational_quadratic_spline_fwd(x, x_pos, y_pos, knot_slopes, range_min, range_max)
    # Handle values outside the range first
    if x <= range_min || x >= range_max
        return x, 0.0
    end

    # Search for the correct bin
    # This is faster than searchsortedfirst on GPU
    # Use of `x .>` instead of `>=` ensures that we find the bin `x` is in, even at boundaries.
    k = sum(x .> x_pos[2:end-1]) + 1

    # Parameters for the selected bin
    x_k = x_pos[k]
    x_kp1 = x_pos[k+1]
    y_k = y_pos[k]
    y_kp1 = y_pos[k+1]
    s_k = knot_slopes[k]
    s_kp1 = knot_slopes[k+1]

    # Normalize x to [0, 1]
    z = (x - x_k) / (x_kp1 - x_k)

    # Transformation
    bin_height = y_kp1 - y_k
    bin_width = x_kp1 - x_k
    bin_slope = bin_height / bin_width

    slopes_term = s_kp1 + s_k - 2 * bin_slope
    numerator = bin_height * (bin_slope * z^2 + s_k * z * (1 - z))
    denominator = bin_slope + slopes_term * z * (1 - z)
    y = y_k + numerator / denominator

    # Log-determinant of the Jacobian
    logdet = 2 * log(bin_slope) + log(s_kp1 * z^2 + 2 * bin_slope * z * (1 - z) + s_k * (1 - z)^2) - 2 * log(denominator)

    return y, logdet
end


# Core inverse transformation for a scalar
function _rational_quadratic_spline_inv(y, x_pos, y_pos, knot_slopes, range_min, range_max)
    # Handle values outside the range first
    if y <= range_min || y >= range_max
        return y, 0.0
    end

    k = sum(y .> y_pos[2:end-1]) + 1

    x_k, x_kp1 = x_pos[k], x_pos[k+1]
    y_k, y_kp1 = y_pos[k], y_pos[k+1]
    s_k, s_kp1 = knot_slopes[k], knot_slopes[k+1]

    bin_width = x_kp1 - x_k
    bin_height = y_kp1 - y_k
    bin_slope = bin_height / bin_width

    w = (y - y_k) / bin_height

    slopes_term = s_kp1 + s_k - 2 * bin_slope

    c = -bin_slope * w
    b = s_k - slopes_term * w
    a = bin_slope - b

    # Numerically stable quadratic root finder
    discr = b^2 - 4 * a * c
    # Due to floating point errors, discr can be slightly negative
    z = -2 * c / (b + sqrt(max(discr, 0.0)))

    x = z * bin_width + x_k

    # Log-determinant of the Jacobian for the inverse
    logdet_inv = -(2 * log(bin_slope) + log(s_kp1 * z^2 + 2 * bin_slope * z * (1 - z) + s_k * (1 - z)^2) - 2 * log(bin_slope + slopes_term * z * (1 - z)))

    return x, logdet_inv
end

# Scalar input
function Bijectors.with_logabsdet_jacobian(b::RationalQuadraticSpline, x::Real)
    _validate_broadcasting(b.x_pos, x)
    x_pos_slice = _get_param_slice(b.x_pos)
    y_pos_slice = _get_param_slice(b.y_pos)
    knot_slopes_slice = _get_param_slice(b.knot_slopes)
    return _rational_quadratic_spline_fwd(x, x_pos_slice, y_pos_slice, knot_slopes_slice, b.range_min, b.range_max)
end

# Array input with comprehensive broadcasting
function Bijectors.with_logabsdet_jacobian(b::RationalQuadraticSpline, x::AbstractArray)
    _validate_broadcasting(b.x_pos, x)

    y = similar(x)
    logdet = similar(x)

    # Handle different input dimensions with broadcasting
    if ndims(x) == 1
        # Vector case: x has dimensions (N,)
        for i in eachindex(x)
            x_pos_slice = _get_param_slice(b.x_pos, i)
            y_pos_slice = _get_param_slice(b.y_pos, i)
            knot_slopes_slice = _get_param_slice(b.knot_slopes, i)

            y[i], logdet[i] = _rational_quadratic_spline_fwd(
                x[i], x_pos_slice, y_pos_slice, knot_slopes_slice, b.range_min, b.range_max,
            )
        end
        return y, sum(logdet)

    elseif ndims(x) == 2
        # Matrix case: x has dimensions (N, D)
        for col in 1:size(x, 2), row in 1:size(x, 1)
            x_pos_slice = _get_param_slice(b.x_pos, row, col)
            y_pos_slice = _get_param_slice(b.y_pos, row, col)
            knot_slopes_slice = _get_param_slice(b.knot_slopes, row, col)

            y[row, col], logdet[row, col] = _rational_quadratic_spline_fwd(
                x[row, col], x_pos_slice, y_pos_slice, knot_slopes_slice, b.range_min, b.range_max,
            )
        end
        return y, vec(sum(logdet, dims = 2))

    else
        # Higher-dimensional arrays
        for idx in CartesianIndices(x)
            indices = Tuple(idx)
            x_pos_slice = _get_param_slice(b.x_pos, indices...)
            y_pos_slice = _get_param_slice(b.y_pos, indices...)
            knot_slopes_slice = _get_param_slice(b.knot_slopes, indices...)

            y[idx], logdet[idx] = _rational_quadratic_spline_fwd(
                x[idx], x_pos_slice, y_pos_slice, knot_slopes_slice, b.range_min, b.range_max,
            )
        end

        # Sum logdet over the last dimension to preserve input shape
        return y, dropdims(sum(logdet, dims = ndims(logdet)), dims = ndims(logdet))
    end
end

transform(b::RationalQuadraticSpline, x) = with_logabsdet_jacobian(b, x)[1]
logabsdetjac(b::RationalQuadraticSpline, x) = with_logabsdet_jacobian(b, x)[2]


# Inverse for scalar input
function Bijectors.with_logabsdet_jacobian(ib::Inverse{<:RationalQuadraticSpline}, y::Real)
    b = ib.orig
    _validate_broadcasting(b.x_pos, y)
    x_pos_slice = _get_param_slice(b.x_pos)
    y_pos_slice = _get_param_slice(b.y_pos)
    knot_slopes_slice = _get_param_slice(b.knot_slopes)
    return _rational_quadratic_spline_inv(y, x_pos_slice, y_pos_slice, knot_slopes_slice, b.range_min, b.range_max)
end

# Inverse for array input with comprehensive broadcasting
function Bijectors.with_logabsdet_jacobian(ib::Inverse{<:RationalQuadraticSpline}, y::AbstractArray)
    b = ib.orig
    _validate_broadcasting(b.x_pos, y)

    x = similar(y)
    logdet = similar(y)

    # Handle different input dimensions with broadcasting
    if ndims(y) == 1
        # Vector case: y has dimensions (N,)
        for i in eachindex(y)
            x_pos_slice = _get_param_slice(b.x_pos, i)
            y_pos_slice = _get_param_slice(b.y_pos, i)
            knot_slopes_slice = _get_param_slice(b.knot_slopes, i)

            x[i], logdet[i] = _rational_quadratic_spline_inv(
                y[i], x_pos_slice, y_pos_slice, knot_slopes_slice, b.range_min, b.range_max,
            )
        end
        return x, sum(logdet)

    elseif ndims(y) == 2
        # Matrix case: y has dimensions (N, D)
        for col in 1:size(y, 2), row in 1:size(y, 1)
            x_pos_slice = _get_param_slice(b.x_pos, row, col)
            y_pos_slice = _get_param_slice(b.y_pos, row, col)
            knot_slopes_slice = _get_param_slice(b.knot_slopes, row, col)

            x[row, col], logdet[row, col] = _rational_quadratic_spline_inv(
                y[row, col], x_pos_slice, y_pos_slice, knot_slopes_slice, b.range_min, b.range_max,
            )
        end
        return x, vec(sum(logdet, dims = 2))

    else
        # Higher-dimensional arrays
        for idx in CartesianIndices(y)
            indices = Tuple(idx)
            x_pos_slice = _get_param_slice(b.x_pos, indices...)
            y_pos_slice = _get_param_slice(b.y_pos, indices...)
            knot_slopes_slice = _get_param_slice(b.knot_slopes, indices...)

            x[idx], logdet[idx] = _rational_quadratic_spline_inv(
                y[idx], x_pos_slice, y_pos_slice, knot_slopes_slice, b.range_min, b.range_max,
            )
        end

        # Sum logdet over the last dimension to preserve input shape
        return x, dropdims(sum(logdet, dims = ndims(logdet)), dims = ndims(logdet))
    end
end

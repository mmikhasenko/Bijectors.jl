
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

struct RationalQuadraticSpline{T, N} <: Bijector
    x_pos::T
    y_pos::T
    knot_slopes::T
    range_min::Float64
    range_max::Float64
    boundary_slopes::Symbol

    function RationalQuadraticSpline(
        params::AbstractArray,
        range_min::Real,
        range_max::Real;
        boundary_slopes::Symbol = :unconstrained,
        min_bin_size::Real = 1e-4,
        min_knot_slope::Real = 1e-4,
    )
        @assert range_min < range_max
        @assert min_bin_size > 0
        @assert min_knot_slope > 0

        # Determine number of bins and features
        P, D... = size(params)
        num_bins = (P - 1) ÷ 3
        @assert 3 * num_bins + 1 == P "Last dimension of `params` must have size `3 * num_bins + 1`"

        # Extract unnormalized parameters
        if ndims(params) == 1
            unnormalized_bin_widths = params[1:num_bins]
            unnormalized_bin_heights = params[num_bins+1:2*num_bins]
            unnormalized_knot_slopes = params[2*num_bins+1:end]
        else
            unnormalized_bin_widths = params[1:num_bins, :]
            unnormalized_bin_heights = params[num_bins+1:2*num_bins, :]
            unnormalized_knot_slopes = params[2*num_bins+1:end, :]
        end

        # Helper for broadcasting pads
        _pad_shape(pads) = reshape(pads, 1, size(pads)[2:end]...)

        # Normalize bin sizes
        range_size = range_max - range_min

        # _normalize_bin_sizes from distrax
        bin_widths = LogExpFunctions.softmax(unnormalized_bin_widths; dims = 1)
        bin_widths = bin_widths .* (range_size - num_bins * min_bin_size) .+ min_bin_size

        bin_heights = LogExpFunctions.softmax(unnormalized_bin_heights; dims = 1)
        bin_heights = bin_heights .* (range_size - num_bins * min_bin_size) .+ min_bin_size

        # Compute bin positions
        x_pos_inter = cumsum(bin_widths; dims = 1)
        y_pos_inter = cumsum(bin_heights; dims = 1)

        # Add boundaries
        if ndims(params) == 1
            x_pos = vcat([range_min], range_min .+ x_pos_inter[1:end-1], [range_max])
            y_pos = vcat([range_min], range_min .+ y_pos_inter[1:end-1], [range_max])
        else
            pad_below = fill(range_min, (1, D...))
            x_pos = vcat(pad_below, range_min .+ x_pos_inter[1:end-1, :], fill(range_max, (1, D...)))
            y_pos = vcat(pad_below, range_min .+ y_pos_inter[1:end-1, :], fill(range_max, (1, D...)))
        end

        # _normalize_knot_slopes from distrax
        # The offset is such that the normalized knot slope will be equal to 1
        # whenever the unnormalized knot slope is equal to 0.
        if min_knot_slope >= 1.0
            throw(ArgumentError("The minimum knot slope must be less than 1; got $(min_knot_slope)."))
        end
        offset = log(exp(1.0 - min_knot_slope) - 1.0)
        knot_slopes_ = LogExpFunctions.softplus.(unnormalized_knot_slopes .+ offset) .+ min_knot_slope

        if boundary_slopes === :unconstrained
            knot_slopes = knot_slopes_
        elseif boundary_slopes === :identity
            if ndims(params) == 1
                knot_slopes = vcat([one(eltype(params))], knot_slopes_[2:end-1], [one(eltype(params))])
            else
                ones_pad = ones(eltype(params), (1, D...))
                knot_slopes = vcat(ones_pad, knot_slopes_[2:end-1, :], ones_pad)
            end
        else
            throw(ArgumentError("Unknown boundary_slopes: $boundary_slopes"))
        end

        T = typeof(x_pos)
        N_ = ndims(params) - 1
        new{T, N_}(x_pos, y_pos, knot_slopes, range_min, range_max, boundary_slopes)
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

# Scalar input, single spline
function Bijectors.with_logabsdet_jacobian(b::RationalQuadraticSpline{<:AbstractVector}, x::Real)
    _rational_quadratic_spline_fwd(x, b.x_pos, b.y_pos, b.knot_slopes, b.range_min, b.range_max)
end
transform(b::RationalQuadraticSpline{<:AbstractVector}, x::Real) = _rational_quadratic_spline_fwd(x, b.x_pos, b.y_pos, b.knot_slopes, b.range_min, b.range_max)[1]
logabsdetjac(b::RationalQuadraticSpline{<:AbstractVector}, x::Real) = _rational_quadratic_spline_fwd(x, b.x_pos, b.y_pos, b.knot_slopes, b.range_min, b.range_max)[2]

# Multi-dim input, multi-spline
function Bijectors.with_logabsdet_jacobian(b::RationalQuadraticSpline{<:AbstractMatrix}, x::AbstractVecOrMat)
    y = similar(x)
    logdet = similar(x)

    # Broadcasting over features and batch
    if ndims(x) == 1
        # Vector case: each element uses its corresponding feature
        for i in eachindex(x)
            y[i], logdet[i] = _rational_quadratic_spline_fwd(
                x[i],
                view(b.x_pos, :, i),
                view(b.y_pos, :, i),
                view(b.knot_slopes, :, i),
                b.range_min,
                b.range_max,
            )
        end
    else
        # Matrix case: iterate over all elements, using column index for feature
        for col in 1:size(x, 2), row in 1:size(x, 1)
            y[row, col], logdet[row, col] = _rational_quadratic_spline_fwd(
                x[row, col],
                view(b.x_pos, :, col),
                view(b.y_pos, :, col),
                view(b.knot_slopes, :, col),
                b.range_min,
                b.range_max,
            )
        end
    end

    if ndims(x) == 1
        return y, sum(logdet)
    else # Matrix
        return y, vec(sum(logdet, dims = 2))
    end
end
transform(b::RationalQuadraticSpline, x::AbstractVecOrMat) = with_logabsdet_jacobian(b, x)[1]
logabsdetjac(b::RationalQuadraticSpline, x::AbstractVecOrMat) = with_logabsdet_jacobian(b, x)[2]


# Inverse for scalar input, single spline
function Bijectors.with_logabsdet_jacobian(ib::Inverse{<:RationalQuadraticSpline{<:AbstractVector}}, y::Real)
    b = ib.orig
    _rational_quadratic_spline_inv(y, b.x_pos, b.y_pos, b.knot_slopes, b.range_min, b.range_max)
end

# Inverse for multi-dim input, multi-spline
function Bijectors.with_logabsdet_jacobian(ib::Inverse{<:RationalQuadraticSpline{<:AbstractMatrix}}, y::AbstractVecOrMat)
    b = ib.orig
    x = similar(y)
    logdet = similar(y)

    if ndims(y) == 1
        # Vector case: each element uses its corresponding feature
        for i in eachindex(y)
            x[i], logdet[i] = _rational_quadratic_spline_inv(
                y[i],
                view(b.x_pos, :, i),
                view(b.y_pos, :, i),
                view(b.knot_slopes, :, i),
                b.range_min,
                b.range_max,
            )
        end
    else
        # Matrix case: iterate over all elements, using column index for feature
        for col in 1:size(y, 2), row in 1:size(y, 1)
            x[row, col], logdet[row, col] = _rational_quadratic_spline_inv(
                y[row, col],
                view(b.x_pos, :, col),
                view(b.y_pos, :, col),
                view(b.knot_slopes, :, col),
                b.range_min,
                b.range_max,
            )
        end
    end

    if ndims(y) == 1
        return x, sum(logdet)
    else # Matrix
        return x, vec(sum(logdet, dims = 2))
    end
end

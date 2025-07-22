using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using NormalizingFlows
using Bijectors
using Bijectors.LogExpFunctions
using NormalizingFlows.Distributions
using NormalizingFlows.Optimisers
using DifferentiationInterface
using NormalizingFlows
using Plots
import Flux: Dense, Chain, relu
using Enzyme: Enzyme
using Mooncake: Mooncake

using Random
rng = MersenneTwister(42)



struct NewRationalQuadraticSpline{T, N} <: Bijectors.Bijector
    x_pos::AbstractArray{T, N}
    y_pos::AbstractArray{T, N}
    knot_slopes::AbstractArray{T, N}
end

transform(b::NewRationalQuadraticSpline, x) = with_logabsdet_jacobian(b, x)[1]
logabsdetjac(b::NewRationalQuadraticSpline, x) = with_logabsdet_jacobian(b, x)[2]

# Type-stable constructor for 1D parameters (AbstractVector)
function NewRationalQuadraticSpline(
    params::AbstractVector{T},
) where {T <: Real}

    # Convert to consistent types
    range_min = T(0.0)
    range_max = T(1.0)

    # Determine number of bins
    P = length(params)
    num_bins = (P - 1) ÷ 3
    3 * num_bins + 1 == P || throw(ArgumentError("Length of `params` must be `3 * num_bins + 1`"))

    # Extract unnormalized parameters
    unnormalized_bin_widths = @view params[1:num_bins]
    unnormalized_bin_heights = @view params[num_bins+1:2*num_bins]
    unnormalized_knot_slopes = @view params[2*num_bins+1:end]

    # _normalize_bin_sizes from distrax
    bin_widths = LogExpFunctions.softmax(unnormalized_bin_widths)
    bin_heights = LogExpFunctions.softmax(unnormalized_bin_heights)

    # Compute bin positions
    x_pos_inter = cumsum(bin_widths)
    y_pos_inter = cumsum(bin_heights)

    # Add boundaries - 1D case
    x_pos = vcat([range_min], range_min .+ x_pos_inter[1:end-1], [range_max])
    y_pos = vcat([range_min], range_min .+ y_pos_inter[1:end-1], [range_max])

    offset = log(exp(one(T)) - one(T))
    knot_slopes = LogExpFunctions.softplus.(unnormalized_knot_slopes .+ offset)

    NewRationalQuadraticSpline{T, 1}(x_pos, y_pos, knot_slopes)
end


# Core forward transformation for a scalar
function _rational_quadratic_spline_fwd(x, x_pos, y_pos, knot_slopes)

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
function _rational_quadratic_spline_inv(y, x_pos, y_pos, knot_slopes)

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
function Bijectors.with_logabsdet_jacobian(b::NewRationalQuadraticSpline, x::Real)
    x_pos_slice = b.x_pos
    y_pos_slice = b.y_pos
    knot_slopes_slice = b.knot_slopes
    return _rational_quadratic_spline_fwd(x, x_pos_slice, y_pos_slice, knot_slopes_slice)
end

# Array input with comprehensive broadcasting
function Bijectors.with_logabsdet_jacobian(b::NewRationalQuadraticSpline, x::AbstractVector)
    y = similar(x)
    logdet = similar(x)

    x_pos_slice = b.x_pos
    y_pos_slice = b.y_pos
    knot_slopes_slice = b.knot_slopes

    for i in eachindex(x)
        y[i], logdet[i] = _rational_quadratic_spline_fwd(
            x[i], x_pos_slice, y_pos_slice, knot_slopes_slice,
        )
    end
    return y, sum(logdet)
end


# Inverse for scalar input
function Bijectors.with_logabsdet_jacobian(ib::Inverse{<:NewRationalQuadraticSpline}, y::Real)
    b = ib.orig
    x_pos_slice = b.x_pos
    y_pos_slice = b.y_pos
    knot_slopes_slice = b.knot_slopes
    return _rational_quadratic_spline_inv(y, x_pos_slice, y_pos_slice, knot_slopes_slice)
end

# Inverse for array input with comprehensive broadcasting
function Bijectors.with_logabsdet_jacobian(ib::Inverse{<:NewRationalQuadraticSpline}, y::AbstractArray)
    b = ib.orig

    x = similar(y)
    logdet = similar(y)

    # Vector case: y has dimensions (N,)
    x_pos_slice = b.x_pos
    y_pos_slice = b.y_pos
    knot_slopes_slice = b.knot_slopes
    for i in eachindex(y)
        x[i], logdet[i] = _rational_quadratic_spline_inv(
            y[i], x_pos_slice, y_pos_slice, knot_slopes_slice,
        )
    end
    return x, sum(logdet)
end





const conditioner = Dense(1, 7, relu)

flow = let
    q₀ = Distributions.Product([Uniform(0, 1), Uniform(0, 1)])
    mask = Bijectors.PartitionMask(2, [1], [2])
    ts = Coupling(NewRationalQuadraticSpline ∘ conditioner, mask)
    transformed(q₀, ts)
end

# more complicated function
p_target = MultivariateNormal(
    [0.5, 0.5] .|> Float32,
    [0.01 0.0; 0.0 0.01] .|> Float32)
data = rand(p_target, 100)

NormalizingFlows.loglikelihood(rng, flow, data)

let
    θ, re = Optimisers.destructure(flow)
    loss(θ) = -NormalizingFlows.loglikelihood(rng, re(θ), data)
    loss(θ)
end


let
    θ, re = Optimisers.destructure(flow)
    loss(θ) = -NormalizingFlows.loglikelihood(rng, re(θ), data)


    ADbackend = AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Const)

    prep = prepare_gradient(loss, ADbackend, θ)
    gradient(loss, prep, ADbackend, θ)
end


let
    θ, re = Optimisers.destructure(flow)
    loss(θ) = -NormalizingFlows.loglikelihood(rng, re(θ), data)


    ADbackend = AutoMooncake(; config = nothing)

    prep = prepare_gradient(loss, ADbackend, θ)
    gradient(loss, prep, ADbackend, θ)
end






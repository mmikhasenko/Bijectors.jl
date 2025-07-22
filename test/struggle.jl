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

    num_bins = 2
    x_pos = @view params[1:num_bins]
    y_pos = @view params[num_bins+1:2*num_bins]
    knot_slopes = @view params[2*num_bins+1:end-1]

    NewRationalQuadraticSpline{T, 1}(x_pos, y_pos, knot_slopes)
end


# Core forward transformation for a scalar
function _rational_quadratic_spline_fwd(x, x_pos, y_pos, knot_slopes)
    k = sum(x .> x_pos[2:end-1]) + 1
    return y_pos[k], knot_slopes[k]
end


# Core inverse transformation for a scalar
function _rational_quadratic_spline_inv(y, x_pos, y_pos, knot_slopes)

    k = sum(y .> y_pos[2:end-1]) + 1
    return x_pos[k], -knot_slopes[k]
end

# Scalar input
Bijectors.with_logabsdet_jacobian(b::NewRationalQuadraticSpline, x::Real) =
    _rational_quadratic_spline_fwd(x, b.x_pos, b.y_pos, b.knot_slopes)

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





flow = let
    q₀ = Distributions.Product([Uniform(0, 1), Uniform(0, 1)])
    mask12 = Bijectors.PartitionMask(2, [1], [2])
    mask21 = Bijectors.PartitionMask(2, [2], [1])
    layers = [
        Coupling(NewRationalQuadraticSpline ∘ Dense(1, 7), mask12),
        Coupling(NewRationalQuadraticSpline ∘ Dense(1, 7), mask21),
    ]
    ts = reduce(∘, layers)
    transformed(q₀, ts)
end

# more complicated function
const data = rand(Float32, 2, 2)
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

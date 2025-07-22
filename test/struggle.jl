using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

import Bijectors: Coupling, RationalQuadraticSpline, PartitionMask, transformed
using Bijectors.Distributions
import NormalizingFlows.Optimisers

import Flux: Dense
using Enzyme

using Random
rng = MersenneTwister(42)

flow = let
    q₀ = Distributions.Product([Uniform(0, 1), Uniform(0, 1)])
    mask12 = PartitionMask(2, [1], [2])
    mask21 = PartitionMask(2, [2], [1])
    layers = [
        Coupling(RationalQuadraticSpline ∘ Dense(1, 7), mask12),
        Coupling(RationalQuadraticSpline ∘ Dense(1, 7), mask21),
    ]
    ts = reduce(∘, layers)
    transformed(q₀, ts)
end

let
    θ, re = Optimisers.destructure(flow)
    loss(θ) = -pdf(re(θ), Float32[0.5, 0.5])
    loss(θ)
end # works

# let # fails
#     θ, re = Optimisers.destructure(flow)
#     loss(θ) = -pdf(re(θ), Float32[0.5, 0.5])

#     ADbackend = AutoEnzyme(;
#         mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
#         function_annotation = Enzyme.Const)

#     prep = prepare_gradient(loss, ADbackend, θ)
#     DifferentiationInterface.gradient(loss, prep, ADbackend, θ)
# end

let # fails
    θ, re = Optimisers.destructure(flow)
    loss(θ) = -logpdf(re(θ), rand(Float32, 2))
    Enzyme.gradient(Enzyme.Reverse, loss, θ)
end

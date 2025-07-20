using Flux
using Bijectors
using Bijectors: RationalQuadraticSpline
using Distributions
using Random

using Plots
theme(:boxed)


const mask12 = Bijectors.PartitionMask(2, [1], [2]) # acts on 1, conditioned on 2
const mask21 = Bijectors.PartitionMask(2, [2], [1]) # acts on 2, conditioned on 1

conditioner = Flux.Dense(0.01 .* randn(31, 1), 0.01 .* randn(31)) # 1->31, 62 parameters

function create_bijector(condition_z)
    z_dims = size(condition_z)
    target_dims = z_dims[2:end]

    if isempty(target_dims)
        return RationalQuadraticSpline(conditioner(condition_z), 0, 1)
    else
        return Bijectors.Reshape(target_dims, z_dims) ∘
               RationalQuadraticSpline(conditioner(condition_z), 0, 1) ∘
               Bijectors.Reshape(z_dims, target_dims)
    end
end


# basic test
q_test = let
    _base = Distributions.Product([Uniform(0, 1), Uniform(0, 1)])
    _mask = Bijectors.PartitionMask(2, [2], [1])
    _flow = Coupling(RationalQuadraticSpline ∘ conditioner, _mask)
    Bijectors.transformed(_base, _flow)
end

pdf(q_test, rand(2))
rand(q_test)


θ_flat, re = Optimisers.destructure(q_advanced)

size(θ_flat)

samples = rand(q_test, 100_000)

# Create the visualization
plot(layout = grid(1, 2), size = (1000, 500),
    heatmap(range(0, 1, 100), range(0, 1, 100), (x, y) -> pdf(q_test, [x, y]), aspect_ratio = 1),
    histogram2d(samples[1, :], samples[2, :], aspect_ratio = 1, bins = 50))
# 


# advanced test

using FunctionChains
# Functors

function create_flow(n_layers::Int, q₀)
    Ls = map(1:n_layers) do i
        isodd(i) ?
        Coupling(RationalQuadraticSpline ∘ conditioner, mask12) :
        Coupling(RationalQuadraticSpline ∘ conditioner, mask21)
    end
    ts = fchain(Ls)
    return transformed(q₀, ts)
end


q_advanced = let
    _base = Distributions.Product([Uniform(0, 1), Uniform(0, 1)])
    create_flow(10, _base)
end

samples = rand(q_advanced, 100_000)

# Create the visualization
plot(layout = grid(1, 2), size = (1000, 500),
    heatmap(range(0, 1, 100), range(0, 1, 100), (x, y) -> pdf(q_advanced, [x, y]), aspect_ratio = 1),
    histogram2d(samples[1, :], samples[2, :], aspect_ratio = 1, bins = 50))
# 


θ_flat, re = Optimisers.destructure(q_advanced)
θ_flat


Optimisers.destructure(Dense(3, 13))

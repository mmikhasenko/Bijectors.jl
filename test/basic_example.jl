using Flux
using Bijectors
using Bijectors: RationalQuadraticSpline
using Distributions
using Random

using Plots
theme(:boxed)


const mask12 = Bijectors.PartitionMask(2, [1], [2]) # acts on 1, conditioned on 2
const mask21 = Bijectors.PartitionMask(2, [2], [1]) # acts on 2, conditioned on 1

conditioner = Flux.Dense(randn(7, 1), zeros(7), relu)

# function create_bijector(condition_z)
#     z_dims = size(condition_z)
#     target_dims = z_dims[2:end]

#     if isempty(target_dims)
#         return RationalQuadraticSpline(conditioner(condition_z), 0, 1)
#     else
#         return Bijectors.Reshape(target_dims, z_dims) ∘
#                RationalQuadraticSpline(conditioner(condition_z), 0, 1) ∘
#                Bijectors.Reshape(z_dims, target_dims)
#     end
# end

# basic test
q_test = let
    _base = Distributions.Product([Uniform(0, 1), Uniform(0, 1)])
    _mask = Bijectors.PartitionMask(2, [2], [1])
    _flow = Coupling(RationalQuadraticSpline ∘ conditioner, _mask)
    Bijectors.transformed(_base, _flow)
end

pdf(q_test, rand(2))
rand(q_test)


# θ_flat, re = Optimisers.destructure(q_advanced)

# size(θ_flat)

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
    create_flow(4, _base)
end

samples = rand(q_advanced, 100_000)

# Create the visualization
plot(layout = grid(1, 2), size = (1000, 500),
    heatmap(range(0, 1, 100), range(0, 1, 100), (x, y) -> pdf(q_advanced, [x, y]), aspect_ratio = 1),
    histogram2d(samples[1, :], samples[2, :], aspect_ratio = 1, bins = 50))
# 

using NormalizingFlows
using NormalizingFlows.ADTypes
using Enzyme

d_target = Distributions.Product([Normal(0.5, 0.2), Normal(0.5, 0.2)])

let
    logp = Base.Fix1(logpdf, d_target)
    NormalizingFlows.elbo_single_sample(q_advanced, logp, [0.2, 0.3])
end

sample_per_iter = 100
flow_trained, stats, _ = train_flow(
    NormalizingFlows.elbo,
    q_advanced,
    Base.Fix1(logpdf, d_target),
    sample_per_iter;
    max_iters = 20,
    optimiser = Optimisers.ADAM(0.01),
    ADbackend = ADTypes.AutoEnzyme(;
        function_annotation = Enzyme.Duplicated,
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse)),
)

# y -> logpdf(d_target, y)


# plot()
let
    plot(layout = grid(1, 2), size = (1000, 500),
        heatmap(range(0, 1, 100), range(0, 1, 100), (x, y) -> pdf(d_target, [x, y]), aspect_ratio = 1),
        heatmap(range(0, 1, 100), range(0, 1, 100), (x, y) -> pdf(flow_trained, [x, y]), aspect_ratio = 1),
    )
end


using Flux
using Bijectors
using Bijectors: RationalQuadraticSpline
using Distributions
using Random

using Plots
theme(:boxed)


# FF = RationalQuadraticSpline(rand(31, 10), 0, 1)
# FF(rand(1, 10))

const mask12 = Bijectors.PartitionMask(2, [1], [2]) # acts on 1, conditioned on 2
const mask21 = Bijectors.PartitionMask(2, [2], [1]) # acts on 2, conditioned on 1
const conditioner = Flux.Dense(randn(31, 1), randn(31)) # 1->31, 62 parameters

# Use existing Reshape bijector instead of custom implementations
# Reshape expects (in_shape, out_shape) as constructor arguments

layer12 = Bijectors.Coupling(mask12) do z
    z_dims = size(z)
    @assert z_dims[1] == 1
    target_dims = z_dims[2:end]

    if isempty(target_dims)
        return RationalQuadraticSpline(conditioner(z), 0, 1)
    else
        return Bijectors.Reshape(target_dims, z_dims) ∘
               RationalQuadraticSpline(conditioner(z), 0, 1) ∘
               Bijectors.Reshape(z_dims, target_dims)
    end
end

layer12([0.1, 0.2])
layer12(rand(2, 5)) # does not work consustently with my rules


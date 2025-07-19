using Bijectors
using Bijectors: RationalQuadraticSpline
using Distributions
using LogExpFunctions: softmax, softplus
using Random

Random.seed!(123)

println("--- Constructor Tests ---")
# Test scalar case
println("Scalar case:")
K = 10
D = 1
params = randn(3 * K + 1)
b = RationalQuadraticSpline(params, -4.0, 4.0)
println("size(b.x_pos) == (K + 1,): ", size(b.x_pos) == (K + 1,))
println("size(b.y_pos) == (K + 1,): ", size(b.y_pos) == (K + 1,))
println("size(b.knot_slopes) == (K + 1,): ", size(b.knot_slopes) == (K + 1,))
println("b.range_min == -4.0: ", b.range_min == -4.0)
println("b.range_max == 4.0: ", b.range_max == 4.0)

# Test matrix case
println("\nMatrix case:")
D = 5
params_mat = randn(3 * K + 1, D)
b_mat = RationalQuadraticSpline(params_mat, -4.0, 4.0)
println("size(b_mat.x_pos) == (K + 1, D): ", size(b_mat.x_pos) == (K + 1, D))
println("size(b_mat.y_pos) == (K + 1, D): ", size(b_mat.y_pos) == (K + 1, D))
println("size(b_mat.knot_slopes) == (K + 1, D): ", size(b_mat.knot_slopes) == (K + 1, D))

println("\n--- Scalar Transform Tests ---")
K = 10
params = randn(3 * K + 1)
b = RationalQuadraticSpline(params, -4.0, 4.0)

x_scalar = 1.0
y_scalar, logdet_scalar = with_logabsdet_jacobian(b, x_scalar)

println("y_scalar isa Real: ", y_scalar isa Real)
println("logdet_scalar isa Real: ", logdet_scalar isa Real)

# Test identity outside of range with identity boundary slopes
println("\nIdentity outside of range:")
b_ident = RationalQuadraticSpline(params, -4.0, 4.0, boundary_slopes = :identity)
println("transform(b_ident, -5.0) == -5.0: ", transform(b_ident, -5.0) == -5.0)
println("transform(b_ident, 5.0) == 5.0: ", transform(b_ident, 5.0) == 5.0)
println("logabsdetjac(b_ident, -5.0) == 0.0: ", logabsdetjac(b_ident, -5.0) == 0.0)
println("logabsdetjac(b_ident, 5.0) == 0.0: ", logabsdetjac(b_ident, 5.0) == 0.0)

# Invertibility
println("\nInvertibility:")
x_recon_scalar = transform(inverse(b), y_scalar)
println("x_recon_scalar ≈ x_scalar: ", isapprox(x_recon_scalar, x_scalar))
println("x_scalar: $x_scalar, x_recon_scalar: $x_recon_scalar")

# Logdet property
println("\nLogdet property:")
logdet_inv_scalar = logabsdetjac(inverse(b), y_scalar)
println("logdet_scalar + logdet_inv_scalar ≈ 0.0: ", isapprox(logdet_scalar + logdet_inv_scalar, 0.0, atol = 1e-6))
println("logdet_scalar + logdet_inv_scalar = $(logdet_scalar + logdet_inv_scalar)")


println("\n--- Vector Transform Tests ---")
K = 10
D_vec = 5
params_vec = randn(3 * K + 1, D_vec)
b_vec = RationalQuadraticSpline(params_vec, -4.0, 4.0)

x_vec = randn(D_vec)
y_vec, logdet_vec = with_logabsdet_jacobian(b_vec, x_vec)

println("y_vec isa AbstractVector: ", y_vec isa AbstractVector)
println("size(y_vec) == (D_vec,): ", size(y_vec) == (D_vec,))
println("logdet_vec isa Real: ", logdet_vec isa Real)

# Invertibility
println("\nInvertibility:")
x_recon_vec = transform(inverse(b_vec), y_vec)
println("x_recon_vec ≈ x_vec: ", isapprox(x_recon_vec, x_vec))

# Logdet property
println("\nLogdet property:")
logdet_inv_vec = logabsdetjac(inverse(b_vec), y_vec)
println("logdet_vec + logdet_inv_vec ≈ 0.0: ", isapprox(logdet_vec + logdet_inv_vec, 0.0, atol = 1e-6))
println("logdet_vec + logdet_inv_vec = $(logdet_vec + logdet_inv_vec)")


println("\n--- Matrix Transform Tests ---")
K = 10
D_mat = 5
batch_size = 2
params_mat = randn(3 * K + 1, D_mat)
b_mat = RationalQuadraticSpline(params_mat, -4.0, 4.0)

x_mat = randn(batch_size, D_mat)
y_mat = transform(b_mat, x_mat)
logdet_mat = logabsdetjac(b_mat, x_mat)

println("size(y_mat) == (batch_size, D_mat): ", size(y_mat) == (batch_size, D_mat))
println("size(logdet_mat) == (batch_size,): ", size(logdet_mat) == (batch_size,))

# Invertibility
println("\nInvertibility:")
x_recon_mat = transform(inverse(b_mat), y_mat)
println("x_recon_mat ≈ x_mat: ", isapprox(x_recon_mat, x_mat))

# Logdet property
println("\nLogdet property:")
logdet_inv_mat = logabsdetjac(inverse(b_mat), y_mat)
println("logdet_mat + logdet_inv_mat ≈ zeros(batch_size): ", isapprox(logdet_mat + logdet_inv_mat, zeros(batch_size), atol = 1e-6))
println("logdet_mat + logdet_inv_mat = $(logdet_mat + logdet_inv_mat)")

println("\n--- Distribution Transform Tests ---")
K = 10
D_dist = 2
params_dist = randn(3 * K + 1, D_dist)
b_dist = RationalQuadraticSpline(params_dist, -4.0, 4.0, boundary_slopes = :identity)

d_dist = MvNormal(zeros(D_dist), ones(D_dist))
td_dist = transformed(d_dist, b_dist)

println("td_dist isa Distribution: ", td_dist isa Distribution)

# Test sampling
println("\nSampling:")
y_sample = rand(td_dist, 10)
println("size(y_sample) == (D_dist, 10): ", size(y_sample) == (D_dist, 10))

# Test logpdf
println("\nLogpdf:")
x_dist = randn(D_dist)
y_dist = transform(b_dist, x_dist)

lp_x = logpdf(d_dist, x_dist)
ladj = logabsdetjac(b_dist, x_dist)
lp_y = logpdf(td_dist, y_dist)

println("logpdf(td_dist, y_dist) ≈ lp_x - ladj: ", isapprox(lp_y, lp_x - ladj))
println("logpdf(td_dist, y_dist): $lp_y, lp_x - ladj: $(lp_x - ladj)")

println("\n--- [0,1] Range Tests ---")
K_01 = 5
params_01 = randn(3 * K_01 + 1)
b_01 = RationalQuadraticSpline(params_01, 0.0, 1.0)

# Test values within [0,1] range
x_01_in = [0.1, 0.5, 0.9]
for x in x_01_in
    y, logdet = with_logabsdet_jacobian(b_01, x)
    x_recon = transform(inverse(b_01), y)
    logdet_inv = logabsdetjac(inverse(b_01), y)

    println("x=$x: invertible=$(isapprox(x, x_recon, atol=1e-12)), logdet_property=$(isapprox(logdet + logdet_inv, 0.0, atol=1e-12))")
end

# Test boundary behavior
println("transform(b_01, -0.1) == -0.1: ", transform(b_01, -0.1) == -0.1)
println("transform(b_01, 1.1) == 1.1: ", transform(b_01, 1.1) == 1.1)
println("logabsdetjac(b_01, -0.1) == 0.0: ", logabsdetjac(b_01, -0.1) == 0.0)
println("logabsdetjac(b_01, 1.1) == 0.0: ", logabsdetjac(b_01, 1.1) == 0.0)

println("\n--- Identity Function Test (Analytical Verification) ---")
# Create a spline that should be close to identity
# For 3 bins, we need 3*3 + 1 = 10 parameters
K_id = 3
range_min_id, range_max_id = -2.0, 2.0

# Set up parameters for near-identity transformation
# Equal bin widths and heights (slopes ≈ 1), and knot slopes = 1
unnorm_widths = zeros(K_id)  # After softmax, these become equal
unnorm_heights = zeros(K_id)  # After softmax, these become equal  
unnorm_slopes = zeros(K_id + 1)  # After softplus + offset, these become ≈ 1

params_id = vcat(unnorm_widths, unnorm_heights, unnorm_slopes)
b_id = RationalQuadraticSpline(params_id, range_min_id, range_max_id, boundary_slopes = :identity)

println("Identity spline created with range [$range_min_id, $range_max_id]")

# Test several points - for an identity function, y should ≈ x and logdet should ≈ 0
test_points = [-1.5, -0.5, 0.0, 0.5, 1.5]
println("Testing near-identity transformation:")

let
    max_error = 0.0
    max_logdet = 0.0

    for x in test_points
        y, logdet = with_logabsdet_jacobian(b_id, x)
        error = abs(y - x)
        max_error = max(max_error, error)
        max_logdet = max(max_logdet, abs(logdet))

        println("  x=$x -> y=$y (error=$(round(error, digits=6))), logdet=$logdet")
    end

    println("Maximum transformation error: $(round(max_error, digits=6))")
    println("Maximum logdet magnitude: $(round(max_logdet, digits=6))")
    println("Near-identity test passed: $(max_error < 0.1 && max_logdet < 0.5)")
end

# Test invertibility for the identity case
x_test_id = 0.3
y_test_id = transform(b_id, x_test_id)
x_recon_id = transform(inverse(b_id), y_test_id)
println("Identity invertibility: x=$x_test_id -> y=$y_test_id -> x_recon=$x_recon_id")
println("Identity invertibility error: $(abs(x_test_id - x_recon_id))")

println("\n--- Extended Test with 5 Nodes ---")
# Test with 5 bins to ensure implementation handles different numbers correctly
K_ext = 5
params_ext = randn(3 * K_ext + 1)
b_ext = RationalQuadraticSpline(params_ext, -3.0, 3.0)

println("Testing 5-node spline with range [-3.0, 3.0]")
extended_test_points = [-2.5, -1.0, 0.0, 1.0, 2.5]

let
    all_invertible = true
    all_logdet_correct = true

    for x in extended_test_points
        y, logdet = with_logabsdet_jacobian(b_ext, x)
        x_recon = transform(inverse(b_ext), y)
        logdet_inv = logabsdetjac(inverse(b_ext), y)

        invertible = isapprox(x, x_recon, atol = 1e-12)
        logdet_correct = isapprox(logdet + logdet_inv, 0.0, atol = 1e-10)

        all_invertible = all_invertible && invertible
        all_logdet_correct = all_logdet_correct && logdet_correct

        println("  x=$x: y=$y, invertible=$invertible, logdet_sum=$(logdet + logdet_inv)")
    end

    println("All points invertible: $all_invertible")
    println("All logdet properties satisfied: $all_logdet_correct")
end

# Test monotonicity within each bin (optional verification)
println("Testing within-bin behavior:")
bin_test_points = [-2.9, -2.1, -0.9, -0.1, 0.1, 0.9, 2.1, 2.9]
for x in bin_test_points
    y, logdet = with_logabsdet_jacobian(b_ext, x)
    println("  x=$x -> y=$(round(y, digits=4)), logdet=$(round(logdet, digits=4))")
end

# Extra advanced tests

# FF = RationalQuadraticSpline(rand(31, 10), 0, 1)
# FF(rand(1, 10))

# const amask = Bijectors.PartitionMask(11, [1], 1:10) # acts on 1.

# partitioned = Bijectors.partition(amask, rand(11, 55))
# @assert partitioned[2] |> size == (10, 55)

# layer = Bijectors.Coupling(amask) do z
#     RationalQuadraticSpline(z, 0.0, 1.0)
# end

# layer(rand(11, 55))


# using Plots
# theme(:boxed)

# ft = RationalQuadraticSpline([0.4, 0.6, 0.4, 0.6, 1.0, 0.01, 5.0], 0, 1)
# let
#     plot(x -> ft(x), 0, 1)
#     scatter!(ft.x_pos, ft.y_pos)
# end

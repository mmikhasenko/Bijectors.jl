using Bijectors
using Bijectors: RationalQuadraticSpline
using Distributions
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
N_vec = 5

# Rule 2.1: x (N,), params (3*K+1,) - broadcast same spline
println("\nRule 2.1: x (N,), params (3*K+1,) - broadcast same spline")
let
    params_broadcast = randn(3 * K + 1)
    b_broadcast = RationalQuadraticSpline(params_broadcast, -4.0, 4.0)

    x_vec = randn(N_vec)
    y_vec, logdet_vec = with_logabsdet_jacobian(b_broadcast, x_vec)

    println("y_vec isa AbstractVector: ", y_vec isa AbstractVector)
    println("size(y_vec) == (N_vec,): ", size(y_vec) == (N_vec,))
    println("logdet_vec isa Real: ", logdet_vec isa Real)

    # Invertibility
    x_recon_vec = transform(inverse(b_broadcast), y_vec)
    println("Invertible: ", isapprox(x_recon_vec, x_vec))

    # Logdet property
    logdet_inv_vec = logabsdetjac(inverse(b_broadcast), y_vec)
    println("LogDet property: ", isapprox(logdet_vec + logdet_inv_vec, 0.0, atol = 1e-6))
end

# Rule 2.2: x (N,), params (3*K+1, N) - N splines applied
println("\nRule 2.2: x (N,), params (3*K+1, N) - N splines applied")
let
    params_n_splines = randn(3 * K + 1, N_vec)
    b_n_splines = RationalQuadraticSpline(params_n_splines, -4.0, 4.0)

    x_vec = randn(N_vec)
    y_vec, logdet_vec = with_logabsdet_jacobian(b_n_splines, x_vec)

    println("y_vec isa AbstractVector: ", y_vec isa AbstractVector)
    println("size(y_vec) == (N_vec,): ", size(y_vec) == (N_vec,))
    println("logdet_vec isa Real: ", logdet_vec isa Real)

    # Invertibility
    x_recon_vec = transform(inverse(b_n_splines), y_vec)
    println("Invertible: ", isapprox(x_recon_vec, x_vec))

    # Logdet property
    logdet_inv_vec = logabsdetjac(inverse(b_n_splines), y_vec)
    println("LogDet property: ", isapprox(logdet_vec + logdet_inv_vec, 0.0, atol = 1e-6))
end


println("\n--- Matrix Transform Tests ---")
K = 10
N_mat = 3  # First dimension (batch size)
D_mat = 4  # Second dimension (feature size)

# Rule 1.1: x (N, D), params (3*K+1,) - broadcast same spline
println("\nRule 1.1: x (N, D), params (3*K+1,) - broadcast same spline")
let
    params_broadcast = randn(3 * K + 1)
    b_broadcast = RationalQuadraticSpline(params_broadcast, -4.0, 4.0)

    x_mat = randn(N_mat, D_mat)
    y_mat = transform(b_broadcast, x_mat)
    logdet_mat = logabsdetjac(b_broadcast, x_mat)

    println("size(y_mat) == (N_mat, D_mat): ", size(y_mat) == (N_mat, D_mat))
    println("size(logdet_mat) == (N_mat,): ", size(logdet_mat) == (N_mat,))

    # Invertibility
    x_recon_mat = transform(inverse(b_broadcast), y_mat)
    println("Invertible: ", isapprox(x_recon_mat, x_mat))

    # Logdet property
    logdet_inv_mat = logabsdetjac(inverse(b_broadcast), y_mat)
    println("LogDet property: ", isapprox(logdet_mat + logdet_inv_mat, zeros(N_mat), atol = 1e-6))
end

# Rule 1.2: x (N, D), params (3*K+1, N) - N splines, broadcast over D
println("\nRule 1.2: x (N, D), params (3*K+1, N) - N splines, broadcast over D")
let
    params_n_splines = randn(3 * K + 1, N_mat)
    b_n_splines = RationalQuadraticSpline(params_n_splines, -4.0, 4.0)

    x_mat = randn(N_mat, D_mat)
    y_mat = transform(b_n_splines, x_mat)
    logdet_mat = logabsdetjac(b_n_splines, x_mat)

    println("size(y_mat) == (N_mat, D_mat): ", size(y_mat) == (N_mat, D_mat))
    println("size(logdet_mat) == (N_mat,): ", size(logdet_mat) == (N_mat,))

    # Invertibility
    x_recon_mat = transform(inverse(b_n_splines), y_mat)
    println("Invertible: ", isapprox(x_recon_mat, x_mat))

    # Logdet property
    logdet_inv_mat = logabsdetjac(inverse(b_n_splines), y_mat)
    println("LogDet property: ", isapprox(logdet_mat + logdet_inv_mat, zeros(N_mat), atol = 1e-6))
end

# Rule 1.3: x (N, D), params (3*K+1, N, D) - N×D splines applied correctly
println("\nRule 1.3: x (N, D), params (3*K+1, N, D) - N×D splines applied correctly")
let
    params_nd_splines = randn(3 * K + 1, N_mat, D_mat)
    b_nd_splines = RationalQuadraticSpline(params_nd_splines, -4.0, 4.0)

    x_mat = randn(N_mat, D_mat)
    y_mat = transform(b_nd_splines, x_mat)
    logdet_mat = logabsdetjac(b_nd_splines, x_mat)

    println("size(y_mat) == (N_mat, D_mat): ", size(y_mat) == (N_mat, D_mat))
    println("size(logdet_mat) == (N_mat,): ", size(logdet_mat) == (N_mat,))

    # Invertibility
    x_recon_mat = transform(inverse(b_nd_splines), y_mat)
    println("Invertible: ", isapprox(x_recon_mat, x_mat))

    # Logdet property
    logdet_inv_mat = logabsdetjac(inverse(b_nd_splines), y_mat)
    println("LogDet property: ", isapprox(logdet_mat + logdet_inv_mat, zeros(N_mat), atol = 1e-6))
end

println("\n--- Distribution Transform Tests ---")
K = 10
D_dist = 2

# Test both broadcasting scenarios for distributions
println("\nRule 2.1: Distribution with broadcast params")
let
    params_broadcast = randn(3 * K + 1)  # Same spline for all dimensions
    b_broadcast = RationalQuadraticSpline(params_broadcast, -4.0, 4.0, boundary_slopes = :identity)

    d_dist = MvNormal(zeros(D_dist), ones(D_dist))
    td_dist = transformed(d_dist, b_broadcast)

    println("td_dist isa Distribution: ", td_dist isa Distribution)

    # Test sampling
    y_sample = rand(td_dist, 10)
    println("size(y_sample) == (D_dist, 10): ", size(y_sample) == (D_dist, 10))

    # Test logpdf
    x_dist = randn(D_dist)
    y_dist = transform(b_broadcast, x_dist)

    lp_x = logpdf(d_dist, x_dist)
    ladj = logabsdetjac(b_broadcast, x_dist)
    lp_y = logpdf(td_dist, y_dist)

    println("LogPDF property: ", isapprox(lp_y, lp_x - ladj))
end

println("\nRule 2.2: Distribution with per-dimension params")
let
    params_per_dim = randn(3 * K + 1, D_dist)  # Different spline per dimension
    b_per_dim = RationalQuadraticSpline(params_per_dim, -4.0, 4.0, boundary_slopes = :identity)

    d_dist = MvNormal(zeros(D_dist), ones(D_dist))
    td_dist = transformed(d_dist, b_per_dim)

    println("td_dist isa Distribution: ", td_dist isa Distribution)

    # Test sampling
    y_sample = rand(td_dist, 10)
    println("size(y_sample) == (D_dist, 10): ", size(y_sample) == (D_dist, 10))

    # Test logpdf
    x_dist = randn(D_dist)
    y_dist = transform(b_per_dim, x_dist)

    lp_x = logpdf(d_dist, x_dist)
    ladj = logabsdetjac(b_per_dim, x_dist)
    lp_y = logpdf(td_dist, y_dist)

    println("LogPDF property: ", isapprox(lp_y, lp_x - ladj))
end

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



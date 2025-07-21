using Test
using LinearAlgebra
using Distributions
using FiniteDifferences
using ForwardDiff
using ReverseDiff
using Enzyme
using Mooncake
using Tracker
using Bijectors
using Bijectors: RationalQuadraticSpline

@testset "AD for RationalQuadraticSpline" begin
    # Test scalar case (wrapped in array for AD testing)
    @testset "Scalar input" begin
        K = 3
        params = randn(3 * K + 1)
        b = RationalQuadraticSpline(params, -2.0, 2.0)
        binv = inverse(b)

        x = [1.0]  # Wrap scalar in array for AD testing
        y = transform(b, x[1])

        # Test forward transformation
        test_ad(x) do x
            transform(b, x[1])
        end

        # Test log jacobian determinant
        test_ad(x) do x
            logabsdetjac(b, x[1])
        end

        # Test inverse transformation
        test_ad([y]) do y
            transform(binv, y[1])
        end

        # Test combined transformation
        test_ad(x) do x
            y, logdet = with_logabsdet_jacobian(b, x[1])
            return y + logdet
        end
    end

    # Test vector case with broadcast parameters
    @testset "Vector input - broadcast parameters" begin
        K = 3
        N = 4
        params = randn(3 * K + 1)  # Single spline broadcast to all elements
        b = RationalQuadraticSpline(params, -2.0, 2.0)
        binv = inverse(b)

        x = randn(N)
        y = transform(b, x)

        # Test forward transformation
        test_ad(x) do x
            sum(transform(b, x))
        end

        # Test log jacobian determinant
        test_ad(x) do x
            logabsdetjac(b, x)
        end

        # Test inverse transformation
        test_ad(y) do y
            sum(transform(binv, y))
        end
    end

    # Test vector case with per-element parameters
    @testset "Vector input - per-element parameters" begin
        K = 3
        N = 4
        params = randn(3 * K + 1, N)  # Different spline for each element
        b = RationalQuadraticSpline(params, -2.0, 2.0)
        binv = inverse(b)

        x = randn(N)
        y = transform(b, x)

        # Test forward transformation
        test_ad(x) do x
            sum(transform(b, x))
        end

        # Test log jacobian determinant
        test_ad(x) do x
            logabsdetjac(b, x)
        end

        # Test inverse transformation  
        test_ad(y) do y
            sum(transform(binv, y))
        end
    end

    # Test matrix case
    @testset "Matrix input" begin
        K = 3
        N = 3
        D = 2
        params = randn(3 * K + 1, N, D)  # Different spline for each element
        b = RationalQuadraticSpline(params, -2.0, 2.0)
        binv = inverse(b)

        x = randn(N, D)
        y = transform(b, x)

        # Test forward transformation
        test_ad(x) do x
            sum(transform(b, x))
        end

        # Test log jacobian determinant
        test_ad(x) do x
            sum(logabsdetjac(b, x))
        end

        # Test inverse transformation
        test_ad(y) do y
            sum(transform(binv, y))
        end
    end

    # Test with different boundary conditions
    @testset "Identity boundary slopes" begin
        K = 3
        params = randn(3 * K + 1)
        b = RationalQuadraticSpline(params, -2.0, 2.0, boundary_slopes = :identity)
        binv = inverse(b)

        x = [1.0]  # Wrap scalar in array for AD testing
        y = transform(b, x[1])

        # Test forward transformation
        test_ad(x) do x
            transform(b, x[1])
        end

        # Test log jacobian determinant
        test_ad(x) do x
            logabsdetjac(b, x[1])
        end

        # Test inverse transformation
        test_ad([y]) do y
            transform(binv, y[1])
        end
    end

    # Test in distribution context
    @testset "Distribution usage" begin
        K = 3
        D = 2
        params = randn(3 * K + 1, D)
        b = RationalQuadraticSpline(params, -2.0, 2.0, boundary_slopes = :identity)

        d = MvNormal(zeros(D), I)
        td = transformed(d, b)

        x = randn(D)

        # Test logpdf computation (this involves both transform and logabsdetjac)
        test_ad(x) do x
            logpdf(td, transform(b, x))
        end
    end
end

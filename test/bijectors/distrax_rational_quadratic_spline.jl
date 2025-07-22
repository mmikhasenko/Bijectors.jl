using Test
using Bijectors: RationalQuadraticSpline
using Distributions
using InteractiveUtils
@testset "RationalQuadraticSpline Type Stability" begin
    @testset "RationalQuadraticSpline Constructor Type Stability" begin
        # Test parameters
        params_1d = rand(Float64, 31)  # 3*10 + 1 = 31 for 10 bins
        params_2d = rand(Float64, 31, 10)  # Multi-dimensional parameters
        params_mixed = rand(Float32, 31)
        range_min, range_max = 0.0, 1.0
        range_min_f64, range_max_f64 = 0.0, 1.0  # Float64

        @testset "1D Constructor" begin
            # Basic 1D constructor with vector params
            @test @inferred(RationalQuadraticSpline(params_1d, range_min, range_max)) isa RationalQuadraticSpline

            # Constructor with keyword arguments
            @test @inferred(RationalQuadraticSpline(params_1d, range_min, range_max; boundary_slopes = :identity)) isa RationalQuadraticSpline
        end

        @testset "2D Constructor" begin
            # Basic 2D constructor with matrix params
            @test @inferred(RationalQuadraticSpline(params_2d, range_min, range_max)) isa RationalQuadraticSpline
        end

        @testset "Mixed Type Constructor" begin
            # Mixed types should still work (may promote types)
            rqs_mixed = RationalQuadraticSpline(params_mixed, range_min_f64, range_max_f64)
            @test rqs_mixed isa RationalQuadraticSpline
        end
    end


    @testset "Promote Type Behavior" begin
        using Bijectors: promote_type, eltype

        # Test promote_type with different numeric types
        widths_f32 = Float32[1.0, 2.0, 3.0]
        heights_f64 = Float64[1.0, 2.0, 3.0]
        derivatives_mixed = Float32[1.0, 2.0, 3.0]
        x_f16 = Float16(0.5)

        promote_result = promote_type(eltype(widths_f32), eltype(heights_f64), eltype(derivatives_mixed), eltype(x_f16))
        @test promote_result == Float64
    end

    @testset "Type-stable Specialized Constructors" begin
        @testset "Float64 Constructors" begin
            params_f64_1d = rand(Float64, 31)
            params_f64_2d = rand(Float64, 31, 10)

            # 1D Float64 constructor should be type stable
            rqs_f64_1d = @inferred RationalQuadraticSpline(params_f64_1d, 0.0, 1.0)
            @test rqs_f64_1d isa RationalQuadraticSpline

            # 2D Float64 constructor should be type stable
            rqs_f64_2d = @inferred RationalQuadraticSpline(params_f64_2d, 0.0, 1.0)
            @test rqs_f64_2d isa RationalQuadraticSpline
        end

        @testset "Float32 Constructors" begin
            params_f32_1d = rand(Float32, 31)
            params_f32_2d = rand(Float32, 31, 5)

            # 1D Float32 constructor should be type stable
            rqs_f32_1d = @inferred RationalQuadraticSpline(params_f32_1d, 0.0f0, 1.0f0)
            @test rqs_f32_1d isa RationalQuadraticSpline

            # 2D Float32 constructor should be type stable
            rqs_f32_2d = @inferred RationalQuadraticSpline(params_f32_2d, 0.0f0, 1.0f0)
            @test rqs_f32_2d isa RationalQuadraticSpline
        end
    end
end

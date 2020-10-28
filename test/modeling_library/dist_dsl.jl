@testset "dist DSL (untyped)" begin
  # Test transformed distributions
  @dist f(x) = exp(normal(x, 0.001))
  @test isapprox(1, f(0); atol = 5)

  # Test relabeled distributions with labels provided as an Array
  @dist labeled_cat(labels, probs) = labels[categorical(probs)]
  @test labeled_cat([:a, :b], [0., 1.]) == :b
  @test isapprox(logpdf(labeled_cat, :b, [:a, :b], [0.5, 0.5]), log(0.5))
  @test logpdf(labeled_cat, :c, [:a, :b], [0.5, 0.5]) == -Inf

  # Test relabeled distributions with labels provided in a Dict
  dict = Dict(1 => :a, 2 => :b)
  @dist dict_cat(probs) = dict[categorical(probs)]
  @test dict_cat([0., 1.]) == :b
  @test isapprox(logpdf(dict_cat, :b, [0.5, 0.5]), log(0.5))
  @test logpdf(dict_cat, :c, [0.5, 0.5]) == -Inf

  # Test relabeled distributions with Enum labels
  @enum Fruit apple orange
  @dist enum_cat(probs) = Fruit(categorical(probs) - 1)
  @test enum_cat([0., 1.]) == orange
  @test isapprox(logpdf(enum_cat, orange, [0.5, 0.5]), log(0.5))
  @test logpdf(enum_cat, orange, [1.0]) == -Inf

  # Regression test for https://github.com/probcomp/Gen/issues/253
  @dist real_minus_uniform(a, b) = 1 - Gen.uniform(a, b)
  @test real_minus_uniform(1, 2) < 0
  @test logpdf(real_minus_uniform, -0.5, 1, 2) == 0.0

  # Test dist with unusual signatures
  @dist tuple_minus_uniform((a, b)) = 1 - Gen.uniform(a, b)
  @test tuple_minus_uniform((1, 2)) < 0
  @test logpdf(tuple_minus_uniform, -0.5, (1, 2)) == 0.0

  @dist normal_with_meanlingless_args((_,), (_, _), foo, (bar, baz)) = normal(0, 1)
  @test normal_with_meanlingless_args((1), (1, 2), 4, (1, 2)) isa Real
  @test logpdf(normal_with_meanlingless_args, 2.3, (1), (1, 2), 4, (1, 2)) == logpdf(normal, 2.3, 0, 1)
end

# User-defined type for testing purposes
struct MyLabel
  name::Symbol
end

@testset "dist DSL (typed)" begin
  # Test typed relabeled distributions
  @dist symbol_cat(labels::Vector{Symbol}, probs) = labels[categorical(probs)]
  @test symbol_cat([:a, :b], [0., 1.]) == :b
  @test_throws MethodError symbol_cat(["a", "b"], [0., 1.])
  @test logpdf(symbol_cat, :c, [:a, :b], [0.5, 0.5]) == -Inf
  @test_throws MethodError logpdf(symbol_cat, "c", [:a, :b], [0.5, 0.5])

  # Test typed parameters
  @dist int_bounded_uniform(low::Int, high::Int) = uniform(low, high)
  @test 0.0 <= int_bounded_uniform(0, 1) <= 1
  @test_throws MethodError int_bounded_uniform(-0.5, 0.5)
  @test logpdf(int_bounded_uniform, 0.5, 0, 1) == 0
  @test_throws MethodError logpdf(int_bounded_uniform, 0.0, -0.5, 0.5)

  # Test relabeled distributions with user-defined types
  @dist mylabel_cat(labels::Vector{MyLabel}, probs) = labels[categorical(probs)]
  @test mylabel_cat([MyLabel(:a), MyLabel(:b)], [0., 1.]) == MyLabel(:b)
  @test_throws MethodError mylabel_cat([:a, :b], [0., 1.])
  @test logpdf(mylabel_cat, MyLabel(:a), [MyLabel(:a)], [1.0]) == 0
  @test_throws MethodError logpdf(mylabel_cat, :a, [MyLabel(:a)], [1.0])

  # Test dist with unusual signatures
  @dist strangedist((_, (foo, bar))::Tuple{Int, Tuple{Int, Float64}}, baz, ::Int) = normal(bar, 1)
  @test_throws MethodError strangedist((1, (1, "hi")), 1, 1)
  @test strangedist((1, (1, 1.)), 1, 1) isa Real
  @test logpdf(strangedist, 1.2, (1, (1, 1.)), 1, 1) == logpdf(normal, 1.2, 1, 1)
end

@testset "dist dsl as generative function" begin
  @dist labeled_cat(labels, probs) = labels[categorical(probs)]

  (tr, weight) = generate(labeled_cat, ([1, 2, 3], [.2, .2, .6]), Value(3))
  @test get_score(tr) == weight
  @test get_retval(tr) == 3

  (new_tr, weight, _, _) = update(tr, ([1, 2, 3], [.2, .2, .6]), (NoChange(), NoChange()), Value(2))
  @test isapprox(weight, log(.2) - log(.6))
  @test get_retval(new_tr) == 2
end
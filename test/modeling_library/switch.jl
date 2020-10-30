@testset "switch combinator" begin
    @gen (static, diffs) function a(x, y)
        val ~ normal(x, y)
        return val
    end

    @gen (static, diffs) function b(x, y)
        return x + y
    end

    @gen (static, diffs) function c(x, y)
        use_a ~ bernoulli(0.5)
        val ~ Switch(a, b)(switchint(use_a), x, y)
        return val
    end
    load_generated_functions()

    tr, weight = generate(c, (1, 1), choicemap((:use_a, false)))
    @test get_retval(tr) === 2

    tr, weight = generate(c, (1, 1), choicemap((:use_a, true), (:val => 1 => :val, 1.0)))
    @test isapprox(get_score(tr), logpdf(bernoulli, true, 0.5) + logpdf(normal, 1, 1, 1))

    tr, weight, retdiff, discard = update(tr, (1, 2), (NoChange(), UnknownChange()), EmptyAddressTree())
    @test retdiff === NoChange()
    @test isempty(discard)
    @test isapprox(weight, logpdf(normal, 1, 1, 2) - logpdf(normal, 1, 1, 1))
    @test isapprox(get_score(tr), logpdf(bernoulli, true, 0.5) + logpdf(normal, 1, 1, 2))

    tr, weight, retdiff, discard = update(tr, (1, 2), (NoChange(), NoChange()), choicemap((:use_a, false)))
    @test discard == choicemap((:use_a, true), (:val => 1 => :val, 1.0))
    @test retdiff === UnknownChange()
    @test get_retval(tr) == 3
    @test isapprox(weight, -logpdf(normal, 1, 1, 2))
end
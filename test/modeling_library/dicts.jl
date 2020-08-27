@testset "lazy dict maps" begin
    keys = Set([-1,0,1,2,3])
    dict = lazy_set_to_dict_map(x->x^2, keys)
    @test dict == Dict(-1 => 1, 0 => 0, 1 => 1, 2 => 4, 3 => 9)

    sqrted = lazy_val_map(sqrt, dict)
    @test sqrted == Dict(-1 => 1, 0 => 0, 1 => 1, 2 => 2, 3 => 3)
end

@dist add_noise(x) = normal(x, 1)
@testset "dictmap combinator" begin
    dict = Dict(:a => 1, :b => 2, :c => 3, :d => 4)
    tr = simulate(DictMap(add_noise), (dict,))
    @test get_retval(tr) isa AbstractDict

    tr, weight = generate(DictMap(add_noise), (dict,), choicemap(
        (:a, 1.1), (:b, 2.1), (:c, 3.1), (:d, 3.9)
    ))
    @test isapprox(get_score(tr), 4*logpdf(normal, 0.1, 0, 1))
    @test isapprox(weight, get_score(tr))
    @test get_retval(tr) == Dict([(:a, 1.1), (:b, 2.1), (:c, 3.1), (:d, 3.9)])

    dict = Dict(:b => 2, :c => 3, :d => -4, :e => 5)
    diff = DictDiff(Dict(:e => 5), Set([:a]), Dict(:d => UnknownChange()))
    new_tr, weight, retdiff, discard = update(tr, (dict,), (diff,), choicemap((:e, 5.1), (:b, 2.2)))
    @test isapprox(get_score(new_tr), 2*logpdf(normal, 0.1, 0, 1) + logpdf(normal, 3.9, -4, 1) + logpdf(normal, 0.2, 0, 1))
    @test isapprox(weight, logpdf(normal, 3.9, -4, 1) + logpdf(normal, 0.2, 0, 1) - 2*logpdf(normal, 0.1, 0, 1))

    @test retdiff isa DictDiff
    @test retdiff.added == Dict(:e => 5.1)
    @test retdiff.deleted == Set([:a])
    @test retdiff.updated == Dict(:b => UnknownChange())

    @test get_retval(new_tr) == Dict(:b => 2.2, :c => 3.1, :d => 3.9, :e => 5.1)
end
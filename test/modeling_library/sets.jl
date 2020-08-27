using FunctionalCollections: push, disj, PersistentSet
@testset "multiset" begin
    ms = MultiSet()
    @test ms isa MultiSet{Any}

    ms = MultiSet{Int}()
    ms = push(ms, 1)
    @test length(ms) == 1
    ms = push(ms, 1)
    @test length(ms) == 2
    @test collect(ms) == [1, 1]

    ms = push(ms, 2)
    @test length(ms) == 3
    @test 2 in ms
    @test 1 in ms
    ms = remove_one(ms, 1)
    @test 1 in ms
    @test 2 in ms
    ms = push(ms, 2)
    @test length(ms) == 3
    ms = disj(ms, 2)
    @test length(ms) == 1

    @test push(push(ms, 2), 2) == MultiSet([1, 2, 2])
    @test MultiSet([2, 1, 2, 5]) == MultiSet([2, 5, 2, 1])

    total = 0
    for el in MultiSet([2, 1, 2, 5])
        total += el
    end
    @test total == 2+1+2+5

    @test set_map(x -> x^2, Set([-2, -1, 0, 1])) == MultiSet([4, 1, 1, 0])
    @test no_collision_set_map(x -> x^3, Set([-2, -1, 0, 1])) == PersistentSet([-8, -1, 0, 1])
end

@testset "SetMap combinator" begin
    priors = [
        [0.1, 0.3, 0.6],
        [0.2, 0.6, 0.2],
        [0.6, 0.2, 0.2]
    ]
    tr, weight = generate(SetMap(categorical), (Set(priors),), choicemap((priors[1], 2)))
    @test tr[priors[1]] == 2
    @test tr[priors[2]] in (1, 2, 3)
    @test tr[priors[3]] in (1, 2, 3)
    @test isapprox(weight, log(0.3))

    tr = simulate(SetMap(categorical), (Set(priors),))
    exp_score = sum(logpdf(categorical, tr[priors[i]], priors[i]) for i=1:3)
    @test isapprox(get_score(tr), exp_score)

    current1 = tr[priors[1]]
    new = current1 == 1 ? 2 : 1

    new_tr, weight, _, discard = update(tr, (Set(priors),), (NoChange(),), choicemap((priors[1], new)))
    expected_weight = logpdf(categorical, new, priors[1]) - logpdf(categorical, tr[priors[1]], priors[1])
    @test isapprox(weight, expected_weight)
    @test isapprox(get_score(new_tr) - get_score(tr), expected_weight)
    @test discard == choicemap((priors[1], tr[priors[1]]))

    new_tr, weight, _, discard = update(tr, (Set(priors[1:2]),), (UnknownChange(),), EmptyAddressTree())
    expected_weight = -logpdf(categorical, tr[priors[3]], priors[3])
    @test isapprox(weight, expected_weight)
    @test isapprox(get_score(new_tr), sum(logpdf(categorical, tr[priors[i]], priors[i]) for i=1:2))
    @test discard == choicemap((priors[3], tr[priors[3]]))
end

@testset "NoCollisionSetMap combinator" begin
    priors = [
        [0.5, 0.5, 0., 0., 0., 0.],
        [0., 0., 0.3, 0.7, 0., 0.],
        [0., 0., 0., 0., 0.1, 0.9]
    ]
    tr = simulate(NoCollisionSetMap(categorical), (Set(priors),))
    exp_score = sum(logpdf(categorical, tr[priors[i]], priors[i]) for i=1:3)
    @test isapprox(get_score(tr), exp_score)

    tr, weight = generate(NoCollisionSetMap(categorical), (Set(priors),), choicemap((priors[1], 2)))
    @test tr[priors[1]] == 2
    @test tr[priors[2]] in (3, 4)
    @test tr[priors[3]] in (5, 6)
    @test get_retval(tr) isa PersistentSet
    @test isapprox(weight, log(0.5))

    new_tr, weight, retdiff, discard = update(tr, (Set(priors[1:2]),), (SetDiff(Set(), Set{Any}([priors[3]])),), EmptyAddressTree(), AllSelection())
    @test retdiff isa SetDiff
    @test length(retdiff.deleted) == 1
    @test tr[priors[3]] in retdiff.deleted
    @test length(retdiff.added) == 0
    @test isapprox(weight, -logpdf(categorical, tr[priors[3]], priors[3]))
    @test length(collect(get_subtrees_shallow(discard))) == 1
    @test discard[priors[3]] == tr[priors[3]]
    @test isapprox(get_score(new_tr) - get_score(tr), weight)


    new_tr, weight, retdiff, discard = update(tr, (Set(priors[1:2]),), (SetDiff(Set(), Set([priors[3]])),), choicemap((priors[1], 1)), AllSelection())
    @test retdiff isa SetDiff
    @test retdiff.deleted == Set([tr[priors[3]], 2])
    @test retdiff.added == Set([1])
    @test isapprox(weight, -logpdf(categorical, tr[priors[3]], priors[3]))
    @test length(collect(get_subtrees_shallow(discard))) == 2
    @test discard[priors[3]] == tr[priors[3]]
    @test discard[priors[1]] == 2
    @test isapprox(get_score(new_tr) - get_score(tr), weight)
end

@testset "tracked product set" begin
    s1 = Set([1, 2, 3, 4])
    s2 = Set(["a", "b", "c", "d"])
    s3 = Set([0.1, 0.2, 0.3, 0.4])
    tr = simulate(tracked_product_set, ([s1, s2, s3],))
    @test get_retval(tr) == Set(Iterators.product(s1, s2, s3))

    s1 = Set([1, 2, 3])
    s2 = Set(["a", "b", "c", "d", "e"])
    diff = VectorDiff(
        3, 3, Dict(
            1 => SetDiff(Set(), Set([4])),
            2 => SetDiff(Set(["e"]), Set())
        )
    )
    new_tr, _, retdiff, _ = update(tr, ([s1, s2, s3],), (diff,), EmptyAddressTree(), AllSelection())
    @test retdiff isa SetDiff
    @test retdiff.added == Set([(i, "e", j) for i=1:3, j=0.1:0.1:0.4])
    @test retdiff.deleted == Set([(4, a, b) for a in ("a", "b", "c", "d"), b=0.1:0.1:0.4])
    @test get_retval(new_tr) == Set(Iterators.product(s1, s2, s3))
end
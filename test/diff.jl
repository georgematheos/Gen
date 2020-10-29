@testset "diff arithmetic" begin

    @testset "+" begin
        a = Diffed(1, NoChange())
        b = Diffed(2, NoChange())
        @test get_diff(a + b) == NoChange()
        @test strip_diff(a + b) == 3

        a = 1
        b = Diffed(2, NoChange())
        @test get_diff(a + b) == NoChange()
        @test strip_diff(a + b) == 3

        a = Diffed(1, NoChange())
        b = 2
        @test get_diff(a + b) == NoChange()
        @test strip_diff(a + b) == 3

        a = Diffed(1, UnknownChange())
        b = Diffed(2, UnknownChange())
        @test get_diff(a + b) == UnknownChange()
        @test strip_diff(a + b) == 3

        a = 1
        b = Diffed(2, UnknownChange())
        @test get_diff(a + b) == UnknownChange()
        @test strip_diff(a + b) == 3

        a = Diffed(1, UnknownChange())
        b = 2
        @test get_diff(a + b) == UnknownChange()
        @test strip_diff(a + b) == 3
    end

    # TODO test other binary operators
end

@testset "diff vectors" begin

    @testset "length" begin

    v = Diffed([1, 2, 3], NoChange())
    @test strip_diff(length(v)) == 3
    @test get_diff(length(v)) == NoChange()

    v = Diffed([1, 2, 3], UnknownChange())
    @test strip_diff(length(v)) == 3
    @test get_diff(length(v)) == UnknownChange()

    v = Diffed([1, 2, 3], VectorDiff(3, 4, Dict{Int,Diff}()))
    @test strip_diff(length(v)) == 3
    @test get_diff(length(v)) == IntDiff(-1)

    end

    @testset "getindex" begin
    
    v = [1, 2, 3]
    i = Diffed(2, UnknownChange())
    @test strip_diff(v[i]) == 2
    @test get_diff(v[i]) == UnknownChange()

    v = [1, 2, 3]
    i = Diffed(2, NoChange())
    @test strip_diff(v[i]) == 2
    @test get_diff(v[i]) == NoChange()

    v = [1, 2, 3]
    i = Diffed(2, IntDiff(1))
    @test strip_diff(v[i]) == 2
    @test get_diff(v[i]) == UnknownChange()

    v = Diffed([1, 2, 3], NoChange())
    @test strip_diff(v[2]) == 2
    @test get_diff(v[2]) == NoChange()

    v = Diffed([1, 2, 3], UnknownChange())
    @test strip_diff(v[2]) == 2
    @test get_diff(v[2]) == UnknownChange()

    # the value at v[2] was not changed
    v = Diffed([1, 2, 3], VectorDiff(3, 4, Dict{Int,Diff}()))
    @test strip_diff(v[2]) == 2
    @test get_diff(v[2]) == NoChange()

    # the value at v[2] was updated (reduced by 3)
    element_diff = IntDiff(-3)
    v = Diffed([1, 2, 3], VectorDiff(3, 4, Dict{Int,Diff}(2 => element_diff)))
    @test strip_diff(v[2]) == 2
    @test get_diff(v[2]) == element_diff
    end
end

@testset "diff dictionaries" begin
    println("TODO: write tests for dict diffs")
    added = Dict(1 => 1, 2 => 2)
    deleted = Set([3])
    updated = Dict(6 => UnknownChange())
    # TODO: this fails if I say `isa DictDiff{Int, Int}`
    @test DictDiff(added, deleted, updated) isa DictDiff
end

@testset "diff sets" begin
    println("TODO: write tests for set diffs")
    # TODO: this fails if I say `isa SetDiff{Int}`
    @test SetDiff(Set([1, 2]), Set([3])) isa SetDiff
end

@testset "diff properties" begin
    struct Foo51
        x::Int
    end
    @test Diffed(Foo51(1), NoChange()).x == Diffed(1, NoChange())
    @test Diffed(Foo51(1), UnknownChange()).x == Diffed(1, UnknownChange())
end

@testset "map diff" begin
    list = [1, 2, 3, 4, 5]
    @test map(x -> 2*x, Diffed(list, NoChange())) == Diffed([2, 4, 6, 8, 10], NoChange())
    @test map(x -> 2*x, Diffed(list, UnknownChange())) == Diffed([2, 4, 6, 8, 10], UnknownChange())

    # TODO: test propagating VectorDiffs
end
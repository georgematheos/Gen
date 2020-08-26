function set_map(f, set)
    vals = [f(el) for el in set]
    MultiSet(vals)
end

########################
# No collision set map #
########################

struct DeterministicNoCollisionSetMap <: CustomUpdateGF{PersistentSet, PersistentSet} end
no_collision_set_map = DeterministicNoCollisionSetMap()
(d::DeterministicNoCollisionSetMap)(args...) = Gen.apply_with_state(d, args)[1]
function Gen.apply_with_state(::DeterministicNoCollisionSetMap, args::Tuple{<:Function, <:Any})
    (f, set) = args
    mapped = PersistentSet([f(el) for el in set])
    return (mapped, mapped)
end
function Gen.update_with_state(::DeterministicNoCollisionSetMap, out_set, args::Tuple{<:Function, <:Any},
    argdiffs::Tuple{NoChange, SetDiff}
)
    f = args[1]
    setdiff = argdiffs[2]
    removed = Set()
    added = Set()
    for el in setdiff.deleted
        obj = f(el)
        out_set = disj(out_set, obj)
        push!(removed, obj)
    end
    for el in setdiff.added
        obj = f(el)
        out_set = push(out_set, obj)
        push!(added, obj)
    end
    (out_set, out_set, SetDiff(added, removed))
end
function Gen.update_with_state(d::DeterministicNoCollisionSetMap, out_set, args::Tuple{<:Function, <:Any},
    argdiffs::Tuple{<:Diff, <:Diff}
)
    (Gen.apply_with_state(d, args)..., UnknownChange())
end

# function no_collision_set_map(f, set)
#     vals = [f(el) for el in set]
#     PersistentSet(vals)
# end

# function no_collision_set_map(f::Diffed{<:Function, NoChange}, set::Diffed{<:Any, <:SetDiff})

# end
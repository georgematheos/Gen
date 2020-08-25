function persistent_union(subsets::AbstractSet{<:AbstractSet{T}}) where T
    set = PersistentSet{T}()
    for subset in subsets
        for obj in subset
            set = push(set, obj)
        end
    end
    return set
end

#################
# tracked_union #
#################
struct TrackedUnion <: CustomUpdateGF{PersistentSet, PersistentSet} end
tracked_union = TrackedUnion()
function apply_with_state(::TrackedUnion, (subsets,))
    set = persistent_union(subsets)
    return (set, set)
end
function update_with_state(::TrackedUnion, old_tset, (subsets,), (subsets_diff,))
    (new_set, diff) = _update_tracked_union(old_tset, subsets, subsets_diff)
    return (new_set, new_set, diff)
end
_update_tracked_union(_, subsets, ::UnknownChange) = (persistent_union(subsets), UnknownChange())
_update_tracked_union(_, subsets, ::NoChange) = (persistent_union(subsets), NoChange())
function _update_tracked_union(old_set, subsets, diff::SetDiff)
    set = old_set
    removed = Set()
    added = Set()
    for subset in diff.removed
        for obj in subset
            set = disj(set, obj)
            push!(removed, obj)
        end
    end
    for subset in diff.added
        for obj in subset
            set = push(set, obj)
            push!(added, obj)
        end
    end
    return (set, SetDiff(removed, added))
end

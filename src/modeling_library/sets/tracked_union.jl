function persistent_union(subsets::Dict{<:Any, <:AbstractSet})
    set = PersistentSet()
    for (_, subset) in subsets
        for obj in subset
            set = push(set, obj)
        end
    end
    return set
end

#################
# tracked_union #
#################
struct TrackedUnionState
    union::PersistentSet
    subsets
end
struct TrackedUnion <: CustomUpdateGF{PersistentSet, TrackedUnionState} end
tracked_union = TrackedUnion()
function apply_with_state(::TrackedUnion, (subsets,))
    set = persistent_union(subsets)
    return (set, TrackedUnionState(set, subsets))
end
function update_with_state(::TrackedUnion, st, (subsets,), (subsets_diff,))
    (new_set, diff) = _update_tracked_union(st, subsets, subsets_diff)
    return (TrackedUnionState(new_set, subsets), new_set, diff)
end
_update_tracked_union(_, subsets, ::UnknownChange) = (persistent_union(subsets), UnknownChange())
_update_tracked_union(_, subsets, ::NoChange) = (persistent_union(subsets), NoChange())
function _update_tracked_union(st, new_subsets, diff::DictDiff)
    set = st.union
    old_subsets = st.subsets
    removed = Set()
    added = Set()

    set = handle_subset_additions_and_deletions!(set, old_subsets, diff, added, removed)
    set = handle_subset_changes!(set, old_subsets, new_subsets, diff, added, removed)

    return (set, SetDiff(added, removed))
end

function handle_subset_additions_and_deletions!(set, old_subsets, diff, added, removed)
    for key in diff.deleted
        for obj in old_subsets[key]
            set = disj(set, obj)
            push!(removed, obj)
        end
    end
    for (key, subset) in diff.added
        for obj in subset
            set = push(set, obj)
            push!(added, obj)
        end
    end
    return set
end

function handle_subset_changes!(set, old_subsets, new_subsets, diff, added, removed)
    for (key, set_diff) in diff.updated
        if set_diff === NoChange()
            continue;
        elseif set_diff isa SetDiff
            news = set_diff.added
            remvds = set_diff.deleted
        else
            news = new_subsets[key]
            remvds = old_subsets[key]
        end
        for obj in remvds
            set = disj(set, obj)
            push!(removed, obj)
        end
        for obj in news
            set = push(set, obj)
            push!(added, obj)
        end
    end
    return set
end
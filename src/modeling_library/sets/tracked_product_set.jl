struct TrackedProductSetState
    product_set::PersistentSet
    sets
end
struct TrackedProductSet <: CustomUpdateGF{PersistentSet, TrackedProductSetState} end
tracked_product_set = TrackedProductSet()
function apply_with_state(::TrackedProductSet, (sets,))
    tuples = PersistentSet()
    for tup in Iterators.product(sets...) 
        tuples = push(tuples, tup)
    end
    return (tuples, TrackedProductSetState(tuples, sets))
end

substitute_at_index(lst, i, val) = (lst[1:i-1]..., val, lst[i+1:end]...)
_handle_deletions(product_set, _, _, ::NoChange) = product_set
_handle_additions(product_set, _, _, ::NoChange) = product_set
function _handle_deletions(product_set, original_sets, i, diff::SetDiff, deleted)
    for tup in Iterators.product(substitute_at_index(original_sets, i, diff.deleted)...)
        product_set = disj(product_set, tup)
        push!(deleted, tup)
    end
    return product_set
end
function _handle_additions(product_set, new_sets, i, diff::SetDiff, added)
    for tup in Iterators.product(substitute_at_index(new_sets, i, diff.added)...)
        product_set = push(product_set, tup)
        push!(added, tup)
    end
    return product_set
end
function update_with_state(t::TrackedProductSet, st::TrackedProductSetState, (sets,), (sets_diff,)::Tuple{<:VectorDiff})
    num_sets_changed = sets_diff.new_length !== sets_diff.prev_length
    given_unrecognized_set_diff = !all((diff isa SetDiff || diff isa NoChange) for (_, diff) in sets_diff.updated)
    if num_sets_changed || given_unrecognized_set_diff
        return update_with_state(t, st, (sets,), (UnknownChange(),))
    end

    deleted = Set()
    added = Set()
    product_set = st.product_set
    for (i, diff) in sets_diff.updated
        product_set = _handle_deletions(product_set, st.sets, i, diff, deleted)
    end
    for (i, diff) in sets_diff.updated
        product_set = _handle_additions(product_set, sets, i, diff, added)
    end

    return (TrackedProductSetState(product_set, sets), product_set, SetDiff(added, deleted))
end
function update_with_state(t::TrackedProductSet, ::TrackedProductSetState, (sets,), (diff,)::Tuple{<:Diff})
    (retval, st) = apply_with_state(t, (sets,))
    (st, retval, UnknownChange())
end
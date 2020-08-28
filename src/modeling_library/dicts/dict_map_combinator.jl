struct DictTrace{KeyType, RetType, TraceType} <: Trace
    gen_fn::GenerativeFunction{<:AbstractDict{KeyType, RetType}}
    subtraces::PersistentHashMap{KeyType, TraceType}
    args::Tuple
    score::Float64
    noise::Float64
end
struct DictTraceChoiceMap <: AddressTree{Value}
    tr::DictTrace
end
function get_subtree(dtcm::DictTraceChoiceMap, key)
    haskey(dtcm.tr.subtraces, key) ? get_choices(dtcm.tr.subtraces[key]) : EmptyAddressTree()
end
get_subtree(dtcm::DictTraceChoiceMap, addr::Pair) = _get_subtree(dtcm, addr)
get_subtrees_shallow(dtcm::DictTraceChoiceMap) = ((key, get_choices(tr)) for (key, tr) in dtcm.tr.subtraces)

get_choices(trace::DictTrace) = DictTraceChoiceMap(trace)
get_retval(trace::DictTrace) = lazy_val_map(get_retval, trace.subtraces)
get_args(trace::DictTrace) = trace.args
get_score(trace::DictTrace) = trace.score
project(trace::DictTrace, ::EmptyAddressTree) = trace.noise
Base.getindex(tr::DictTrace, key) = tr.subtraces[key][]
Base.getindex(tr::DictTrace, key::Pair) = tr.subtraces[key[2]][key[2]]
"""
    DictMap(kernel)

A generative function which calls `kernel` on each value in an input dictionary,
and returns the dictionary of `key => ~kernel(val)` for each `(key, val)` in the
input dictionary.

## Example
```julia
@dist add_noise(x) = normal(x, 1)
@gen function add_noise_to_values(dict::AbstractDict)
    vals_with_noise ~ DictMap(add_noise)(dict)
    return vals_with_noise
end
```
"""
struct DictMap{KeyType, RetType, TraceType} <: Gen.GenerativeFunction{LazyValMapDict{KeyType, RetType}, DictTrace{<:KeyType, RetType, TraceType}}
    kernel::GenerativeFunction{RetType}
end
function DictMap(kernel::GenerativeFunction{RetType}) where {RetType}
    DictMap{Any, RetType, get_trace_type(kernel)}(kernel)
end
has_argument_grads(gf::DictMap) = has_argument_grads(gf.kernel)
accepts_output_grad(gf::DictMap) = accepts_output_grad(gf.kernel)

function simulate(dm::DictMap{KeyType, RetType, TraceType}, (dict,)::Tuple{<:AbstractDict}) where {KeyType, RetType, TraceType}
    subtraces = PersistentHashMap{KeyType, TraceType}()
    score = 0.
    noise = 0.
    for (key, val) in dict
        subtr = simulate(dm.kernel, (val,))
        subtraces = assoc(subtraces, key, subtr)
        score += get_score(subtr)
        noise += project(subtr, EmptyAddressTree())
    end
    return DictTrace{KeyType, RetType, TraceType}(dm, subtraces, (dict,), score, noise)
end
function generate(dm::DictMap{KeyType, RetType, TraceType}, (dict,)::Tuple{<:AbstractDict}, constraints::ChoiceMap) where {KeyType, RetType, TraceType}
    subtraces = PersistentHashMap{KeyType, TraceType}()
    score = 0.
    noise = 0.
    weight = 0.
    for (key, val) in dict
        subtr, wt = generate(dm.kernel, (val,), get_subtree(constraints, key))
        subtraces = assoc(subtraces, key, subtr)
        score += get_score(subtr)
        noise += project(subtr, EmptyAddressTree())
        weight += wt
    end
    return (DictTrace{KeyType, RetType, TraceType}(dm, subtraces, (dict,), score, noise), weight)
end

function update(tr::DictTrace{KeyType, RetType, TraceType}, (dict,)::Tuple, (diff,)::Tuple{<:Union{NoChange, <:DictDiff}}, spec::UpdateSpec, eca::Selection) where {KeyType, TraceType, RetType}
    # If this is a leaf--so we can't count on `get_subtrees_shallow`--resort to our no-argdiff update.
    if spec isa AddressTreeLeaf && spec !== EmptyAddressTree(); update(tr, (dict,), (UnknownChange(),), spec, eca); end

    subtraces = tr.subtraces
    weight = 0.
    score = tr.score
    noise = tr.noise
    discard = choicemap()

    spec_diff_dict = Dict(
        key => NoChange()
        for (key, sub) in get_subtrees_shallow(spec)
        if !isempty(sub)
    )
    if diff isa DictDiff
        in_added = diff.added
        in_deleted = diff.deleted
        in_updated = merge!(
            Dict{KeyType, Diff}(diff.updated),
            filter(((k,v),)->!haskey(in_added,k), spec_diff_dict)
        )
    elseif diff === NoChange()
        in_added = ()
        in_deleted = ()
        in_updated = spec_diff_dict
    end

    added = Dict{KeyType, RetType}()
    deleted = Set{KeyType}()
    updated = Dict{KeyType, Diff}()

    for key in in_deleted
        subtr = subtraces[key]
        subtraces = dissoc(subtraces, key)
        score -= get_score(subtr)
        noise -= project(subtr, EmptyAddressTree())
        weight -= project(subtr, get_subtree(eca, key))
        set_subtree!(discard, key, get_choices(subtr))
        push!(deleted, key)
    end
    for (key, val) in in_added
        subtr, wt = generate(tr.gen_fn.kernel, (val,), get_subtree(spec, key))
        score += get_score(subtr)
        noise += project(subtr, EmptyAddressTree())
        weight += wt
        subtraces = assoc(subtraces, key, subtr)
        added[key] = get_retval(subtr)
    end
    for (key, df) in in_updated
        old_subtr = subtraces[key]
        (new_subtr, wt, retdiff, dsc) = update(old_subtr, (dict[key],), (df,), get_subtree(spec, key), get_subtree(eca, key))
        weight += wt
        score += get_score(new_subtr) - get_score(old_subtr)
        noise += project(new_subtr, EmptyAddressTree()) - project(old_subtr, EmptyAddressTree())
        subtraces = assoc(subtraces, key, new_subtr)
        set_subtree!(discard, key, dsc)
        if retdiff !== NoChange()
            updated[key] = retdiff
        end
    end

    new_tr = DictTrace{KeyType, RetType, TraceType}(tr.gen_fn, subtraces, (dict,), score, noise)
    (new_tr, weight, DictDiff{KeyType, RetType}(added, deleted, updated), discard)
end
function update(tr::DictTrace{KeyType, RetType, TraceType}, (dict,)::Tuple, (diff,)::Tuple{<:Diff}, spec::UpdateSpec, eca::Selection) where {KeyType, TraceType, RetType}
    new_subtraces = PersistentHashMap{KeyType, TraceType}()
    discard = choicemap()
    weight = 0.
    score = 0.
    noise = 0.
    updated = Dict{KeyType, Diff}()
    added = Dict{KeyType, RetType}()
    deleted = Set{KeyType}()
    for (key, val) in dict
        if haskey(tr.subtraces, key)
            subtr = tr.subtraces[key]
            valdiff = get_args(subtr)[1] == val ? NoChange() : UnknownChange()
            (new_tr, wt, retdiff, this_discard) = update(tr.subtraces[key], (val,), (valdiff,), get_subtree(spec, key), get_subtree(eca, key))
            new_subtraces = assoc(new_subtraces, key, new_tr)
            score += get_score(new_tr)
            noise += project(new_tr, EmptyAddressTree())
            weight += wt
            if retdiff !== NoChange()
                push!(updated, key => retdiff)
            end
            set_subtree!(discard, key, this_discard)
        else
            subtr, wt = generate(tr.gen_fn.kernel, (val,), get_subtree(spec, key))
            score += get_score(subtr)
            noise += project(subtr, EmptyAddressTree())
            weight += wt
            new_subtraces = assoc(new_subtraces, key, subtr)
            push!(added, key, get_retval(subtr))
        end
    end
    for (key, subtr) in tr.subtraces
        if !haskey(tr.subtraces, key)
            ext_const = get_subtree(eca, key)
            weight -= project(subtr, addrs(get_selected(get_choices(subtr), ext_const)))
            noise -= project(subtr, EmptyAddressTree())
            set_subtree!(discard, key, get_choices(subtr))
            push!(deleted, key)
        end
    end
    new_tr = DictTrace{KeyType, RetType, TraceType}(tr.gen_fn, new_subtraces, (dict,), score, noise)
    return (new_tr, weight, DictDiff{KeyType, RetType}(added, deleted, updated), discard)
end

#=
TODO: there is probably a way to implement this using the `DictMapCombinator`
and greatly reduce the amount of needed code.
=#

struct SetTrace{allows_collisions, ArgType, TraceType} <: Trace
    gen_fn::GenerativeFunction
    subtraces::PersistentHashMap{ArgType, TraceType}
    args::Tuple
    score::Float64
    noise::Float64
end

struct SetTraceChoiceMap <: AddressTree{Value}
    tr::SetTrace
end
function get_subtree(stcm::SetTraceChoiceMap, addr)
    haskey(stcm.tr.subtraces, addr) ? get_choices(stcm.tr.subtraces[addr]) : EmptyAddressTree()
end
get_subtree(stcm::SetTraceChoiceMap, addr::Pair) = _get_subtree(stcm, addr)
get_subtrees_shallow(stcm::SetTraceChoiceMap) = ((addr, get_choices(tr)) for (addr, tr) in stcm.tr.subtraces)

get_choices(trace::SetTrace) = SetTraceChoiceMap(trace)
get_retval(trace::SetTrace{true}) = set_map(((_, tr),) -> get_retval(tr), trace.subtraces)
get_retval(trace::SetTrace{false}) = no_collision_set_map(((_, tr),) -> get_retval(tr), trace.subtraces)
get_args(trace::SetTrace) = trace.args
get_score(trace::SetTrace) = trace.score
get_gen_fn(trace::SetTrace) = trace.gen_fn
project(trace::SetTrace, ::EmptyAddressTree) = trace.noise
Base.getindex(tr::SetTrace, address) = tr.subtraces[address][]
Base.getindex(tr::SetTrace, address::Pair) = tr.subtraces[address.first][address]

struct SetMap{allows_collisions, SetRetType, TraceType} <: GenerativeFunction{SetRetType, SetTrace{allows_collisions, <:Any, TraceType}}
    kernel::GenerativeFunction # we will have TraceType as specific as possible--but sometimes the gen function may not know the fully specific trace type
end
function SetMap(kernel::GenerativeFunction{RetType, TraceType}) where {RetType, TraceType}
    SetMap{true, MultiSet{RetType}, get_trace_type(kernel)}(kernel)
end
function NoCollisionSetMap(kernel::GenerativeFunction{RetType, TraceType}) where {RetType, TraceType}
    SetMap{false, PersistentSet{RetType}, get_trace_type(kernel)}(kernel)
end
has_argument_grads(gf::SetMap) = has_argument_grads(gf.kernel)
accepts_output_grad(gf::SetMap) = accepts_output_grad(gf.kernel)

# SetMap(gen_fn)(set, shared_arg1, shared_arg2, ..., shared_argN)
function simulate(sm::SetMap{ac , <:Any, TraceType}, (set,)::Tuple{<:AbstractSet{ArgType}}) where {ac, TraceType, ArgType}
    subtraces = PersistentHashMap{ArgType, TraceType}()
    score = 0.
    noise = 0.
    for item in set
        subtr = simulate(sm.kernel, (item,))
        subtraces = assoc(subtraces, item, subtr)
        score += get_score(subtr)
        noise += project(subtr, EmptyAddressTree())
    end
    return SetTrace{ac, ArgType, TraceType}(sm, subtraces, (set,), score, noise)
end

function generate(sm::SetMap{ac, <:Any, TraceType}, (set,)::Tuple{<:AbstractSet{ArgType}}, constraints::ChoiceMap) where {ac, ArgType, TraceType}
    subtraces = PersistentHashMap{ArgType, TraceType}()
    score = 0.
    weight = 0.
    noise = 0.
    for item in set
        constraint = get_subtree(constraints, item)
        subtr, wt = generate(sm.kernel, (item,), constraint)
        weight += wt
        noise += project(subtr, EmptyAddressTree())
        subtraces = assoc(subtraces, item, subtr)
        score += get_score(subtr)
    end
    return (SetTrace{ac, ArgType, TraceType}(sm, subtraces, (set,), score, noise), weight)
end

function update(tr::SetTrace{ac, ArgType, TraceType}, (set,)::Tuple, (diff,)::Tuple{<:Union{NoChange, <:SetDiff}}, spec::UpdateSpec, eca::Selection) where {ac, ArgType, TraceType}
    # If this is a leaf--so we can't count on `get_subtrees_shallow`--resort to our no-argdiff update.
    if spec isa AddressTreeLeaf && spec !== EmptyAddressTree(); update(tr, (set,), (UnknownChange(),), spec, eca); end
    
    subtraces = tr.subtraces
    weight = 0.
    score = tr.score
    noise = tr.noise
    discard = choicemap()
    if !ac
        added = Set()
        deleted = Set()
    end

    for (addr, subspec) in get_subtrees_shallow(spec)
        !haskey(subtraces, addr) && continue
        subtr = subtraces[addr]
        new_subtr, wt, retdiff, dsc = update(subtr, (addr,), (NoChange(),), subspec, get_subtree(eca, addr))
        subtraces = assoc(subtraces, addr, new_subtr) # overwrite with new subtrace
        weight += wt
        score += get_score(new_subtr) - get_score(subtr)
        noise += project(new_subtr, EmptyAddressTree()) - project(subtr, EmptyAddressTree())
        set_subtree!(discard, addr, dsc)
        if !ac && retdiff !== NoChange()
            old = get_retval(subtr)
            new = get_retval(new_subtr)
            if old != new
                push!(added, new)
                push!(deleted, old)
            end
        end
    end

    if diff isa SetDiff
        for removed_addr in diff.deleted
            subtr = subtraces[removed_addr]
            subtraces = dissoc(subtraces, removed_addr)
            score -= get_score(subtr)
            noise -= project(subtr, EmptyAddressTree())
            weight -= project(subtr, addrs(get_selected(get_choices(subtr), get_subtree(eca, removed_addr))))
            set_subtree!(discard, removed_addr, get_choices(subtr))
            if !ac
                push!(deleted, get_retval(subtr))
            end
        end
        for new_addr in diff.added
            subtr, wt = generate(tr.gen_fn.kernel, (new_addr,), get_subtree(spec, new_addr))
            weight += wt
            score += get_score(subtr)
            noise += project(subtr, EmptyAddressTree())
            subtraces = assoc(subtraces, new_addr, subtr)
            if !ac
                push!(added, get_retval(subtr))
            end
        end
    end

    tr = SetTrace{ac, ArgType, TraceType}(tr.gen_fn, subtraces, (set,), score, noise)
    if ac
        retdiff = UnknownChange()
    elseif isempty(added) && isempty(deleted)
        retdiff = NoChange
    else
        retdiff = SetDiff(added, deleted)
    end

    return (tr, weight, retdiff, discard)    
end

function update(tr::SetTrace{ac, ArgType, TraceType}, (set,)::Tuple, argdiffs::Tuple{<:Diff}, spec::UpdateSpec, ext_const_addrs::Selection) where {ac, ArgType, TraceType}
    new_subtraces = PersistentHashMap{ArgType, TraceType}()
    discard = choicemap()
    weight = 0.
    score = 0.
    noise = 0.
    for item in set
        if item in keys(tr.subtraces)
            (new_tr, wt, retdiff, this_discard) = update(
                tr.subtraces[item], (item,),
                (UnknownChange(),),
                get_subtree(spec, item),
                get_subtree(ext_const_addrs, item)
            )
            new_subtraces = assoc(new_subtraces, item, new_tr)
            score += get_score(new_tr)
            noise += project(new_tr, EmptyAddressTree())
            weight += wt
            set_subtree!(discard, item, this_discard)
        else
            subtr, wt = generate(tr.gen_fn.kernel, (item,), get_subtree(spec, item))
            score += get_score(subtr)
            noise += project(subtr, EmptyAddressTree())
            weight += wt
            new_subtraces = assoc(new_subtraces, item, subtr)
        end
    end
    for (item, subtr) in tr.subtraces
        if !(item in set)
            ext_const = get_subtree(ext_const_addrs, item)
            weight -= project(subtr, addrs(get_selected(get_choices(subtr), ext_const)))
            noise -= project(subtr, EmptyAddressTree())
            set_subtree!(discard, item, get_choices(subtr))
        end
    end
    new_tr = SetTrace{ac, ArgType, TraceType}(tr.gen_fn, new_subtraces, (set,), score, noise)

    if ac
        retdiff = UnknownChange()
    else
        added = Set(item for item in get_retval(new_tr) if !(item in get_retval(tr)))
        deleted = Set(item for item in get_retval(tr) if !(item in get_retval(new_tr)))
        retdiff = isempty(added) && isempty(deleted) ? NoChange() : SetDiff(added, deleted)
    end

    return (new_tr, weight, retdiff, discard)
end
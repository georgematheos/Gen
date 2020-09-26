#=
TODO: there is probably a way to implement this using the `DictMapCombinator`
and greatly reduce the amount of needed code.
=#

struct SetTrace{RetType, ArgType, TraceType} <: Trace
    gen_fn::GenerativeFunction
    subtraces::PersistentHashMap{ArgType, TraceType}
    retval::RetType
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
get_retval(trace::SetTrace) = trace.retval
get_args(trace::SetTrace) = trace.args
get_score(trace::SetTrace) = trace.score
get_gen_fn(trace::SetTrace) = trace.gen_fn
project(trace::SetTrace, ::EmptyAddressTree) = trace.noise
Base.getindex(tr::SetTrace, address) = tr.subtraces[address][]
Base.getindex(tr::SetTrace, address::Pair) = tr.subtraces[address.first][address]

struct SetMap{SetRetType, TraceType} <: GenerativeFunction{SetRetType, SetTrace{SetRetType, <:Any, TraceType}}
    kernel::GenerativeFunction
end
function SetMap(kernel::GenerativeFunction{RetType, TraceType}) where {RetType, TraceType}
    SetMap{MultiSet{<:RetType}, get_trace_type(kernel)}(kernel)
end
function NoCollisionSetMap(kernel::GenerativeFunction{RetType, TraceType}) where {RetType, TraceType}
    SetMap{PersistentSet{<:RetType}, get_trace_type(kernel)}(kernel)
end
has_argument_grads(gf::SetMap) = has_argument_grads(gf.kernel)
accepts_output_grad(gf::SetMap) = accepts_output_grad(gf.kernel)

function get_return_val(RetType, subtraces)
    mapper = RetType <: MultiSet ? set_map : no_collision_set_map
    mapper(((_, tr),) -> get_retval(tr), subtraces)
end

# SetMap(gen_fn)(set, shared_arg1, shared_arg2, ..., shared_argN)
function simulate(sm::SetMap{RetType, TraceType}, (set,)::Tuple{<:AbstractSet{ArgType}}) where {RetType, TraceType, ArgType}
    subtraces = PersistentHashMap{ArgType, TraceType}()
    score = 0.
    noise = 0.
    for item in set
        subtr = simulate(sm.kernel, (item,))
        subtraces = assoc(subtraces, item, subtr)
        score += get_score(subtr)
        noise += project(subtr, EmptyAddressTree())
    end
    retval = get_return_val(RetType, subtraces)
    return SetTrace{RetType, ArgType, TraceType}(sm, subtraces, retval, (set,), score, noise)
end

function generate(sm::SetMap{RetType, TraceType}, (set,)::Tuple{<:AbstractSet{ArgType}}, constraints::ChoiceMap) where {RetType, ArgType, TraceType}
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
    retval = get_return_val(RetType, subtraces)
    return (SetTrace{RetType, ArgType, TraceType}(sm, subtraces, retval, (set,), score, noise), weight)
end

function update(tr::SetTrace{RetType, ArgType, TraceType}, (set,)::Tuple, (diff,)::Tuple{<:Union{NoChange, <:SetDiff}}, spec::UpdateSpec, eca::Selection) where {RetType, ArgType, TraceType}
    # If this is a leaf--so we can't count on `get_subtrees_shallow`--resort to our no-argdiff update.
    if spec isa AddressTreeLeaf && spec !== EmptyAddressTree(); update(tr, (set,), (UnknownChange(),), spec, eca); end
    
    subtraces = tr.subtraces
    weight = 0.
    score = tr.score
    noise = tr.noise
    discard = choicemap()
    if !(RetType <: MultiSet)
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
        if !(RetType <: MultiSet) && retdiff !== NoChange()
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
            if !(RetType <: MultiSet)
                push!(deleted, get_retval(subtr))
            end
        end
        for new_addr in diff.added
            subtr, wt = generate(tr.gen_fn.kernel, (new_addr,), get_subtree(spec, new_addr))
            weight += wt
            score += get_score(subtr)
            noise += project(subtr, EmptyAddressTree())
            subtraces = assoc(subtraces, new_addr, subtr)
            if !(RetType <: MultiSet)
                push!(added, get_retval(subtr))
            end
        end
    end

    if RetType <: MultiSet
        retdiff = UnknownChange()
        # TODO: this is a linear time operation!  can we do any better??
        new_retval = get_return_val(RetType, subtraces)
    elseif isempty(added) && isempty(deleted)
        retdiff = NoChange()
        new_retval = get_retval(tr)
    else
        retdiff = SetDiff(added, deleted)
        new_retval = get_retval(tr)
        for item in deleted
            new_retval = disj(new_retval, item)
        end
        for item in added
            new_retval = push(new_retval, item)
        end
    end
    new_tr = SetTrace{RetType, ArgType, TraceType}(tr.gen_fn, subtraces, new_retval, (set,), score, noise)

    return (new_tr, weight, retdiff, discard)    
end

function update(tr::SetTrace{RetType, ArgType, TraceType}, (set,)::Tuple, argdiffs::Tuple{<:Diff}, spec::UpdateSpec, ext_const_addrs::Selection) where {RetType, ArgType, TraceType}
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

    retval = get_return_val(RetType, new_subtraces)
    new_tr = SetTrace{RetType, ArgType, TraceType}(tr.gen_fn, new_subtraces, retval, (set,), score, noise)
    if RetType <: MultiSet
        retdiff = UnknownChange()
    else
        added = Set(item for item in retval if !(item in get_retval(tr)))
        deleted = Set(item for item in get_retval(tr) if !(item in retval))
        retdiff = isempty(added) && isempty(deleted) ? NoChange() : SetDiff(added, deleted)
    end

    return (new_tr, weight, retdiff, discard)
end
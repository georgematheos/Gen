struct SetTrace{allows_collisions, ArgType, RetType, TraceType} <: Trace
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
    get_choices(stcm.tr.subtraces[addr])
end
get_subtree(stcm::SetTraceChoiceMap, addr::Pair) = _get_subtree(stcm, addr)
get_subtrees_shallow(stcm::SetTraceChoiceMap) = ((addr, get_choices(tr) for (addr, tr) in stcm.tr.subtraces))

get_choices(trace::SetTrace) = SetTraceChoiceMap(trace)
get_retval(trace::SetTrace{true}) = set_map(((_, tr),) -> get_retval(tr), trace.subtraces)
get_retval(trace::SetTrace{false}) = no_collision_set_map(((_, tr),) -> get_retval(tr), trace.subtraces)
get_args(trace::SetTrace) = trace.args
get_score(trace::SetTrace) = trace.score
get_gen_fn(trace::SetTrace) = trace.gen_fn
project(trace::SetTrace, ::EmptyAddressTree) = trace.noise
Base.getindex(tr::SetTrace, address) = tr.subtraces[address][]
Base.getindex(tr::SetTrace, address::Pair) = tr.subtraces[address.first][address]

struct SetMap{allows_collisions, SetRetType, TraceType} <: GenerativeFunction{SetRetType, SetTrace{allows_collisions, <:Any, RetType, TraceType}}
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
function simulate(sm::SetMap{<:Any , RetType, TraceType}, args::Tuple{<:AbstractSet{ArgType}, Vararg}) where {RetType, TraceType, ArgType}
    set, shared_args = args[1], args[2:end]
    subtraces = PersistentHashMap{ArgType, TraceType}()
    score = 0.
    noise = 0.
    for item in set
        subtr = simulate(sm.kernel, (item, shared_args...))
        subtraces = assoc(subtraces, item, subtr)
        score += get_score(subtr)
        noise += project(subtr, EmptyAddressTree())
    end
    return SetTrace{ArgType, RetType, TraceType}(sm, subtraces, (set,), score, noise)
end

function generate(sm::SetMap{ac, RetType, TraceType}, args::Tuple{<:AbstractSet{ArgType}, Vararg}, constraints::ChoiceMap) where {ac, ArgType, RetType, TraceType}
    set, shared_args = args[1], args[2:end]
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
    return (SetTrace{ac, ArgType, RetType, TraceType}(sm, subtraces, (set,), score), weight, noise)
end

function update(tr::SetTrace{ac, ArgType, RetType, TraceType}, args::Tuple, argdiffs::Tuple{Vararg{NoChange}}, spec::UpdateSpec, eca::Selection) where {ac, ArgType, RetType, TraceType}

end
function update(tr::SetTrace{ac, ArgType, RetType, TraceType}, args::Tuple, argdiffs::Tuple{Vararg{NoChange}}, spec::UpdateSpec, eca::Selection) where {ac, ArgType, RetType, TraceType}
    
end

function update(tr::SetTrace{ac, ArgType, RetType, TraceType}, (set,)::Tuple, ::Tuple{<:Diff}, spec::UpdateSpec, ext_const_addrs::Selection) where {ac, ArgType, RetType, TraceType}
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
            tr, weight = generate(tr.gen_fn.kernel, (item,), get_subspec(spec, item))
            score += get_score(tr)
            noise += project(tr, EmptyAddressTree())
            new_subtraces = assoc(new_subtraces, item, tr)
        end
    end
    for (item, tr) in tr.subtraces
        if !(item in set)
            ext_const = get_subtree(ext_const_addrs, item)
            weight -= project(tr, addrs(get_selected(get_choices(tr), ext_const)))
            set_subtree!(discard, item, get_choices(tr))
        end
    end
    tr = SetTrace{ac, ArgType, RetType, TraceType}(tr.gen_fn, new_subtraces, (set,), score, noise)
    return (tr, weight, UnknownChange(), discard)
end
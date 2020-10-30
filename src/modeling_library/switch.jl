# TODO: Be more careful with type safety

struct SwitchTrace <: Trace
    gen_fn
    branch::Int
    subtrace
end
get_choices(tr::SwitchTrace) = CallAtChoiceMap(tr.branch, get_choices(tr.subtrace))
get_gen_fn(tr::SwitchTrace) = tr.gen_fn
get_retval(tr::SwitchTrace) = get_retval(tr.subtrace)
get_args(tr::SwitchTrace) = get_args(tr.subtrace)
get_score(tr::SwitchTrace) = get_score(tr.subtrace)
project(tr::SwitchTrace, sel::Selection) = project(tr.subtrace, sel)
project(tr::SwitchTrace, ::EmptyAddressTree) = project(tr.subtrace, EmptyAddressTree())
# TODO: Base.getindex ?

struct Switch <: Gen.GenerativeFunction{Any, SwitchTrace}
    kernels::Tuple{Vararg{GenerativeFunction}}
    Switch(kernels...) = new(kernels)
end
has_argument_grads(gf::Switch) = all(has_argument_grads(gen_fn) for gen_fn in gf.kernels)
accepts_output_grad(gf::Switch) = all(accepts_output_grad(gen_fn) for gen_fn in gf.kernels)

# TODO: Once Julia 1.5.3 is out, we should be able to do (i, args...)
# in the signature and have it work
function simulate(s::Switch, args::Tuple)
    i = args[1]; args = args[2:end]
    SwitchTrace(s, i, simulate(s.kernels[i], args))
end

function generate(s::Switch, args::Tuple, constraints::ChoiceMap)
    i = args[1]; args = args[2:end]
    (subtr, weight) = generate(s.kernels[i], args, get_subtree(constraints, i)) 
    return (SwitchTrace(s, i, subtr), weight)
end

function update(
    tr::SwitchTrace,
    args::Tuple,
    diffs::Tuple{NoChange, Vararg{<:Diff}},
    constraints::UpdateSpec,
    eca::Selection
)
    i = args[1]; args = args[2:end]
    diffs = diffs[2:end]
    subtr, weight, retdiff, discard = update(tr.subtrace, args, diffs, get_subtree(constraints, i), get_subtree(eca, i))
    newtr = SwitchTrace(get_gen_fn(tr), i, subtr)
    (newtr, weight, retdiff, CallAtChoiceMap(i, discard))
end

function update(
    tr::SwitchTrace,
    args::Tuple,
    diffs::Tuple{<:Diff, Vararg{<:Diff}},
    constraints::UpdateSpec,
    eca::Selection
)
    i = args[1]; args = args[2:end]
    diffs = diffs[2:end]
    if tr.branch == i
        update(tr, (i, args...), (NoChange(), diffs...), constraints, eca)
    else
        (newtr, weight) = generate(get_gen_fn(tr).kernels[i], args, get_subtree(constraints, i))
        weight = weight - project(tr.subtrace, get_subtree(eca, tr.branch))
        diff = get_retval(newtr) == get_retval(tr) ? NoChange() : UnknownChange()
        return (SwitchTrace(get_gen_fn(tr), i, newtr), weight, diff, get_choices(tr))
    end
end

"""
    switchint(conditions...)

Returns the index of the first true condition, or `length(conditions) + 1` if none are true.
"""
function switchint(conditions::Vararg{Bool})
    first = findfirst(conditions)
    if first === nothing
        length(conditions) + 1
    else
        first
    end
end
function switchint(conditions::Vararg{Diffed{Bool, NoChange}})
    Diffed(switchint(map(strip_diff, conditions)...), NoChange())
end
function switchint(conditions::Vararg{Diffed{Bool}})
    diff = NoChange()
    for (i, cond) in enumerate(conditions)
        if get_diff(cond) !== NoChange()
            diff = UnknownChange()
        end
        if strip_diff(cond)
            return Diffed(i, diff)
        end
    end
    return Diffed(length(conditions) + 1, diff)
end

export Switch, switchint
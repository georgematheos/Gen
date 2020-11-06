### lazy_val_map ###
struct LazyValMapDict <: AbstractDict{Any, Any}
    f::Function
    keys_to_vals
end

Base.getindex(dict::LazyValMapDict, key) = dict.f(dict.keys_to_vals[key])
function Base.get(dict::LazyValMapDict, key, v)
    haskey(dict, key) ? dict[key] : v
end
Base.haskey(dict::LazyValMapDict, key) = haskey(dict.keys_to_vals, key)
Base.length(dict::LazyValMapDict) = length(dict.keys_to_vals)

_iterator(dict::LazyValMapDict) = (key => dict.f(val) for (key, val) in dict.keys_to_vals)
Base.iterate(dict::LazyValMapDict) = iterate(_iterator(dict))
Base.iterate(dict::LazyValMapDict, st) = iterate(_iterator(dict), st)

"""
    lazy_val_map(f, dict)

Return a dictionary of `key => f(val)` for `(key, val)` in `dict`.
Calls to `f` are performed lazily.
"""
lazy_val_map(f, dict) = LazyValMapDict(f, dict)
function lazy_val_map(f::Diffed{<:Function, NoChange}, dict::Diffed{<:Any, DictDiff})
    f = strip_diff(f)
    diff = get_diff(dict)
    newdiff = DictDiff(
        lazy_val_map(f, diff.added),
        diff.deleted,
        lazy_set_to_dict_map(key -> strip_diff(f(dict[key])), keys(diff.updated))
    )
    Diffed(LazyValMapDict(f, strip_diff(dict)), newdiff)
end
function lazy_val_map(f::Diffed, dict::Diffed)
    Diffed(lazy_val_map(strip_diff(f), strip_diff(dict)), UnknownChange())
end
function lazy_val_map(f::Diffed{<:Function, NoChange}, dict::Diffed{<:Any, NoChange})
    Diffed(lazy_val_map(strip_diff(f), strip_diff(dict)), NoChange())
end

### lazy_set_to_dict_map ###
struct LazySetToDictMap{K} <: AbstractDict{K, Any}
    f::Function
    keys::AbstractSet{K}
end
Base.getindex(dict::LazySetToDictMap{K}, key::K) where K = dict.f(key)
Base.get(dict::LazySetToDictMap{K}, key::K, v) where K = haskey(dict.keys, key) ? dict.f(key) : v
Base.haskey(dict::LazySetToDictMap{K}, key::K) where {K} = key in dict.keys
Base.length(dict::LazySetToDictMap) = length(dict.keys)

_iterator(dict::LazySetToDictMap) = (key => dict.f(key) for key in dict.keys)
Base.iterate(dict::LazySetToDictMap) = iterate(_iterator(dict))
Base.iterate(dict::LazySetToDictMap, st) = iterate(_iterator(dict), st)

"""
    lazy_set_to_dict_map(f, keys)

Return a dictionary of `key => f(key)` for each `key` in `keys`.
Calls to `f` are performed lazily.
"""
lazy_set_to_dict_map(f, keys) = LazySetToDictMap(f, keys)
function lazy_set_to_dict_map(f::Diffed{<:Function, NoChange}, keys::Diffed{<:Any, <:SetDiff})
    f = strip_diff(f)
    indiff = get_diff(keys)
    outdiff = DictDiff(
        lazy_set_to_dict_map(f, indiff.added),
        indiff.deleted,
        Dict{Any, Diff}()
    )
    Diffed(lazy_set_to_dict_map(f, strip_diff(keys)), outdiff)
end
function lazy_set_to_dict_map(f::Diffed{<:Function, NoChange}, keys::Diffed{<:Any, NoChange})
    Diffed(lazy_set_to_dict_map(strip_diff(f), strip_diff(keys)), NoChange())
end
function lazy_set_to_dict_map(f::Diffed, keys::Diffed)
    Diffed(lazy_set_to_dict_map(strip_diff(f), strip_diff(keys)), UnknownChange())
end

### lazy_bijection_set_map ###
struct LazyBijectionSetMap{K} <: AbstractSet{Any}
    f::Function
    f_inv::Function
    keys::AbstractSet{K}
end
Base.in(set::LazyBijectionSetMap{K}, v) where K = f_inv(v) in set.keys
Base.length(set::LazyBijectionSetMap) = length(set.keys)

function _iterator(set::LazyBijectionSetMap)
    try
        collect((set.f(item) for item in set.keys))
    catch e
        println("caught error! keys are:")
        display(set.keys)
        println("f(firstkey) is ", set.f(first(set.keys)))
        throw(e)
    end
    (set.f(item) for item in set.keys)
end
Base.iterate(set::LazyBijectionSetMap) = iterate(_iterator(set))
Base.iterate(set::LazyBijectionSetMap, st) = iterate(_iterator(set), st)

"""
    lazy_bijection_set_map(f, keys)

Return a set of elements `f(key)` for each key in keys,
where `f` is a bijection with inverse `f_inv`.
Calls to `f` and `f_inv` are performed lazily.
"""
lazy_bijection_set_map(f, f_inv, keys) = LazyBijectionSetMap(f, f_inv, keys)
function lazy_bijection_set_map(f::Diffed{<:Function, NoChange}, f_inv::Diffed{<:Function, NoChange}, keys::Diffed{<:Any, <:SetDiff})
    f = strip_diff(f)
    f_inv = strip_diff(f_inv)
    indiff = get_diff(keys)
    outdiff = SetDiff(
        lazy_bijection_set_map(f, f_inv, indiff.added),
        lazy_bijection_set_map(f, f_inv, indiff.deleted)
    )
    Diffed(lazy_bijection_set_map(f, f_inv, strip_diff(keys)), outdiff)
end
function lazy_bijection_set_map(f::Diffed{<:Function, NoChange}, f_inv::Diffed{<:Function, NoChange}, keys::Diffed{<:Any, NoChange})
    Diffed(lazy_bijection_set_map(strip_diff(f), strip_diff(f_inv), strip_diff(keys)), NoChange())
end
function lazy_bijection_set_map(f::Diffed, f_inv::Diffed, keys::Diffed)
    Diffed(lazy_bijection_set_map(strip_diff(f), strip_diff(f_inv), strip_diff(keys)), UnknownChange())
end

### lazy_map ###
struct LazyMap{InType} <: AbstractVector{Any}
    f::Function
    in::InType
    function LazyMap(f::Function, in::T) where {
        T <: Union{Tuple, AbstractVector}
    }
        new{T}(f, in)
    end
end
Base.size(l::LazyMap{<:AbstractVector}) = size(l.in)
Base.size(l::LazyMap{<:Tuple}) = (length(l.in),)
Base.getindex(l::LazyMap, i) = l.f(getindex(l.in, i))

"""
    lazy_map(f, in_vector)

Returns a vector [f(x) for x in in_vector] which is constructed lazily.
"""
lazy_map(f, in) = LazyMap(f, in)
function lazy_map(f::Diffed{<:Function, NoChange}, in::Diffed{<:Any, <:VectorDiff})
    f = strip_diff(f)
    diff = get_diff(in)
    passthrough_updated = lazy_val_map(i -> strip_diff(f(in[i])), keys(diff.updated))
    Diffed(
        lazy_map(strip_diff(f), strip_diff(in)),
        VectorDiff(diff.new_length, diff.prev_length, passthrough_updated)
    )
end
lazy_map(f::Diffed{<:Function, NoChange}, in::Diffed{<:Any, NoChange}) = Diffed(lazy_map(strip_diff(f), strip_diff(in)), NoChange())
lazy_map(f::Diffed{<:Function, <:Any}, in::Diffed{<:Any, <:Any}) = Diffed(lazy_map(strip_diff(f), strip_diff(in)), UnknownChange())

export lazy_val_map, lazy_set_to_dict_map, lazy_bijection_set_map, lazy_map
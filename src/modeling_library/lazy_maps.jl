### lazy_val_map ###
struct LazyValMapDict{K, V} <: AbstractDict{K, V}
    f::Function
    keys_to_vals::AbstractDict{K, V}
end
Base.getindex(dict::LazyValMapDict{K}, key::K) where {K} = dict.f(dict.keys_to_vals[key])
function Base.get(dict::LazyValMapDict{K}, key::K, v) where {K}
    haskey(dict, key) ? dict[key] : v
end
Base.haskey(dict::LazyValMapDict{K}, key::K) where {K} = haskey(dict.keys_to_vals, key)
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

### lazy_no_collision_set_map ###
struct LazyNoCollisionSetMap{K} <: AbstractSet{Any}
    f::Function
    keys::AbstractSet{K}
end
Base.in(set::LazyNoCollisionSetMap{K}, v) where K = set.f(v) in set.keys
Base.length(set::LazyNoCollisionSetMap) = length(set.keys)

_iterator(set::LazyNoCollisionSetMap) = (set.f(item) for item in dict.keys)
Base.iterate(set::LazyNoCollisionSetMap) = iterate(_iterator(set))
Base.iterate(set::LazyNoCollisionSetMap, st) = iterate(_iterator(set), st)

"""
    lazy_no_collision_set_map(f, keys)

Return a set of elements `f(key)` for each key in keys.
Calls to `f` are performed lazily.
"""
lazy_no_collision_set_map(f, keys) = LazyNoCollisionSetMap(f, keys)
function lazy_no_collision_set_map(f::Diffed{<:Function, NoChange}, keys::Diffed{<:Any, <:SetDiff})
    f = strip_diff(f)
    indiff = get_diff(keys)
    outdiff = SetDiff(
        lazy_no_collision_set_map(f, indiff.added),
        lazy_no_collision_set_map(f, indiff.deleted)
    )
    Diffed(lazy_no_collision_set_map(f, strip_diff(keys)), outdiff)
end
function lazy_no_collision_set_map(f::Diffed{<:Function, NoChange}, keys::Diffed{<:Any, NoChange})
    Diffed(lazy_no_collision_set_map(strip_diff(f), strip_diff(keys)), NoChange())
end
function lazy_no_collision_set_map(f::Diffed, keys::Diffed)
    Diffed(lazy_no_collision_set_map(strip_diff(f), strip_diff(keys)), UnknownChange())
end

### TODO: lazy_set_map ###

### lazy_map ###
struct LazyMap{K} <: AbstractVector{Any}
    f::Function
    in::AbstractVector{K}
end
Base.size(l::LazyMap) = size(l.in)
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

export lazy_val_map, lazy_set_to_dict_map, lazy_no_collision_set_map, lazy_map
include("address_schema.jl")

"""
    AddressTree{LeafType}

Abstract type for trees where each node's subtrees are labelled with
an address.  All leaf nodes are of `LeafType` (or are an `EmptyAddressTree`).
"""
abstract type AddressTree{LeafType} end

"""
    AddressTreeLeaf

Abstract type for address tree leaf nodes.

## Note: 
When declaring a subtype `T` of `AddressTreeLeaf`,
declare `T <: AddressTreeLeaf{T}` to ensure
`T <: AddressTree{T}`.
"""
abstract type AddressTreeLeaf{Type} <: AddressTree{Type} end

"""
    EmptyAddressTree

An empty address tree with no subtrees.
"""
struct EmptyAddressTree <: AddressTreeLeaf{EmptyAddressTree} end

"""
    Value{T}

An address tree leaf node storing a value of type `T`.
"""
struct Value{T} <: AddressTreeLeaf{Value}
    val::T
end
@inline get_value(v::Value) = v.val

# Note that I don't set `Value{T} <: AddressTreeLeaf{Value{T}}`.
# I have run into issues when I do this,
# because then `Value{T} <: AddressTree{Value}` is not true.
# There may be some way to make this work, but I haven't been
# able to figure it out, and I don't currently have any need
# for specifying `AddressTree{Value{T}}` types, so I'm not going
# to worry about it for now.

"""
    SelectionLeaf

Abstract type for a `Selection` which cannot be naturally decomposed
into "subtrees" as an address tree.  (Often this is because infinitely
many addresses should have `get_subtree` return a nonempty tree.)
"""
abstract type SelectionLeaf <: AddressTreeLeaf{SelectionLeaf} end

"""
    AllSelection

An address tree leaf node representing that all sub-addresses
from this point are selected.
"""
struct AllSelection <: SelectionLeaf end

"""
    CustomUpdateSpec

Supertype for custom update specifications.
"""
abstract type CustomUpdateSpec <: AddressTreeLeaf{CustomUpdateSpec} end
const UpdateSpec = AddressTree{<:Union{Value, SelectionLeaf, EmptyAddressTree, CustomUpdateSpec}}

"""
    get_subtree(tree::AddressTree{T}, addr)::Union{AddressTree{T}, EmptyAddressTree}

Get the subtree at address `addr` or return `EmptyAddressTree`
if there is no subtree at this address.

Invariant: `get_subtree(::AddressTree{LeafType}, addr)` either returns
an object of `LeafType` or an `EmptyAddressTree`.
"""
function get_subtree(::AddressTree{LeafType}, addr)::AddressTree{LeafType} where {LeafType} end

function _get_subtree(t::AddressTree, addr::Pair)
    get_subtree(get_subtree(t, addr.first), addr.second)
end

"""
    get_subtrees_shallow(tree::AddressTree{T})

Return an iterator over tuples `(address, subtree::AddressTree{T})` for each
top-level address associated with `tree`.

The length of this iterator must nonzero if this is not a leaf node.
"""
function get_subtrees_shallow end

get_leaf_type(T::Type{AddressTree{U}}) where {U} = U

"""
schema = get_address_schema(::Type{T}) where {T <: AddressTree}

Return the (top-level) address schema for the given address tree type.
"""
function get_address_schema end
@inline get_address_schema(::Type{EmptyAddressTree}) = EmptyAddressSchema()
@inline get_address_schema(::Type{AllSelection}) = AllAddressSchema()

@inline get_address_schema(::Type{Value}) = error("I don't think this currently gets called, and it's not part of the user-facing interface.  If we need this, set the appropriate value then.")

Base.isempty(::Value) = false
Base.isempty(::AllSelection) = false
Base.isempty(::EmptyAddressTree) = true
Base.isempty(::AddressTreeLeaf) = error("Not implemented")
Base.isempty(t::AddressTree) = all(((_, subtree),) -> isempty(subtree), get_subtrees_shallow(t))

@inline get_subtree(::AddressTreeLeaf, _) = EmptyAddressTree()
@inline get_subtrees_shallow(::AddressTreeLeaf) = ()

@inline get_subtree(::AllSelection, _) = AllSelection()

function Base.:(==)(a::AddressTree, b::AddressTree)
    for (addr, subtree) in get_subtrees_shallow(a)
        if get_subtree(b, addr) != subtree
            return false
        end
    end
    for (addr, subtree) in get_subtrees_shallow(b)
        if get_subtree(a, addr) != subtree
            return false
        end
    end
    return true
end
@inline Base.:(==)(a::Value, b::Value) = a.val == b.val
Base.:(==)(a::AddressTreeLeaf, b::AddressTreeLeaf) = false
Base.:(==)(::T, ::T) where {T <: AddressTreeLeaf} = true

Base.isapprox(a::Value, b::Value) = isapprox(a.val, b.val)
Base.isapprox(::EmptyAddressTree, ::EmptyAddressTree) = true
Base.isapprox(::AllSelection, ::AllSelection) = true
function Base.isapprox(::AddressTreeLeaf{T}, ::AddressTreeLeaf{U}) where {T, U}
    if T != U
        false
    else
        error("Not implemented")
    end
end
function Base.isapprox(a::AddressTree, b::AddressTree)
    for (addr, subtree) in get_subtrees_shallow(a)
        if !isapprox(get_subtree(b, addr), subtree)
            return false
        end
    end
    for (addr, subtree) in get_subtrees_shallow(b)
        if !isapprox(get_subtree(a, addr), subtree)
            return false
        end
    end
    return true
end

"""
    Base.merge(a::AddressTree, b::AddressTree)

Merge two address trees.
"""
function Base.merge(a::AddressTree{T}, b::AddressTree{U}) where {T, U}
    tree = DynamicAddressTree{Union{T, U}}()
    for (key, subtree) in get_subtrees_shallow(a)
        set_subtree!(tree, key, merge(subtree, get_subtree(b, key)))
    end
    for (key, subtree) in get_subtrees_shallow(b)
        if isempty(get_subtree(a, key))
            set_subtree!(tree, key, subtree)
        end
    end
    tree
end
Base.merge(t::AddressTree, ::EmptyAddressTree) = t
Base.merge(::EmptyAddressTree, t::AddressTree) = t
Base.merge(t::AddressTreeLeaf, ::EmptyAddressTree) = t
Base.merge(::EmptyAddressTree, t::AddressTreeLeaf) = t

Base.merge(::AddressTreeLeaf, ::AddressTree) = error("Not implemented")
Base.merge(::AddressTree, ::AddressTreeLeaf) = error("Not implemented")

"""
Variadic merge of address trees.
"""
function Base.merge(first::AddressTree, rest::AddressTree...)
    reduce(Base.merge, rest; init=first)
end

function _show_pretty(io::IO, tree::AddressTree, pre, vert_bars::Tuple)
    VERT = '\u2502'
    PLUS = '\u251C'
    HORZ = '\u2500'
    LAST = '\u2514'
    indent_vert = vcat(Char[' ' for _ in 1:pre], Char[VERT, '\n'])
    indent_vert_last = vcat(Char[' ' for _ in 1:pre], Char[VERT, '\n'])
    indent = vcat(Char[' ' for _ in 1:pre], Char[PLUS, HORZ, HORZ, ' '])
    indent_last = vcat(Char[' ' for _ in 1:pre], Char[LAST, HORZ, HORZ, ' '])
    for i in vert_bars
        indent_vert[i] = VERT
        indent[i] = VERT
        indent_last[i] = VERT
    end
    indent_vert_str = join(indent_vert)
    indent_vert_last_str = join(indent_vert_last)
    indent_str = join(indent)
    indent_last_str = join(indent_last)
    key_and_subtrees = collect(get_subtrees_shallow(tree))
    n = length(key_and_subtrees)
    cur = 1
    for (key, subtree) in key_and_subtrees
        print(io, indent_vert_str)
        if subtree isa AddressTreeLeaf
            print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key)) : $subtree\n")
        else
            if isempty(subtree); continue; end;
            print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key))\n")
            _show_pretty(io, subtree, pre + 4, cur == n ? (vert_bars...,) : (vert_bars..., pre+1))
        end
        cur += 1
    end
end

function _show_pretty_terse(io, tree::AddressTree)
    print(io, "{")

    itr = collect(get_subtrees_shallow(tree))
    for (i, (addr, subtree)) in enumerate(itr)
        print(io, "$(repr(addr)) => ")
        _show_pretty_terse(io, subtree)
        if i != length(itr)
            print(io, ", ")
        end
    end

    print(io, "}")
end
_show_pretty_terse(io, tree::AddressTreeLeaf) = Base.show_default(io, tree)

function Base.show(io::IO, ::MIME"text/plain", tree::AddressTree)
    _show_pretty(io, tree, 0, ())
end
Base.show(io::IO, ::MIME"text/plain", t::AddressTreeLeaf) = print(io, t)

function Base.show(io::IO, tree::AddressTree)
    _show_pretty_terse(io, tree)
end

function nonempty_subtree_itr(itr)
    ((addr, subtree) for (addr, subtree) in itr if !isempty(subtree))
end

include("dynamic_address_tree.jl")
include("static_address_tree.jl")

include("choicemap.jl")
include("selection.jl")

"""
    regenchoicemap()

Construct an empty mutable update specification which may constrain
values at addresses and specify for other values to be regenerated.
"""
function regenchoicemap()
    DynamicAddressTree{Union{Value, SelectionLeaf}}()
end
"""
    regenchoicemap(tuples...)

Construct a mutable update specification which may constrain
values at addresses and specify for other values to be regenerated.
This update spec will be initialized with:
- For each `(addr, selection::Selection)` in `tuples`, the given `selection` will be at address `addr`
- For each `(addr, val)` in `tuples` (where `val` is not a `Selection), the given value
will be at address `addr`
"""
function regenchoicemap(tuples...)
    rcm = regenchoicemap()
    for (addr, val) in tuples
        rcm[addr] = val
    end
    rcm
end

"""
    all_values_deep(tree::Gen.AddressTree)

Returns an iterator over all values in the `tree`, where each value is the contents of a `Value` leaf node.
"""
all_values_deep(v::Value) = (get_value(v),)
all_values_deep(::AddressTreeLeaf) = ()
function all_values_deep(tree::AddressTree)
    Iterators.flatten(
        (all_values_deep(subtree) for (_, subtree) in get_subtrees_shallow(tree))
    )
end

function Base.setindex!(rcm::DynamicAddressTree{Union{Value, SelectionLeaf}}, sel::Selection, addr)
    set_subtree!(rcm, addr, sel)
end
function Base.setindex!(rcm::DynamicAddressTree{Union{Value, SelectionLeaf}}, val, addr)
    set_subtree!(rcm, addr, Value(val))
end

export get_subtree, get_subtrees_shallow
export EmptyAddressTree, Value, AllSelection, SelectionLeaf, CustomUpdateSpec, UpdateSpec
export get_address_schema
export regenchoicemap, all_values_deep
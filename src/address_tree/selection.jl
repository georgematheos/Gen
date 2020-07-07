const Selection = AddressTree{<:Union{AllSelection, EmptyAddressTree}}

const StaticSelection = StaticAddressTree{AllSelection}
const EmptySelection = EmptyAddressTree

"""
    in(addr, selection::Selection)

Whether the address is selected in the given selection.
"""
@inline function Base.in(addr, selection::Selection) 
    get_subtree(selection, addr) === AllSelection()
end

# indexing returns subtrees for selections
Base.getindex(selection::AddressTree{AllSelection}, addr) = get_subtree(selection, addr)

# TODO: deprecate indexing syntax and only use this
get_subselection(s::Selection, addr) = get_subtree(s, addr)

get_subselections(s::Selection) = get_subtrees_shallow(s)

Base.merge(::AllSelection, ::Selection) = AllSelection()
Base.merge(::Selection, ::AllSelection) = AllSelection()
Base.merge(::AllSelection, ::AllSelection) = AllSelection()

"""
    filtered = SelectionFilteredAddressTree(tree, selection)

An address tree containing only the nodes in `tree` whose addresses are selected
in `selection.`
"""
struct SelectionFilteredAddressTree{T} <: AddressTree{T}
    tree::AddressTree{T}
    sel::Selection
end
SelectionFilteredAddressTree(t::AddressTree, ::AllSelection) = t
SelectionFilteredAddressTree(t::AddressTreeLeaf, ::AllSelection) = t
SelectionFilteredAddressTree(::AddressTree, ::EmptyAddressTree) = EmptyAddressTree()
SelectionFilteredAddressTree(::AddressTreeLeaf, ::EmptyAddressTree) = EmptyAddressTree()
SelectionFilteredAddressTree(::AddressTreeLeaf, ::Selection) = EmptyAddressTree() # if we hit a leaf node before a selected value, the node is not selected

function get_subtree(t::SelectionFilteredAddressTree, addr)
    subselection = get_subtree(t.sel, addr)
    if subselection === EmptyAddressTree()
        EmptyAddressTree()
    else
        SelectionFilteredAddressTree(get_subtree(t.tree, addr), subselection)
    end
end

function get_subtrees_shallow(t::SelectionFilteredAddressTree)
    nonempty_subtree_itr(
        (addr, SelectionFilteredAddressTree(subtree, get_subtree(t.sel, addr)))
        for (addr, subtree) in get_subtrees_shallow(t.tree)
    )
end

"""
    selected = get_selected(tree::AddressTree, selection::Selection)

Filter the address tree `tree` to only include leaf nodes at selected
addresses.
"""
get_selected(tree::AddressTree, selection::Selection) = SelectionFilteredAddressTree(tree, selection)

"""
    struct DynamicSelection <: HierarchicalSelection .. end
A hierarchical, mutable, selection with arbitrary addresses.
Can be mutated with the following methods:
    Base.push!(selection::DynamicSelection, addr)
Add the address and all of its sub-addresses to the selection.
Example:
```julia
selection = select()
@assert !(:x in selection)
push!(selection, :x)
@assert :x in selection
```
    set_subselection!(selection::DynamicSelection, addr, other::Selection)
Change the selection status of the given address and its sub-addresses that defined by `other`.
Example:
```julia
selection = select(:x)
@assert :x in selection
subselection = select(:y)
set_subselection!(selection, :x, subselection)
@assert (:x => :y) in selection
@assert !(:x in selection)
```
Note that `set_subselection!` does not copy data in `other`, so `other` may be mutated by a later calls to `set_subselection!` for addresses under `addr`.
"""
const DynamicSelection = DynamicAddressTree{AllSelection}
Base.push!(s::DynamicSelection, addr) = set_subtree!(s, addr, AllSelection())
set_subselection!(s::DynamicSelection, addr, sub::Selection) = set_subtree!(s, addr, sub)

function select(addrs...)
    selection = DynamicSelection()
    for addr in addrs
        set_subtree!(selection, addr, AllSelection())
    end
    selection
end

"""
    AddressSelection(::AddressTree)

A selection containing all of the addresses in the given address tree with a nonempty leaf node.
"""
struct AddressSelection{T} <: AddressTree{AllSelection}
    a::T
    AddressSelection(a::T) where {T <: AddressTree} = new{T}(a)
end
AddressSelection(::AddressTreeLeaf) = AllSelection()
AddressSelection(::EmptyAddressTree) = EmptyAddressTree()
get_subtree(a::AddressSelection, addr) = AddressSelection(get_subtree(a.a, addr))
function get_subtrees_shallow(a::AddressSelection)
    nonempty_subtree_itr((addr, AddressSelection(subtree)) for (addr, subtree) in get_subtrees_shallow(a.a))
end
get_address_schema(::Type{AddressSelection{T}}) where {T} = get_address_schema(T)

"""
    addrs(::AddressTree)

Returns a selection containing all of the addresses in the tree with a nonempty leaf node.
"""
addrs(a::AddressTree) = AddressSelection(a)

"""
    SelectionDiff(sel::Selection, minus::Selection)

Returns the selection containing every address in `sel` but not in `minus`.
"""
struct SelectionDiff <: AddressTree{AllSelection}
    sel::Selection
    minus::Selection
end
SelectionDiff(::EmptyAddressTree, ::Selection) = EmptyAddressTree()
SelectionDiff(::Selection, ::AllSelection) = EmptyAddressTree()
SelectionDiff(::EmptyAddressTree, ::AllSelection) = EmptyAddressTree()
SelectionDiff(s::Selection, ::EmptyAddressTree) = s
SelectionDiff(::EmptyAddressTree, ::EmptyAddressTree) = EmptyAddressTree()
get_subtree(a::SelectionDiff, addr) = SelectionDiff(get_subtree(a.sel, addr), get_subtree(a.minus, addr))
function get_subtrees_shallow(a::SelectionDiff)
    nonempty_subtree_itr(
        (addr, SelectionDiff(subtree, get_subtree(a.minus, addr)))
        for (addr, subtree) in get_subtrees_shallow(a.sel)
    )
end
# TODO: address schema?

"""
    selection1 - selection2

The selection containing every address in `selection1` but not in `selection2`.
"""
Base.:(-)(sel::Selection, minus::Selection) = SelectionDiff(sel, minus)

"""
    SelectedAddrs(::AddressTree)

The selection containg all addresses in the given address tree
which contain an `AllSelection` (or are downstream from an `AllSelection`).
"""
struct SelectedAddrs <: AddressTree{AllSelection}
    tree::AddressTree
    SelectedAddrs(tree::AddressTree{T}) where {AllSelection <: T <: Any} = new(tree)
end
SelectedAddrs(::AllSelection) = AllSelection()
SelectedAddrs(::AddressTree) = EmptyAddressTree()
get_subtree(s::SelectedAddrs, addr) = SelectedAddrs(get_subtree(s.tree, addr))
function get_subtrees_shallow(s::SelectedAddrs)
    nonempty_subtree_itr((addr, SelectedAddrs(sub)) for (addr, sub) in get_subtrees_shallow(s.tree))
end

"""
    underlying_selection(::AddressTree)

Returns the "underlying selection" in the given address tree, ie. the selection
containg all addresses in the address tree
which point to an `AllSelection` (or are downstream from an `AllSelection`).
"""
underlying_selection(t::AddressTree) = SelectedAddrs(t)

export select, get_selected, underlying_selection, addrs, get_subselection, get_subselections
export Selection, DynamicSelection, EmptySelection, StaticSelection
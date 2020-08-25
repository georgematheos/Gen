import FunctionalCollections
using FunctionalCollections: PersistentSet, disj

# include("uniform_from_set.jl")
# include("tracked_union.jl")
include("multiset.jl")
include("set_map.jl")
# include("set_map_combinators.jl")

export SetMap, NoCollisionSetMap
export set_map, no_collision_set_map
# export tracked_union
export MultiSet
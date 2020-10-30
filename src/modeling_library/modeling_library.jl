#############################
# probability distributions #
#############################

import Distributions
using SpecialFunctions: loggamma, logbeta, digamma

# built-in distributions
include("distributions/distributions.jl")

# @dist DSL
include("dist_dsl/dist_dsl.jl")

### lazy maps ###
include("lazy_maps.jl")

###############
# combinators #
###############

# code shared by vector-shaped combinators
include("vector.jl")

# built-in generative function combinators
include("call_at/call_at.jl")
include("map/map.jl")
include("unfold/unfold.jl")
include("recurse/recurse.jl")
include("switch.jl")
include("dict_map_combinator.jl")

#############################################################
# abstractions for constructing custom generative functions #
#############################################################

include("custom_determ.jl")

### sets functionality (and setmap combinator) ###
include("sets/sets.jl")
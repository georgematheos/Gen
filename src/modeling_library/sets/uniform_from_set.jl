struct UniformFromSet <: Distribution{Any} end
const uniform_from_set = UniformFromSet()
random(::UniformFromSet, set) = collect(set)[uniform_discrete(1: length(set))]
logpdf(::UniformFromSet, obj, set) = obj in set ? -log(length(set)) : -Inf
has_argument_grads(::UniformFromSet) = false
accepts_output_grad(::UniformFromSet) = false
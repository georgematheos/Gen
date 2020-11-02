struct IntBernoulli <: Distribution{Bool} end

"""
    int_bernoulli(prob1::Real)

Samples 1 with probability `prob1` and 0 otherwise.
"""
const int_bernoulli = IntBernoulli()

function logpdf(::IntBernoulli, x::Integer, prob::Real)
    x === 1 ? log(prob) : (x === 0 ? log(1. - prob) : -Inf)
end

function logpdf_grad(::IntBernoulli, x::Integer, prob::Real)
    # TODO: what should happen if x is not 0 or 1?  should we error?  should we return 1?
    prob_grad = x === 1 ? 1. / prob : (x === 0 ? -1. / (1-prob) : error())
    (nothing, prob_grad)
end

random(::IntBernoulli, prob::Real) = bernoulli(prob) ? 1 : 0

is_discrete(::IntBernoulli) = true

(::IntBernoulli)(prob) = random(IntBernoulli(), prob)

has_output_grad(::IntBernoulli) = false
has_argument_grads(::IntBernoulli) = (true,)

export int_bernoulli

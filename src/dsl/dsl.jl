export @gen, @param, @trace

import MacroTools

const DSL_STATIC_ANNOTATION = :static
const DSL_ARG_GRAD_ANNOTATION = :grad
const DSL_RET_GRAD_ANNOTATION = :grad
const DSL_TRACK_DIFFS_ANNOTATION = :diffs
const DSL_NO_JULIA_CACHE_ANNOTATION = :nojuliacache
const DSL_MACROS = Set([Symbol("@trace"), Symbol("@param")])

struct Argument
    name::Union{Symbol, Nothing}
    typ::Union{Symbol,Expr}
    annotations::Set{Symbol}
    default::Union{Some{Any}, Nothing}
end

Argument(name, typ) = Argument(name, typ, Set{Symbol}(), nothing)
Argument(name, typ, annotations) = Argument(name, typ, annotations, nothing)

function parse_annotations(annotations_expr)
    annotations = Set{Symbol}()
    if isa(annotations_expr, Symbol)
        push!(annotations, annotations_expr)
    elseif isa(annotations_expr, Expr) && annotations_expr.head == :tuple
        for annotation in annotations_expr.args
            if !isa(annotation, Symbol)
                error("syntax error in annotations_expr at $annotation")
            else
                push!(annotations, annotation)
            end
        end
    else
        error("syntax error in annotations at $annotations")
    end
    annotations
end

"""
    handle_unpacking!(unpack_stmts, expr)

Given an argument name declaration, if needed since the declaration
is an unpacking-statement like `:((a, b))`, adds a statement to `unpack_stmts`
to extract the needed variable names.  Returns the name used to refer
to the overall variable.

Eg. `handle_unpacking!(unpack_stmts, :(((a, b), c)))`
will create a token `sym` and push `((a, b), c) = sym`
into `unpack_stmts`.  `sym` will be returned.
"""
handle_unpacking!(_, argname::Symbol) = argname
handle_unpacking!(_, ::Nothing) = nothing
function handle_unpacking!(unpack_stmts, argname::Expr)
    @assert (argname.head === :tuple) "Unrecognized argname construction in gen fn definition: $argname"
    name = gensym(sprint(print, argname))
    push!(unpack_stmts, :($argname = $name))
    return name
end

"""
    parse_arg!(unpack_stmts::Expr[], arg_expr)

Parses the given argument expression, producing an `Argument`
and possibly pushing statements into `unpack_stmts`
to handle tuple-form variable declarations (like `:((a, b)::Tuple{Int, Float64})`).
"""
function parse_arg!(unpack_stmts, expr)
    if isa(expr, Expr) && expr.head === :call
        # annotated, like (grad,foo)(x::Int)
        annotations_expr = expr.args[1]
        sub_arg = parse_arg!(unpack_stmts, expr.args[2])
        annotations = parse_annotations(annotations_expr)
        Argument(sub_arg.name, sub_arg.typ, annotations, sub_arg.default)
    else
        # non-annotated
        (argname, argtype, _, default) = MacroTools.splitarg(expr)
        name = handle_unpacking!(unpack_stmts, argname)
        Argument(name, argtype, Set{Symbol}(), default === nothing ? nothing : Some(default))
    end
end

include("dynamic.jl")
include("static.jl")

function desugar_tildes(expr)
    trace_ref = GlobalRef(@__MODULE__, Symbol("@trace"))
    line_num = LineNumberNode(1, :none)
    MacroTools.postwalk(expr) do e
        # Replace tilde statements with :gentrace expressions
        if MacroTools.@capture(e, {*} ~ rhs_call)
            Expr(:gentrace, rhs, nothing)
        elseif MacroTools.@capture(e, {addr_} ~ rhs_call)
            Expr(:gentrace, rhs, Some(addr))
        elseif MacroTools.@capture(e, lhs_Symbol ~ rhs_call)
            addr = QuoteNode(lhs)
            Expr(:(=), lhs, Expr(:gentrace, rhs, Some(addr)))
        elseif MacroTools.@capture(e, lhs_ ~ rhs_call)
            error("Syntax error: Invalid left-hand side: $(e)." *
                  "Only a variable or address can appear on the left of a `~`.")
        elseif MacroTools.@capture(e, lhs_ ~ rhs_)
            error("Syntax error: Invalid right-hand side in: $(e)")
        else
            e
        end
    end
end

function extract_quoted_exprs(expr)
    quoted_exprs = []
    expr = MacroTools.prewalk(expr) do e
        if MacroTools.@capture(e, :(quoted_)) && !isa(quoted, Symbol)
            push!(quoted_exprs, e)
            Expr(:placeholder, length(quoted_exprs))
        else
            e
        end
    end
    return expr, quoted_exprs
end

function insert_quoted_exprs(expr, quoted_exprs)
    expr = MacroTools.prewalk(expr) do e
        if MacroTools.@capture(e, p_placeholder)
            idx = p.args[1]
            quoted_exprs[idx]
        else
            e
        end
    end
    return expr
end

function preprocess_body(expr, __module__)
    # Expand all macros relative to the calling module
    expr = macroexpand(__module__, expr)
    # Protect quoted expressions from pre-processing by extracting them
    expr, quoted_exprs = extract_quoted_exprs(expr)
    # Desugar tilde calls to :gentrace expressions
    expr = desugar_tildes(expr)
    # Reinsert quoted expressions after pre-processing
    expr = insert_quoted_exprs(expr, quoted_exprs)
    return expr
end

function parse_gen_function(ast, annotations, __module__)
    ast = MacroTools.longdef(ast)
    if ast.head != :function
        error("syntax error at $ast in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error at $ast in $(ast.args)")
    end
    signature = ast.args[1]
    if signature.head == :(::)
        (call_signature, return_type) = signature.args
    elseif signature.head == :call
        (call_signature, return_type) = (signature, :Any)
    else
        error("syntax error at $(signature)")
    end
    body = preprocess_body(ast.args[2], __module__)
    name = call_signature.args[1]

    # unpack the args, and add any necessary statements to unpack arguments into the body
    unpack_stmts = Expr[]
    args = [parse_arg!(unpack_stmts, arg) for arg in call_signature.args[2:end]]
    body = Expr(:block, unpack_stmts..., body.args...)

    static = DSL_STATIC_ANNOTATION in annotations
    if static
        make_static_gen_function(name, args, body, return_type, annotations)
    else
        make_dynamic_gen_function(name, args, body, return_type, annotations)
    end
end

macro gen(annotations_expr, ast::Expr)
    # parse the annotations
    annotations = parse_annotations(annotations_expr)
    # parse the function definition
    parse_gen_function(ast, annotations, __module__)
end

macro gen(ast::Expr)
    parse_gen_function(ast, Set{Symbol}(), __module__)
end

macro trace(expr::Expr)
    return Expr(:gentrace, esc(expr), nothing)
end

macro trace(expr::Expr, addr)
    return Expr(:gentrace, esc(expr), Some(addr))
end

macro param(expr::Expr)
    if (expr.head != :(::)) error("Syntax in error in @param at $(expr)") end
    name, type = expr.args
    return Expr(:genparam, esc(name), esc(type))
end

macro param(sym::Symbol)
    return Expr(:genparam, esc(sym), esc(:Any))
end

mutable struct Tree
    parent
    num_children
    children
    #_size 
end

function Tree()
    parent = nothing
    num_children = 0
    children = []
    #_size = nothing
    #return Tree(parent, num_children, children, _size)
    return Tree(parent, num_children, children)
end

function String_tree( t::Tree, level=0)
    for child in t.children
        if child isa typeof(t)
            ret = ""
            ret *= String_tree(child, level+1)
        else
            ret *= "\t"*string(level) * string(child) * "\n"
        end
    end
    return ret        
    
end
        
function add_child(t::Tree,c)
    if c isa typeof(t)	
        c.parent = t
    end
    push!(t.children, c)
    t.num_children = t.num_children + 1 
end

#function sizeof(t::Tree)
#    if !isnothing(t._size)
#	println("NOT__EMPTY") 
#        return t._size
#    end
#    size =1
#    for i in 1:(t.num_children)
#        size = size + sizeof(t.children[i])
#    end
#    t._size = size
#    return size
#end


function children_vector(t::Tree)
    r_list = []
    for i in 1:(t.num_children)
        if t.children[i] isa typeof(t)
            push!(r_list, 4)
        else
            push!(r_list, t.children[i])
        end
    end
    return r_list 
end     
                
function to_string(t::Tree)
    r_list = []
    for i in 1:(t.num_children)
        if t.children[i] isa typeof(t)
            push!(r_list, "( " * to_string(t.children[i]) * " )")
        else
            push!(r_list, string(t.children[i]))
        end
    end
    return join(r_list, " ") 
end
                    
                    
function to_list(t::Tree, form_manager)
    r_list = []
    for i in 1:(t.num_children)
        if t.children[i] isa typeof(t)
            push!(r_list, get_symbol_idx(form_manager, "("))
            cl = to_list(t.children[i], form_manager)
            for k in 1:length(cl)
                push!(r_list, cl[k])
            end
            push!(r_list, get_symbol_idx(form_manager, ")"))
        else
            push!(r_list, t.children[i])
        end
    end
    return r_list
end
    

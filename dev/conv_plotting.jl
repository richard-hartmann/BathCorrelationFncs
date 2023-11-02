function cl(l)
    l.attributes.color
end


function cplx_lines!(ax, x, cplx_y; args...)
    l_real = lines!(ax, x, real(cplx_y); args...)
    
    # skip the 'label' argument
    args_imag = filter(
        x -> first(x) != :label && 
             first(x) != :color &&
             first(x) != :linestyle
        , args
        )
    l_cplx = lines!(ax, x, imag(cplx_y), color=cl(l_real), linestyle=:dash; args_imag...)
    return l_real, l_cplx
end
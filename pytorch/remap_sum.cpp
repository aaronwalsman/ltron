#include <torch/extension.h>

#include <iostream>

torch::Tensor remap_sum_forward(
        torch::Tensor source,
        torch::Tensor remap,
        std::optional<long> dim){
    
    torch::Tensor destination = torch::zeros({source.size(0)});
    
    long dim_value = 0
    if(dim){
        dim_value = dim.value()
    }
    
    for(int i = 0; i < source.size(dim_value); i++){
        auto remap_index = remap[i];
        destination[remap_index] += source[i];
    }
    
    return destination;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &remap_sum_forward, "remap_sum forward");
    //m.def("backward", &remap_sum_backward, "remap_sum backward");
}

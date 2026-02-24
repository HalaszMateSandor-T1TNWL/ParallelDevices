    __kernel void hello_kernel(__global int* buffer, int n) {
       if (get_global_id(0) % 2 == 0) {
           buffer[get_global_id(0)] = 11;
       } else {
           buffer[get_global_id(0)] = 22;
       }
    }
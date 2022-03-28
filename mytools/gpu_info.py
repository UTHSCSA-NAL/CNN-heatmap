import torch
from models import ClassificationModel3D_mri, ClassificationModel3D_4
import GPUtil as GPU
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

#free_gpu_cache() 
GPU.showUtilization()

print('All available ordered by id: '),
print(GPU.getAvailable(order='first', limit=999))



device = 0
 

print(f"device:{device}")
print(torch.cuda.current_device())

print(f"how many GPUs: {torch.cuda.device_count()}")

print(torch.cuda.get_device_name(0))


print(torch.cuda.get_device_properties(0).total_memory)



print(torch.cuda.memory_summary(device=0, abbreviated=False))

t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0) 
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved

print(t)
print(r)
print(a)
print(f)
model = ClassificationModel3D_4().cuda(0)
model = ClassificationModel3D_mri().cuda(device)
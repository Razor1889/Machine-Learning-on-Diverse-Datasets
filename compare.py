import torch, sys
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.utils.parametrize as P

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if len(sys.argv) > 1:
    device = sys.argv[1]
if device == 'cpu':
    model = models.resnet152()
    inputs = torch.randn(5, 3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            with P.cached():
                model(inputs)
            
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
else:
    model = models.resnet152().cuda()
    inputs = torch.randn(5, 3, 224, 224).cuda()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            with P.cached():
                model(inputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=20))
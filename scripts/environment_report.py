import json,platform,torch,torchvision
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
 'torchvision':torchvision.__version__,'cuda':torch.version.cuda,
 'nccl':torch.cuda.nccl.version() if torch.cuda.is_available() else None},indent=2))

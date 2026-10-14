import torch

print("Testing torch.compile availability...")
try:
    @torch.compile
    def fn(x):
        return x * 2 + 1
    a = torch.randn(10, device="cuda")
    print("Result:", fn(a))
    print("torch.compile works!")
except Exception as e:
    print("torch.compile failed:", e)

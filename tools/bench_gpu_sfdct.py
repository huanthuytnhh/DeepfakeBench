"""
bench_gpu_sfdct.py — measure REAL training-step throughput (fwd+bwd+opt.step) at 256x256
for both the B4 baseline and the B4+SFDCT method, on whatever GPU is present (here a 3050 Ti),
then extrapolate epoch / run / campaign time to RTX 4090 and RTX 5090.

Faithful to the actual model: EfficientNet-B4 (efficientnet_pytorch) -> [B,1792,h,w]
  baseline : backbone -> GAP -> Linear(1792,2)
  sfdct    : backbone -> GatedCrossAttnFusion(<- ContentDCT) -> GAP -> Linear(1792,2)

Auto-finds the largest batch that fits 4GB, measures steady-state img/s, prints a table.
Run:  python3 tools/bench_gpu_sfdct.py
"""
import os, sys, time, math
import torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "detectors"))
from sfdct_core import ContentDCT, GatedCrossAttnFusion
from efficientnet_pytorch import EfficientNet

torch.backends.cudnn.benchmark = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
RES = 256
IMGS_PER_EPOCH = 114884          # measured from FF++ json (5 labels x ~719 vids x 32 frames)

class B4Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet.from_name("efficientnet-b4")  # weights irrelevant to timing
        self.head = nn.Linear(1792, 2)
    def forward(self, x):
        f = self.net.extract_features(x)               # [B,1792,h,w]
        return self.head(f.mean((2, 3)))

class B4SFDCT(nn.Module):
    def __init__(self, freq_repr="global48"):
        super().__init__()
        self.net = EfficientNet.from_name("efficientnet-b4")
        self.dct = ContentDCT(freq_repr=freq_repr, grid=8)
        self.fusion = GatedCrossAttnFusion(spatial_ch=1792, token_in=self.dct.token_in,
                                           n_tokens=self.dct.n_tokens)
        self.head = nn.Linear(1792, 2)
    def forward(self, x):
        f = self.net.extract_features(x)               # [B,1792,h,w]
        _, tok = self.dct(x)
        f = self.fusion(f, tok)
        return self.head(f.mean((2, 3)))

def time_model(make, label, warmup=8, iters=25):
    for bs in (32, 16, 8, 4):
        try:
            torch.cuda.empty_cache() if dev == "cuda" else None
            model = make().to(dev).train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            x = torch.randn(bs, 3, RES, RES, device=dev)
            y = torch.randint(0, 2, (bs,), device=dev)
            lossf = nn.CrossEntropyLoss()
            def step():
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(x), y); loss.backward(); opt.step()
            for _ in range(warmup): step()
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters): step()
            if dev == "cuda": torch.cuda.synchronize()
            dt = (time.time() - t0) / iters
            ips = bs / dt
            mem = torch.cuda.max_memory_allocated() / 1024**3 if dev == "cuda" else 0
            print(f"  [{label:16}] bs={bs:2d}  {dt*1000:7.1f} ms/iter  {ips:7.1f} img/s  peakVRAM={mem:.2f} GB")
            del model, opt, x, y
            if dev == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
            return ips, bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [{label:16}] bs={bs:2d}  OOM -> smaller batch")
                del model
                if dev == "cuda": torch.cuda.empty_cache()
                continue
            raise
    return None, None

print(f"GPU: {torch.cuda.get_device_name(0) if dev=='cuda' else 'CPU'}   torch {torch.__version__}   res={RES}  fp32")
print(f"Imgs/epoch (FF++ full, frame_num=32) = {IMGS_PER_EPOCH:,}\n")
print("Measuring steady-state training-step throughput (fwd+bwd+Adam.step):")
ips_base, _ = time_model(B4Base, "B4 baseline")
ips_sf,   _ = time_model(lambda: B4SFDCT("global48"),  "SFDCT global48")
ips_bg,   _ = time_model(lambda: B4SFDCT("blockgrid"), "SFDCT blockgrid")

def report(name, ips_3050):
    if not ips_3050:
        print(f"\n{name}: could not measure"); return
    print(f"\n=== Extrapolation for: {name}  (measured {ips_3050:.0f} img/s on 3050 Ti) ===")
    # Empirical CNN-training speedup of desktop cards vs a 3050 Ti Laptop (fp32, batch 32).
    # Range reflects small-batch underutilisation on the laptop chip; central = typical.
    factors = {"RTX 4090": (8.0, 10.0, 12.0), "RTX 5090": (11.0, 13.0, 16.0)}
    for gpu, (lo, mid, hi) in factors.items():
        for tag, f in (("low", lo), ("typ", mid), ("high", hi)):
            ips = ips_3050 * f
            sec_ep = IMGS_PER_EPOCH / ips
            run_h = (sec_ep * 10) / 3600 * 1.25     # 10 epochs + ~25% eval/IO overhead
            if tag == "typ":
                print(f"  {gpu:9} x{f:>4.1f} ({tag}):  {ips:6.0f} img/s  |  {sec_ep/60:4.1f} min/epoch  |  ~{run_h:.2f} h / run(10ep)  |  10 runs ~{run_h*10:.0f} h")
            else:
                print(f"  {gpu:9} x{f:>4.1f} ({tag}):  {ips:6.0f} img/s  |  {sec_ep/60:4.1f} min/epoch  |  ~{run_h:.2f} h / run")

report("B4 baseline", ips_base)
report("SFDCT (global48, the method)", ips_sf)
if ips_base and ips_sf:
    print(f"\nSFDCT fusion overhead vs B4 baseline: {(ips_base/ips_sf - 1)*100:+.1f}% slower (global48)")
    if ips_bg:
        print(f"SFDCT blockgrid overhead vs B4 baseline: {(ips_base/ips_bg - 1)*100:+.1f}% slower")

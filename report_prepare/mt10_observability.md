# MT-10 · Observability + Production-readiness (← novelty #10 AWS ops của Trí)

Học từ Trí: middleware request_id+latency, `/health` psutil, log rotate, model nạp 1 lần. **CODE-FIRST** — gắn vào `backend/` và `DeepfakeBench/serving/infer_server.py`.

## 1. Middleware request_id + latency (FastAPI — backend & SFDCT)
```python
import time, uuid
from starlette.middleware.base import BaseHTTPMiddleware

class AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t = time.perf_counter()
        resp = await call_next(request)
        dt = (time.perf_counter() - t) * 1000
        resp.headers["x-request-id"] = rid
        print(f'{{"rid":"{rid}","method":"{request.method}","path":"{request.url.path}",'
              f'"status":{resp.status_code},"ms":{dt:.1f}}}')   # 1 dòng JSON → dễ forward log
        return resp
app.add_middleware(AccessLog)
```

## 2. `/health` giàu thông tin (psutil + cờ model-loaded)
```python
import psutil, time
START = time.time()
@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time()-START),
            "cpu_pct": psutil.cpu_percent(), "mem_pct": psutil.virtual_memory().percent,
            "model_loaded": MODEL is not None, "model_version": MODEL_VERSION}
```
> Nâng cấp hơn Trí: tách `/livez` (process sống) vs `/readyz` (model_loaded == True) để LB/K8s không gửi traffic khi model chưa nạp.

## 3. Model nạp 1 lần lúc startup (SFDCT infer_server.py)
Đảm bảo `@app.on_event("startup")` nạp `.pth` vào biến global (đã đúng ở infer_server hiện tại — giữ nguyên, đừng nạp trong handler).

## 4. CI chạy THẬT (GitHub Actions) — KHÔNG `|| true`
```yaml
name: api-test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -r requirements.txt   # KHÔNG '|| true'
      - run: pip install pytest httpx
      - run: pytest report_prepare/mt10_api_test_template.py -v
```

## 5. Train-serve consistency (MT-6 liên quan)
SFDCT `:8501` phải dùng **đúng** MTCNN-crop + block-DCT params + mean/std như lúc train. Lưu các tham số này kèm checkpoint (hoặc config versioned) và nạp lại y hệt → tránh drift làm tụt AUC demo + giữ FPR≤5% tái lập.

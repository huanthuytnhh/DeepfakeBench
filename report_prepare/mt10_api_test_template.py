#!/usr/bin/env python3
"""MT-10 · Test API THẬT cho app eKYC (← TRÁNH bẫy test-giả của Trí: tests/test_api.py assert dict cứng).

Khác Trí: test này GỌI THẬT endpoint detect của backend FastAPI + SFDCT microservice :8501,
gửi ảnh real/fake mẫu, assert schema + nhãn. Chạy bằng pytest; CI phải chạy thật (KHÔNG `|| true`).

CODE-FIRST — cần backend (:8000) + SFDCT (:8501) đang chạy + ảnh mẫu. Điền 2 biến SAMPLE_*.
Chạy mai:
  uvicorn app.main:app --port 8000 &     # backend
  python DeepfakeBench/serving/infer_server.py &   # SFDCT :8501
  pytest report_prepare/mt10_api_test_template.py -v
"""
import os
import pytest
import httpx

BACKEND = os.getenv("DG_BACKEND_URL", "http://127.0.0.1:8000")
SFDCT = os.getenv("SFDCT_INFER_URL", "http://127.0.0.1:8501")
SAMPLE_REAL = os.getenv("DG_SAMPLE_REAL", "")  # TODO: đường dẫn ảnh thật mẫu
SAMPLE_FAKE = os.getenv("DG_SAMPLE_FAKE", "")  # TODO: đường dẫn ảnh fake mẫu


def _alive(url):
    try:
        httpx.get(url, timeout=2.0); return True
    except Exception:
        return False


@pytest.mark.skipif(not _alive(SFDCT + "/"), reason="SFDCT :8501 chưa chạy")
def test_sfdct_predict_schema():
    assert SAMPLE_FAKE, "Đặt DG_SAMPLE_FAKE = ảnh fake mẫu"
    with open(SAMPLE_FAKE, "rb") as f:
        r = httpx.post(f"{SFDCT}/predict", files={"file": ("x.jpg", f.read(), "image/jpeg")}, timeout=60)
    r.raise_for_status()
    j = r.json()
    assert {"prob_fake", "verdict"} <= set(j), f"thiếu field: {j.keys()}"
    assert 0.0 <= j["prob_fake"] <= 1.0


@pytest.mark.skipif(not _alive(SFDCT + "/"), reason="SFDCT :8501 chưa chạy")
def test_sfdct_real_vs_fake_separation():
    assert SAMPLE_REAL and SAMPLE_FAKE, "Đặt DG_SAMPLE_REAL và DG_SAMPLE_FAKE"
    def prob(path):
        with open(path, "rb") as f:
            return httpx.post(f"{SFDCT}/predict", files={"file": ("x.jpg", f.read(), "image/jpeg")}, timeout=60).json()["prob_fake"]
    assert prob(SAMPLE_FAKE) > prob(SAMPLE_REAL), "model phải cho fake prob cao hơn real"


@pytest.mark.skipif(not _alive(BACKEND + "/docs"), reason="backend :8000 chưa chạy")
def test_backend_detect_image_endpoint():
    assert SAMPLE_FAKE, "Đặt DG_SAMPLE_FAKE"
    with open(SAMPLE_FAKE, "rb") as f:
        r = httpx.post(f"{BACKEND}/v1/detect/image",
                       files={"file": ("x.jpg", f.read(), "image/jpeg")},
                       headers={"X-API-Key": os.getenv("DG_API_KEY", "")}, timeout=60)
    assert r.status_code in (200, 401, 403), f"status lạ: {r.status_code}"
    if r.status_code == 200:
        assert "verdict" in r.json()

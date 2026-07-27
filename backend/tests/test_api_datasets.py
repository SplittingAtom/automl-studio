"""Contract tests for the datasets API."""

import io


def _upload(client, content: bytes, filename="data.csv"):
    return client.post(
        "/api/datasets", files={"file": (filename, io.BytesIO(content), "text/csv")}
    )


SIMPLE_CSV = b"age,sex,outcome\n" + b"".join(
    f"{20 + i % 50}.5,{'male' if i % 2 else 'female'},{i % 2}\n".encode()
    for i in range(100)
)


class TestEnvelope:
    def test_success_shape(self, client):
        body = client.get("/api/datasets").json()
        assert body["success"] is True
        assert body["error"] is None
        assert "data" in body and "meta" in body

    def test_error_shape(self, client):
        body = client.get("/api/datasets/ds_nope").json()
        assert body["success"] is False
        assert body["data"] is None
        assert body["error"]["code"] == "DATASET_NOT_FOUND"
        assert body["error"]["message"]


class TestListAndSamples:
    def test_samples_are_seeded(self, client):
        data = client.get("/api/datasets").json()["data"]
        names = [d["name"] for d in data]
        assert any("Titanic" in n for n in names)
        assert all(d["source"] == "sample" for d in data)


class TestUpload:
    def test_valid_csv_returns_profile(self, client):
        resp = _upload(client, SIMPLE_CSV)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["row_count"] == 100
        assert data["column_count"] == 3
        kinds = {c["name"]: c["kind"] for c in data["columns"]}
        assert kinds["age"] == "numeric"
        assert kinds["sex"] == "categorical"

    def test_uploaded_dataset_is_retrievable(self, client):
        ds_id = _upload(client, SIMPLE_CSV).json()["data"]["id"]
        body = client.get(f"/api/datasets/{ds_id}").json()
        assert body["success"] is True
        assert body["data"]["id"] == ds_id

    def test_non_csv_rejected(self, client):
        resp = _upload(client, b"hello", filename="notes.txt")
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_FILE_TYPE"

    def test_empty_csv_rejected(self, client):
        resp = _upload(client, b"a,b\n")
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "EMPTY_DATASET"

    def test_unparseable_csv_rejected(self, client):
        resp = _upload(client, b"\x00\x01\x02binary garbage")
        assert resp.status_code == 422

    def test_semicolon_delimiter_sniffed(self, client):
        csv = b"a;b\n" + b"".join(f"{i}.5;{i % 3}\n".encode() for i in range(60))
        data = _upload(client, csv).json()["data"]
        assert data["column_count"] == 2


class TestPreview:
    def test_preview_rows(self, client):
        ds_id = _upload(client, SIMPLE_CSV).json()["data"]["id"]
        data = client.get(f"/api/datasets/{ds_id}/preview?rows=5").json()["data"]
        assert data["columns"] == ["age", "sex", "outcome"]
        assert len(data["rows"]) == 5

from civetwatch import jobs
from civetwatch.config import camera_from_filename, load_config
from civetwatch.dashboard import create_app


def test_camera_from_filename():
    assert camera_from_filename("A01_20260405000000.mp4") == "cam1"
    assert camera_from_filename("camera 4-2026-04-05_12-00-00.mov") == "cam4"
    assert camera_from_filename("recording_20260405000000.mp4") is None


def test_flat_folder_metadata_is_grouped_by_camera(tmp_path):
    for name in ("A01_20260405000000.mp4", "A01_20260405010000.mp4",
                 "A02_20260405000000.mp4", "A04_20260405000000.mp4", "other.mp4"):
        (tmp_path / name).touch()

    grouped = jobs.group_videos_by_camera(str(tmp_path))
    assert {camera: len(files) for camera, files in grouped.items()} == {"cam1": 2, "cam2": 1, "cam4": 1}

    root = tmp_path / "runs"
    root.mkdir()
    app = create_app(str(root), load_config())
    client = app.test_client()
    page = client.get("/new")
    response = client.post("/api/footage-metadata", json={"source": str(tmp_path)})

    assert b'id="autofill"' in page.data
    assert response.status_code == 200
    assert response.get_json() == {
        "date": "2026-04-05",
        "cameras": {"cam1": {"files": 2}, "cam2": {"files": 1}, "cam4": {"files": 1}},
    }

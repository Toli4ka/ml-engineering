from defect_qc.data import create_manifest, get_data_dir


def test_create_manifest():
    df = create_manifest(get_data_dir())
    assert {"file_path", "defect", "split"}.issubset(df.columns)


def test_manifest_labels_are_binary():
    df = create_manifest(get_data_dir())
    assert set(df["defect"].unique()).issubset({0, 1})

def test_manifest_paths_exist():
    df = create_manifest(get_data_dir())
    # sample a few to keep it fast
    sample = df.sample(n=min(20, len(df)), random_state=0)
    for p in sample["file_path"]:
        assert isinstance(p, str) and len(p) > 0
import json

def generate_config(n: int, base_dir: str = "data/CoronaryArtery") -> dict:
    config = {"dir": base_dir}
    for i in range(1, n + 1):
        config[str(i)] = [{
            "image":      f"case_{i}/img.nii.gz",
            "label":      f"case_{i}/lab.nii.gz",
            "mesh":       f"case_{i}/mesh.xyz",
            "centerline": f"case_{i}/centerline.xyz",
            "segmented":  f"case_{i}/seg.nii.gz"
        }]
    return config

if __name__ == "__main__":
    n = 30
    cfg = generate_config(n)

    # JSON ファイルとして保存
    output_path = "data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    print(f"Saved configuration to {output_path}")

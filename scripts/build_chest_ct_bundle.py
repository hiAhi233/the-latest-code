"""Assemble a private, portable CT training bundle from the verified local inputs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
from zipfile import ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
TOP = "MedADM_RRG_ChestCT"
SOURCE = "subset_100_validation"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def case_table(chest: Path) -> tuple[str, list[dict]]:
    source = chest / SOURCE
    entries = json.loads((source / "manifest.json").read_text(encoding="utf-8"))["images"]
    text = {}
    with (source / "validation_region_report.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["Anatomy"].strip():
                key = row["Volumename"]
                if key in text:
                    raise ValueError(f"Duplicate study-level reference: {key}")
                text[key] = row["Sentence"].strip()
    names = sorted(row["name"] for row in entries)
    if len(names) != 100 or set(names) != set(text) or not all(text.values()):
        raise ValueError("Expected 100 unique CTs with one nonempty study-level reference per CT")
    patients = {name: re.fullmatch(r"(valid_\d+)_[a-z]+_\d+\.nii\.gz", name)[1] for name in names}
    if len(set(patients.values())) != 100:
        raise ValueError("Expected 100 unique validation patients")
    random.Random(0).shuffle(names)
    split = {name: label for start, stop, label in [(0, 70, "train"), (70, 80, "val"), (80, 100, "test")]
             for name in names[start:stop]}
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=("study_id", "patient_id", "split", "image",
                                               "findings", "impression", "source_split"))
    writer.writeheader()
    for row in sorted(entries, key=lambda item: item["name"]):
        name = row["name"]
        writer.writerow(dict(study_id=name.removesuffix(".nii.gz"), patient_id=patients[name],
                             split=split[name], image=row["repo_path"], findings=text[name],
                             impression="", source_split="official_validation"))
    return target.getvalue(), entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chest", type=Path, required=True)
    parser.add_argument("--msd", type=Path, required=True,
                        help="Folder containing Task06_Lung with split.json and MSD_dataset.json beside it")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--unimed", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True, help="Existing Hugging Face hub cache")
    parser.add_argument("--resnet", type=Path, required=True)
    parser.add_argument("--scorers", type=Path, required=True, help="EKAGen's pycocoevalcap folder")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    temporary = args.out.with_suffix(".zip.part")
    if args.out.exists() or temporary.exists():
        raise FileExistsError(f"Archive or partial archive already exists: {args.out}")

    table, entries = case_table(args.chest)
    source = args.chest / SOURCE
    files = {}

    def add(path: Path, dest: str):
        if not path.is_file():
            raise FileNotFoundError(path)
        if dest in files:
            raise ValueError(f"Duplicate destination: {dest}")
        files[dest] = path

    def tree(root: Path, dest: str, suffixes: set[str]):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if (path.is_file() and path.suffix in suffixes
                    and not any(part.startswith(".") or part == "__pycache__" for part in rel.parts)):
                add(path, f"{dest}/{rel.as_posix()}")

    for row in entries:
        path = args.chest / row["repo_path"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"CT does not match the downloaded manifest: {path}")
        add(path, f"data/chest_ct/{row['repo_path']}")
    for name in ("manifest.json", "README.md", "validation_region_report.csv", "validation_case_disorders.csv"):
        add(source / name, f"data/chest_ct/source/{name}")
    masks = source / "valid_region_mask"
    for row in entries:
        folder = "seg_" + row["name"].removesuffix(".nii.gz")
        mask_files = sorted((masks / folder).glob("*.nii.gz"))
        if len(mask_files) != 10:
            raise ValueError(f"Expected 10 anatomy masks: {folder}")
        for mask in mask_files:
            add(mask, f"data/chest_ct/source/valid_region_mask/{folder}/{mask.name}")

    msd = args.msd / "Task06_Lung"
    split = json.loads((msd / "split.json").read_text(encoding="utf-8"))
    if set(split) != {"support", "val", "test"} or any(len(rows) != 5 for rows in split.values()):
        raise ValueError("Expected 5 support, 5 calibration, 5 test cases in MSD")
    add(msd / "split.json", "data/msd_lung/split.json")
    add(args.msd / "MSD_dataset.json", "data/msd_lung/source_dataset.json")
    for rows in split.values():
        for row in rows:
            for rel in row.values():
                rel = rel.removeprefix("./")
                add(msd / rel, f"data/msd_lung/{rel}")

    tree(ROOT / "medrrg", "medrrg", {".py"})
    tree(ROOT / "external_tools", "external_tools", {".py"})
    tree(ROOT / "configs/chest_ct", "configs/chest_ct", {".yaml", ".json"})
    add(ROOT / "configs/adm_model.yaml", "configs/adm_model.yaml")
    for name in ("step01_build_manifest.py", "step03_train.py", "step04_generate.py",
                 "evaluate_standard_reports.py", "run_chest_ct.py"):
        add(ROOT / "scripts" / name, f"scripts/{name}")
    for name in ("requirements-chest-ct.txt", "requirements-cloud.txt", "pyproject.toml"):
        add(ROOT / name, name)
    add(ROOT / "docs/ChestCT_K5_训练方案.md", "START_HERE.md")

    for name in ("__init__.py", "config.py", "losses.py", "paths.py", "run_localization.py"):
        add(args.detector / name, f"vendor/paclip/{name}")
    for folder in ("data", "evaluation", "localization_v3", "models", "prompts"):
        tree(args.detector / folder, f"vendor/paclip/{folder}", {".py"})
    tree(args.unimed / "UniMed-CLIP/src/open_clip", "vendor/unimed/UniMed-CLIP/src/open_clip",
         {".py", ".json", ".gz", ".txt"})
    for name in ("LICENSE", "openclip_LICENSE"):
        add(args.unimed / "UniMed-CLIP" / name, f"vendor/unimed/UniMed-CLIP/{name}")
    tree(args.unimed / "weights/biomedbert", "vendor/unimed/weights/biomedbert", {".json", ".txt", ".bin", ".safetensors"})
    add(args.unimed / "weights/unimed-clip-vit-b16.pt", "vendor/unimed/weights/unimed-clip-vit-b16.pt")
    for model in ("models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                  "models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract"):
        repo = args.hf_cache / model
        for path in repo.rglob("*"):
            if path.is_file():
                add(path, f"vendor/hf/hub/{model}/{path.relative_to(repo).as_posix()}")
    add(args.resnet, f"vendor/torch/hub/checkpoints/{args.resnet.name}")
    for folder in ("bleu", "rouge", "meteor"):
        tree(args.scorers / folder, f"vendor/pycocoevalcap/{folder}", {".py", ".jar", ".gz"})
    for name in ("__init__.py", "license.txt"):
        add(args.scorers / name, f"vendor/pycocoevalcap/{name}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(path.stat().st_size for path in files.values())
    print(f"Packaging {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    with ZipFile(temporary, "x", compression=ZIP_STORED) as archive:
        archive.writestr(f"{TOP}/data/chest_ct/cases.csv", table)
        for dest, path in sorted(files.items()):
            archive.write(path, f"{TOP}/{dest}")
    with ZipFile(temporary) as archive:
        failed = archive.testzip()
        if failed is not None:
            raise RuntimeError(f"Archive CRC failure: {failed}")
    checksum = digest(temporary)
    temporary.replace(args.out)
    args.out.with_suffix(".zip.sha256").write_text(f"{checksum}  {args.out.name}\n", encoding="utf-8")
    print(f"Verified archive: {args.out}", flush=True)


if __name__ == "__main__":
    main()

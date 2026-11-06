"""Bundle existing measured cases and untouched upstream functions for Online.

python methods/qab_dequip/matlab/package_online.py --results outputs/python_denoising --repos external
No code from the input repositories is executed by this packager.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--repos", default=str(Path(__file__).resolve().parents[3] / "external"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    results, repos = Path(args.results).resolve(), Path(args.repos).resolve()
    destination = Path(args.out).resolve() if args.out else results / "matlab_online_bundle.zip"
    mapping = {
        "dequip64": [results / "report_dequip" / "case_image_64.mat"],
        "dequip_all": [results / "report_dequip24" / "case_image_24.mat"] +
                      [results / "report_dequip" / f"case_image_{n}.mat" for n in (64, 128, 256)],
        "qab_ms3": [results / "report_qab_ms3" / "case_image_64.mat"],
        "qab_ms20": [results / "report_qab_ms20" / "case_image_64.mat"],
        "signal": [results / "qab_signal" / "case_signal_512.mat"],
    }
    for files in mapping.values():
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
    files = [(project / "matlab/run_online.m", "run_online.m"),
             (project / "matlab/benchmark_upstream.m", "matlab/benchmark_upstream.m"),
             (project / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")]
    for name in ("QAB-denoising", "De-QuIP-Denoising"):
        repo = repos / name
        for source in sorted(repo.rglob("*.m")):
            files.append((source, "repos/" + name + "/" + source.relative_to(repo).as_posix()))
        for filename in ("README.md", "LICENSE"):
            if (repo / filename).exists():
                files.append((repo / filename, f"repos/{name}/{filename}"))
    for group, paths in mapping.items():
        files.extend((path, f"cases/{group}/{path.name}") for path in paths)
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}
    readme = """# So MATLAB Online với Python trên cùng dữ liệu

1. Upload ZIP này lên MATLAB Online, giải nén, mở thư mục chứa run_online.m.
2. Chạy `run_online` để đo De-QuIP 64x64 trước.
3. Chạy `run_online('dequip_all')` để đo 24/64/128/256.
4. QAB: `run_online('qab_ms3')` hoặc `run_online('qab_ms20')`.
5. Tải các file cases/.../case_*_matlab.mat về cạnh case_*.mat tương ứng
   trong outputs/python_denoising/report_dequip, report_dequip24,
   report_qab_ms3, report_qab_ms20 hoặc qab_signal trên máy local.
6. Chạy `python -m quantum_denoising compare-matlab <thư mục kết quả>`.

Các hàm MATLAB gốc được giữ nguyên. Mảng clean/noisy/config và times Python
đã được xuất từ lần chạy thật. Không tạo lại noise bên MATLAB.
Tính toàn bộ denoiser, warmup 1 lần + 3 lần đo, 1 luồng tính toán.
Các hàm không gọi mạng, không dùng toolbox Image Processing trong harness.
Harness từ chối ghi đè file đo cũ. QAB nguyên bản có vòng tái dựng lớn;
bắt đầu với De-QuIP64 trước khi chạy cả bộ.

Ngay cả khi dùng chung dữ liệu, MATLAB Online và CPU local vẫn khác phần cứng;
tỷ lệ đo mô tả hai môi trường cụ thể, không cô lập riêng ảnh hưởng ngôn ngữ.
"""
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("README_VI.md", readme)
        archive.writestr("SHA256.json", json.dumps(manifest, indent=2))
    print(destination)


if __name__ == "__main__":
    main()

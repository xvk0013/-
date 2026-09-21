"""Read-only dependency/backend check; no installation and no dataset access."""
import importlib
import json


def main():
    report = {}
    for name in ("numpy", "scipy", "soundfile", "torch", "torchaudio", "mamba_ssm", "selective_scan_cuda"):
        try:
            module = importlib.import_module(name)
            report[name] = {"version": str(getattr(module, "__version__", "available"))}
        except Exception as exc:
            report[name] = {"error": str(exc)}
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name()
            report["tensor_check"] = torch.ones(4, device="cuda").sum().item()
    except Exception as exc:
        report["cuda_error"] = str(exc)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

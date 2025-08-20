import warnings
from packaging import version
import torch
import subprocess


def validate_environment(strict: bool = False) -> bool:
	"""Validate core runtime versions for Genie Phase 1."""
	success = True

	# PyTorch version
	torch_ver = version.parse(torch.__version__.split("+")[0])
	if not (version.parse("2.1.0") <= torch_ver < version.parse("2.2.0")):
		warnings.warn(f"Genie targets PyTorch 2.1.x; found {torch.__version__}")
		success = False

	# CUDA availability is optional in Phase 1

	# Optional: DPDK presence (warn if missing)
	try:
		res = subprocess.run(["pkg-config", "--modversion", "libdpdk"], capture_output=True, text=True)
		if res.returncode != 0:
			warnings.warn("DPDK not found; zero-copy features disabled")
	except Exception:
		warnings.warn("Could not verify DPDK version")

	return success or not strict



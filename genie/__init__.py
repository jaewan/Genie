from .version_validator import validate_environment  # noqa: F401

# Initialize core subsystems on package import
# - Validate environment (non-strict)
# - Register backend + default device
# - Register torch.library implementations

# Perform a soft environment validation
try:
	validate_environment(strict=False)
except Exception:
	# Do not block import on validation issues
	pass

# Import side-effect modules to ensure registration
from .core import device as _genie_device  # noqa: F401,E402
from .core import library as _genie_library  # noqa: F401,E402
from .core import factory_intercept as _genie_factory  # noqa: F401,E402
from .core import enhanced_dispatcher as _genie_enhanced  # noqa: F401,E402

# Ensure a default device exists (triggers backend registration)
try:
	_genie_device.get_device(0)
except Exception:
	# Keep import resilient even if backend not fully available
	pass

# Install factory function interceptors (randn/zeros/ones/empty/full)
try:
	_genie_factory.install_all()
except Exception:
	pass


# Public convenience API
def set_lazy_mode(enabled: bool) -> None:
	"""Enable or disable lazy execution globally.

	This toggles both the torch.library-based path and the enhanced dispatcher
	for consistent behavior across code paths.
	"""
	try:
		_genie_library.set_lazy_mode(enabled)
	except Exception:
		pass
	try:
		from .core.dispatcher import set_lazy_mode as _set_enhanced  # noqa: WPS433
		_set_enhanced(enabled)
	except Exception:
		pass


def is_remote_accelerator_available() -> bool:
	"""Return whether the remote_accelerator backend is available."""
	try:
		return _genie_device.is_available()
	except Exception:
		return False




def get_capture_stats() -> dict:
	"""Return combined capture statistics from the unified dispatcher and library.

	Includes counts of registered operations, fallback-captured operations, and
	other helpful signals for understanding coverage and behavior.
	"""
	stats = {}
	try:
		from .core.enhanced_dispatcher import get_enhanced_stats  # noqa: WPS433
		disp = get_enhanced_stats()
		stats["dispatcher"] = disp
	except Exception:
		stats["dispatcher"] = {}
	try:
		from .core.library import get_library_stats  # noqa: WPS433
		lib = get_library_stats()
		stats["library"] = lib
	except Exception:
		stats["library"] = {}
	# Convenience rollups
	try:
		d_registered = int(stats["dispatcher"].get("successful_registrations", 0))
		d_failed = int(stats["dispatcher"].get("failed_registrations", 0))
		d_fallback_ops = int(stats["dispatcher"].get("fallback_ops", 0))
		d_fallback_caps = int(stats["dispatcher"].get("fallback_capture_count", 0))
		stats["summary"] = {
			"dispatcher_registered_ops": d_registered,
			"dispatcher_failed_ops": d_failed,
			"dispatcher_fallback_ops": d_fallback_ops,
			"dispatcher_fallback_capture_count": d_fallback_caps,
		}
	except Exception:
		stats["summary"] = {}
	return stats


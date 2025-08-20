import genie
from genie.core.library import get_library_stats


def test_library_registration_stats():
	# Ensure import side-effects registered ops
	stats = get_library_stats()
	assert stats["registered_ops"] >= 10
	assert isinstance(stats["operation_count"], int)
	assert stats["lazy_mode"] is True




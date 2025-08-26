from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
	name="genie",
	version="0.1.0",
	packages=find_packages(),
	ext_modules=[
		CppExtension(
			"genie._C",
			["genie/csrc/device.cpp"],
			extra_compile_args=["-std=c++17"],
		)
		,
		CppExtension(
			"genie._runtime",
			["genie/csrc/runtime.cpp"],
			extra_compile_args=["-std=c++17", "-Wno-unknown-pragmas"],
		)
	],
	cmdclass={"build_ext": BuildExtension},
	install_requires=[
		"torch>=2.2.0,<2.6.0",
		"numpy>=1.24.0",
		"networkx>=3.0",
	],
)



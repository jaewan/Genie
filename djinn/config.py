"""
Centralized configuration system for Djinn framework.

Replaces hardcoded values throughout the codebase with a unified configuration system.
Supports loading from YAML files, environment variables, and defaults.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Optional, Dict, Any
import yaml
from .core.types import TransportType, ExecutionPhase


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class NetworkConfig:
    """Network-related configuration."""
    # Remote server configuration
    remote_server_address: Optional[str] = None  # e.g., 'localhost:5556'
    
    # Ports
    control_port: int = 5555
    data_port: int = 5556
    metrics_port: int = 9095
    result_port_offset: int = 2000  # Offset from data_port for result ports

    # UDP (legacy)
    default_udp_port: int = 5556

    # Transport preferences
    prefer_dpdk: bool = True
    require_dpdk: bool = False
    tcp_fallback: bool = True

    # Connection settings
    max_connections_per_target: int = 5
    connection_timeout: float = 30.0
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 90.0

    # Buffer sizes
    chunk_size: int = 1024 * 1024  # 1MB chunks
    max_message_size: int = 1024 * 1024  # 1MB max message
    max_payload_size: int = 1394  # Ethernet MTU - headers

    # Retry logic
    max_retry_attempts: int = 3
    retry_backoff: float = 0.1

    # Batching
    batch_size_threshold: int = 5
    batch_timeout: float = 0.01

    @classmethod
    def from_env(cls) -> 'NetworkConfig':
        """Load network config from environment variables."""
        return cls(
            remote_server_address=os.getenv('GENIE_SERVER_ADDRESS', None),
            control_port=int(os.getenv('GENIE_CONTROL_PORT', 5555)),
            data_port=int(os.getenv('GENIE_DATA_PORT', 5556)),
            metrics_port=int(os.getenv('GENIE_METRICS_PORT', 9095)),
            default_udp_port=int(os.getenv('GENIE_UDP_PORT', 5556)),
            prefer_dpdk=os.getenv('GENIE_PREFER_DPDK', 'true').lower() == 'true',
            require_dpdk=os.getenv('GENIE_REQUIRE_DPDK', 'false').lower() == 'true',
            tcp_fallback=os.getenv('GENIE_TCP_FALLBACK', 'true').lower() == 'true',
            max_connections_per_target=int(os.getenv('GENIE_MAX_CONNECTIONS', 5)),
            connection_timeout=float(os.getenv('GENIE_CONNECTION_TIMEOUT', 30.0)),
            heartbeat_interval=float(os.getenv('GENIE_HEARTBEAT_INTERVAL', 30.0)),
            heartbeat_timeout=float(os.getenv('GENIE_HEARTBEAT_TIMEOUT', 90.0)),
            chunk_size=int(os.getenv('GENIE_CHUNK_SIZE', 1024 * 1024)),
            max_message_size=int(os.getenv('GENIE_MAX_MESSAGE_SIZE', 1024 * 1024)),
            max_retry_attempts=int(os.getenv('GENIE_MAX_RETRIES', 3)),
            retry_backoff=float(os.getenv('GENIE_RETRY_BACKOFF', 0.1)),
            batch_size_threshold=int(os.getenv('GENIE_BATCH_THRESHOLD', 5)),
            batch_timeout=float(os.getenv('GENIE_BATCH_TIMEOUT', 0.01))
        )


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    # Timeouts
    operation_timeout: float = 300.0
    transfer_timeout: float = 30.0
    negotiation_timeout: float = 10.0

    # Caching
    pattern_cache_size: int = 1000
    shape_cache_size: int = 10000
    metadata_cache_size: int = 5000
    cache_ttl_seconds: int = 3600  # 1 hour

    # Memory management
    max_memory_mb: int = 500
    memory_pressure_threshold: float = 0.8  # 80% memory usage

    # GPU settings
    gpu_memory_fraction: float = 0.9  # Use 90% of GPU memory
    gpu_id: int = 0  # Default GPU

    # Profiling
    enable_profiling: bool = False
    profile_sample_rate: float = 0.1  # Sample 10% of operations

    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Load performance config from environment variables."""
        return cls(
            operation_timeout=float(os.getenv('GENIE_OPERATION_TIMEOUT', 300.0)),
            transfer_timeout=float(os.getenv('GENIE_TRANSFER_TIMEOUT', 30.0)),
            negotiation_timeout=float(os.getenv('GENIE_NEGOTIATION_TIMEOUT', 10.0)),
            pattern_cache_size=int(os.getenv('GENIE_PATTERN_CACHE_SIZE', 1000)),
            shape_cache_size=int(os.getenv('GENIE_SHAPE_CACHE_SIZE', 10000)),
            metadata_cache_size=int(os.getenv('GENIE_METADATA_CACHE_SIZE', 5000)),
            cache_ttl_seconds=int(os.getenv('GENIE_CACHE_TTL', 3600)),
            max_memory_mb=int(os.getenv('GENIE_MAX_MEMORY_MB', 500)),
            memory_pressure_threshold=float(os.getenv('GENIE_MEMORY_THRESHOLD', 0.8)),
            gpu_memory_fraction=float(os.getenv('GENIE_GPU_MEMORY_FRACTION', 0.9)),
            gpu_id=int(os.getenv('GENIE_GPU_ID', 0)),
            enable_profiling=os.getenv('GENIE_ENABLE_PROFILING', 'false').lower() == 'true',
            profile_sample_rate=float(os.getenv('GENIE_PROFILE_SAMPLE_RATE', 0.1))
        )


@dataclass
class RuntimeConfig:
    """Runtime initialization configuration."""
    # Thread pool settings
    thread_pool_size: int = 4
    auto_connect: bool = True
    auto_init: bool = True
    
    # Profiling
    enable_profiling: bool = False
    profile_sample_rate: float = 0.1
    
    @classmethod
    def from_env(cls) -> 'RuntimeConfig':
        """Load runtime config from environment variables."""
        return cls(
            thread_pool_size=int(os.getenv('GENIE_THREAD_POOL_SIZE', 4)),
            auto_connect=os.getenv('GENIE_AUTO_CONNECT', 'true').lower() == 'true',
            auto_init=os.getenv('GENIE_AUTO_INIT', 'true').lower() == 'true',
            enable_profiling=os.getenv('GENIE_ENABLE_PROFILING', 'false').lower() == 'true',
            profile_sample_rate=float(os.getenv('GENIE_PROFILE_SAMPLE_RATE', 0.1))
        )


@dataclass
class ServerConfig:
    """Server-specific configuration."""
    node_id: str = "genie-server-0"
    max_concurrent_transfers: int = 32
    max_queue_size: int = 1000

    # Resource management
    enable_resource_discovery: bool = True
    resource_check_interval: float = 60.0  # seconds

    # Load balancing
    enable_load_balancing: bool = True
    load_balance_threshold: float = 0.7  # 70% utilization

    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """Load server config from environment variables."""
        return cls(
            node_id=os.getenv('GENIE_NODE_ID', 'genie-server-0'),
            max_concurrent_transfers=int(os.getenv('GENIE_MAX_TRANSFERS', 32)),
            max_queue_size=int(os.getenv('GENIE_MAX_QUEUE_SIZE', 1000)),
            enable_resource_discovery=os.getenv('GENIE_RESOURCE_DISCOVERY', 'true').lower() == 'true',
            resource_check_interval=float(os.getenv('GENIE_RESOURCE_CHECK_INTERVAL', 60.0)),
            enable_load_balancing=os.getenv('GENIE_LOAD_BALANCING', 'true').lower() == 'true',
            load_balance_threshold=float(os.getenv('GENIE_LOAD_THRESHOLD', 0.7))
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5

    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Load logging config from environment variables."""
        return cls(
            level=LogLevel(os.getenv('GENIE_LOG_LEVEL', 'info')),
            format=os.getenv('GENIE_LOG_FORMAT', "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=os.getenv('GENIE_LOG_FILE'),
            max_file_size_mb=int(os.getenv('GENIE_LOG_MAX_SIZE_MB', 100)),
            backup_count=int(os.getenv('GENIE_LOG_BACKUP_COUNT', 5))
        )


@dataclass
class OptimizationConfig:
    """Optimization features configuration (Tensor Registry & SRG Fusion)."""
    # Tensor Registry
    enable_tensor_registry: bool = True              # Enable tensor caching across requests
    tensor_registry_max_models: int = 5             # Max models to cache
    tensor_registry_max_bytes_per_model: Optional[int] = None  # Max bytes per model
    tensor_registry_max_total_bytes: Optional[int] = None      # Max total bytes
    
    # SRG-Driven Fusion
    enable_srg_fusion: bool = True                  # Enable pattern-based fusion
    enable_fusion_torchscript: bool = False         # Enable Tier 2 TorchScript (experimental)
    enable_fusion_compilation: bool = False         # Enable Tier 3 TensorRT (experimental)
    
    # Profiling
    profile_registry_overhead: bool = True          # Track registry overhead
    profile_fusion_overhead: bool = True            # Track fusion overhead
    
    @classmethod
    def from_env(cls) -> 'OptimizationConfig':
        """Load optimization config from environment variables."""
        return cls(
            enable_tensor_registry=os.getenv('GENIE_TENSOR_REGISTRY', 'true').lower() == 'true',
            tensor_registry_max_models=int(os.getenv('GENIE_REGISTRY_MAX_MODELS', 5)),
            tensor_registry_max_bytes_per_model=(
                int(os.getenv('GENIE_REGISTRY_MAX_BYTES_PER_MODEL'))
                if os.getenv('GENIE_REGISTRY_MAX_BYTES_PER_MODEL') else None
            ),
            tensor_registry_max_total_bytes=(
                int(os.getenv('GENIE_REGISTRY_MAX_TOTAL_BYTES'))
                if os.getenv('GENIE_REGISTRY_MAX_TOTAL_BYTES') else None
            ),
            enable_srg_fusion=os.getenv('GENIE_SRG_FUSION', 'true').lower() == 'true',
            enable_fusion_torchscript=os.getenv('GENIE_FUSION_TORCHSCRIPT', 'false').lower() == 'true',
            enable_fusion_compilation=os.getenv('GENIE_FUSION_COMPILATION', 'false').lower() == 'true',
            profile_registry_overhead=os.getenv('GENIE_PROFILE_REGISTRY', 'true').lower() == 'true',
            profile_fusion_overhead=os.getenv('GENIE_PROFILE_FUSION', 'true').lower() == 'true',
        )


@dataclass
class DjinnConfig:
    """Main configuration class for Djinn framework."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Global settings
    debug_mode: bool = False
    config_file: Optional[str] = None

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'DjinnConfig':
        """
        Load configuration from YAML file and/or environment variables.

        Args:
            path: Path to YAML config file (optional)

        Returns:
            DjinnConfig instance with loaded settings
        """
        config = cls()

        # Load from YAML file if provided
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    yaml_data = yaml.safe_load(f)

                # Update config with YAML data
                if yaml_data:
                    config = cls._from_dict(yaml_data)
                    config.config_file = path

            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")
                print("Using defaults and environment variables")

        # Override with environment variables
        config.network = NetworkConfig.from_env()
        config.performance = PerformanceConfig.from_env()
        config.runtime = RuntimeConfig.from_env()
        config.server = ServerConfig.from_env()
        config.logging = LoggingConfig.from_env()
        config.optimization = OptimizationConfig.from_env()

        # Global environment overrides
        config.debug_mode = os.getenv('GENIE_DEBUG', 'false').lower() == 'true'

        return config

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'DjinnConfig':
        """Create config from dictionary (YAML data)."""
        network_data = data.get('network', {})
        performance_data = data.get('performance', {})
        runtime_data = data.get('runtime', {})
        server_data = data.get('server', {})
        logging_data = data.get('logging', {})
        optimization_data = data.get('optimization', {})

        return cls(
            network=NetworkConfig(**network_data),
            performance=PerformanceConfig(**performance_data),
            runtime=RuntimeConfig(**runtime_data),
            server=ServerConfig(**server_data),
            logging=LoggingConfig(**logging_data),
            optimization=OptimizationConfig(**optimization_data),
            debug_mode=data.get('debug_mode', False)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'network': {
                'control_port': self.network.control_port,
                'data_port': self.network.data_port,
                'metrics_port': self.network.metrics_port,
                'prefer_dpdk': self.network.prefer_dpdk,
                'require_dpdk': self.network.require_dpdk,
                'tcp_fallback': self.network.tcp_fallback,
                'max_connections_per_target': self.network.max_connections_per_target,
                'connection_timeout': self.network.connection_timeout,
                'heartbeat_interval': self.network.heartbeat_interval,
                'heartbeat_timeout': self.network.heartbeat_timeout,
                'chunk_size': self.network.chunk_size,
                'max_message_size': self.network.max_message_size,
                'max_retry_attempts': self.network.max_retry_attempts,
                'retry_backoff': self.network.retry_backoff,
                'batch_size_threshold': self.network.batch_size_threshold,
                'batch_timeout': self.network.batch_timeout
            },
            'performance': {
                'operation_timeout': self.performance.operation_timeout,
                'transfer_timeout': self.performance.transfer_timeout,
                'negotiation_timeout': self.performance.negotiation_timeout,
                'pattern_cache_size': self.performance.pattern_cache_size,
                'shape_cache_size': self.performance.shape_cache_size,
                'metadata_cache_size': self.performance.metadata_cache_size,
                'cache_ttl_seconds': self.performance.cache_ttl_seconds,
                'max_memory_mb': self.performance.max_memory_mb,
                'memory_pressure_threshold': self.performance.memory_pressure_threshold,
                'gpu_memory_fraction': self.performance.gpu_memory_fraction,
                'gpu_id': self.performance.gpu_id,
                'enable_profiling': self.performance.enable_profiling,
                'profile_sample_rate': self.performance.profile_sample_rate
            },
            'runtime': {
                'thread_pool_size': self.runtime.thread_pool_size,
                'auto_connect': self.runtime.auto_connect,
                'auto_init': self.runtime.auto_init,
                'enable_profiling': self.runtime.enable_profiling,
                'profile_sample_rate': self.runtime.profile_sample_rate
            },
            'server': {
                'node_id': self.server.node_id,
                'max_concurrent_transfers': self.server.max_concurrent_transfers,
                'max_queue_size': self.server.max_queue_size,
                'enable_resource_discovery': self.server.enable_resource_discovery,
                'resource_check_interval': self.server.resource_check_interval,
                'enable_load_balancing': self.server.enable_load_balancing,
                'load_balance_threshold': self.server.load_balance_threshold
            },
            'logging': {
                'level': self.logging.level.value,
                'format': self.logging.format,
                'file_path': self.logging.file_path,
                'max_file_size_mb': self.logging.max_file_size_mb,
                'backup_count': self.logging.backup_count
            },
            'optimization': {
                'enable_tensor_registry': self.optimization.enable_tensor_registry,
                'tensor_registry_max_models': self.optimization.tensor_registry_max_models,
                'tensor_registry_max_bytes_per_model': self.optimization.tensor_registry_max_bytes_per_model,
                'tensor_registry_max_total_bytes': self.optimization.tensor_registry_max_total_bytes,
                'enable_srg_fusion': self.optimization.enable_srg_fusion,
                'enable_fusion_torchscript': self.optimization.enable_fusion_torchscript,
                'enable_fusion_compilation': self.optimization.enable_fusion_compilation,
                'profile_registry_overhead': self.optimization.profile_registry_overhead,
                'profile_fusion_overhead': self.optimization.profile_fusion_overhead,
            },
            'debug_mode': self.debug_mode
        }

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# Global configuration instance
_config: Optional[DjinnConfig] = None


def get_config() -> DjinnConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = DjinnConfig.load()
    return _config


def set_config(config: DjinnConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(path: Optional[str] = None) -> DjinnConfig:
    """Load configuration from file and/or environment."""
    return DjinnConfig.load(path)

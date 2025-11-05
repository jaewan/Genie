"""
Health Check Service for Djinn Cluster

Performs comprehensive health checks on cluster nodes including:
- GPU health and availability
- Network connectivity
- Memory availability
- Node responsiveness
- Overall cluster health
"""

import asyncio
import logging
import time
import os
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .node_info import NodeInfo, NodeStatus

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result"""
    check_name: str
    status: HealthStatus
    message: str
    details: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }


@dataclass
class HealthReport:
    """Overall health report"""
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: float = field(default_factory=time.time)
    node_id: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'overall_status': self.overall_status.value,
            'node_id': self.node_id,
            'checks': [c.to_dict() for c in self.checks],
            'timestamp': self.timestamp,
            'summary': {
                'total_checks': len(self.checks),
                'healthy': sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
                'degraded': sum(1 for c in self.checks if c.status == HealthStatus.DEGRADED),
                'unhealthy': sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY),
                'critical': sum(1 for c in self.checks if c.status == HealthStatus.CRITICAL)
            }
        }


class HealthChecker:
    """
    Performs health checks on cluster node.
    
    Checks:
    - GPU availability and health
    - CPU usage
    - Memory availability
    - Disk space
    - Network connectivity
    - Process status
    """
    
    def __init__(self, local_node: NodeInfo):
        """
        Initialize health checker.
        
        Args:
            local_node: Local node to check
        """
        self.local_node = local_node
        self.last_report: Optional[HealthReport] = None
    
    async def perform_health_check(self) -> HealthReport:
        """
        Perform comprehensive health check.
        
        Returns:
            HealthReport with all check results
        """
        checks = []
        
        # GPU health check
        checks.append(await self._check_gpu_health())
        
        # CPU usage check
        checks.append(await self._check_cpu_usage())
        
        # Memory availability check
        checks.append(await self._check_memory())
        
        # Disk space check
        checks.append(await self._check_disk_space())
        
        # Network connectivity check
        checks.append(await self._check_network())
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        # Create report
        report = HealthReport(
            overall_status=overall_status,
            checks=checks,
            node_id=self.local_node.node_id
        )
        
        self.last_report = report
        
        # Log summary
        summary = report.to_dict()['summary']
        logger.info(
            f"Health check complete: {overall_status.value} "
            f"({summary['healthy']}/{summary['total_checks']} healthy)"
        )
        
        return report
    
    async def _check_gpu_health(self) -> HealthCheck:
        """Check GPU health and availability"""
        try:
            gpus = self.local_node.gpus
            
            if not gpus:
                return HealthCheck(
                    check_name="gpu_health",
                    status=HealthStatus.HEALTHY,
                    message="No GPUs configured (normal for client)",
                    details={'gpu_count': 0}
                )
            
            # Count healthy GPUs
            healthy_gpus = sum(1 for gpu in gpus if gpu.health_status == 'healthy')
            total_gpus = len(gpus)
            
            # Check for issues
            overheated = [gpu.gpu_id for gpu in gpus 
                         if gpu.temperature_celsius and gpu.temperature_celsius > 85]
            high_utilization = [gpu.gpu_id for gpu in gpus 
                              if gpu.utilization_percent > 95]
            
            details = {
                'total_gpus': total_gpus,
                'healthy_gpus': healthy_gpus,
                'overheated_gpus': overheated,
                'high_utilization_gpus': high_utilization
            }
            
            # Determine status
            if healthy_gpus == 0:
                status = HealthStatus.CRITICAL
                message = f"All GPUs unhealthy ({total_gpus} GPUs)"
            elif healthy_gpus <= total_gpus * 0.5:
                status = HealthStatus.UNHEALTHY
                message = f"Most GPUs unhealthy ({healthy_gpus}/{total_gpus} healthy)"
            elif overheated or high_utilization:
                status = HealthStatus.DEGRADED
                message = f"Some GPUs degraded ({healthy_gpus}/{total_gpus} healthy)"
            else:
                status = HealthStatus.HEALTHY
                message = f"All GPUs healthy ({healthy_gpus}/{total_gpus})"
            
            return HealthCheck(
                check_name="gpu_health",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            logger.error(f"GPU health check failed: {e}", exc_info=True)
            return HealthCheck(
                check_name="gpu_health",
                status=HealthStatus.UNHEALTHY,
                message=f"GPU health check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_cpu_usage(self) -> HealthCheck:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            details = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count
            }
            
            if cpu_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"CPU critical: {cpu_percent:.1f}%"
            elif cpu_percent > 85:
                status = HealthStatus.UNHEALTHY
                message = f"CPU very high: {cpu_percent:.1f}%"
            elif cpu_percent > 70:
                status = HealthStatus.DEGRADED
                message = f"CPU high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU normal: {cpu_percent:.1f}%"
            
            return HealthCheck(
                check_name="cpu_usage",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                check_name="cpu_usage",
                status=HealthStatus.UNHEALTHY,
                message=f"CPU check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_memory(self) -> HealthCheck:
        """Check memory availability"""
        try:
            memory = psutil.virtual_memory()
            
            details = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent
            }
            
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Memory critical: {memory.percent:.1f}%"
            elif memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Memory very high: {memory.percent:.1f}%"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Memory high: {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory normal: {memory.percent:.1f}%"
            
            return HealthCheck(
                check_name="memory",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                check_name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_disk_space(self) -> HealthCheck:
        """Check disk space availability"""
        try:
            disk = psutil.disk_usage('/')
            
            details = {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'used_percent': disk.percent
            }
            
            if disk.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk critical: {disk.percent:.1f}%"
            elif disk.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Disk very high: {disk.percent:.1f}%"
            elif disk.percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Disk high: {disk.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk normal: {disk.percent:.1f}%"
            
            return HealthCheck(
                check_name="disk_space",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                check_name="disk_space",
                status=HealthStatus.DEGRADED,
                message=f"Disk check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_network(self) -> HealthCheck:
        """Check network connectivity"""
        try:
            # Check if network interfaces are up
            net_io = psutil.net_io_counters()
            
            details = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout
            }
            
            # Check for network errors
            total_errors = net_io.errin + net_io.errout
            total_packets = net_io.packets_sent + net_io.packets_recv
            
            if total_packets > 0:
                error_rate = (total_errors / total_packets) * 100
                details['error_rate_percent'] = error_rate
                
                if error_rate > 5.0:
                    status = HealthStatus.UNHEALTHY
                    message = f"High network error rate: {error_rate:.2f}%"
                elif error_rate > 1.0:
                    status = HealthStatus.DEGRADED
                    message = f"Elevated network errors: {error_rate:.2f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Network healthy"
            else:
                status = HealthStatus.HEALTHY
                message = "Network initialized"
            
            return HealthCheck(
                check_name="network",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                check_name="network",
                status=HealthStatus.DEGRADED,
                message=f"Network check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall status from individual checks"""
        if not checks:
            return HealthStatus.UNHEALTHY
        
        # If any check is critical, overall is critical
        if any(c.status == HealthStatus.CRITICAL for c in checks):
            return HealthStatus.CRITICAL
        
        # Count unhealthy checks
        unhealthy_count = sum(1 for c in checks 
                             if c.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL])
        
        if unhealthy_count >= len(checks) // 2:
            return HealthStatus.UNHEALTHY
        elif unhealthy_count > 0:
            return HealthStatus.DEGRADED
        
        # Check for degraded
        degraded_count = sum(1 for c in checks if c.status == HealthStatus.DEGRADED)
        if degraded_count >= len(checks) // 2:
            return HealthStatus.DEGRADED
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def get_last_report(self) -> Optional[HealthReport]:
        """Get the last health report"""
        return self.last_report


def create_health_checker(local_node: NodeInfo) -> HealthChecker:
    """
    Create a health checker instance.
    
    Args:
        local_node: Local node to check
    
    Returns:
        HealthChecker instance
    """
    return HealthChecker(local_node)


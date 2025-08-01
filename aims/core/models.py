"""
Core data models for the AIMS system
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import uuid


class DeviceType(str, Enum):
    GPU_SERVER = "gpu_server"
    COOLING_UNIT = "cooling_unit"
    POWER_UNIT = "power_unit"
    NETWORK_SWITCH = "network_switch"
    STORAGE_UNIT = "storage_unit"


class DeviceStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    WARNING = "warning"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"


class MetricType(str, Enum):
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class Device(BaseModel):
    """Represents a managed device in the infrastructure"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    device_type: DeviceType
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    location: Optional[str] = None
    rack_position: Optional[str] = None
    status: DeviceStatus = DeviceStatus.OFFLINE
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class GPUDevice(Device):
    """GPU-specific device information"""
    
    device_type: DeviceType = DeviceType.GPU_SERVER
    gpu_count: int = 0
    gpu_model: Optional[str] = None
    gpu_memory_total: Optional[int] = None  # In MB
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None


class Metric(BaseModel):
    """Represents a time-series metric"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str
    metric_name: str
    metric_type: MetricType
    value: Union[float, int]
    unit: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class GPUMetric(Metric):
    """GPU-specific metrics"""
    
    gpu_index: Optional[int] = None
    
    @validator('metric_name')
    def validate_gpu_metric_name(cls, v):
        valid_metrics = [
            'gpu_utilization', 'memory_utilization', 'temperature',
            'power_draw', 'fan_speed', 'memory_used', 'memory_free',
            'compute_processes', 'graphics_processes', 'ecc_errors'
        ]
        if v not in valid_metrics:
            raise ValueError(f"Invalid GPU metric name: {v}")
        return v


class Event(BaseModel):
    """Represents a discrete event in the system"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: Optional[str] = None
    event_type: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class Alert(BaseModel):
    """Represents an alert condition"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str
    device_id: Optional[str] = None
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class AlertRule(BaseModel):
    """Defines conditions for triggering alerts"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "greater_than", "less_than", "equals"
    threshold: Union[float, int]
    duration: int = 0  # Duration in seconds before triggering
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    device_filter: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class InventoryItem(BaseModel):
    """Hardware or software inventory item"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str
    category: str  # hardware, software, firmware
    name: str
    version: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    part_number: Optional[str] = None
    installation_date: Optional[datetime] = None
    warranty_expiry: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Command(BaseModel):
    """Represents a command to be executed on a device"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str
    command_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str


class CommandResult(BaseModel):
    """Result of a command execution"""
    
    command_id: str
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: Optional[int] = None


class KubernetesWorkload(BaseModel):
    """Kubernetes workload information"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    namespace: str
    name: str
    workload_type: str  # pod, deployment, job, etc.
    node_name: Optional[str] = None
    gpu_requests: Dict[str, Any] = Field(default_factory=dict)
    gpu_limits: Dict[str, Any] = Field(default_factory=dict)
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Performance benchmark result"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str
    benchmark_name: str
    benchmark_type: str  # gpu_compute, memory_bandwidth, network_throughput, etc.
    score: float
    unit: str
    details: Dict[str, Any] = Field(default_factory=dict)
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: int
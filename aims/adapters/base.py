"""
Base protocol adapter interface and registry
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type
import asyncio
from datetime import datetime
import logging
from ..core.models import Metric, Command, CommandResult, Device

logger = logging.getLogger(__name__)


class ProtocolAdapter(ABC):
    """Base class for all protocol adapters"""
    
    def __init__(self, device: Device, config: Dict[str, Any]):
        self.device = device
        self.config = config
        self.connected = False
        self.last_connection_attempt = None
        self.connection_failures = 0
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the device"""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the device"""
        pass
        
    @abstractmethod
    async def collect_metrics(self) -> List[Metric]:
        """Collect metrics from the device"""
        pass
        
    @abstractmethod
    async def execute_command(self, command: Command) -> CommandResult:
        """Execute a command on the device"""
        pass
        
    @abstractmethod
    async def discover_inventory(self) -> List[Dict[str, Any]]:
        """Discover hardware/software inventory"""
        pass
        
    async def is_healthy(self) -> bool:
        """Check if the adapter and device are healthy"""
        try:
            if not self.connected:
                return False
            # Try a simple operation to verify connection
            await self._health_check()
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.device.name}: {e}")
            return False
            
    async def _health_check(self) -> None:
        """Override this method to implement adapter-specific health checks"""
        pass
        
    async def _reconnect_if_needed(self) -> bool:
        """Attempt to reconnect if not connected"""
        if self.connected:
            return True
            
        if self.connection_failures >= self.max_retries:
            return False
            
        try:
            self.last_connection_attempt = datetime.utcnow()
            success = await self.connect()
            if success:
                self.connection_failures = 0
                logger.info(f"Successfully reconnected to {self.device.name}")
            else:
                self.connection_failures += 1
                logger.warning(f"Reconnection attempt {self.connection_failures} failed for {self.device.name}")
            return success
        except Exception as e:
            self.connection_failures += 1
            logger.error(f"Reconnection failed for {self.device.name}: {e}")
            return False
            
    async def safe_execute(self, operation):
        """Execute an operation with automatic reconnection"""
        if not await self._reconnect_if_needed():
            raise ConnectionError(f"Cannot connect to device {self.device.name}")
            
        try:
            return await operation()
        except ConnectionError:
            self.connected = False
            # Try once more after reconnecting
            if await self._reconnect_if_needed():
                return await operation()
            raise


class AdapterRegistry:
    """Registry for protocol adapters"""
    
    _adapters: Dict[str, Type[ProtocolAdapter]] = {}
    _instances: Dict[str, ProtocolAdapter] = {}
    
    @classmethod
    def register(cls, protocol_name: str, adapter_class: Type[ProtocolAdapter]):
        """Register a protocol adapter"""
        cls._adapters[protocol_name] = adapter_class
        logger.info(f"Registered adapter for protocol: {protocol_name}")
        
    @classmethod
    def get_adapter_class(cls, protocol_name: str) -> Optional[Type[ProtocolAdapter]]:
        """Get adapter class by protocol name"""
        return cls._adapters.get(protocol_name)
        
    @classmethod
    def create_adapter(cls, protocol_name: str, device: Device, config: Dict[str, Any]) -> Optional[ProtocolAdapter]:
        """Create an adapter instance"""
        adapter_class = cls.get_adapter_class(protocol_name)
        if not adapter_class:
            logger.error(f"No adapter found for protocol: {protocol_name}")
            return None
            
        try:
            adapter = adapter_class(device, config)
            cls._instances[f"{device.id}_{protocol_name}"] = adapter
            return adapter
        except Exception as e:
            logger.error(f"Failed to create adapter for {protocol_name}: {e}")
            return None
            
    @classmethod
    def get_adapter(cls, device_id: str, protocol_name: str) -> Optional[ProtocolAdapter]:
        """Get existing adapter instance"""
        return cls._instances.get(f"{device_id}_{protocol_name}")
        
    @classmethod
    def remove_adapter(cls, device_id: str, protocol_name: str):
        """Remove adapter instance"""
        key = f"{device_id}_{protocol_name}"
        if key in cls._instances:
            adapter = cls._instances[key]
            asyncio.create_task(adapter.disconnect())
            del cls._instances[key]
            
    @classmethod
    def list_protocols(cls) -> List[str]:
        """List all registered protocols"""
        return list(cls._adapters.keys())
        
    @classmethod
    async def disconnect_all(cls):
        """Disconnect all adapter instances"""
        tasks = []
        for adapter in cls._instances.values():
            tasks.append(adapter.disconnect())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        cls._instances.clear()


def register_adapter(protocol_name: str):
    """Decorator to register a protocol adapter"""
    def decorator(adapter_class: Type[ProtocolAdapter]):
        AdapterRegistry.register(protocol_name, adapter_class)
        return adapter_class
    return decorator
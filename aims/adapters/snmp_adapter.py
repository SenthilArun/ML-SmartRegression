"""
SNMP adapter for DCIM systems
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pysnmp.hlapi.asyncio import *
from pysnmp.proto.rfc1902 import Integer, OctetString
from ..core.models import Metric, Command, CommandResult, MetricType
from .base import ProtocolAdapter, register_adapter

logger = logging.getLogger(__name__)


@register_adapter("snmp")
class SNMPAdapter(ProtocolAdapter):
    """SNMP adapter for monitoring DCIM equipment"""
    
    # Common SNMP OIDs for different equipment types
    COMMON_OIDS = {
        # System information
        'system_description': '1.3.6.1.2.1.1.1.0',
        'system_uptime': '1.3.6.1.2.1.1.3.0',
        'system_name': '1.3.6.1.2.1.1.5.0',
        'system_location': '1.3.6.1.2.1.1.6.0',
        
        # Interface statistics
        'interface_count': '1.3.6.1.2.1.2.1.0',
        'interface_table': '1.3.6.1.2.1.2.2.1',
        
        # Generic power measurements
        'power_consumption': '1.3.6.1.4.1.534.6.6.7.1.2.1.4',  # APC PDU
        'voltage': '1.3.6.1.4.1.534.6.6.7.1.2.1.3',
        'current': '1.3.6.1.4.1.534.6.6.7.1.2.1.6',
    }
    
    # Cooling system specific OIDs
    COOLING_OIDS = {
        'temperature_inlet': '1.3.6.1.4.1.674.10892.2.3.1.2.0',  # Dell
        'temperature_outlet': '1.3.6.1.4.1.674.10892.2.3.1.3.0',
        'fan_speed': '1.3.6.1.4.1.674.10892.2.4.1.1.0',
        'flow_rate': '1.3.6.1.4.1.674.10892.2.5.1.1.0',
        'humidity': '1.3.6.1.4.1.674.10892.2.6.1.1.0',
    }
    
    # Power system specific OIDs  
    POWER_OIDS = {
        'ups_battery_status': '1.3.6.1.2.1.33.1.2.1.0',
        'ups_output_power': '1.3.6.1.2.1.33.1.4.4.1.4',
        'ups_load_percentage': '1.3.6.1.2.1.33.1.4.4.1.5',
        'pdu_outlet_count': '1.3.6.1.4.1.534.6.6.7.1.1.1.8.0',
        'pdu_total_power': '1.3.6.1.4.1.534.6.6.7.1.2.1.4.0',
    }
    
    def __init__(self, device, config):
        super().__init__(device, config)
        self.community = config.get('community', 'public')
        self.version = config.get('version', '2c')
        self.timeout = config.get('timeout', 5)
        self.retries = config.get('retries', 3)
        self.port = config.get('port', 161)
        
        # Version-specific authentication
        if self.version == '3':
            self.username = config.get('username')
            self.auth_key = config.get('auth_key')
            self.priv_key = config.get('priv_key')
            self.auth_protocol = config.get('auth_protocol', 'usmHMACMD5AuthProtocol')
            self.priv_protocol = config.get('priv_protocol', 'usmDESPrivProtocol')
        
        self._setup_oids()
        
    def _setup_oids(self):
        """Setup OIDs based on device type"""
        self.oids = dict(self.COMMON_OIDS)
        
        if self.device.device_type == "cooling_unit":
            self.oids.update(self.COOLING_OIDS)
        elif self.device.device_type == "power_unit":
            self.oids.update(self.POWER_OIDS)
            
        # Add custom OIDs from config
        custom_oids = self.config.get('custom_oids', {})
        self.oids.update(custom_oids)
        
    async def connect(self) -> bool:
        """Test SNMP connectivity"""
        try:
            # Test with a simple system description query
            result = await self._snmp_get([self.oids['system_description']])
            if result:
                self.connected = True
                logger.info(f"SNMP connection established to {self.device.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"SNMP connection failed to {self.device.name}: {e}")
            return False
            
    async def disconnect(self) -> None:
        """Disconnect SNMP (no persistent connection needed)"""
        self.connected = False
        
    async def _health_check(self) -> None:
        """Check SNMP connectivity"""
        result = await self._snmp_get([self.oids['system_uptime']])
        if not result:
            raise ConnectionError("SNMP health check failed")
            
    async def collect_metrics(self) -> List[Metric]:
        """Collect metrics via SNMP"""
        metrics = []
        
        try:
            # Collect based on device type
            if self.device.device_type == "cooling_unit":
                metrics.extend(await self._collect_cooling_metrics())
            elif self.device.device_type == "power_unit":
                metrics.extend(await self._collect_power_metrics())
            else:
                metrics.extend(await self._collect_common_metrics())
                
        except Exception as e:
            logger.error(f"Failed to collect SNMP metrics from {self.device.name}: {e}")
            
        return metrics
        
    async def _collect_cooling_metrics(self) -> List[Metric]:
        """Collect cooling system metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Temperature metrics
        for metric_name, oid in [
            ('temperature_inlet', 'temperature_inlet'),
            ('temperature_outlet', 'temperature_outlet'),
            ('humidity', 'humidity')
        ]:
            if oid in self.oids:
                value = await self._snmp_get_single(self.oids[oid])
                if value is not None:
                    metrics.append(Metric(
                        device_id=self.device.id,
                        metric_name=metric_name,
                        metric_type=MetricType.GAUGE,
                        value=float(value),
                        unit="celsius" if "temperature" in metric_name else "percent",
                        timestamp=timestamp,
                        tags={"source": "snmp", "device_type": "cooling"}
                    ))
                    
        # Fan speed and flow rate
        for metric_name, oid in [
            ('fan_speed', 'fan_speed'),
            ('flow_rate', 'flow_rate')
        ]:
            if oid in self.oids:
                value = await self._snmp_get_single(self.oids[oid])
                if value is not None:
                    metrics.append(Metric(
                        device_id=self.device.id,
                        metric_name=metric_name,
                        metric_type=MetricType.GAUGE,
                        value=float(value),
                        unit="rpm" if "fan" in metric_name else "cfm",
                        timestamp=timestamp,
                        tags={"source": "snmp", "device_type": "cooling"}
                    ))
                    
        return metrics
        
    async def _collect_power_metrics(self) -> List[Metric]:
        """Collect power system metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Power consumption
        power_value = await self._snmp_get_single(self.oids.get('power_consumption'))
        if power_value is not None:
            metrics.append(Metric(
                device_id=self.device.id,
                metric_name="power_consumption",
                metric_type=MetricType.GAUGE,
                value=float(power_value),
                unit="watts",
                timestamp=timestamp,
                tags={"source": "snmp", "device_type": "power"}
            ))
            
        # Voltage and current
        for metric_name, oid_key in [
            ('voltage', 'voltage'),
            ('current', 'current')
        ]:
            if oid_key in self.oids:
                value = await self._snmp_get_single(self.oids[oid_key])
                if value is not None:
                    metrics.append(Metric(
                        device_id=self.device.id,
                        metric_name=metric_name,
                        metric_type=MetricType.GAUGE,
                        value=float(value),
                        unit="volts" if metric_name == "voltage" else "amps",
                        timestamp=timestamp,
                        tags={"source": "snmp", "device_type": "power"}
                    ))
                    
        # UPS specific metrics
        if 'ups_load_percentage' in self.oids:
            load_value = await self._snmp_get_single(self.oids['ups_load_percentage'])
            if load_value is not None:
                metrics.append(Metric(
                    device_id=self.device.id,
                    metric_name="ups_load_percentage",
                    metric_type=MetricType.GAUGE,
                    value=float(load_value),
                    unit="percent",
                    timestamp=timestamp,
                    tags={"source": "snmp", "device_type": "power", "component": "ups"}
                ))
                
        return metrics
        
    async def _collect_common_metrics(self) -> List[Metric]:
        """Collect common system metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        # System uptime
        uptime = await self._snmp_get_single(self.oids.get('system_uptime'))
        if uptime is not None:
            metrics.append(Metric(
                device_id=self.device.id,
                metric_name="system_uptime",
                metric_type=MetricType.COUNTER,
                value=int(uptime),
                unit="centiseconds",
                timestamp=timestamp,
                tags={"source": "snmp", "device_type": str(self.device.device_type)}
            ))
            
        return metrics
        
    async def _snmp_get(self, oids: List[str]) -> Optional[Dict[str, Any]]:
        """Perform SNMP GET operation"""
        try:
            auth_data = self._get_auth_data()
            transport_target = UdpTransportTarget((self.device.ip_address, self.port))
            
            results = {}
            for oid in oids:
                iterator = getCmd(
                    SnmpEngine(),
                    auth_data,
                    transport_target,
                    ContextData(),
                    ObjectType(ObjectIdentity(oid)),
                    timeout=self.timeout,
                    retries=self.retries
                )
                
                error_indication, error_status, error_index, var_binds = await iterator
                
                if error_indication:
                    logger.error(f"SNMP error indication: {error_indication}")
                    continue
                    
                if error_status:
                    logger.error(f"SNMP error status: {error_status} at {error_index}")
                    continue
                    
                for var_bind in var_binds:
                    oid_key = str(var_bind[0])
                    value = var_bind[1]
                    results[oid_key] = value
                    
            return results if results else None
            
        except Exception as e:
            logger.error(f"SNMP GET operation failed: {e}")
            return None
            
    async def _snmp_get_single(self, oid: str) -> Optional[Any]:
        """Get a single SNMP value"""
        result = await self._snmp_get([oid])
        return result.get(oid) if result else None
        
    def _get_auth_data(self):
        """Get SNMP authentication data based on version"""
        if self.version == '1':
            return CommunityData(self.community, mpModel=0)
        elif self.version == '2c':
            return CommunityData(self.community, mpModel=1)
        elif self.version == '3':
            if self.username:
                auth_protocol = getattr(usmHMACMD5AuthProtocol, self.auth_protocol, usmHMACMD5AuthProtocol)
                priv_protocol = getattr(usmDESPrivProtocol, self.priv_protocol, usmDESPrivProtocol)
                
                return UsmUserData(
                    self.username,
                    authKey=self.auth_key,
                    privKey=self.priv_key,
                    authProtocol=auth_protocol,
                    privProtocol=priv_protocol
                )
            else:
                return UsmUserData(self.username)
        else:
            raise ValueError(f"Unsupported SNMP version: {self.version}")
            
    async def execute_command(self, command: Command) -> CommandResult:
        """Execute SNMP SET command"""
        start_time = datetime.utcnow()
        
        try:
            command_type = command.command_type
            parameters = command.parameters
            
            if command_type == "snmp_set":
                oid = parameters.get('oid')
                value = parameters.get('value')
                value_type = parameters.get('type', 'integer')
                
                if not oid or value is None:
                    return CommandResult(
                        command_id=command.id,
                        success=False,
                        error_message="Missing OID or value for SNMP SET"
                    )
                    
                # Convert value based on type
                if value_type == 'integer':
                    snmp_value = Integer(value)
                elif value_type == 'string':
                    snmp_value = OctetString(value)
                else:
                    return CommandResult(
                        command_id=command.id,
                        success=False,
                        error_message=f"Unsupported value type: {value_type}"
                    )
                    
                # Perform SNMP SET
                auth_data = self._get_auth_data()
                transport_target = UdpTransportTarget((self.device.ip_address, self.port))
                
                iterator = setCmd(
                    SnmpEngine(),
                    auth_data,
                    transport_target,
                    ContextData(),
                    ObjectType(ObjectIdentity(oid), snmp_value),
                    timeout=self.timeout,
                    retries=self.retries
                )
                
                error_indication, error_status, error_index, var_binds = await iterator
                
                if error_indication or error_status:
                    error_msg = f"SNMP SET failed: {error_indication or error_status}"
                    return CommandResult(
                        command_id=command.id,
                        success=False,
                        error_message=error_msg,
                        execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    )
                    
                return CommandResult(
                    command_id=command.id,
                    success=True,
                    result_data={"oid": oid, "value": str(snmp_value)},
                    execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                )
                
            else:
                return CommandResult(
                    command_id=command.id,
                    success=False,
                    error_message=f"Unsupported command type: {command_type}"
                )
                
        except Exception as e:
            return CommandResult(
                command_id=command.id,
                success=False,
                error_message=str(e),
                execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
            
    async def discover_inventory(self) -> List[Dict[str, Any]]:
        """Discover device inventory via SNMP"""
        inventory = []
        
        try:
            # Get system information
            system_info = await self._snmp_get([
                self.oids.get('system_description', ''),
                self.oids.get('system_name', ''),
                self.oids.get('system_location', '')
            ])
            
            if system_info:
                for oid, value in system_info.items():
                    if oid == self.oids.get('system_description'):
                        inventory.append({
                            'category': 'system',
                            'name': 'System Description',
                            'value': str(value),
                            'source': 'snmp'
                        })
                    elif oid == self.oids.get('system_name'):
                        inventory.append({
                            'category': 'system',
                            'name': 'System Name', 
                            'value': str(value),
                            'source': 'snmp'
                        })
                        
        except Exception as e:
            logger.error(f"Failed to discover inventory for {self.device.name}: {e}")
            
        return inventory
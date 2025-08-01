"""
Modbus adapter for DCIM systems
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pymodbus.client.asynchronous.tcp import AsyncModbusTCPClient
from pymodbus.client.asynchronous.serial import AsyncModbusSerialClient
from pymodbus.exceptions import ModbusException
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder
from ..core.models import Metric, Command, CommandResult, MetricType
from .base import ProtocolAdapter, register_adapter

logger = logging.getLogger(__name__)


@register_adapter("modbus")
class ModbusAdapter(ProtocolAdapter):
    """Modbus adapter for industrial DCIM equipment"""
    
    # Common Modbus register maps for different equipment types
    COOLING_REGISTERS = {
        'temperature_inlet': {'address': 40001, 'type': 'holding', 'scale': 0.1, 'unit': 'celsius'},
        'temperature_outlet': {'address': 40002, 'type': 'holding', 'scale': 0.1, 'unit': 'celsius'},
        'humidity': {'address': 40003, 'type': 'holding', 'scale': 0.1, 'unit': 'percent'},
        'fan_speed_1': {'address': 40010, 'type': 'holding', 'scale': 1, 'unit': 'rpm'},
        'fan_speed_2': {'address': 40011, 'type': 'holding', 'scale': 1, 'unit': 'rpm'},
        'flow_rate': {'address': 40020, 'type': 'holding', 'scale': 0.01, 'unit': 'cfm'},
        'compressor_status': {'address': 10001, 'type': 'coil', 'scale': 1, 'unit': 'boolean'},
        'alarm_status': {'address': 10010, 'type': 'discrete', 'scale': 1, 'unit': 'boolean'},
    }
    
    POWER_REGISTERS = {
        'voltage_l1': {'address': 30001, 'type': 'input', 'scale': 0.1, 'unit': 'volts'},
        'voltage_l2': {'address': 30002, 'type': 'input', 'scale': 0.1, 'unit': 'volts'},
        'voltage_l3': {'address': 30003, 'type': 'input', 'scale': 0.1, 'unit': 'volts'},
        'current_l1': {'address': 30010, 'type': 'input', 'scale': 0.01, 'unit': 'amps'},
        'current_l2': {'address': 30011, 'type': 'input', 'scale': 0.01, 'unit': 'amps'},
        'current_l3': {'address': 30012, 'type': 'input', 'scale': 0.01, 'unit': 'amps'},
        'power_total': {'address': 30020, 'type': 'input', 'scale': 1, 'unit': 'watts'},
        'energy_total': {'address': 30030, 'type': 'input', 'scale': 0.01, 'unit': 'kwh'},
        'frequency': {'address': 30040, 'type': 'input', 'scale': 0.01, 'unit': 'hz'},
        'power_factor': {'address': 30041, 'type': 'input', 'scale': 0.01, 'unit': 'ratio'},
    }
    
    def __init__(self, device, config):
        super().__init__(device, config)
        self.host = device.ip_address
        self.port = config.get('port', 502)
        self.unit_id = config.get('unit_id', 1)
        self.timeout = config.get('timeout', 5)
        self.connection_type = config.get('connection_type', 'tcp')  # tcp, rtu, ascii
        
        # Serial connection parameters (for RTU/ASCII)
        self.serial_port = config.get('serial_port', '/dev/ttyUSB0')
        self.baudrate = config.get('baudrate', 9600)
        self.bytesize = config.get('bytesize', 8)
        self.parity = config.get('parity', 'N')
        self.stopbits = config.get('stopbits', 1)
        
        self.client = None
        self._setup_registers()
        
    def _setup_registers(self):
        """Setup register maps based on device type"""
        self.registers = {}
        
        if self.device.device_type == "cooling_unit":
            self.registers.update(self.COOLING_REGISTERS)
        elif self.device.device_type == "power_unit":
            self.registers.update(self.POWER_REGISTERS)
            
        # Add custom registers from config
        custom_registers = self.config.get('custom_registers', {})
        self.registers.update(custom_registers)
        
    async def connect(self) -> bool:
        """Establish Modbus connection"""
        try:
            if self.connection_type == 'tcp':
                self.client = AsyncModbusTCPClient(
                    host=self.host,
                    port=self.port,
                    timeout=self.timeout
                )
            elif self.connection_type in ['rtu', 'ascii']:
                self.client = AsyncModbusSerialClient(
                    method=self.connection_type,
                    port=self.serial_port,
                    baudrate=self.baudrate,
                    bytesize=self.bytesize,
                    parity=self.parity,
                    stopbits=self.stopbits,
                    timeout=self.timeout
                )
            else:
                logger.error(f"Unsupported Modbus connection type: {self.connection_type}")
                return False
                
            # Attempt to connect
            connection_result = await self.client.connect()
            if connection_result:
                self.connected = True
                logger.info(f"Modbus connection established to {self.device.name}")
                return True
            else:
                logger.error(f"Failed to establish Modbus connection to {self.device.name}")
                return False
                
        except Exception as e:
            logger.error(f"Modbus connection failed to {self.device.name}: {e}")
            return False
            
    async def disconnect(self) -> None:
        """Close Modbus connection"""
        if self.client:
            self.client.close()
        self.connected = False
        
    async def _health_check(self) -> None:
        """Check Modbus connectivity"""
        try:
            # Try to read a single coil as a health check
            result = await self.client.read_coils(0, 1, unit=self.unit_id)
            if result.isError():
                raise ConnectionError("Modbus health check failed")
        except Exception as e:
            raise ConnectionError(f"Modbus health check failed: {e}")
            
    async def collect_metrics(self) -> List[Metric]:
        """Collect metrics via Modbus"""
        metrics = []
        timestamp = datetime.utcnow()
        
        try:
            # Group registers by type for efficient batch reading
            register_groups = {
                'coil': [],
                'discrete': [],
                'input': [],
                'holding': []
            }
            
            for name, config in self.registers.items():
                reg_type = config['type']
                if reg_type in register_groups:
                    register_groups[reg_type].append((name, config))
                    
            # Read each group
            for reg_type, registers in register_groups.items():
                if not registers:
                    continue
                    
                # Sort by address for efficient reading
                registers.sort(key=lambda x: x[1]['address'])
                
                # Read registers in batches
                batch_size = 50  # Adjust based on device capabilities
                for i in range(0, len(registers), batch_size):
                    batch = registers[i:i + batch_size]
                    if not batch:
                        continue
                        
                    start_addr = batch[0][1]['address']
                    end_addr = batch[-1][1]['address']
                    count = end_addr - start_addr + 1
                    
                    # Read the batch
                    values = await self._read_registers(reg_type, start_addr, count)
                    if values is None:
                        continue
                        
                    # Process individual values
                    for name, config in batch:
                        addr = config['address']
                        index = addr - start_addr
                        if index < len(values):
                            raw_value = values[index]
                            scaled_value = raw_value * config.get('scale', 1)
                            
                            metrics.append(Metric(
                                device_id=self.device.id,
                                metric_name=name,
                                metric_type=MetricType.GAUGE,
                                value=scaled_value,
                                unit=config.get('unit', ''),
                                timestamp=timestamp,
                                tags={
                                    "source": "modbus",
                                    "device_type": str(self.device.device_type),
                                    "register_type": reg_type,
                                    "address": str(addr)
                                }
                            ))
                            
        except Exception as e:
            logger.error(f"Failed to collect Modbus metrics from {self.device.name}: {e}")
            
        return metrics
        
    async def _read_registers(self, reg_type: str, address: int, count: int) -> Optional[List[int]]:
        """Read registers of a specific type"""
        try:
            if reg_type == 'coil':
                result = await self.client.read_coils(address, count, unit=self.unit_id)
            elif reg_type == 'discrete':
                result = await self.client.read_discrete_inputs(address, count, unit=self.unit_id)
            elif reg_type == 'input':
                result = await self.client.read_input_registers(address, count, unit=self.unit_id)
            elif reg_type == 'holding':
                result = await self.client.read_holding_registers(address, count, unit=self.unit_id)
            else:
                logger.error(f"Unknown register type: {reg_type}")
                return None
                
            if result.isError():
                logger.error(f"Modbus read error: {result}")
                return None
                
            if reg_type in ['coil', 'discrete']:
                return result.bits[:count]
            else:
                return result.registers
                
        except Exception as e:
            logger.error(f"Failed to read {reg_type} registers at {address}: {e}")
            return None
            
    async def execute_command(self, command: Command) -> CommandResult:
        """Execute Modbus command"""
        start_time = datetime.utcnow()
        
        try:
            command_type = command.command_type
            parameters = command.parameters
            
            if command_type == "modbus_write":
                reg_type = parameters.get('type', 'holding')
                address = parameters.get('address')
                value = parameters.get('value')
                
                if address is None or value is None:
                    return CommandResult(
                        command_id=command.id,
                        success=False,
                        error_message="Missing address or value for Modbus write"
                    )
                    
                # Perform write operation
                if reg_type == 'coil':
                    result = await self.client.write_coil(address, bool(value), unit=self.unit_id)
                elif reg_type == 'holding':
                    result = await self.client.write_register(address, int(value), unit=self.unit_id)
                else:
                    return CommandResult(
                        command_id=command.id,
                        success=False,
                        error_message=f"Unsupported write register type: {reg_type}"
                    )
                    
                if result.isError():
                    return CommandResult(
                        command_id=command.id,
                        success=False,
                        error_message=f"Modbus write failed: {result}",
                        execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    )
                    
                return CommandResult(
                    command_id=command.id,
                    success=True,
                    result_data={"address": address, "value": value, "type": reg_type},
                    execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                )
                
            elif command_type == "modbus_read":
                reg_type = parameters.get('type', 'holding')
                address = parameters.get('address')
                count = parameters.get('count', 1)
                
                if address is None:
                    return CommandResult(
                        command_id=command.id,
                        success=False,
                        error_message="Missing address for Modbus read"
                    )
                    
                values = await self._read_registers(reg_type, address, count)
                if values is None:
                    return CommandResult(
                        command_id=command.id,
                        success=False,
                        error_message="Modbus read operation failed",
                        execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    )
                    
                return CommandResult(
                    command_id=command.id,
                    success=True,
                    result_data={"address": address, "values": values, "type": reg_type},
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
        """Discover device inventory via Modbus"""
        inventory = []
        
        try:
            # Read device identification registers (if supported)
            # Many Modbus devices have identification information in specific registers
            device_info_registers = {
                'device_id': 40001,
                'firmware_version': 40002,
                'hardware_version': 40003,
                'serial_number': 40004,
            }
            
            for info_type, address in device_info_registers.items():
                try:
                    values = await self._read_registers('holding', address, 1)
                    if values and values[0] != 0:
                        inventory.append({
                            'category': 'device_info',
                            'name': info_type.replace('_', ' ').title(),
                            'value': str(values[0]),
                            'source': 'modbus',
                            'address': address
                        })
                except Exception:
                    # Continue if specific register is not available
                    continue
                    
            # Add register map as inventory
            inventory.append({
                'category': 'configuration',
                'name': 'Register Map',
                'value': f"{len(self.registers)} configured registers",
                'source': 'modbus',
                'details': list(self.registers.keys())
            })
            
        except Exception as e:
            logger.error(f"Failed to discover inventory for {self.device.name}: {e}")
            
        return inventory
        
    async def _read_float(self, address: int, reg_type: str = 'holding') -> Optional[float]:
        """Read a 32-bit float from two consecutive registers"""
        try:
            values = await self._read_registers(reg_type, address, 2)
            if values and len(values) >= 2:
                # Combine two 16-bit registers into a 32-bit float
                decoder = BinaryPayloadDecoder.fromRegisters(
                    values, 
                    byteorder=Endian.Big, 
                    wordorder=Endian.Big
                )
                return decoder.decode_32bit_float()
            return None
        except Exception as e:
            logger.error(f"Failed to read float at address {address}: {e}")
            return None
            
    async def _write_float(self, address: int, value: float, reg_type: str = 'holding') -> bool:
        """Write a 32-bit float to two consecutive registers"""
        try:
            # Convert float to two 16-bit registers
            from pymodbus.payload import BinaryPayloadBuilder
            builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Big)
            builder.add_32bit_float(value)
            payload = builder.to_registers()
            
            if reg_type == 'holding':
                result = await self.client.write_registers(address, payload, unit=self.unit_id)
                return not result.isError()
            else:
                logger.error(f"Cannot write to {reg_type} registers")
                return False
                
        except Exception as e:
            logger.error(f"Failed to write float {value} at address {address}: {e}")
            return False
"""
REST API adapter for modern DCIM systems
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import aiohttp
from urllib.parse import urljoin, urlparse
from ..core.models import Metric, Command, CommandResult, MetricType
from .base import ProtocolAdapter, register_adapter

logger = logging.getLogger(__name__)


@register_adapter("rest_api")
class RestApiAdapter(ProtocolAdapter):
    """REST API adapter for modern DCIM systems"""
    
    # Common API endpoint patterns for different DCIM vendors
    ENDPOINT_PATTERNS = {
        # Generic patterns
        'devices': '/api/v1/devices',
        'metrics': '/api/v1/metrics',
        'sensors': '/api/v1/sensors',
        'alerts': '/api/v1/alerts',
        'inventory': '/api/v1/inventory',
        
        # Schneider Electric EcoStruxure
        'schneider': {
            'power_metrics': '/api/v1/measurements/electrical',
            'environmental': '/api/v1/measurements/environmental',
            'devices': '/api/v1/devices',
            'alarms': '/api/v1/alarms'
        },
        
        # Vertiv Liebert
        'vertiv': {
            'units': '/api/v1/units',
            'measurements': '/api/v1/measurements',
            'alarms': '/api/v1/alarms',
            'events': '/api/v1/events'
        },
        
        # Emerson NetSure
        'emerson': {
            'systems': '/api/v1/systems',
            'measurements': '/api/v1/measurements',
            'status': '/api/v1/status',
            'configuration': '/api/v1/configuration'
        },
        
        # Delta InfraSuite
        'delta': {
            'devices': '/api/v1/devices',
            'realtime_data': '/api/v1/realtime',
            'historical_data': '/api/v1/historical',
            'events': '/api/v1/events'
        }
    }
    
    def __init__(self, device, config):
        super().__init__(device, config)
        self.base_url = config.get('base_url', f"http://{device.ip_address}")
        self.api_version = config.get('api_version', 'v1')
        self.vendor = config.get('vendor', 'generic').lower()
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2)
        
        # Authentication configuration
        self.auth_type = config.get('auth_type', 'api_key')  # api_key, oauth2, basic
        self.api_key = config.get('api_key')
        self.username = config.get('username')
        self.password = config.get('password')
        self.oauth_token = config.get('oauth_token')
        self.oauth_refresh_token = config.get('oauth_refresh_token')
        self.oauth_client_id = config.get('oauth_client_id')
        self.oauth_client_secret = config.get('oauth_client_secret')
        
        # Request configuration
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'AIMS/1.0'
        }
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 100)  # requests per minute
        self.rate_limit_window = 60  # seconds
        self.request_timestamps = []
        
        self.session = None
        self._setup_endpoints()
        
    def _setup_endpoints(self):
        """Setup API endpoints based on vendor"""
        if self.vendor in self.ENDPOINT_PATTERNS:
            self.endpoints = self.ENDPOINT_PATTERNS[self.vendor]
        else:
            self.endpoints = self.ENDPOINT_PATTERNS
            
        # Add custom endpoints from config
        custom_endpoints = self.config.get('custom_endpoints', {})
        if isinstance(self.endpoints, dict):
            self.endpoints.update(custom_endpoints)
        
    async def connect(self) -> bool:
        """Establish HTTP session and authenticate"""
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.headers
            )
            
            # Authenticate
            auth_success = await self._authenticate()
            if auth_success:
                # Test connectivity with a simple API call
                test_success = await self._test_connectivity()
                if test_success:
                    self.connected = True
                    logger.info(f"REST API connection established to {self.device.name}")
                    return True
                    
            logger.error(f"REST API authentication or connectivity test failed for {self.device.name}")
            return False
            
        except Exception as e:
            logger.error(f"REST API connection failed to {self.device.name}: {e}")
            return False
            
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
        self.connected = False
        
    async def _authenticate(self) -> bool:
        """Authenticate with the API based on auth_type"""
        try:
            if self.auth_type == 'api_key':
                if self.api_key:
                    # Add API key to headers (common patterns)
                    self.headers['X-API-Key'] = self.api_key
                    self.headers['Authorization'] = f'Bearer {self.api_key}'
                    return True
                else:
                    logger.error("API key required but not provided")
                    return False
                    
            elif self.auth_type == 'basic':
                if self.username and self.password:
                    auth = aiohttp.BasicAuth(self.username, self.password)
                    self.session._default_auth = auth
                    return True
                else:
                    logger.error("Username and password required for basic auth")
                    return False
                    
            elif self.auth_type == 'oauth2':
                return await self._oauth_authenticate()
                
            else:
                logger.warning(f"Unknown auth type: {self.auth_type}")
                return True  # Assume no auth required
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
            
    async def _oauth_authenticate(self) -> bool:
        """Perform OAuth 2.0 authentication"""
        try:
            if self.oauth_token:
                # Use existing token
                self.headers['Authorization'] = f'Bearer {self.oauth_token}'
                return True
                
            if self.oauth_client_id and self.oauth_client_secret:
                # Get new token
                token_url = urljoin(self.base_url, '/oauth/token')
                data = {
                    'grant_type': 'client_credentials',
                    'client_id': self.oauth_client_id,
                    'client_secret': self.oauth_client_secret
                }
                
                async with self.session.post(token_url, data=data) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.oauth_token = token_data.get('access_token')
                        self.headers['Authorization'] = f'Bearer {self.oauth_token}'
                        return True
                        
            logger.error("OAuth credentials not available or invalid")
            return False
            
        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            return False
            
    async def _test_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            # Try to access a basic endpoint
            test_endpoints = ['/api/health', '/api/status', '/api/v1/health', '/health']
            
            for endpoint in test_endpoints:
                try:
                    url = urljoin(self.base_url, endpoint)
                    async with self.session.get(url) as response:
                        if response.status < 500:  # Accept any non-server error
                            return True
                except:
                    continue
                    
            # If health endpoints fail, try main device endpoint
            if isinstance(self.endpoints, dict) and 'devices' in self.endpoints:
                url = urljoin(self.base_url, self.endpoints['devices'])
                async with self.session.get(url) as response:
                    return response.status < 500
                    
            return False
            
        except Exception as e:
            logger.debug(f"Connectivity test failed: {e}")
            return False
            
    async def _health_check(self) -> None:
        """Check API connectivity and authentication"""
        if not await self._test_connectivity():
            raise ConnectionError("REST API health check failed")
            
    async def collect_metrics(self) -> List[Metric]:
        """Collect metrics via REST API"""
        metrics = []
        timestamp = datetime.utcnow()
        
        try:
            # Rate limiting check
            await self._check_rate_limit()
            
            # Collect based on device type and vendor
            if self.device.device_type == "cooling_unit":
                metrics.extend(await self._collect_cooling_metrics(timestamp))
            elif self.device.device_type == "power_unit":
                metrics.extend(await self._collect_power_metrics(timestamp))
            else:
                metrics.extend(await self._collect_generic_metrics(timestamp))
                
        except Exception as e:
            logger.error(f"Failed to collect REST API metrics from {self.device.name}: {e}")
            
        return metrics
        
    async def _collect_cooling_metrics(self, timestamp: datetime) -> List[Metric]:
        """Collect cooling system metrics"""
        metrics = []
        
        try:
            # Environmental metrics endpoint
            if isinstance(self.endpoints, dict):
                endpoint = self.endpoints.get('environmental', self.endpoints.get('measurements'))
            else:
                endpoint = '/api/v1/environmental'
                
            if endpoint:
                data = await self._make_request('GET', endpoint)
                if data:
                    metrics.extend(self._parse_environmental_data(data, timestamp))
                    
        except Exception as e:
            logger.error(f"Failed to collect cooling metrics: {e}")
            
        return metrics
        
    async def _collect_power_metrics(self, timestamp: datetime) -> List[Metric]:
        """Collect power system metrics"""
        metrics = []
        
        try:
            # Power metrics endpoint
            if isinstance(self.endpoints, dict):
                endpoint = self.endpoints.get('power_metrics', self.endpoints.get('measurements'))
            else:
                endpoint = '/api/v1/power'
                
            if endpoint:
                data = await self._make_request('GET', endpoint)
                if data:
                    metrics.extend(self._parse_power_data(data, timestamp))
                    
        except Exception as e:
            logger.error(f"Failed to collect power metrics: {e}")
            
        return metrics
        
    async def _collect_generic_metrics(self, timestamp: datetime) -> List[Metric]:
        """Collect generic device metrics"""
        metrics = []
        
        try:
            # Try sensors or measurements endpoint
            endpoints_to_try = [
                self.endpoints.get('sensors') if isinstance(self.endpoints, dict) else None,
                self.endpoints.get('measurements') if isinstance(self.endpoints, dict) else None,
                '/api/v1/sensors',
                '/api/v1/measurements',
                '/api/v1/data'
            ]
            
            for endpoint in endpoints_to_try:
                if endpoint:
                    try:
                        data = await self._make_request('GET', endpoint)
                        if data:
                            metrics.extend(self._parse_generic_data(data, timestamp))
                            break
                    except:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to collect generic metrics: {e}")
            
        return metrics
        
    def _parse_environmental_data(self, data: Any, timestamp: datetime) -> List[Metric]:
        """Parse environmental data from API response"""
        metrics = []
        
        try:
            # Handle different response formats
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
                elif 'measurements' in data:
                    data = data['measurements']
                    
            if isinstance(data, list):
                for item in data:
                    metrics.extend(self._parse_measurement_item(item, timestamp, 'environmental'))
            elif isinstance(data, dict):
                metrics.extend(self._parse_measurement_item(data, timestamp, 'environmental'))
                
        except Exception as e:
            logger.error(f"Failed to parse environmental data: {e}")
            
        return metrics
        
    def _parse_power_data(self, data: Any, timestamp: datetime) -> List[Metric]:
        """Parse power data from API response"""
        metrics = []
        
        try:
            # Handle different response formats
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
                elif 'measurements' in data:
                    data = data['measurements']
                    
            if isinstance(data, list):
                for item in data:
                    metrics.extend(self._parse_measurement_item(item, timestamp, 'power'))
            elif isinstance(data, dict):
                metrics.extend(self._parse_measurement_item(data, timestamp, 'power'))
                
        except Exception as e:
            logger.error(f"Failed to parse power data: {e}")
            
        return metrics
        
    def _parse_generic_data(self, data: Any, timestamp: datetime) -> List[Metric]:
        """Parse generic measurement data"""
        metrics = []
        
        try:
            # Handle different response formats
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
                elif 'sensors' in data:
                    data = data['sensors']
                elif 'measurements' in data:
                    data = data['measurements']
                    
            if isinstance(data, list):
                for item in data:
                    metrics.extend(self._parse_measurement_item(item, timestamp, 'generic'))
            elif isinstance(data, dict):
                metrics.extend(self._parse_measurement_item(data, timestamp, 'generic'))
                
        except Exception as e:
            logger.error(f"Failed to parse generic data: {e}")
            
        return metrics
        
    def _parse_measurement_item(self, item: Dict[str, Any], timestamp: datetime, category: str) -> List[Metric]:
        """Parse individual measurement item"""
        metrics = []
        
        try:
            # Common field mappings
            field_mappings = {
                'temperature': ['temperature', 'temp', 'temperatureValue'],
                'humidity': ['humidity', 'relativeHumidity', 'humidityValue'],
                'voltage': ['voltage', 'voltageValue', 'V'],
                'current': ['current', 'currentValue', 'A'],
                'power': ['power', 'powerValue', 'W', 'activePower'],
                'energy': ['energy', 'energyValue', 'kWh', 'totalEnergy'],
                'frequency': ['frequency', 'freq', 'Hz'],
                'pressure': ['pressure', 'pressureValue'],
                'flow_rate': ['flowRate', 'flow', 'airFlow']
            }
            
            for metric_name, field_names in field_mappings.items():
                for field_name in field_names:
                    if field_name in item:
                        try:
                            value = float(item[field_name])
                            unit = item.get(f"{field_name}Unit", item.get("unit", ""))
                            
                            # Create tags
                            tags = {
                                "source": "rest_api",
                                "device_type": str(self.device.device_type),
                                "category": category,
                                "vendor": self.vendor
                            }
                            
                            # Add additional tags from item
                            if 'id' in item:
                                tags['sensor_id'] = str(item['id'])
                            if 'name' in item:
                                tags['sensor_name'] = str(item['name'])
                            if 'location' in item:
                                tags['location'] = str(item['location'])
                                
                            metrics.append(Metric(
                                device_id=self.device.id,
                                metric_name=metric_name,
                                metric_type=MetricType.GAUGE,
                                value=value,
                                unit=unit,
                                timestamp=timestamp,
                                tags=tags
                            ))
                            break  # Found this metric, move to next
                        except (ValueError, TypeError):
                            continue
                            
        except Exception as e:
            logger.error(f"Failed to parse measurement item: {e}")
            
        return metrics
        
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Any]:
        """Make HTTP request with retry logic"""
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                await self._check_rate_limit()
                
                if method.upper() == 'GET':
                    async with self.session.get(url) as response:
                        return await self._handle_response(response)
                elif method.upper() == 'POST':
                    async with self.session.post(url, json=data) as response:
                        return await self._handle_response(response)
                elif method.upper() == 'PUT':
                    async with self.session.put(url, json=data) as response:
                        return await self._handle_response(response)
                elif method.upper() == 'DELETE':
                    async with self.session.delete(url) as response:
                        return await self._handle_response(response)
                        
            except aiohttp.ClientError as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
                    
        return None
        
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Optional[Any]:
        """Handle HTTP response"""
        if response.status == 200:
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                return await response.json()
            else:
                return await response.text()
        elif response.status == 401:
            # Try to re-authenticate
            if await self._authenticate():
                raise aiohttp.ClientError("Authentication required - retry with new credentials")
            else:
                raise aiohttp.ClientError("Authentication failed")
        elif response.status == 429:
            # Rate limited
            retry_after = int(response.headers.get('Retry-After', 60))
            await asyncio.sleep(retry_after)
            raise aiohttp.ClientError("Rate limited - retrying")
        else:
            logger.warning(f"HTTP {response.status}: {await response.text()}")
            return None
            
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.utcnow()
        
        # Remove old timestamps
        cutoff = now.timestamp() - self.rate_limit_window
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
        
        # Check if we're over the limit
        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = self.rate_limit_window - (now.timestamp() - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        # Add current timestamp
        self.request_timestamps.append(now.timestamp())
        
    async def execute_command(self, command: Command) -> CommandResult:
        """Execute REST API command"""
        start_time = datetime.utcnow()
        
        try:
            command_type = command.command_type
            parameters = command.parameters
            
            if command_type == "api_request":
                method = parameters.get('method', 'GET').upper()
                endpoint = parameters.get('endpoint')
                data = parameters.get('data')
                
                if not endpoint:
                    return CommandResult(
                        command_id=command.id,
                        success=False,
                        error_message="Missing endpoint for API request"
                    )
                    
                result = await self._make_request(method, endpoint, data)
                
                return CommandResult(
                    command_id=command.id,
                    success=True,
                    result_data={"response": result},
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
        """Discover device inventory via REST API"""
        inventory = []
        
        try:
            # Try inventory endpoint
            if isinstance(self.endpoints, dict) and 'inventory' in self.endpoints:
                endpoint = self.endpoints['inventory']
            else:
                endpoint = '/api/v1/inventory'
                
            data = await self._make_request('GET', endpoint)
            if data:
                inventory.extend(self._parse_inventory_data(data))
                
            # Try devices endpoint for additional info
            if isinstance(self.endpoints, dict) and 'devices' in self.endpoints:
                endpoint = self.endpoints['devices']
            else:
                endpoint = '/api/v1/devices'
                
            device_data = await self._make_request('GET', endpoint)
            if device_data:
                inventory.extend(self._parse_device_data(device_data))
                
        except Exception as e:
            logger.error(f"Failed to discover inventory for {self.device.name}: {e}")
            
        return inventory
        
    def _parse_inventory_data(self, data: Any) -> List[Dict[str, Any]]:
        """Parse inventory data from API response"""
        inventory = []
        
        try:
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
                
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        inventory.append({
                            'category': item.get('category', 'hardware'),
                            'name': item.get('name', item.get('description', 'Unknown')),
                            'version': item.get('version', item.get('firmwareVersion')),
                            'manufacturer': item.get('manufacturer', item.get('vendor')),
                            'model': item.get('model', item.get('productName')),
                            'serial_number': item.get('serialNumber', item.get('serial')),
                            'source': 'rest_api',
                            'metadata': item
                        })
                        
        except Exception as e:
            logger.error(f"Failed to parse inventory data: {e}")
            
        return inventory
        
    def _parse_device_data(self, data: Any) -> List[Dict[str, Any]]:
        """Parse device data for inventory information"""
        inventory = []
        
        try:
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
                
            if isinstance(data, list):
                for device in data:
                    if isinstance(device, dict):
                        inventory.append({
                            'category': 'device_info',
                            'name': 'Device Information',
                            'value': device.get('name', device.get('description', 'Unknown Device')),
                            'source': 'rest_api',
                            'metadata': device
                        })
                        
        except Exception as e:
            logger.error(f"Failed to parse device data: {e}")
            
        return inventory
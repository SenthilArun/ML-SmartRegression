"""
Configuration management for the AIMS system
"""

import os
from typing import Dict, List, Optional, Any
from pydantic import BaseSettings, Field
import yaml


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    
    influxdb_url: str = Field(default="http://localhost:8086", env="INFLUXDB_URL")
    influxdb_token: Optional[str] = Field(default=None, env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="aims", env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="metrics", env="INFLUXDB_BUCKET")
    
    mongodb_url: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    mongodb_database: str = Field(default="aims", env="MONGODB_DATABASE")
    
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")


class AlertingConfig(BaseSettings):
    """Alerting configuration"""
    
    smtp_server: Optional[str] = Field(default=None, env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")
    
    slack_token: Optional[str] = Field(default=None, env="SLACK_TOKEN")
    slack_channel: Optional[str] = Field(default="#alerts", env="SLACK_CHANNEL")
    
    pagerduty_api_key: Optional[str] = Field(default=None, env="PAGERDUTY_API_KEY")
    pagerduty_service_id: Optional[str] = Field(default=None, env="PAGERDUTY_SERVICE_ID")


class SecurityConfig(BaseSettings):
    """Security configuration"""
    
    secret_key: str = Field(default="your-secret-key-change-this", env="SECRET_KEY")
    api_key_enabled: bool = Field(default=True, env="API_KEY_ENABLED")
    oauth_enabled: bool = Field(default=False, env="OAUTH_ENABLED")
    oauth_provider: Optional[str] = Field(default=None, env="OAUTH_PROVIDER")
    oauth_client_id: Optional[str] = Field(default=None, env="OAUTH_CLIENT_ID")
    oauth_client_secret: Optional[str] = Field(default=None, env="OAUTH_CLIENT_SECRET")
    
    tls_cert_path: Optional[str] = Field(default=None, env="TLS_CERT_PATH")
    tls_key_path: Optional[str] = Field(default=None, env="TLS_KEY_PATH")


class MonitoringConfig(BaseSettings):
    """Monitoring configuration"""
    
    collection_interval: int = Field(default=30, env="COLLECTION_INTERVAL")  # seconds
    batch_size: int = Field(default=100, env="BATCH_SIZE")
    retention_days: int = Field(default=90, env="RETENTION_DAYS")
    
    gpu_monitoring_enabled: bool = Field(default=True, env="GPU_MONITORING_ENABLED")
    dcim_monitoring_enabled: bool = Field(default=True, env="DCIM_MONITORING_ENABLED")
    k8s_monitoring_enabled: bool = Field(default=False, env="K8S_MONITORING_ENABLED")


class DCIMConfig(BaseSettings):
    """DCIM integration configuration"""
    
    snmp_enabled: bool = Field(default=True, env="SNMP_ENABLED")
    snmp_community: str = Field(default="public", env="SNMP_COMMUNITY")
    snmp_version: str = Field(default="2c", env="SNMP_VERSION")
    snmp_timeout: int = Field(default=5, env="SNMP_TIMEOUT")
    
    modbus_enabled: bool = Field(default=True, env="MODBUS_ENABLED")
    modbus_timeout: int = Field(default=5, env="MODBUS_TIMEOUT")
    
    rest_api_enabled: bool = Field(default=True, env="REST_API_ENABLED")
    rest_api_timeout: int = Field(default=30, env="REST_API_TIMEOUT")


class KubernetesConfig(BaseSettings):
    """Kubernetes integration configuration"""
    
    kubeconfig_path: Optional[str] = Field(default=None, env="KUBECONFIG_PATH")
    namespace: str = Field(default="default", env="K8S_NAMESPACE")
    service_account_token: Optional[str] = Field(default=None, env="K8S_SERVICE_ACCOUNT_TOKEN")
    api_server_url: Optional[str] = Field(default=None, env="K8S_API_SERVER_URL")


class WebConfig(BaseSettings):
    """Web interface configuration"""
    
    host: str = Field(default="0.0.0.0", env="WEB_HOST")
    port: int = Field(default=8000, env="WEB_PORT")
    workers: int = Field(default=4, env="WEB_WORKERS")
    reload: bool = Field(default=False, env="WEB_RELOAD")
    
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    static_files_path: str = Field(default="static", env="STATIC_FILES_PATH")
    templates_path: str = Field(default="templates", env="TEMPLATES_PATH")


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="{time} | {level} | {name} | {message}", env="LOG_FORMAT")
    rotation: str = Field(default="10 MB", env="LOG_ROTATION")
    retention: str = Field(default="10 days", env="LOG_RETENTION")
    
    file_path: Optional[str] = Field(default="logs/aims.log", env="LOG_FILE_PATH")
    json_format: bool = Field(default=False, env="LOG_JSON_FORMAT")


class AIMSConfig(BaseSettings):
    """Main AIMS configuration"""
    
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    alerting: AlertingConfig = AlertingConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    dcim: DCIMConfig = DCIMConfig()
    kubernetes: KubernetesConfig = KubernetesConfig()
    web: WebConfig = WebConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Device configurations
    devices: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_config(config_path: Optional[str] = None) -> AIMSConfig:
    """Load configuration from file and environment variables"""
    
    config_data = {}
    
    # Load from YAML file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    elif os.path.exists("config/config.yaml"):
        with open("config/config.yaml", 'r') as f:
            config_data = yaml.safe_load(f) or {}
    
    # Create configuration with loaded data
    # Environment variables will override file settings
    return AIMSConfig(**config_data)


def save_config(config: AIMSConfig, config_path: str = "config/config.yaml") -> None:
    """Save configuration to YAML file"""
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Convert to dict and remove None values
    config_dict = config.dict(exclude_none=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Global configuration instance
config: Optional[AIMSConfig] = None


def get_config() -> AIMSConfig:
    """Get the global configuration instance"""
    global config
    if config is None:
        config = load_config()
    return config


def init_config(config_path: Optional[str] = None) -> AIMSConfig:
    """Initialize the global configuration"""
    global config
    config = load_config(config_path)
    return config
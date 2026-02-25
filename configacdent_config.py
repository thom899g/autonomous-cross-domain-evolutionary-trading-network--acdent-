"""
Configuration management for ACDENT system.
Centralized configuration with environment-aware settings and validation.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingMode(Enum):
    """Operational modes for the trading system."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class RiskLevel(Enum):
    """Risk management levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class FirebaseConfig:
    """Firebase configuration with validation."""
    project_id: str = field(default_factory=lambda: os.getenv("FIREBASE_PROJECT_ID", ""))
    credentials_path: str = field(default_factory=lambda: os.getenv("FIREBASE_CREDENTIALS", "firebase_credentials.json"))
    database_url: str = field(default_factory=lambda: os.getenv("FIREBASE_DATABASE_URL", ""))
    
    def validate(self) -> bool:
        """Validate Firebase configuration."""
        if not self.project_id:
            logging.error("Firebase project_id is not configured")
            return False
        if not os.path.exists(self.credentials_path):
            logging.error(f"Firebase credentials file not found: {self.credentials_path}")
            return False
        return True

@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    mode: TradingMode = TradingMode.PAPER
    risk_level: RiskLevel = RiskLevel.MODERATE
    max_position_size: float = 0.1  # 10% of portfolio per position
    stop_loss_pct: float = 2.0  # 2% stop loss
    take_profit_pct: float = 5.0  # 5% take profit
    max_daily_loss: float = 3.0  # 3% max daily loss
    trade_cooldown_minutes: int = 5  # Minutes between trades
    
    def validate(self) -> List[str]:
        """Validate trading configuration and return any errors."""
        errors = []
        if self.max_position_size <= 0 or self.max_position_size > 1:
            errors.append("max_position_size must be between 0 and 1")
        if self.stop_loss_pct <= 0:
            errors.append("stop_loss_pct must be positive")
        if self.take_profit_pct <= self.stop_loss_pct:
            errors.append("take_profit_pct must be greater than stop_loss_pct")
        return errors

@dataclass
class ModelConfig:
    """Neural network model configuration."""
    attention_heads: int = 8
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    sequence_length: int = 100  # Number of time steps for LSTM/Attention
    
    def get_model_params(self) -> Dict[str, Any]:
        """Return model parameters as dictionary."""
        return {
            "attention_heads": self.attention_heads,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length
        }

class ACDENTConfig:
    """Main configuration class for ACDENT system."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.firebase = FirebaseConfig()
        self.trading = TradingConfig()
        self.model = ModelConfig()
        self.exchanges: List[str] = ["binance", "coinbase", "kraken"]
        self.data_update_interval: int = 60  # seconds
        self.evolution_interval: int = 3600  # seconds (1 hour)
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Initialize logging
        self._setup_logging()
    
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                
            # Update configurations
            if 'trading' in data:
                self.trading.mode = TradingMode(data['trading'].get('mode', 'paper'))
                self.trading.risk_level = RiskLevel(data['trading'].get('risk_level', 'moderate'))
                self.trading.max_position_size = data['trading'].get('max_position_size', 0.1)
                
            if 'model' in data:
                self.model.hidden_layers = data['model'].get('hidden_layers', [256, 128, 64])
                self.model.learning_rate = data['model'].get('learning_rate', 0.001)
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.error(f"Error loading config file {config_file}: {e}")
    
    def _setup_logging(self) -> None:
        """Configure logging based on log level."""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('acdent
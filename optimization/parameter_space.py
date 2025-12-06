"""
Parameter Space Definition for IB Breakout Optimizer.

Defines all optimizable parameters with their ranges, types, and constraints.
Used by both grid search and Bayesian optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import itertools
import numpy as np
import yaml
from pathlib import Path


class ParameterType(Enum):
    """Type of parameter."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CATEGORICAL = "categorical"
    TIME = "time"


@dataclass
class ParameterConfig:
    """
    Configuration for a single optimizable parameter.

    Attributes:
        name: Parameter name (must match StrategyParams attribute)
        param_type: Type of parameter
        default: Default value
        min_value: Minimum value (for int/float)
        max_value: Maximum value (for int/float)
        step: Step size for grid search (for int/float)
        choices: List of choices (for categorical)
        enabled: Whether to include in optimization
        condition: Optional condition for this parameter
    """
    name: str
    param_type: ParameterType
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    enabled: bool = True
    condition: Optional[str] = None  # e.g., "trailing_stop_enabled == True"

    def get_grid_values(self) -> List[Any]:
        """Get all values for grid search."""
        if not self.enabled:
            return [self.default]

        if self.param_type == ParameterType.BOOL:
            return [True, False]

        elif self.param_type == ParameterType.CATEGORICAL:
            return self.choices or [self.default]

        elif self.param_type == ParameterType.INT:
            if self.min_value is not None and self.max_value is not None:
                step = int(self.step) if self.step else 1
                return list(range(int(self.min_value), int(self.max_value) + 1, step))
            return [self.default]

        elif self.param_type == ParameterType.FLOAT:
            if self.min_value is not None and self.max_value is not None:
                step = self.step if self.step else 0.1
                values = np.arange(self.min_value, self.max_value + step/2, step)
                return [round(v, 4) for v in values]
            return [self.default]

        elif self.param_type == ParameterType.TIME:
            # Time parameters handled specially
            return self.choices or [self.default]

        return [self.default]

    def get_optuna_config(self) -> Dict[str, Any]:
        """Get configuration for Optuna suggest methods."""
        if self.param_type == ParameterType.BOOL:
            return {"type": "categorical", "choices": [True, False]}

        elif self.param_type == ParameterType.CATEGORICAL:
            return {"type": "categorical", "choices": self.choices}

        elif self.param_type == ParameterType.INT:
            return {
                "type": "int",
                "low": int(self.min_value),
                "high": int(self.max_value),
                "step": int(self.step) if self.step else 1
            }

        elif self.param_type == ParameterType.FLOAT:
            return {
                "type": "float",
                "low": self.min_value,
                "high": self.max_value,
                "step": self.step
            }

        elif self.param_type == ParameterType.TIME:
            return {"type": "categorical", "choices": self.choices}

        return {"type": "categorical", "choices": [self.default]}


class ParameterSpace:
    """
    Defines the complete parameter space for optimization.

    Includes all parameters from StrategyParams with their optimization ranges.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize parameter space.

        Args:
            config_file: Optional path to YAML config file
        """
        self.parameters: Dict[str, ParameterConfig] = {}
        self._setup_default_parameters()

        if config_file:
            self.load_from_yaml(config_file)

    def _setup_default_parameters(self):
        """Setup default parameter configurations."""

        # IB Parameters
        self.add_parameter(ParameterConfig(
            name="ib_duration_minutes",
            param_type=ParameterType.INT,
            default=30,
            min_value=15,
            max_value=60,
            step=5,
            enabled=True
        ))

        self.add_parameter(ParameterConfig(
            name="ib_proximity_percent",
            param_type=ParameterType.FLOAT,
            default=0.0,
            min_value=0.0,
            max_value=2.0,
            step=0.25,
            enabled=False  # Disabled by default
        ))

        # Entry Parameters
        self.add_parameter(ParameterConfig(
            name="trade_direction",
            param_type=ParameterType.CATEGORICAL,
            default="both",
            choices=["both", "long_only", "short_only"],
            enabled=True
        ))

        self.add_parameter(ParameterConfig(
            name="trading_start_time",
            param_type=ParameterType.TIME,
            default="09:00",
            choices=["09:00", "09:30", "09:45", "10:00"],
            enabled=False
        ))

        self.add_parameter(ParameterConfig(
            name="trading_end_time",
            param_type=ParameterType.TIME,
            default="15:00",
            choices=["14:00", "14:30", "15:00", "15:30"],
            enabled=False
        ))

        # Position Sizing
        self.add_parameter(ParameterConfig(
            name="fixed_share_size",
            param_type=ParameterType.INT,
            default=100,
            min_value=50,
            max_value=500,
            step=50,
            enabled=False
        ))

        # Exit Parameters
        self.add_parameter(ParameterConfig(
            name="profit_target_percent",
            param_type=ParameterType.FLOAT,
            default=0.5,
            min_value=0.3,
            max_value=2.0,
            step=0.1,
            enabled=True
        ))

        self.add_parameter(ParameterConfig(
            name="stop_loss_type",
            param_type=ParameterType.CATEGORICAL,
            default="opposite_ib",
            choices=["opposite_ib", "match_target"],
            enabled=True
        ))

        # Advanced Exits
        self.add_parameter(ParameterConfig(
            name="trailing_stop_enabled",
            param_type=ParameterType.BOOL,
            default=False,
            enabled=True
        ))

        self.add_parameter(ParameterConfig(
            name="trailing_stop_atr_mult",
            param_type=ParameterType.FLOAT,
            default=2.0,
            min_value=1.0,
            max_value=4.0,
            step=0.5,
            enabled=True,
            condition="trailing_stop_enabled == True"
        ))

        self.add_parameter(ParameterConfig(
            name="break_even_enabled",
            param_type=ParameterType.BOOL,
            default=False,
            enabled=True
        ))

        self.add_parameter(ParameterConfig(
            name="break_even_pct",
            param_type=ParameterType.FLOAT,
            default=0.7,
            min_value=0.5,
            max_value=0.9,
            step=0.1,
            enabled=True,
            condition="break_even_enabled == True"
        ))

        self.add_parameter(ParameterConfig(
            name="max_bars_enabled",
            param_type=ParameterType.BOOL,
            default=False,
            enabled=False
        ))

        self.add_parameter(ParameterConfig(
            name="max_bars",
            param_type=ParameterType.INT,
            default=60,
            min_value=15,
            max_value=120,
            step=15,
            enabled=False,
            condition="max_bars_enabled == True"
        ))

        self.add_parameter(ParameterConfig(
            name="eod_exit_time",
            param_type=ParameterType.TIME,
            default="15:55",
            choices=["15:30", "15:45", "15:55"],
            enabled=False
        ))

        # Filter Parameters
        self.add_parameter(ParameterConfig(
            name="use_qqq_filter",
            param_type=ParameterType.BOOL,
            default=False,
            enabled=True  # Enabled - optimizer will load QQQ data when needed
        ))

        self.add_parameter(ParameterConfig(
            name="min_ib_range_percent",
            param_type=ParameterType.FLOAT,
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            enabled=True
        ))

        self.add_parameter(ParameterConfig(
            name="max_ib_range_percent",
            param_type=ParameterType.FLOAT,
            default=10.0,
            min_value=1.0,
            max_value=5.0,
            step=0.5,
            enabled=True
        ))

        self.add_parameter(ParameterConfig(
            name="max_breakout_time",
            param_type=ParameterType.TIME,
            default="14:00",
            choices=["11:00", "12:00", "13:00", "14:00", "15:00"],
            enabled=True
        ))

        # Day of Week Filters
        for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            self.add_parameter(ParameterConfig(
                name=f"trade_{day}",
                param_type=ParameterType.BOOL,
                default=True,
                enabled=False  # Enable for day-of-week optimization
            ))

    def add_parameter(self, config: ParameterConfig):
        """Add a parameter configuration."""
        self.parameters[config.name] = config

    def get_parameter(self, name: str) -> Optional[ParameterConfig]:
        """Get parameter configuration by name."""
        return self.parameters.get(name)

    def enable_parameter(self, name: str, enabled: bool = True):
        """Enable or disable a parameter for optimization."""
        if name in self.parameters:
            self.parameters[name].enabled = enabled

    def enable_parameters(self, names: List[str]):
        """Enable multiple parameters."""
        for name in names:
            self.enable_parameter(name, True)

    def disable_all(self):
        """Disable all parameters."""
        for param in self.parameters.values():
            param.enabled = False

    def enable_all(self):
        """Enable all parameters."""
        for param in self.parameters.values():
            param.enabled = True

    def get_enabled_parameters(self) -> Dict[str, ParameterConfig]:
        """Get only enabled parameters."""
        return {k: v for k, v in self.parameters.items() if v.enabled}

    def get_grid_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for grid search.

        Returns:
            List of parameter dictionaries
        """
        enabled = self.get_enabled_parameters()

        if not enabled:
            return [self.get_defaults()]

        # Get values for each enabled parameter
        param_names = list(enabled.keys())
        param_values = [enabled[name].get_grid_values() for name in param_names]

        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_values):
            combo = dict(zip(param_names, values))

            # Add defaults for disabled parameters
            for name, config in self.parameters.items():
                if name not in combo:
                    combo[name] = config.default

            # Apply conditions
            combo = self._apply_conditions(combo)
            combinations.append(combo)

        return combinations

    def _apply_conditions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conditional logic to parameters."""
        result = params.copy()

        # If trailing_stop_enabled is False, use default for trailing_stop_atr_mult
        if not result.get("trailing_stop_enabled", False):
            result["trailing_stop_atr_mult"] = self.parameters["trailing_stop_atr_mult"].default

        # If break_even_enabled is False, use default for break_even_pct
        if not result.get("break_even_enabled", False):
            result["break_even_pct"] = self.parameters["break_even_pct"].default

        # If max_bars_enabled is False, use default for max_bars
        if not result.get("max_bars_enabled", False):
            result["max_bars"] = self.parameters["max_bars"].default

        return result

    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return {name: config.default for name, config in self.parameters.items()}

    def get_grid_size(self) -> int:
        """Calculate total number of grid combinations."""
        enabled = self.get_enabled_parameters()
        if not enabled:
            return 1

        size = 1
        for config in enabled.values():
            size *= len(config.get_grid_values())
        return size

    def load_from_yaml(self, filepath: str):
        """Load parameter configurations from YAML file."""
        path = Path(filepath)
        if not path.exists():
            return

        with open(path) as f:
            config = yaml.safe_load(f)

        # Update parameters from config
        for section in ['entry', 'exit', 'filters', 'portfolio']:
            if section in config:
                for name, settings in config[section].items():
                    if name in self.parameters:
                        param = self.parameters[name]

                        if 'range' in settings:
                            param.min_value = settings['range'][0]
                            param.max_value = settings['range'][1]

                        if 'step' in settings:
                            param.step = settings['step']

                        if 'values' in settings:
                            param.choices = settings['values']

                        if 'default' in settings:
                            param.default = settings['default']

                        if 'enabled' in settings:
                            param.enabled = settings.get('enabled', True)

    def save_to_yaml(self, filepath: str):
        """Save current parameter configurations to YAML."""
        config = {}

        for name, param in self.parameters.items():
            param_config = {
                'type': param.param_type.value,
                'default': param.default,
                'enabled': param.enabled
            }

            if param.min_value is not None:
                param_config['range'] = [param.min_value, param.max_value]

            if param.step is not None:
                param_config['step'] = param.step

            if param.choices:
                param_config['values'] = param.choices

            config[name] = param_config

        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def summary(self) -> str:
        """Get summary of parameter space."""
        lines = ["Parameter Space Summary", "=" * 50]

        enabled = self.get_enabled_parameters()
        disabled = {k: v for k, v in self.parameters.items() if not v.enabled}

        lines.append(f"\nEnabled Parameters ({len(enabled)}):")
        for name, config in enabled.items():
            values = config.get_grid_values()
            lines.append(f"  {name}: {len(values)} values")
            if len(values) <= 10:
                lines.append(f"    -> {values}")

        lines.append(f"\nDisabled Parameters ({len(disabled)}):")
        for name in disabled:
            lines.append(f"  {name}")

        lines.append(f"\nTotal Grid Combinations: {self.get_grid_size():,}")

        return "\n".join(lines)


# Preset configurations for common optimization scenarios
# Each preset targets a specific combination count for predictable runtime
OPTIMIZATION_PRESETS = {
    "quick": {
        "description": "Quick optimization - 96 combinations, ~3 sec",
        "enabled": [
            "ib_duration_minutes",
            "profit_target_percent",
            "stop_loss_type",
            "trade_direction"
        ],
        "step_overrides": {
            "ib_duration_minutes": 15,          # 15, 30, 45, 60 (4 values)
            "profit_target_percent": 0.5,       # 0.5, 1.0, 1.5, 2.0 (4 values)
        }
        # 4 * 4 * 2 * 3 = 96 combinations
    },
    "standard": {
        "description": "Standard optimization - 288 combinations, ~5 sec (recommended)",
        "enabled": [
            "ib_duration_minutes",
            "profit_target_percent",
            "stop_loss_type",
            "trade_direction",
            "min_ib_range_percent"
        ],
        "step_overrides": {
            "ib_duration_minutes": 15,          # 15, 30, 45, 60 (4 values)
            "profit_target_percent": 0.5,       # 0.5, 1.0, 1.5, 2.0 (4 values)
            "min_ib_range_percent": 0.5,        # 0.0, 0.5, 1.0 (3 values)
        }
        # 4 * 4 * 2 * 3 * 3 = 288 combinations
    },
    "full": {
        "description": "Full optimization - 1152 combinations, ~15 sec",
        "enabled": [
            "ib_duration_minutes",
            "profit_target_percent",
            "stop_loss_type",
            "trade_direction",
            "trailing_stop_enabled",
            "break_even_enabled",
            "min_ib_range_percent"
        ],
        "step_overrides": {
            "ib_duration_minutes": 15,          # 4 values
            "profit_target_percent": 0.5,       # 4 values
            "min_ib_range_percent": 0.5,        # 3 values
        }
        # 4 * 4 * 2 * 3 * 2 * 2 * 3 = 1,152 combinations
    },
    "thorough": {
        "description": "Thorough optimization - ~2300 combinations, ~30 sec",
        "enabled": [
            "ib_duration_minutes",
            "profit_target_percent",
            "stop_loss_type",
            "trade_direction",
            "trailing_stop_enabled",
            "break_even_enabled",
            "min_ib_range_percent"
        ],
        "step_overrides": {
            "ib_duration_minutes": 15,          # 4 values: 15, 30, 45, 60
            "profit_target_percent": 0.2,       # 9 values: 0.3 to 1.9
            "min_ib_range_percent": 0.5,        # 3 values: 0.0, 0.5, 1.0
        }
        # 4 * 9 * 2 * 3 * 2 * 2 * 3 = 2,592 combinations
    }
}


def create_parameter_space(preset: str = "standard") -> ParameterSpace:
    """
    Create a parameter space with a preset configuration.

    Args:
        preset: Name of preset ("quick", "standard", "full", "fast_full", "exits_only")

    Returns:
        Configured ParameterSpace
    """
    space = ParameterSpace()
    space.disable_all()

    if preset in OPTIMIZATION_PRESETS:
        preset_config = OPTIMIZATION_PRESETS[preset]
        space.enable_parameters(preset_config["enabled"])

        # Apply step overrides if present (for fast_full preset)
        if "step_overrides" in preset_config:
            for param_name, new_step in preset_config["step_overrides"].items():
                if param_name in space.parameters:
                    space.parameters[param_name].step = new_step

    return space


if __name__ == "__main__":
    # Test parameter space
    print("Testing Parameter Space\n")

    # Create with standard preset
    space = create_parameter_space("standard")
    print(space.summary())

    # Get grid combinations
    combos = space.get_grid_combinations()
    print(f"\nFirst 5 combinations:")
    for i, combo in enumerate(combos[:5]):
        print(f"  {i+1}. {combo}")

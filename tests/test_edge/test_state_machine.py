import pytest
import time
from unittest.mock import MagicMock

from src.edge.state_machine import (
    DeviceStateType,
    Action,
    DeviceState,
    BatteryState,
    DeviceStateMachine,
    StateTransition,
)


class TestDeviceState:

    def test_default_values(self):
        state = DeviceState(device_id='dev_001', device_type='generic')

        assert state.device_id == 'dev_001'
        assert state.device_type == 'generic'
        assert state.state_type == DeviceStateType.IDLE
        assert state.power_current == 0.0
        assert state.is_healthy is True

    def test_time_in_state(self):
        state = DeviceState(device_id='dev_001', device_type='generic')

        assert state.time_in_state < 1.0

        time.sleep(0.1)
        assert state.time_in_state >= 0.1

    def test_time_since_signal(self):
        state = DeviceState(device_id='dev_001', device_type='generic')

        assert state.time_since_signal is None

        state.last_signal_at = time.time()
        time.sleep(0.1)
        assert state.time_since_signal is not None
        assert state.time_since_signal >= 0.1


class TestBatteryState:

    def test_default_values(self):
        state = BatteryState(device_id='bat_001', device_type='battery')

        assert state.soc == 0.5
        assert state.capacity_kwh == 10.0
        assert state.max_charge_kw == 5.0
        assert state.charge_efficiency == 0.95

    def test_post_init_sets_device_type(self):
        state = BatteryState(device_id='bat_001', device_type='wrong')

        assert state.device_type == 'battery'


class TestStateTransition:

    def test_creation(self):
        transition = StateTransition(
            from_state=DeviceStateType.IDLE,
            to_state=DeviceStateType.DECIDING,
            trigger='signal_received',
        )

        assert transition.from_state == DeviceStateType.IDLE
        assert transition.to_state == DeviceStateType.DECIDING
        assert transition.trigger == 'signal_received'
        assert transition.timestamp > 0

    def test_explicit_timestamp(self):
        transition = StateTransition(
            from_state=DeviceStateType.IDLE,
            to_state=DeviceStateType.DECIDING,
            trigger='test',
            timestamp=12345.0,
        )

        assert transition.timestamp == 12345.0


class TestDeviceStateTypeEnum:

    def test_all_states_defined(self):
        states = list(DeviceStateType)

        assert DeviceStateType.IDLE in states
        assert DeviceStateType.DECIDING in states
        assert DeviceStateType.RESPONDING in states
        assert DeviceStateType.OFFLINE in states


class TestActionEnum:

    def test_all_actions_defined(self):
        actions = list(Action)

        assert Action.HOLD in actions
        assert Action.CHARGE in actions
        assert Action.DISCHARGE in actions
        assert Action.STOP in actions

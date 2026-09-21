from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Callable, Dict, List, Any
import time


class DeviceStateType(Enum):
    IDLE = auto()
    DECIDING = auto()
    RESPONDING = auto()
    OFFLINE = auto()


class Action(Enum):
    HOLD = auto()
    CHARGE = auto()
    DISCHARGE = auto()
    STOP = auto()


@dataclass
class DeviceState:
    device_id: str
    device_type: str

    state_type: DeviceStateType = DeviceStateType.IDLE

    power_current: float = 0.0
    power_target: float = 0.0

    state_entered_at: float = field(default_factory=time.time)
    last_signal_at: Optional[float] = None
    last_response_at: Optional[float] = None

    current_action: Action = Action.HOLD
    response_duration: float = 0.0

    is_healthy: bool = True
    fault_code: Optional[str] = None

    @property
    def time_in_state(self) -> float:
        return time.time() - self.state_entered_at

    @property
    def time_since_signal(self) -> Optional[float]:
        if self.last_signal_at is None:
            return None
        return time.time() - self.last_signal_at


@dataclass
class BatteryState(DeviceState):
    soc: float = 0.5
    capacity_kwh: float = 10.0
    max_charge_kw: float = 5.0
    max_discharge_kw: float = 5.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    temperature: float = 25.0
    daily_cycles: int = 0

    def __post_init__(self):
        self.device_type = 'battery'


class StateTransition:

    def __init__(
        self,
        from_state: DeviceStateType,
        to_state: DeviceStateType,
        trigger: str,
        timestamp: Optional[float] = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger
        self.timestamp = timestamp or time.time()


class DeviceStateMachine(ABC):

    VALID_TRANSITIONS = {
        (DeviceStateType.IDLE, DeviceStateType.DECIDING): 'signal_received',
        (DeviceStateType.IDLE, DeviceStateType.OFFLINE): 'fault',
        (DeviceStateType.DECIDING, DeviceStateType.RESPONDING): 'decision_respond',
        (DeviceStateType.DECIDING, DeviceStateType.IDLE): 'decision_hold',
        (DeviceStateType.DECIDING, DeviceStateType.OFFLINE): 'fault',
        (DeviceStateType.RESPONDING, DeviceStateType.IDLE): 'response_complete',
        (DeviceStateType.RESPONDING, DeviceStateType.OFFLINE): 'fault',
        (DeviceStateType.OFFLINE, DeviceStateType.IDLE): 'recovery',
    }

    def __init__(self, state: DeviceState):
        self.state = state
        self.transition_history: List[StateTransition] = []
        self._callbacks: Dict[str, List[Callable]] = {
            'on_enter': [],
            'on_exit': [],
            'on_transition': [],
        }

    @property
    def current_state(self) -> DeviceStateType:
        return self.state.state_type

    def can_transition(self, to_state: DeviceStateType) -> bool:
        return (self.current_state, to_state) in self.VALID_TRANSITIONS

    def transition_to(self, to_state: DeviceStateType, trigger: str) -> bool:
        if not self.can_transition(to_state):
            return False

        transition = StateTransition(
            from_state=self.current_state,
            to_state=to_state,
            trigger=trigger,
        )

        for callback in self._callbacks['on_exit']:
            callback(self.current_state)

        old_state = self.current_state
        self.state.state_type = to_state
        self.state.state_entered_at = time.time()
        self.transition_history.append(transition)

        for callback in self._callbacks['on_enter']:
            callback(to_state)

        for callback in self._callbacks['on_transition']:
            callback(transition)

        return True

    def register_callback(self, event: str, callback: Callable) -> None:
        if event not in self._callbacks:
            raise ValueError(f"Unknown event: {event}")
        self._callbacks[event].append(callback)

    def on_signal_received(self) -> None:
        self.state.last_signal_at = time.time()

        if self.current_state == DeviceStateType.IDLE:
            self.transition_to(DeviceStateType.DECIDING, 'signal_received')
        elif self.current_state == DeviceStateType.OFFLINE:
            self.transition_to(DeviceStateType.IDLE, 'recovery')
            self.transition_to(DeviceStateType.DECIDING, 'signal_received')

    def on_decision_made(self, action: Action) -> None:
        if self.current_state != DeviceStateType.DECIDING:
            return

        self.state.current_action = action

        if action == Action.HOLD:
            self.transition_to(DeviceStateType.IDLE, 'decision_hold')
        else:
            self.transition_to(DeviceStateType.RESPONDING, 'decision_respond')
            self.state.last_response_at = time.time()

    def on_response_complete(self) -> None:
        if self.current_state == DeviceStateType.RESPONDING:
            self.state.response_duration = time.time() - (self.state.last_response_at or time.time())
            self.state.current_action = Action.HOLD
            self.transition_to(DeviceStateType.IDLE, 'response_complete')

    def on_fault(self, fault_code: str) -> None:
        self.state.is_healthy = False
        self.state.fault_code = fault_code
        self.transition_to(DeviceStateType.OFFLINE, 'fault')

    def on_recovery(self) -> None:
        if self.current_state == DeviceStateType.OFFLINE:
            self.state.is_healthy = True
            self.state.fault_code = None
            self.transition_to(DeviceStateType.IDLE, 'recovery')

    @abstractmethod
    def update(self, dt: float) -> None:
        pass

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            'device_id': self.state.device_id,
            'device_type': self.state.device_type,
            'state': self.current_state.name,
            'action': self.state.current_action.name,
            'power_kw': self.state.power_current,
            'time_in_state': self.state.time_in_state,
            'is_healthy': self.state.is_healthy,
        }



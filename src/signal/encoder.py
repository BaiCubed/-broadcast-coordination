from dataclasses import dataclass
from typing import Optional
import struct


@dataclass
class EPSSignal:

    version: int
    timestamp_seq: int
    region_id: int
    supply_demand: int
    intensity: int
    price: int
    priority: int
    crc: int = 0

    @property
    def version_ts(self) -> int:
        return ((self.version & 0x0F) << 4) | (self.timestamp_seq & 0x0F)

    @property
    def intensity_percent(self) -> float:
        return self.intensity / 40.95

    @property
    def price_value(self) -> float:
        return self.price / 100.0

    @property
    def is_shortage(self) -> bool:
        return self.supply_demand > 7

    @property
    def is_surplus(self) -> bool:
        return self.supply_demand < 7


class EPSSignalEncoder:

    CRC_POLY = 0x180F
    CRC_BITS = 12

    FIELD_SPECS = {
        'version_ts': {'bits': 8, 'shift': 56, 'max': 255},
        'region_id': {'bits': 12, 'shift': 44, 'max': 4095},
        'supply_demand': {'bits': 4, 'shift': 40, 'max': 15},
        'intensity': {'bits': 12, 'shift': 28, 'max': 4095},
        'price': {'bits': 12, 'shift': 16, 'max': 4095},
        'priority': {'bits': 4, 'shift': 12, 'max': 15},
        'crc': {'bits': 12, 'shift': 0, 'max': 4095},
    }

    def __init__(self, version: int = 1):
        if not 0 <= version <= 15:
            raise ValueError(f"Version must be 0-15, got {version}")
        self.version = version
        self._timestamp_seq = 0

    def _compute_crc12(self, data: int, data_bits: int = 52) -> int:
        remainder = data << self.CRC_BITS

        divisor = self.CRC_POLY << data_bits

        for i in range(data_bits, -1, -1):
            if remainder & (1 << (i + self.CRC_BITS)):
                remainder ^= (self.CRC_POLY << i)

        return remainder & 0xFFF

    def _validate_field(self, name: str, value: int) -> None:
        spec = self.FIELD_SPECS[name]
        if not 0 <= value <= spec['max']:
            raise ValueError(
                f"{name} must be 0-{spec['max']}, got {value}"
            )

    def encode(
        self,
        region_id: int,
        supply_demand: int,
        intensity: int,
        price: int,
        priority: int,
        timestamp_seq: Optional[int] = None,
    ) -> bytes:
        self._validate_field('region_id', region_id)
        self._validate_field('supply_demand', supply_demand)
        self._validate_field('intensity', intensity)
        self._validate_field('price', price)
        self._validate_field('priority', priority)

        if timestamp_seq is None:
            ts = self._timestamp_seq
            self._timestamp_seq = (self._timestamp_seq + 1) % 16
        else:
            if not 0 <= timestamp_seq <= 15:
                raise ValueError(f"timestamp_seq must be 0-15, got {timestamp_seq}")
            ts = timestamp_seq

        version_ts = ((self.version & 0x0F) << 4) | (ts & 0x0F)

        data = 0
        data |= (version_ts & 0xFF) << 44
        data |= (region_id & 0xFFF) << 32
        data |= (supply_demand & 0x0F) << 28
        data |= (intensity & 0xFFF) << 16
        data |= (price & 0xFFF) << 4
        data |= (priority & 0x0F)

        crc = self._compute_crc12(data, data_bits=52)

        signal = 0
        signal |= (version_ts & 0xFF) << 56
        signal |= (region_id & 0xFFF) << 44
        signal |= (supply_demand & 0x0F) << 40
        signal |= (intensity & 0xFFF) << 28
        signal |= (price & 0xFFF) << 16
        signal |= (priority & 0x0F) << 12
        signal |= (crc & 0xFFF)

        return struct.pack('>Q', signal)

    def encode_signal(self, signal: EPSSignal) -> bytes:
        return self.encode(
            region_id=signal.region_id,
            supply_demand=signal.supply_demand,
            intensity=signal.intensity,
            price=signal.price,
            priority=signal.priority,
            timestamp_seq=signal.timestamp_seq,
        )

    @classmethod
    def encode_from_physical(
        cls,
        region_id: int,
        supply_demand: int,
        intensity_percent: float,
        price_value: float,
        priority: int,
        version: int = 1,
    ) -> bytes:
        encoder = cls(version=version)

        intensity = int(round(intensity_percent * 40.95))
        intensity = max(0, min(4095, intensity))

        price = int(round(price_value * 100))
        price = max(0, min(4095, price))

        return encoder.encode(
            region_id=region_id,
            supply_demand=supply_demand,
            intensity=intensity,
            price=price,
            priority=priority,
        )

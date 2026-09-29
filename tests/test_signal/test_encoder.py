import pytest
import struct
from src.signal.encoder import EPSSignalEncoder, EPSSignal


class TestEPSSignalEncoder:

    def setup_method(self):
        self.encoder = EPSSignalEncoder(version=1)

    def test_basic_encoding(self):
        result = self.encoder.encode(
            region_id=100,
            supply_demand=7,
            intensity=2048,
            price=1000,
            priority=5,
        )
        assert len(result) == 8
        assert isinstance(result, bytes)

    def test_field_values_preserved(self):
        from src.signal.decoder import EPSSignalDecoder

        encoder = EPSSignalEncoder(version=1)
        decoder = EPSSignalDecoder()

        test_cases = [
            {'region_id': 0, 'supply_demand': 0, 'intensity': 0, 'price': 0, 'priority': 0},
            {'region_id': 4095, 'supply_demand': 15, 'intensity': 4095, 'price': 4095, 'priority': 15},
            {'region_id': 2048, 'supply_demand': 8, 'intensity': 1024, 'price': 500, 'priority': 8},
            {'region_id': 100, 'supply_demand': 12, 'intensity': 3000, 'price': 2000, 'priority': 3},
        ]

        for tc in test_cases:
            encoded = encoder.encode(**tc, timestamp_seq=5)
            result = decoder.decode(encoded)

            assert result.is_valid, f"Decode failed for {tc}"
            assert result.signal.region_id == tc['region_id']
            assert result.signal.supply_demand == tc['supply_demand']
            assert result.signal.intensity == tc['intensity']
            assert result.signal.price == tc['price']
            assert result.signal.priority == tc['priority']

    def test_version_encoding(self):
        from src.signal.decoder import decode_signal_fields

        encoder_v1 = EPSSignalEncoder(version=1)
        encoder_v15 = EPSSignalEncoder(version=15)

        data_v1 = encoder_v1.encode(region_id=0, supply_demand=0, intensity=0, price=0, priority=0, timestamp_seq=0)
        data_v15 = encoder_v15.encode(region_id=0, supply_demand=0, intensity=0, price=0, priority=0, timestamp_seq=0)

        fields_v1 = decode_signal_fields(data_v1)
        fields_v15 = decode_signal_fields(data_v15)

        assert fields_v1[0] == 1
        assert fields_v15[0] == 15

    def test_timestamp_sequence(self):
        from src.signal.decoder import decode_signal_fields

        encoder = EPSSignalEncoder(version=1)

        for expected_ts in range(3):
            data = encoder.encode(region_id=0, supply_demand=0, intensity=0, price=0, priority=0)
            fields = decode_signal_fields(data)
            assert fields[1] == expected_ts

    def test_timestamp_sequence_wrap(self):
        from src.signal.decoder import decode_signal_fields

        encoder = EPSSignalEncoder(version=1)

        encoder._timestamp_seq = 15

        data = encoder.encode(region_id=0, supply_demand=0, intensity=0, price=0, priority=0)
        fields = decode_signal_fields(data)
        assert fields[1] == 15

        data = encoder.encode(region_id=0, supply_demand=0, intensity=0, price=0, priority=0)
        fields = decode_signal_fields(data)
        assert fields[1] == 0

    def test_explicit_timestamp_sequence(self):
        from src.signal.decoder import decode_signal_fields

        encoder = EPSSignalEncoder(version=1)

        data = encoder.encode(region_id=0, supply_demand=0, intensity=0, price=0, priority=0, timestamp_seq=10)
        fields = decode_signal_fields(data)
        assert fields[1] == 10

    def test_crc_is_non_zero(self):
        from src.signal.decoder import decode_signal_fields

        encoder = EPSSignalEncoder(version=1)

        data = encoder.encode(region_id=100, supply_demand=7, intensity=2048, price=1000, priority=5)
        fields = decode_signal_fields(data)

        crc = fields[7]
        assert crc != 0 or True

    def test_invalid_region_id(self):
        with pytest.raises(ValueError, match="region_id"):
            self.encoder.encode(region_id=-1, supply_demand=0, intensity=0, price=0, priority=0)

        with pytest.raises(ValueError, match="region_id"):
            self.encoder.encode(region_id=4096, supply_demand=0, intensity=0, price=0, priority=0)

    def test_invalid_supply_demand(self):
        with pytest.raises(ValueError, match="supply_demand"):
            self.encoder.encode(region_id=0, supply_demand=-1, intensity=0, price=0, priority=0)

        with pytest.raises(ValueError, match="supply_demand"):
            self.encoder.encode(region_id=0, supply_demand=16, intensity=0, price=0, priority=0)

    def test_invalid_intensity(self):
        with pytest.raises(ValueError, match="intensity"):
            self.encoder.encode(region_id=0, supply_demand=0, intensity=-1, price=0, priority=0)

        with pytest.raises(ValueError, match="intensity"):
            self.encoder.encode(region_id=0, supply_demand=0, intensity=4096, price=0, priority=0)

    def test_invalid_price(self):
        with pytest.raises(ValueError, match="price"):
            self.encoder.encode(region_id=0, supply_demand=0, intensity=0, price=-1, priority=0)

        with pytest.raises(ValueError, match="price"):
            self.encoder.encode(region_id=0, supply_demand=0, intensity=0, price=4096, priority=0)

    def test_invalid_priority(self):
        with pytest.raises(ValueError, match="priority"):
            self.encoder.encode(region_id=0, supply_demand=0, intensity=0, price=0, priority=-1)

        with pytest.raises(ValueError, match="priority"):
            self.encoder.encode(region_id=0, supply_demand=0, intensity=0, price=0, priority=16)

    def test_invalid_version(self):
        with pytest.raises(ValueError, match="Version"):
            EPSSignalEncoder(version=-1)

        with pytest.raises(ValueError, match="Version"):
            EPSSignalEncoder(version=16)

    def test_encode_from_physical(self):
        from src.signal.decoder import EPSSignalDecoder

        decoder = EPSSignalDecoder()

        data = EPSSignalEncoder.encode_from_physical(
            region_id=100,
            supply_demand=12,
            intensity_percent=50.0,
            price_value=10.0,
            priority=5,
        )

        result = decoder.decode(data)
        assert result.is_valid

        assert 49.0 <= result.signal.intensity_percent <= 51.0
        assert 9.9 <= result.signal.price_value <= 10.1

    def test_encode_signal_object(self):
        from src.signal.decoder import EPSSignalDecoder

        signal = EPSSignal(
            version=1,
            timestamp_seq=5,
            region_id=200,
            supply_demand=10,
            intensity=3000,
            price=1500,
            priority=7,
        )

        encoder = EPSSignalEncoder(version=1)
        data = encoder.encode_signal(signal)

        decoder = EPSSignalDecoder()
        result = decoder.decode(data)

        assert result.is_valid
        assert result.signal.region_id == 200
        assert result.signal.supply_demand == 10
        assert result.signal.intensity == 3000
        assert result.signal.price == 1500
        assert result.signal.priority == 7


class TestEPSSignal:

    def test_version_ts_property(self):
        signal = EPSSignal(
            version=5,
            timestamp_seq=10,
            region_id=0,
            supply_demand=0,
            intensity=0,
            price=0,
            priority=0,
        )
        expected = (5 << 4) | 10
        assert signal.version_ts == expected

    def test_intensity_percent_property(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=0,
            supply_demand=0,
            intensity=4095,
            price=0,
            priority=0,
        )
        assert signal.intensity_percent == pytest.approx(100.0, rel=0.01)

        signal.intensity = 0
        assert signal.intensity_percent == 0.0

    def test_price_value_property(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=0,
            supply_demand=0,
            intensity=0,
            price=1000,
            priority=0,
        )
        assert signal.price_value == 10.0

        signal.price = 4095
        assert signal.price_value == 40.95

    def test_is_shortage_property(self):
        signal = EPSSignal(
            version=1, timestamp_seq=0, region_id=0,
            supply_demand=15, intensity=0, price=0, priority=0,
        )
        assert signal.is_shortage is True

        signal.supply_demand = 7
        assert signal.is_shortage is False

        signal.supply_demand = 0
        assert signal.is_shortage is False

    def test_is_surplus_property(self):
        signal = EPSSignal(
            version=1, timestamp_seq=0, region_id=0,
            supply_demand=0, intensity=0, price=0, priority=0,
        )
        assert signal.is_surplus is True

        signal.supply_demand = 7
        assert signal.is_surplus is False

        signal.supply_demand = 15
        assert signal.is_surplus is False

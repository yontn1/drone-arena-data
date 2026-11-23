import pytest

from arena_data.adapters import ArenaCameraPoseSource, BetaflightBlackboxSource, DroneLogFetcher
from arena_data.model import Drone


def test_hardware_adapters_say_what_they_wait_for(tmp_path):
    with pytest.raises(NotImplementedError, match="flight controller"):
        BetaflightBlackboxSource(tmp_path / "log.bbl").read_imu()
    with pytest.raises(NotImplementedError, match="camera"):
        ArenaCameraPoseSource(tmp_path).read_poses()
    with pytest.raises(NotImplementedError, match="come off the drone"):
        DroneLogFetcher().fetch(Drone("d1"), tmp_path)

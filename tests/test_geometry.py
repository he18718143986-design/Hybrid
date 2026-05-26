from src.world.geometry import dist, segment_hits_sun_pts


def test_dist():
    assert dist(0, 0, 3, 4) == 5.0


def test_segment_misses_sun():
    # Horizontal through (50,50) hits sun; use a chord above center.
    assert not segment_hits_sun_pts((10, 70), (90, 70))

def test_pkg_installations():
    try:
        import HousePricePrediction
    except Exception as e:
        assert False
        f"ERROR: {e} , Package HousePricePrediction not installed properly"


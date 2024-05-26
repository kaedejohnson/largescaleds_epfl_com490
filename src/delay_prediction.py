
class DelayPredictorDummy:
    def __init__(self):
        pass

    def predict(self, trip_ids, station_ids, timestamps):
        '''
        Predicts the delay for a list of trips at a list of stations at a list of timestamps.

        Returns: List of delays in seconds.
        '''
        return [60] * len(trip_ids)
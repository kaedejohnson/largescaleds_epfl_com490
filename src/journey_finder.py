from src.delay_prediction import DelayPredictorDummy
from src.confidence_calculation import journey_confidence_on_arrival_delay_predictions
from src.journey_search import find_and_plot_journeys

import numpy as np
import pandas as pd
from typing import Tuple

class JourneyFinder:
    def __init__(self):
        self.timetable = pd.read_csv('data/timetable.csv')
        self.footpaths = pd.read_csv('data/footpaths.csv')
        self.stops_info = pd.read_csv('data/stops.csv') 

    def find_station_id(self, station_name: str) -> int:
        '''
        Finds the station id for a given station name.

        Returns: Station id.
        '''
        return self.stops_info[self.stops_info['stop_name'] == station_name]['stop_id'].values[0]

    def find_journeys(self, start_station_name, end_station_name, arrival_datetime: str, confidence_threshold=0.7):
        '''
        Finds the journeys between two stations that arrive at the destination station at the given time.

        Returns: List of dictionaries
        {
            'journey': List of legs,
            'confidence': Confidence of the journey
        }
        '''

        start_station_id = self.find_station_id(start_station_name)
        end_station_id = self.find_station_id(end_station_name)

        arrival_time = arrival_datetime.split('T')[1]

        journeys = find_and_plot_journeys(
            self.timetable,
            self.footpaths,
            start_station_id,
            end_station_id,
            arrival_time,
            verbose=False
        )

        delay_predictor = DelayPredictorDummy()
        journey_confidences = []
        for journey in journeys:
            stations_ids, timestamps = [], []
            for leg in journey[:-1]:
                start_name = self.stops_info[self.stops_info['stop_id'] == leg['start_stop']]['stop_name'].values[0]
                end_name = self.stops_info[self.stops_info['stop_id'] == leg['arrival_stop']]['stop_name'].values[0]
                end_time = leg['arrival_time']
                stations_ids.append(leg['arrival_stop'])
                timestamps.append(end_time)

            delay_predictions = delay_predictor.predict(stations_ids, timestamps)
            confidence = journey_confidence_on_arrival_delay_predictions(
                journey,
                delay_predictions,
                arrival_datetime
            )
            journey_confidences.append(confidence)

        return [
            {'journey': journey, 'confidence': confidence} 
            for journey, confidence in zip(journeys, journey_confidences) if confidence >= confidence_threshold
        ]

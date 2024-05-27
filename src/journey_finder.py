import numpy as np
import pandas as pd
from typing import Tuple

from src.journey_plotter import JourneyPlotter
from src.delay_prediction import DelayPredictorDummy
from src.confidence_calculation import journey_confidence_on_arrival_delay_predictions


class JourneyFinder:
    def __init__(self, timetable, footpaths, stops_info):
        """
        Initialises the JourneyFinder object.
        
        Args:
            timetable: Timetable DataFrame.
            footpaths: Footpaths DataFrame.
            stops_info: Stops info DataFrame.
        """
        self.timetable = timetable
        self.footpaths = footpaths
        self.stops_info = stops_info


    def __find_station_id(self, station_name_id: str) -> int:
        """
        Finds the station id for a given station name.
        
        Args:
            station_name: Station name.

        Returns: 
            Station id.
        """
        return self.stops_info[self.stops_info['stop_name_id'] == station_name_id]['stop_id'].values[0]


    def find_journeys(self, start_station_id, end_station_id, arrival_datetime: str, confidence_threshold=0.7):
        """
        Finds the journeys between two stations that arrive at the destination station at the given time.

        Args:
            start_station_name: Name of the start station.
            end_station_name: Name of the end station.
            arrival_datetime: Arrival datetime in the format 'YYYY-MM-DDTHH:MM:SS'.
            confidence_threshold: Confidence threshold for the journey.

        Returns: List of dictionaries
        {
            'journey': List of legs,
            'confidence': Confidence of the journey
        }
        """

        #arrival_time = arrival_datetime.split('T')[1]
        arrival_time = arrival_datetime
        
        journeys = self.__find_list_of_journeys(
            start_station_id,
            end_station_id,
            arrival_time,
            num_journeys=5)

        delay_predictor = DelayPredictorDummy()
        journey_confidences = []
        for journey in journeys:
            stations_ids, timestamps = [], []
            for leg in journey[:-1]:
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


    def __connection_scan_latest_arrival(self, timetable: pd.DataFrame, footpaths: pd.DataFrame, source_stop_id: str, destination_stop_id: str, arrival_time: str) -> Tuple[dict, dict]:
        """Use the connection scan algorithm (SCA) to find the latest possible departure from the source stop such
        that the destination stop can be reached on time.

        Args:
            timetable (pd.DataFrame): A DataFrame containing the timetable of all trips.
            footpaths (pd.DataFrame): A DataFrame containing the footpaths between stops.
            source_stop_id (str): The ID of the source stop.
            destination_stop_id (str): The ID of the destination stop.
            arrival_time (str): The desired arrival time at the destination stop.

        Returns:
            S (dict): It maintains the latest possible departure from depart station such that destination stop can eventually be reached on time.
            T (dict): It maintains True or False based on whether the destination can be reached using the given trip.
        """

        total_seconds_init = pd.to_timedelta('00:00:00').total_seconds()
        total_seconds_arrival = pd.to_timedelta(arrival_time).total_seconds()

        timetable_sorted = timetable.sort_values(by='arr_time').reset_index(drop=True)
        stop_ids = set(list(timetable['dep_stop']) + list(timetable['arr_stop']) + list(footpaths['stop_id_a']))

        # T maintains True or False based on whether the destination can be reached using the given trip
        trip_ids = set(timetable['trip_id'])
        T = dict.fromkeys(trip_ids, False)
        
        # S maintains the latest possible departure from depart station such that destination stop can eventually be reached on time
        # Use a dict also as value to improve readability
        S = dict.fromkeys(stop_ids, {
            'transport': None,
            'start_time': total_seconds_init,
            'start_stop': None,
            'arrival_time': None,
            'arrival_stop': None
        })
        
        S[destination_stop_id] = {
            'transport': None,
            'start_time': total_seconds_arrival,
            'start_stop': destination_stop_id,
            'arrival_time': None,
            'arrival_stop': None
        }
        
        # catch case where source == destination so loop quits early
        if source_stop_id == destination_stop_id:
            S[source_stop_id] = {
                'transport': None,
                'start_time': total_seconds_arrival,
                'start_stop': source_stop_id,
                'arrival_time': total_seconds_arrival,
                'arrival_stop': destination_stop_id
            }
            return S, T
            
        # init S entries for footpaths leading to destination stop
        for _, fp in footpaths[footpaths['stop_id_a'] == destination_stop_id].iterrows():
            S[fp['stop_id_b']] = {
                'transport': 'walking',
                'start_time': total_seconds_arrival - fp['duration'],
                'start_stop': fp['stop_id_b'],
                'arrival_time': total_seconds_arrival,
                'arrival_stop': destination_stop_id
            }


        # c_0 is the last connection in the timetable that arrives at its stop before the user's intended arrival time
        c_0 = timetable_sorted[timetable_sorted['arr_time'] <= total_seconds_arrival].iloc[-1]['connection_id'] if np.any(timetable_sorted['arr_time'] <= total_seconds_arrival) else None
        
        # timetable subset: from earliest arrival to c_0 arrival, in reverse order
        ## rss = reverse-sorted subset
        timetable_rss = timetable_sorted.iloc[:timetable_sorted.index[timetable_sorted['connection_id'] == c_0][0] + 1].iloc[::-1].reset_index(drop=True)

        # starting with the last possible connection and working earlier and earlier...
        ## in comments, current connection = connection currently being considered by loop
        for _, row in timetable_rss.iterrows():
            # if arrival time of current connection is lower than S[source stop], algorithm is completed
            ## this is because S[source stop] maintains the latest possible departure from source stop s.t. the destination can be reached
            ## if this condition is met, we've started considering connections that no longer need to be considered
            if S[source_stop_id]['start_time'] >= row['arr_time']:
                break
            # if current connection's departure time is later than all previously scanned connections departing from the same stop 
            # AND
            # if (we know the trip can be used to eventually reach the destination) OR (current connection's arrival time is earlier than departure time for all previously scanned connections leaving from the arrival stop)
            if (T[row['trip_id']] or S[row['arr_stop']]['start_time'] >= row['arr_time']) and (S[row['dep_stop']]['start_time'] < row['dep_time']):
                # indicate that the trip can be used to reach the destination (ensures later connections in same trip are not neglected)
                T[row['trip_id']] = True
                # log the current connection as the new latest-departure connection for the current departure stop
                S[row['dep_stop']] = {
                    'transport': row['trip_id'],
                    'start_time': row['dep_time'],
                    'start_stop': row['dep_stop'],
                    'arrival_time': row['arr_time'],
                    'arrival_stop': row['arr_stop']
                }
                # update latest-departure walking connection for stations close enough to current departure stop to reflect new latest departure from current departure stop 
                for _, fp in footpaths[footpaths['stop_id_a'] == row['dep_stop']].iterrows():
                    if S[fp['stop_id_b']]['start_time'] < ( row['dep_time'] - fp['duration']):
                        S[fp['stop_id_b']] = {
                            'transport': 'walking',
                            'start_time': row['dep_time'] - fp['duration'],
                            'start_stop': fp['stop_id_b'],
                            'arrival_time': row['dep_time'],
                            'arrival_stop': fp['stop_id_a']
                        }
        return S, T


    def __journey_extraction_latest_arrival(self, S: dict, source_stop_id: str, destination_stop_id: str, arrival_time: str) -> list:
        """Extract the journey from the source stop to the destination stop based on the latest possible arrival time.

        Args:
            S (dict): It maintains the latest possible departure from depart station such that destination stop can eventually be reached on time.
            source_stop_id (str): The ID of the source stop.
            destination_stop_id (str): The ID of the destination stop.
            arrival_time (str): The desired arrival time at the destination stop. 

        Returns:
            ideal_journey (list): A list of connections representing the ideal journey from the source stop to the destination stop.
        """

        if source_stop_id == destination_stop_id:
            print('source is destination')
            return S[destination_stop_id]
        # if there is indeed a trip from source stop that starts a journey which eventually reaches the destination...
        if S[source_stop_id]['transport'] != None:
            # initialize the journey to empty, 
            # the mode of transport to None, 
            # the first connection to the source stop's latest-departure connection
            ideal_journey = []
            mode_of_transport = None
            ideal_connection = S[source_stop_id]
            # while mode of transport isn't None (which would indicate the end of the trip, based on how S[destination stop] was initialized)
            while ideal_connection['transport'] != None:
                # if mode of transport isn't previous mode of transport (meaning we have changed trip id or from trip->walk or vice-versa)...
                if ideal_connection['transport'] != mode_of_transport:
                    # if connection isn't the first in the ideal journey...
                    if ideal_connection['start_stop'] != source_stop_id:
                        # set latest trip in ideal journey to arrive where & when previous trip arrived 
                        ideal_journey[-1]['arrival_time'] = previous_connection['arrival_time']
                        ideal_journey[-1]['arrival_stop'] = previous_connection['arrival_stop']
                    # build out ideal journey, log as most recent step
                    ideal_journey.append(ideal_connection)
                    mode_of_transport = ideal_connection['transport']
                    previous_connection = ideal_connection
                    
                # if mode of transport IS previous mode of transport...
                # log as most recent step without adding it to ideal journey yet
                ## think of this like staying seated on the same train as it briefly stops at a platform;
                ## rather than log every intermediate stop where the user doesn't move, 
                ## we just want to log the first and last stop on a given train ride, 
                ## so we don't append any info to the journey yet. Once the user changes trains, we log
                ## that in the journey (this is what happens in the if statement above this else statement)
                previous_connection = ideal_connection
                # Prepare for next connection by finding latest feasible departure from current connection's arrival stop
                ideal_connection = S[ideal_connection['arrival_stop']]
            # leaving the loop means we've reached the final; we simply log the final stop of the final trip, and then end with the final destination.
            if ideal_connection['transport'] != mode_of_transport:
                ideal_journey[-1]['arrival_time'] = previous_connection['arrival_time']
                ideal_journey[-1]['arrival_stop'] = previous_connection['arrival_stop']
            ideal_journey.append(ideal_connection)
            return ideal_journey
        # if there is NOT a connection from source stop that starts a journey which eventually reaches the destination...
        else:
            print('No path exists starting from "{}" and arriving to "{}" by {}'.format(source_stop_id, destination_stop_id, arrival_time))
            return None
        
        
    def __get_latest_arrival_journey(self, source_stop_id: str, destination_stop_id: str, arrival_time: str) -> list:
        """Get the ideal journey from the source stop to the destination stop based on the latest possible arrival time.

        Args:
            timetable (pd.DataFrame): A DataFrame containing the timetable of all trips.
            footpaths (pd.DataFrame): A DataFrame containing the footpaths between stops.
            source_stop_id (str): The ID of the source stop.
            destination_stop_id (str): The ID of the destination stop.
            arrival_time (str): The desired arrival time at the destination stop.

        Returns:
            ideal_journey (list): A list of connections representing the ideal journey from the source stop to the destination stop.
        """
        
        S, _ = self.__connection_scan_latest_arrival(self.timetable, self.footpaths, source_stop_id, destination_stop_id, arrival_time)
        ideal_journey = self.__journey_extraction_latest_arrival(S, source_stop_id, destination_stop_id, arrival_time)
        return ideal_journey
    

    def __find_list_of_journeys(self, source_stop_id: str, destination_stop_id: str, arrival_time: str, num_journeys: str = 5) -> list:
        """Find a list of journeys between two stops that arrive at the destination stop at the given time.

        Args:
            source_stop_id (str): The ID of the source stop.
            destination_stop_id (str): The ID of the destination stop.
            arrival_time (str): The desired arrival time at the destination stop.
            num_journeys (int, optional): The number of journeys to find. Defaults to 5.
            
        Returns:
            journeys (list): A list of journeys.
        """
        journeys = []
        possible = 1
        
        while len(journeys) < num_journeys and possible != 0:
            journey = self.__get_latest_arrival_journey(
                source_stop_id=source_stop_id,
                destination_stop_id=destination_stop_id,
                arrival_time=arrival_time
            )
            if journey == None:
                print('No more journeys possible')
                possible = 0
            else:
                journeys.append(journey)
            
            arrival_time = pd.to_datetime(journey[-2]['arrival_time'] - 1, unit='s').strftime('%H:%M:%S')
        return journeys
            
            
    def find_and_plot_journeys(self, start_station_name_id: str, end_station_name_id: str, arrival_datetime: str, confidence_threshold=0.7) -> list:
        """Find and plot the journeys between two stations that arrive at the destination station at the given time with a confidence above the threshold.

        Args:
            start_station_name_id (_type_): The name of the start station.
            end_station_name_id (_type_): The name of the end station.
            arrival_datetime (str): The arrival datetime in the format 'HH:MM:SS'.
            confidence_threshold (float, optional): The confidence threshold for the journey. Defaults to 0.7.

        Returns:
            list: A list of figures.
        """

        start_station_id = self.__find_station_id(start_station_name_id)
        end_station_id = self.__find_station_id(end_station_name_id)

        journeys_and_confidence = self.find_journeys(
            start_station_id=start_station_id,
            end_station_id=end_station_id,
            arrival_datetime=arrival_datetime,
            confidence_threshold=confidence_threshold
        )
                
        figs = []
        if len(journeys_and_confidence) > 0:
            plotter = JourneyPlotter()
            figs = plotter.plot_journeys(
                journeys=journeys_and_confidence,
                source_stop_id=start_station_id,
                destination_stop_id=end_station_id,
                stops_info=self.stops_info
            )
            return figs
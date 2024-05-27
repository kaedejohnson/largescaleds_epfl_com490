import numpy as np
import pandas as pd
from typing import Tuple
import plotly.graph_objs as go


def connection_scan_latest_arrival(timetable: pd.DataFrame, footpaths: pd.DataFrame, source_stop_id: str, destination_stop_id: str, arrival_time: str) -> Tuple[dict, dict]:
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
    for _, fp in footpaths[footpaths['stop_id_a'] == 'destination_stop_id'].iterrows():
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


def journey_extraction_latest_arrival(S: dict, source_stop_id: str, destination_stop_id: str, arrival_time: str) -> list:
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


def get_latest_arrival_journey(timetable: pd.DataFrame, footpaths: pd.DataFrame, source_stop_id: str, destination_stop_id: str, arrival_time: str) -> list:
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
    
    S, _ = connection_scan_latest_arrival(timetable, footpaths, source_stop_id, destination_stop_id, arrival_time)
    ideal_journey = journey_extraction_latest_arrival(S, source_stop_id, destination_stop_id, arrival_time)
    return ideal_journey


def plot_journey(journey: list, stops_info: pd.DataFrame, source_id: str, desination_id: str, arrival_time: str) -> None:
    """Plot the journey on a map.

    Args:
        journey (list): A list of connections representing the journey.
        stops_info (pd.DataFrame): A DataFrame containing the information about the stops (longitude and latitude).
        source_id (str): The ID of the source stop.
        destination_id (str): The ID of the destination stop.
        arrival_time (str): The desired arrival time at the destination stop.
    """
    plot_df = []
    
    for step in journey:
        row = {}
        dep_info = stops_info[stops_info['stop_id'] == step['start_stop']]
        
        row['dep_stop'] = step['start_stop']
        row['dep_lat'] = dep_info['stop_lat'].values[0]
        row['dep_lon'] = dep_info['stop_lon'].values[0]
        row['dep_time'] = step['start_time']
        row['dep_time'] = pd.to_datetime(row['dep_time'], unit='s').strftime('%H:%M:%S')
        row['transport'] = step['transport'] if step['transport'] == 'walking' else 'trip'
        plot_df.append(row)
        
    plot_df = pd.DataFrame(plot_df)
    
    fig = go.Figure()
    
    # Add lines between stops (red for walking, blue for trips)
    for i in range(len(plot_df)-1):
        dep_lat = plot_df.iloc[i]['dep_lat']
        dep_lon = plot_df.iloc[i]['dep_lon']
        arr_lat = plot_df.iloc[i+1]['dep_lat']
        arr_lon = plot_df.iloc[i+1]['dep_lon']
        color = 'blue' if plot_df.iloc[i]['transport'] == 'trip' else 'red'
        fig.add_trace(go.Scattermapbox(
            lon=[dep_lon, arr_lon],
            lat=[dep_lat, arr_lat],
            mode='lines',
            hoverinfo='none',
            line=go.scattermapbox.Line(
                width=2,
                color=color
            ),
            showlegend=False
        ))   
        
    # Plot all the stops of the journey
    fig.add_trace(go.Scattermapbox(
        lat=plot_df['dep_lat'],
        lon=plot_df['dep_lon'],
        mode='markers',
        name='Stops',
        marker=go.scattermapbox.Marker(
            size=9,
            color='grey'
        ),
        hoverinfo='text',
        hovertemplate="Dep-Time: %{text}}",
        text=plot_df['dep_time'],
    ))
    
    # Add a green marker for the source stop
    start_info = stops_info[stops_info['stop_name'] == source_id]  
    start_time = plot_df.iloc[0]['dep_time']
    fig.add_trace(go.Scattermapbox(
        lon=[start_info['stop_lon'].values[0]],
        lat=[start_info['stop_lat'].values[0]],
        mode='markers',
        name='Start Stop',
        marker=go.scattermapbox.Marker(
            size=9,
            color='green'
        ),
        hoverinfo='text',
        hovertemplate=f"Stop-Name: {start_info['stop_name'].values[0]}<br>Dep-Time: {start_time}<extra></extra>",
    ))
    
    # Add a red marker for the destination stop
    desination_info = stops_info[stops_info['stop_name'] == desination_id]
    arr_time = plot_df.iloc[-1]['dep_time']
    fig.add_trace(go.Scattermapbox(
        lon=[desination_info['stop_lon'].values[0]],
        lat=[desination_info['stop_lat'].values[0]],
        mode='markers',
        name='Destination Stop',
        marker=go.scattermapbox.Marker(
            size=9,
            color='red'
        ),
        hoverinfo='text',
        hovertemplate=f"Stop-Name: {desination_info['stop_name'].values[0]}<br>Arr-Time: {arr_time}<extra></extra>",
    ))
        
    fig.add_trace(go.Scattermapbox(
        lon=[None],
        lat=[None],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Trip'
    ))

    fig.add_trace(go.Scattermapbox(
        lon=[None],
        lat=[None],
        mode='lines',
        line=dict(color='red', width=4),
        name='Walking'
    ))
    
    center_lat = plot_df['dep_lat'].mean()
    center_lon = plot_df['dep_lon'].mean()

    start_name = source_id
    destination_name = desination_id
    fig.update_layout(
        title=f'Journey from {start_name} ({source_id}) at {destination_name} ({desination_id}) arriving at {arrival_time}',
        showlegend=True,
        mapbox=dict(
            style='open-street-map',
            zoom=12,
            center=dict(lat=center_lat, lon=center_lon),
        )
    )
    
    fig.show()


def print_journey_human_readable(journey):
    for leg in journey[0:-1]:
        start_name = stops_info[stops_info['stop_id'] == leg['start_stop']]['stop_name'].values[0]
        end_name = stops_info[stops_info['stop_id'] == leg['arrival_stop']]['stop_name'].values[0]
        start_time = pd.to_datetime(leg['start_time'], unit='s').strftime('%H:%M:%S')
        end_time = pd.to_datetime(leg['arrival_time'], unit='s').strftime('%H:%M:%S')
        if leg['transport'] == 'walking':
            type = 'walk'
        else:
            type = 'ride'
        print(type, start_name, '->', end_name, f'({start_time} to {end_time})')


def find_and_plot_journeys(timetable, footpaths, source_stop_id, destination_stop_id, arrival_time, verbose = True, num_journeys = 5):
    journeys = []
    possible = 1
    
    while len(journeys) < num_journeys and possible != 0:
        journey = get_latest_arrival_journey(
            timetable=timetable,
            footpaths=footpaths,
            source_stop_id=source_stop_id,
            destination_stop_id=destination_stop_id,
            arrival_time=arrival_time
        )
        if journey == None:
            possible = 0
        else:
            journeys.append(journey)
        
        arrival_time = pd.to_datetime(journey[-2]['arrival_time'] - 1, unit='s').strftime('%H:%M:%S')

    if verbose == True:
        for journey in journeys:
            print_journey_human_readable(journey)
            plot_journey(
                journey=journey,
                stops_info=stops_info,
                source_id=source_stop_id,
                desination_id=destination_stop_id,
                arrival_time=pd.to_datetime(journey[-2]['arrival_time'], unit='s').strftime('%H:%M:%S')
            )
    
    return journeys
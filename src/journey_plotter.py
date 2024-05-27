import pandas as pd
import plotly.graph_objects as go


class JourneyPlotter:
    def __init__(self):
        """Initializes the JourneyPlotter."""
        pass

    def __print_journey_human_readable(self, journey: list, stops_info: pd.DataFrame, confidence: float):
        """Print the journey in a human-readable format.

        Args:
            journey (list): The journey to print.
            stops_info (pd.Dataframe): The information about the stops.
            confidence (float): The confidence of the journey.
        """
        print(f'Journey with confidence {round(confidence, 4)}:')
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


    def plot_journeys(self, journeys: list, source_stop_id: str, destination_stop_id: str, stops_info: pd.DataFrame) -> list: 
        """
        Plot the journeys.
        
        Args:
            journeys (list): A list of journeys.
            source_stop_id (str): The ID of the source stop.
            destination_stop_id (str): The ID of the destination stop.
            stops_info (pd.DataFrame): A DataFrame containing the information about the stops (longitude and latitude).
            
        Returns:
            list: A list of figures.
        """   
        
        figs = []

        for journey in journeys:
            
            legs = journey['journey']
            confidence = journey['confidence']
            
            self.__print_journey_human_readable(legs, stops_info, confidence)
            new_fig = self.__plot_single_journey(
                journey=legs,
                stops_info=stops_info,
                source_id=source_stop_id,
                destination_id=destination_stop_id,
                arrival_time=pd.to_datetime(legs[-2]['arrival_time'], unit='s').strftime('%H:%M:%S'),
                confidence=confidence
            )
            figs.append(new_fig)
        
        return figs
    
    
    def __plot_single_journey(self, journey: list, stops_info: pd.DataFrame, source_id: str, destination_id: str, arrival_time: str, confidence: float) -> None:
        """Plot the journey on a map.

        Args:
            journey (list): A list of connections representing the journey.
            stops_info (pd.DataFrame): A DataFrame containing the information about the stops (longitude and latitude).
            source_id (str): The ID of the source stop.
            destination_id (str): The ID of the destination stop.
            arrival_time (str): The desired arrival time at the destination stop.
            confidence (float): The confidence of the journey.
        """
        plot_df = []
        print(journey)
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
        start_info = stops_info[stops_info['stop_id'] == source_id]  
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
        destination_info = stops_info[stops_info['stop_id'] == destination_id]
        arr_time = plot_df.iloc[-1]['dep_time']
        fig.add_trace(go.Scattermapbox(
            lon=[destination_info['stop_lon'].values[0]],
            lat=[destination_info['stop_lat'].values[0]],
            mode='markers',
            name='Destination Stop',
            marker=go.scattermapbox.Marker(
                size=9,
                color='red'
            ),
            hoverinfo='text',
            hovertemplate=f"Stop-Name: {destination_info['stop_name'].values[0]}<br>Arr-Time: {arr_time}<extra></extra>",
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

        start_name = start_info['stop_name'].values[0]
        destination_name = destination_info['stop_name'].values[0]
        fig.update_layout(
            title=f'Journey from {start_name} ({source_id}) to {destination_name} ({destination_id}) arriving at {arrival_time}.<br>Confidence: {round(confidence, 4)}',
            showlegend=True,
            mapbox=dict(
                style='open-street-map',
                zoom=12,
                center=dict(lat=center_lat, lon=center_lon),
            )
        )
        
        fig.show()
        return fig
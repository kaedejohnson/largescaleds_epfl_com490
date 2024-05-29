from pyspark.sql.functions import col, to_timestamp, hour, minute


class DelayPredictor:
    def __init__(self, features_with_stats, loadedPipelineModel, spark):
        self.features_with_stats = features_with_stats
        self.loadedPipelineModel = loadedPipelineModel
        self.spark = spark

    def predict(self, station_ids, timestamps):
        '''
        Predicts the delay for a list of trips at a list of stations at a list of timestamps.
    
        Returns: List of delays in seconds.
        '''
        
        # Create a DataFrame from the station_ids and timestamps
        input_df = self.spark.createDataFrame(zip(station_ids, timestamps), ["station_id", "timestamp"])
    
        # Extract hour and minute from input_df.timestamp
        input_df = input_df.withColumn('timestamp_hour', hour(from_unixtime(col('timestamp'))))
        input_df = input_df.withColumn('timestamp_minute', minute(from_unixtime(col('timestamp'))))

        
        # Extract hour and minute from features_with_stats.arrival_time
        self.features_with_stats = self.features_with_stats.withColumn('arrival_hour', hour(col('arrival_time')))
        self.features_with_stats = self.features_with_stats.withColumn('arrival_minute', minute(col('arrival_time')))

        # Join the input DataFrame with the features_with_stats DataFrame
        features_subset = input_df.join(
            self.features_with_stats,
            on=[input_df.station_id == self.features_with_stats.stop_id, 
                input_df.timestamp_hour == self.features_with_stats.arrival_hour,
                input_df.timestamp_minute == self.features_with_stats.arrival_minute],
            how='inner'
        )

        # Transform the features_subset DataFrame to get predictions
        predictions = self.loadedPipelineModel.transform(features_subset).select("station_id", "timestamp", "prediction")
        # Calculate the average delay for each station and timestamp combination
        avg_delays_df = predictions.groupBy("station_id", "timestamp").agg({'prediction': 'mean'})
        # Convert the result DataFrame to a list of delays
        delays = [row['avg(prediction)'] for row in avg_delays_df.collect()]
    
        return delays
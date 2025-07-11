from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow import keras
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import uuid, os, json, torch, joblib, threading, time

app = Flask(__name__)
TEMP_RESULTS_DIR = "temp_results"
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

lstm_model = keras.models.load_model('models/model_lstm.keras')

def delete_file_later(path, delay=600):
    def _delete():
        time.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Deleted temp file: {path}")
    threading.Thread(target=_delete, daemon=True).start()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route('/results/<session_id>', methods=['GET'])
def get_results(session_id):
    file_path = os.path.join(TEMP_RESULTS_DIR, f"{session_id}.json")
    if not os.path.exists(file_path):
        return jsonify({"error": "Session ID not found"}), 404

    with open(file_path, "r") as f:
        result = json.load(f)

    return jsonify(result)
    
@app.route('/predict', methods=['POST'])
def predict():
    capacity = int(request.form['capacity'])
    units = int(request.form['units'])
    uploaded_files = request.files.getlist("files")
    
    if not uploaded_files:
        return jsonify({"error": "No files received"}), 400
    
    daily_ridership_df = None
    routes_df = None
    stops_df = None

    try:
        for file in uploaded_files:
            filename = file.filename.lower()

            if filename == "daily_ridership.csv":
                daily_ridership_df = pd.read_csv(file)
            elif filename == "routes.csv":
                routes_df = pd.read_csv(file)
            elif filename == "stops.csv":
                stops_df = pd.read_csv(file)
            else:
                return jsonify({"error": f"Unexpected file received: {filename}"}), 400

        if daily_ridership_df is None or routes_df is None or stops_df is None:
            return jsonify({"error": "Missing one or more required files: daily_ridership.csv, routes.csv, stops.csv"}), 400
        '''
        # ----------------- Data Summary -----------------
        initial = combined_df.groupby('day_of_week').agg({
            'trip_distance': ['count', 'mean'],
            'passenger_count': 'sum',
            'trip_duration': 'mean'
        }).round(2)

        initial.columns = ['_'.join(col).strip() for col in initial.columns.values]
        initial = initial.reset_index()

        initial_data = initial.to_dict(orient='records')
        '''
        # ------------------ LSTM Model ------------------
        model_instance = ModelLSTM(daily_ridership_df, lstm_model)
        restructured_df = model_instance.restructureData(daily_ridership_df)
        X, y = model_instance.assignFeatureTarget(restructured_df)
        X_scaled, y_scaled, scaler_y = model_instance.scaleValues(X, y)
        y_pred_scaled = model_instance.predictLSTM(X_scaled, y_scaled)
        y_pred, _ = model_instance.scaleInverseTransform(scaler_y, y_pred_scaled, y_scaled)

        restructured_df['prediction'] = y_pred.flatten()
        
        # Janella's change
        valid_counts = restructured_df.groupby(['route', 'day_of_week']).size().reset_index(name='count')
        valid_pairs = valid_counts[valid_counts['count'] >= 2][['route', 'day_of_week']]
        filtered_for_viz = restructured_df.merge(valid_pairs, on=['route', 'day_of_week'])

        prediction_lstm = filtered_for_viz[['date', 'route', 'day_of_week', 'log_passenger_count_lag7', 'pct_change', 'prediction']].replace({np.nan: None}).to_dict(orient='records')
        
        # ----------------- STGCN Model ------------------
        route_data = load_data_from_csv(routes_df, stops_df)

        route_data = calculate_segment_times(route_data)
        graph_data, stop_to_idx, route_indices, route_info = build_graph(route_data)
        stgcn_model = train_model(graph_data, route_indices)
        optimized_routes, df = optimize_routes(stgcn_model, graph_data, route_data, stop_to_idx, route_indices, route_info)
        '''
        # --------------- Allocation ------------------
        allocator = DynamicAllocator(alpha=0.8)
        allocator.set_user_inputs(capacity, units)
        allocations = allocator.allocate(combined_with_preds, stgcn_output, zone_ids)
        '''
        # ---------------- Results --------------------
        results = {
            "status": "success",
            "files_processed": [f.filename for f in uploaded_files],
            "lstm_results": prediction_lstm,
            "stgcn_results": optimized_routes,
            #"initial": initial_data,
            #"allocation_results": allocations.to_dict(orient='records')
        }
        
        session_id = str(uuid.uuid4())
        file_path = f"{TEMP_RESULTS_DIR}/{session_id}.json"
        with open(file_path, "w") as f:
            json.dump(results, f)

        delete_file_later(file_path, delay=900)

        return jsonify({"session_id": session_id})

    except Exception as e:
        return jsonify({'error': f'Error processing files: {str(e)}'}), 500

class ModelLSTM:
    def __init__(self, df, model_1):
        self.df = df
        self.model_1 = model_1

    def restructureData(self, df):
        res_df = df.groupby(['date', 'route', 'day_of_week'])['passenger_count'].sum().reset_index()
        res_df = res_df.sort_values(['route','day_of_week','date'])
        res_df['passenger_count_lag7'] = res_df.groupby(['route', 'day_of_week'])['passenger_count'].shift(1)
        res_df['pct_change'] = (res_df['passenger_count'] - res_df['passenger_count_lag7']) / (res_df['passenger_count_lag7'] + 1e-6)
        res_df['log_passenger_count'] = np.log1p(res_df['passenger_count'])
        res_df['log_passenger_count_lag7'] = np.log1p(res_df['passenger_count_lag7'])

        return res_df

    def assignFeatureTarget(self, res_df):
        X = res_df[['log_passenger_count_lag7', 'pct_change']].values
        y = res_df['log_passenger_count'].values.reshape(-1, 1)

        return X, y

    def scaleValues(self, X, y):

        # Load the original scalers from training
        scaler_X = joblib.load('scaler_X.save')
        scaler_y = joblib.load('scaler_y.save')

        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y)

        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=1e6, neginf=-1e6)

        self.scaler_y = scaler_y

        return X_scaled, y_scaled, scaler_y

    def predictLSTM(self, X_scaled, y_scaled):
        y_pred_scaled = self.model_1.predict(X_scaled)

        return y_pred_scaled

    def scaleInverseTransform(self, scaler_y, y_pred_scaled, y_scaled):
        y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
        y_pred = np.expm1(y_pred_log)

        y_true = np.expm1(scaler_y.inverse_transform(y_scaled))

        return y_pred, y_true

class STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels + 2, out_channels)

    def forward(self, data, route_indices):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        route_predictions = []
        for start_idx, end_idx in route_indices:
            edge_features = []
            for i in range(start_idx, end_idx):
                src, dst = edge_index[0, i], edge_index[1, i]
                edge_features.append(torch.cat([x[src], edge_attr[i]], dim=0))
            edge_features = torch.stack(edge_features)
            route_pred = self.fc(edge_features).sum()  # Sum segment predictions for total route time
            route_predictions.append(route_pred)
        return torch.stack(route_predictions)

def load_data_from_csv(routes_df, stops_df):
    # Create a dictionary to store route data
    route_data = defaultdict(lambda: defaultdict(dict))

    # Map each RouteID to the corresponding route details
    for _, route in routes_df.iterrows():
        route_id = route['RouteID']
        route_data['peak'][route_id] = {
            'total_travel_time': route['PeakTravelTime'],
            'total_distance': route['Distance'],
            'stops': []
        }
        route_data['off_peak'][route_id] = {
            'total_travel_time': route['OffPeakTravelTime'],
            'total_distance': route['Distance'],
            'stops': []
        }

    # Map each StopID to its corresponding data for both peak and off-peak hours
    for _, stop in stops_df.iterrows():
        route_id = stop['RouteID']
        stop_id = stop['StopID']

        # Add stop data for peak hours
        route_data['peak'][route_id]['stops'].append({
            'stop': stop_id,
            'peak_travel_time': stop['PeakTravelTime'],
            'distance': stop['Distance']
        })

        # Add stop data for off-peak hours
        route_data['off_peak'][route_id]['stops'].append({
            'stop': stop_id,
            'off_peak_travel_time': stop['OffPeakTravelTime'],
            'distance': stop['Distance']
        })

    return route_data

def calculate_segment_times(route_data):
    for period in route_data:
        for direction in route_data[period]:
            stops = route_data[period][direction]['stops']
            # Loop through each stop and set the segment time between consecutive stops
            for i in range(1, len(stops)):
                # Get the travel time for the segment
                if period == 'peak':
                    segment_time = stops[i]['peak_travel_time']
                else:
                    segment_time = stops[i]['off_peak_travel_time']

                # Assign segment time to the stop
                stops[i]['segment_time'] = segment_time

    return route_data

def build_graph(route_data):
    stops = set()
    for period in route_data:
        for direction in route_data[period]:
            for stop in route_data[period][direction]['stops']:
                stops.add(stop['stop'])
    stops = list(stops)
    stop_to_idx = {stop: idx for idx, stop in enumerate(stops)}
    num_nodes = len(stops)

    edge_index = []
    edge_attr = []
    route_targets = []
    route_indices = []
    route_info = []

    for period in route_data:
        for direction in route_data[period]:
            stops = route_data[period][direction]['stops']
            total_time = route_data[period][direction]['total_travel_time']
            start_idx = len(edge_index)
            for i in range(len(stops)-1):
                src = stop_to_idx[stops[i]['stop']]
                dst = stop_to_idx[stops[i+1]['stop']]
                edge_index.append([src, dst])
                edge_attr.append([stops[i+1]['segment_time'], stops[i+1]['distance']])
            route_targets.append(total_time)
            route_indices.append((start_idx, len(edge_index)))
            route_info.append((period, direction))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.eye(num_nodes, dtype=torch.float)
    y = torch.tensor(route_targets, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes), stop_to_idx, route_indices, route_info

def train_model(data, route_indices):
    model = STGCN(in_channels=data.num_nodes, hidden_channels=16, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data, route_indices)
        loss = F.mse_loss(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
    return model

def optimize_routes(model, data, route_data, stop_to_idx, route_indices, route_info):
    model.eval()
    with torch.no_grad():
        predictions = model(data, route_indices).squeeze()

    optimized_routes = {'Route': {}}
    duration_dict = {}  # For the output DataFrame

    for i, (period, route_id) in enumerate(route_info):
        # Treat route_id as unique identifier (e.g., R1_NB)
        if route_id not in duration_dict:
            duration_dict[route_id] = {}

        # Store predicted duration per period
        duration_dict[route_id][f'predicted_{period}_duration'] = predictions[i].item()

        # Also store full route info in nested dict
        if period not in optimized_routes['Route']:
            optimized_routes['Route'][period] = {}
        optimized_routes['Route'][period][route_id] = {
            'predicted_total_travel_time': predictions[i].item(),
            'original_total_travel_time': route_data[period][route_id]['total_travel_time'],
            'total_distance': route_data[period][route_id]['total_distance']
        }

    # Build DataFrame from route_id → {peak/offpeak predictions}
    df = pd.DataFrame([
        {
            'route_id': route_id,
            'predicted_peak_duration': route.get('predicted_peak_duration'),
            'predicted_offpeak_duration': route.get('predicted_off_peak_duration')
        }
        for route_id, route in duration_dict.items()
    ])
        
    return optimized_routes, df

if __name__ == "__main__":
    app.run(debug=True)
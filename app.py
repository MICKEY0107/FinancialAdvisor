from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys
from attached_assets.finalcode_1752907633708 import ExplorerPM, classify_expenses, predict_future_liabilities, detect_anomalies, projections_and_events, sankey_data, asset_rebalancing, health_critical_risk, gauges_data, category_table_data
import pandas as pd

app = Flask(__name__)
CORS(app)

explorer = ExplorerPM()

@app.route('/api/smart-report', methods=['POST'])
def smart_report():
    try:
        print("Received data:", request.json, file=sys.stderr)
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        df = pd.DataFrame([data])
        explorer.original_df = df
        explorer.df = df.copy()
        explorer.feature_columns = list(df.columns)
        report = explorer.generate_smart_report(user_index=0)  # This is a string!

        # Use model-driven outputs for all other fields
        fixed_expenses, variable_expenses = classify_expenses(data)
        future_liabilities = predict_future_liabilities(data)
        anomalies = detect_anomalies(data)
        projections, events = projections_and_events(data)
        sankey = sankey_data(data, fixed_expenses, variable_expenses)
        investment = asset_rebalancing(data)
        insurance = health_critical_risk(data)
        gauges = gauges_data(data)
        category_table = category_table_data(data)

        return jsonify({
            "fixed_expenses": fixed_expenses,
            "variable_expenses": variable_expenses,
            "future_liabilities": future_liabilities,
            "anomalies": anomalies,
            "investment": investment,
            "insurance": insurance,
            "projections": projections,
            "events": events,
            "sankey": sankey,
            "gauges": gauges,
            "category_table": category_table,
            "report": report
        })
    except Exception as e:
        print("Exception in /api/smart-report:", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True) 
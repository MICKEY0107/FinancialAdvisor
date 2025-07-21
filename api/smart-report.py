import sys
import traceback
import json
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../attached_assets'))
from finalcode_1752907633708 import ExplorerPM, classify_expenses, predict_future_liabilities, detect_anomalies, projections_and_events, sankey_data, asset_rebalancing, health_critical_risk, gauges_data, category_table_data
import pandas as pd

def handler(request):
    try:
        if request.method != 'POST':
            return {
                'statusCode': 405,
                'body': json.dumps({'error': 'Method not allowed'}),
                'headers': {'Content-Type': 'application/json'}
            }
        data = request.json() if callable(request.json) else request.json
        if not data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No JSON payload received'}),
                'headers': {'Content-Type': 'application/json'}
            }
        df = pd.DataFrame([data])
        explorer = ExplorerPM()
        explorer.original_df = df
        explorer.df = df.copy()
        explorer.feature_columns = list(df.columns)
        report = explorer.generate_smart_report(user_index=0)
        fixed_expenses, variable_expenses = classify_expenses(data)
        future_liabilities = predict_future_liabilities(data)
        anomalies = detect_anomalies(data)
        projections, events = projections_and_events(data)
        sankey = sankey_data(data, fixed_expenses, variable_expenses)
        investment = asset_rebalancing(data)
        insurance = health_critical_risk(data)
        gauges = gauges_data(data)
        category_table = category_table_data(data)
        return {
            'statusCode': 200,
            'body': json.dumps({
                'fixed_expenses': fixed_expenses,
                'variable_expenses': variable_expenses,
                'future_liabilities': future_liabilities,
                'anomalies': anomalies,
                'investment': investment,
                'insurance': insurance,
                'projections': projections,
                'events': events,
                'sankey': sankey,
                'gauges': gauges,
                'category_table': category_table,
                'report': report
            }),
            'headers': {'Content-Type': 'application/json'}
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e), 'traceback': traceback.format_exc()}),
            'headers': {'Content-Type': 'application/json'}
        } 
# Explorer PM – AI-Powered Financial Advisor

A professional, modern, one-page AI-powered financial advisor web application for personalized financial analysis, recommendations, and reporting.

---

## 🚀 Features
- **Personalized Financial Dashboard**: All your key metrics, charts, and insights in one place.
- **AI-Powered Smart Insights**: Dynamic, model-driven recommendations for:
  - Expense classification (fixed vs variable)
  - Future liabilities (EMIs, tuition, medical, etc.)
  - Spending anomaly detection
  - Investment advice (risk profile, SIP/ELSS/PPF, asset allocation)
  - Insurance gap & risk (life cover, health risk, critical illness)
  - Financial projections (1/3/10-year timeline, events)
  - Gauges (risk, health, savings rate)
  - Sankey diagram (income flow)
  - Category summary table
  - Personalized markdown report
- **Interactive Visualizations**: Pie, bar, line, and Sankey charts (Recharts, Chart.js)
- **Export Reports**: Download your analysis as PDF or CSV
- **Local Storage**: User data is saved in browser (no login required)
- **Responsive Design**: Works on desktop and mobile
- **Indian Rupees (₹) Support**

---

## 🏗️ Architecture & Data Flow
- **Frontend**: React + TypeScript + Vite + Tailwind CSS
  - UI: Modern, responsive, with shadcn/ui and Radix primitives
  - State: React hooks, React Query
  - Charts: Recharts, Chart.js
- **Backend**: Node.js (Express) + Python (ML models)
  - Express API routes (including `/api/smart-report`)
  - Python ML integration via child_process (calls `finalcode_1752907633708.py`)
  - All AI/ML logic in a single Python file for maintainability
- **ML/AI**: XGBoost, LightGBM, RandomForest, Prophet, IsolationForest
  - All models and feature engineering in `finalcode_1752907633708.py`

---

## 🛠️ Requirements

### Python (ML Backend)
Install with:
```sh
pip install -r ./attached_assets/requirements.txt
```
**Contents:**
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- prophet
- matplotlib
- joblib

### Node.js (Frontend/Backend)
Install with:
```sh
npm install
```
**Major dependencies:**
- react, react-dom, react-hook-form, react-markdown, react-icons, recharts, chart.js
- express, cors, express-session, passport, passport-local
- tailwindcss, shadcn/ui, @radix-ui/react-*
- jsPDF, papaparse (for export)
- @tanstack/react-query, wouter, zod

---

## ⚡ Setup Instructions

1. **Clone the repo and enter the main directory:**
   ```sh
   cd AIFinancialAdvisor/AIFinancialAdvisor
   ```
2. **Install Node.js dependencies:**
   ```sh
   npm install
   ```
3. **Set up Python virtual environment and install ML dependencies:**
   ```sh
   python -m venv venv
   # Activate venv (Windows):
   venv\Scripts\activate
   # Activate venv (macOS/Linux):
   source venv/bin/activate
   pip install -r ./attached_assets/requirements.txt
   ```
4. **Start the app:**
   ```sh
   npm run dev
   ```
   - This will start both the frontend (Vite) and backend (Express).
   - The backend will call the Python ML script as needed.
5. **Open the app:**
   - Go to [http://localhost:5173](http://localhost:5173) in your browser.

---

## 💡 Usage
- **Fill out your financial profile** (income, expenses, assets, savings, investments, debt, insurance, etc.)
- **Click Smart Insights** to get:
  - Dynamic, personalized recommendations and analysis
  - All charts, cards, and tables update based on your input
- **Export your report** as PDF or CSV
- **Try different profiles** to see how the AI adapts

---

## 📝 Notes & Tips
- **CORS:** The backend is CORS-enabled for local development.
- **Ports:** Frontend runs on 5173, backend on 5000 (API calls are proxied).
- **No login required:** All data is stored in your browser (localStorage).
- **ML/AI:** All predictions and recommendations are powered by your own input and the trained models in `finalcode_1752907633708.py`.
- **For production:** Train models on real data and save them for fast loading.

---

## 🤝 Contributing
1. Create your branch (`git checkout -b feature/your-feature`)
2. Commit your changes (`git commit -am 'Add new feature'`)
3. Push to the branch (`git push origin feature/your-feature`)
4. Open a Pull Request

---

## 📄 License
MIT


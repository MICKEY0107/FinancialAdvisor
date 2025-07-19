import type { Express } from "express";
import { createServer, type Server } from "http";
import { spawn } from "child_process";
import path from "path";
import { financialDataSchema, mlPredictionSchema } from "@shared/schema";

export async function registerRoutes(app: Express): Promise<Server> {
  
  // ML Prediction endpoint
  app.post("/api/predict", async (req, res) => {
    try {
      // Validate input data
      const validatedData = financialDataSchema.parse(req.body);
      
      // Convert to the format expected by the Python script
      const pythonInput = {
        Age: validatedData.age,
        Occupation: validatedData.occupation,
        "Gross monthly income": validatedData.grossIncome,
        "Net monthly income": validatedData.netIncome,
        Assets: validatedData.assets,
        Savings: validatedData.savings,
        Investments: validatedData.investments,
        Debt: validatedData.debt,
        "Debt Payments": validatedData.monthlyDebtPayments,
        Insurance: validatedData.lifeInsurance,
        // Expense categories
        "Rent/Mortgage": validatedData.housing || 0,
        "Car Payment": validatedData.transportation || 0,
        "Groceries": validatedData.food || 0,
        "Utilities": validatedData.utilities || 0,
        "Movies": validatedData.entertainment || 0,
        "Phone": validatedData.healthcare || 0,
        "Other": validatedData.other || 0,
      };

      // Spawn Python process to run ML prediction
      const pythonScriptPath = path.resolve(process.cwd(), 'attached_assets', 'finalcode_1752907633708.py');
      
      const pythonProcess = spawn('python', [pythonScriptPath], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      // Send input data to Python script
      pythonProcess.stdin.write(JSON.stringify(pythonInput));
      pythonProcess.stdin.end();

      let outputData = '';
      let errorData = '';

      pythonProcess.stdout.on('data', (data) => {
        outputData += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Python script error:', errorData);
          
          // Return fallback predictions if Python script fails
          const fallbackPredictions = {
            expensePredictions: {
              "Housing": validatedData.housing || validatedData.netIncome * 0.3,
              "Transportation": validatedData.transportation || validatedData.netIncome * 0.15,
              "Food": validatedData.food || validatedData.netIncome * 0.12,
              "Utilities": validatedData.utilities || validatedData.netIncome * 0.05,
              "Entertainment": validatedData.entertainment || validatedData.netIncome * 0.08,
              "Healthcare": validatedData.healthcare || validatedData.netIncome * 0.05,
              "Other": validatedData.other || validatedData.netIncome * 0.05,
            },
            investmentRecommendations: {
              riskProfile: validatedData.age < 35 ? "Aggressive" : validatedData.age < 55 ? "Moderate" : "Conservative",
              allocation: {
                stocks: validatedData.age < 35 ? 80 : validatedData.age < 55 ? 60 : 40,
                bonds: validatedData.age < 35 ? 15 : validatedData.age < 55 ? 30 : 50,
                cash: validatedData.age < 35 ? 5 : validatedData.age < 55 ? 10 : 10,
              },
              recommendations: [
                validatedData.age < 35 
                  ? "Focus on growth-oriented investments like stock index funds"
                  : "Balance growth and stability with a mix of stocks and bonds",
                "Consider low-cost index funds to minimize fees",
                "Diversify across different asset classes and geographic regions",
              ],
            },
            insuranceAnalysis: {
              recommendedCoverage: validatedData.grossIncome * 12 * 10,
              gap: Math.max(0, (validatedData.grossIncome * 12 * 10) - validatedData.lifeInsurance),
              recommendations: [
                validatedData.lifeInsurance < (validatedData.grossIncome * 12 * 10)
                  ? "Consider increasing life insurance to 10x annual income"
                  : "Your life insurance coverage appears adequate",
                "Review and update beneficiaries annually",
                "Consider disability insurance to protect your income",
              ],
            },
            financialHealthScore: Math.min(100, Math.max(0, 
              (validatedData.savings / (validatedData.netIncome * 6)) * 25 + // Emergency fund score
              (validatedData.investments / (validatedData.grossIncome * 12 * 0.15)) * 25 + // Investment score
              (Math.max(0, 100 - (validatedData.debt / (validatedData.grossIncome * 12)) * 100) / 100) * 25 + // Debt score
              25 // Base score
            )),
            recommendations: [
              "Build an emergency fund covering 3-6 months of expenses",
              "Invest 10-15% of your income for long-term wealth building",
              "Pay down high-interest debt to improve your financial health",
              "Review and optimize your insurance coverage",
            ],
          };
          
          res.json(fallbackPredictions);
          return;
        }

        try {
          // Parse the output from Python script
          const predictions = JSON.parse(outputData);
          
          // Validate the predictions match our schema
          const validatedPredictions = mlPredictionSchema.parse(predictions);
          
          res.json(validatedPredictions);
        } catch (parseError) {
          console.error('Error parsing Python output:', parseError);
          console.error('Raw output:', outputData);
          
          // Return error response
          res.status(500).json({ 
            error: 'Failed to process ML predictions',
            details: 'The AI model encountered an error processing your data'
          });
        }
      });

      // Handle timeout
      setTimeout(() => {
        if (!pythonProcess.killed) {
          pythonProcess.kill();
          res.status(408).json({ 
            error: 'Request timeout',
            details: 'The AI analysis took too long to complete'
          });
        }
      }, 30000); // 30 second timeout

    } catch (error) {
      console.error('API Error:', error);
      res.status(400).json({ 
        error: 'Invalid request data',
        details: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

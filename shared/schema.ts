import { z } from "zod";
import { pgTable, uuid, varchar, timestamp } from 'drizzle-orm/pg-core';

export const financialDataSchema = z.object({
  // Personal Information
  age: z.number().min(18).max(100),
  occupation: z.string().min(1),
  
  // Income Information
  grossIncome: z.number().min(0),
  netIncome: z.number().min(0),
  
  // Assets & Savings
  assets: z.number().min(0),
  savings: z.number().min(0),
  investments: z.number().min(0),
  
  // Debt Information
  debt: z.number().min(0),
  monthlyDebtPayments: z.number().min(0),
  
  // Insurance & Goals
  lifeInsurance: z.number().min(0),
  goalsPriority: z.enum(['retirement', 'house', 'emergency', 'education', 'travel']),
  
  // Expense Categories (optional, for detailed tracking)
  housing: z.number().min(0).optional(),
  transportation: z.number().min(0).optional(),
  food: z.number().min(0).optional(),
  utilities: z.number().min(0).optional(),
  entertainment: z.number().min(0).optional(),
  healthcare: z.number().min(0).optional(),
  other: z.number().min(0).optional(),
});

export const mlPredictionSchema = z.object({
  expensePredictions: z.record(z.string(), z.number()),
  investmentRecommendations: z.object({
    riskProfile: z.string(),
    allocation: z.record(z.string(), z.number()),
    recommendations: z.array(z.string()),
  }),
  insuranceAnalysis: z.object({
    recommendedCoverage: z.number(),
    gap: z.number(),
    recommendations: z.array(z.string()),
  }),
  financialHealthScore: z.number().min(0).max(100),
  recommendations: z.array(z.string()),
});

export type FinancialData = z.infer<typeof financialDataSchema>;
export type MLPrediction = z.infer<typeof mlPredictionSchema>;

export interface FinancialMetrics {
  netWorth: number;
  debtToIncomeRatio: number;
  savingsRate: number;
  emergencyFundMonths: number;
  investmentRate: number;
}

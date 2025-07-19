import { FinancialData, FinancialMetrics } from "@shared/schema";

export function calculateFinancialMetrics(data: FinancialData): FinancialMetrics {
  const netWorth = data.assets + data.savings + data.investments - data.debt;
  const annualIncome = data.grossIncome * 12;
  const debtToIncomeRatio = annualIncome > 0 ? (data.debt / annualIncome) * 100 : 0;
  
  // Calculate total monthly expenses
  const totalExpenses = (data.housing || 0) + (data.transportation || 0) + 
                       (data.food || 0) + (data.utilities || 0) + 
                       (data.entertainment || 0) + (data.healthcare || 0) + 
                       (data.other || 0) + data.monthlyDebtPayments;
  
  const monthlySurplus = data.netIncome - totalExpenses;
  const savingsRate = data.netIncome > 0 ? (monthlySurplus / data.netIncome) * 100 : 0;
  
  const emergencyFundMonths = totalExpenses > 0 ? data.savings / totalExpenses : 0;
  const investmentRate = data.netIncome > 0 ? (data.investments / (data.netIncome * 12)) * 100 : 0;

  return {
    netWorth,
    debtToIncomeRatio,
    savingsRate,
    emergencyFundMonths,
    investmentRate,
  };
}

export function generateRecommendations(data: FinancialData, metrics: FinancialMetrics): Array<{
  title: string;
  description: string;
  priority: 'High' | 'Medium' | 'Low';
  category: string;
}> {
  const recommendations = [];

  // Emergency fund recommendation
  if (metrics.emergencyFundMonths < 3) {
    recommendations.push({
      title: 'Build Emergency Fund',
      description: `Increase emergency fund to cover 3-6 months of expenses. You currently have ${metrics.emergencyFundMonths.toFixed(1)} months covered.`,
      priority: 'High' as const,
      category: 'Safety'
    });
  }

  // Debt reduction recommendation
  if (metrics.debtToIncomeRatio > 30) {
    recommendations.push({
      title: 'Reduce Debt Burden',
      description: `Your debt-to-income ratio of ${metrics.debtToIncomeRatio.toFixed(1)}% is above the recommended 30%. Consider debt consolidation or acceleration.`,
      priority: 'High' as const,
      category: 'Debt'
    });
  }

  // Investment recommendation
  if (metrics.investmentRate < 10) {
    recommendations.push({
      title: 'Increase Investment Allocation',
      description: 'Consider investing 10-15% of your income for long-term wealth building and retirement planning.',
      priority: 'Medium' as const,
      category: 'Investment'
    });
  }

  // Savings rate recommendation
  if (metrics.savingsRate < 20) {
    recommendations.push({
      title: 'Improve Savings Rate',
      description: `Your current savings rate is ${metrics.savingsRate.toFixed(1)}%. Aim for 20% or higher by reducing discretionary spending.`,
      priority: 'Medium' as const,
      category: 'Savings'
    });
  }

  // Insurance recommendation
  const recommendedLifeInsurance = data.grossIncome * 12 * 10;
  if (data.lifeInsurance < recommendedLifeInsurance) {
    const gap = recommendedLifeInsurance - data.lifeInsurance;
    recommendations.push({
      title: 'Increase Life Insurance',
      description: `Consider increasing life insurance by $${gap.toLocaleString()} to cover 10x annual income.`,
      priority: 'Medium' as const,
      category: 'Insurance'
    });
  }

  return recommendations;
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
}

export function formatPercentage(value: number): string {
  return `${value.toFixed(1)}%`;
}

export function getFinancialHealthScore(metrics: FinancialMetrics): number {
  let score = 0;
  
  // Emergency fund (25 points max)
  score += Math.min(25, metrics.emergencyFundMonths * 4);
  
  // Debt ratio (25 points max)
  if (metrics.debtToIncomeRatio <= 10) score += 25;
  else if (metrics.debtToIncomeRatio <= 20) score += 20;
  else if (metrics.debtToIncomeRatio <= 30) score += 15;
  else if (metrics.debtToIncomeRatio <= 40) score += 10;
  else score += 5;
  
  // Savings rate (25 points max)
  score += Math.min(25, metrics.savingsRate * 1.25);
  
  // Investment rate (25 points max)
  score += Math.min(25, metrics.investmentRate * 1.67);
  
  return Math.round(Math.min(100, score));
}

export function getHealthScoreLabel(score: number): string {
  if (score >= 80) return 'Excellent';
  if (score >= 70) return 'Good';
  if (score >= 60) return 'Fair';
  if (score >= 50) return 'Needs Improvement';
  return 'Poor';
}

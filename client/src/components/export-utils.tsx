import { FinancialData, FinancialMetrics } from "@shared/schema";
import { formatCurrency } from "@/lib/financial-utils";

declare global {
  interface Window {
    jsPDF: any;
    Papa: any;
  }
}

export function exportToPDF(data: FinancialData, metrics: FinancialMetrics, recommendations: any[]) {
  if (!window.jsPDF) {
    console.error('jsPDF not loaded');
    return;
  }

  const { jsPDF } = window;
  const doc = new jsPDF();

  // Header
  doc.setFontSize(20);
  doc.setTextColor(255, 107, 107); // Primary color
  doc.text('Explorer PM Financial Report', 20, 20);

  doc.setFontSize(12);
  doc.setTextColor(0, 0, 0);
  doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 20, 30);

  // Financial Summary
  doc.setFontSize(16);
  doc.text('Financial Summary', 20, 50);
  
  const summaryData = [
    [`Net Worth`, formatCurrency(metrics.netWorth)],
    [`Debt-to-Income Ratio`, `${metrics.debtToIncomeRatio.toFixed(1)}%`],
    [`Savings Rate`, `${metrics.savingsRate.toFixed(1)}%`],
    [`Emergency Fund Coverage`, `${metrics.emergencyFundMonths.toFixed(1)} months`],
    [`Investment Rate`, `${metrics.investmentRate.toFixed(1)}%`],
  ];

  let yPos = 60;
  summaryData.forEach(([label, value]) => {
    doc.text(`${label}: ${value}`, 20, yPos);
    yPos += 10;
  });

  // Recommendations
  if (recommendations.length > 0) {
    yPos += 10;
    doc.setFontSize(16);
    doc.text('AI Recommendations', 20, yPos);
    yPos += 10;

    recommendations.forEach((rec, index) => {
      if (yPos > 250) {
        doc.addPage();
        yPos = 20;
      }
      
      doc.setFontSize(12);
      doc.setTextColor(255, 107, 107);
      doc.text(`${index + 1}. ${rec.title}`, 20, yPos);
      yPos += 7;
      
      doc.setTextColor(0, 0, 0);
      doc.setFontSize(10);
      const descLines = doc.splitTextToSize(rec.description, 170);
      doc.text(descLines, 25, yPos);
      yPos += descLines.length * 5 + 5;
    });
  }

  doc.save('explorer-pm-financial-report.pdf');
}

export function exportToCSV(data: FinancialData, metrics: FinancialMetrics) {
  if (!window.Papa) {
    console.error('PapaParse not loaded');
    return;
  }

  const csvData = [
    ['Metric', 'Value'],
    ['Age', data.age.toString()],
    ['Occupation', data.occupation],
    ['Gross Monthly Income', data.grossIncome.toString()],
    ['Net Monthly Income', data.netIncome.toString()],
    ['Total Assets', data.assets.toString()],
    ['Savings', data.savings.toString()],
    ['Investments', data.investments.toString()],
    ['Total Debt', data.debt.toString()],
    ['Monthly Debt Payments', data.monthlyDebtPayments.toString()],
    ['Life Insurance', data.lifeInsurance.toString()],
    ['Goals Priority', data.goalsPriority],
    ['Net Worth', metrics.netWorth.toString()],
    ['Debt-to-Income Ratio (%)', metrics.debtToIncomeRatio.toFixed(2)],
    ['Savings Rate (%)', metrics.savingsRate.toFixed(2)],
    ['Emergency Fund (Months)', metrics.emergencyFundMonths.toFixed(2)],
    ['Investment Rate (%)', metrics.investmentRate.toFixed(2)],
  ];

  const csv = window.Papa.unparse(csvData);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', 'explorer-pm-financial-data.csv');
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

import { useEffect, useRef } from "react";
import { FinancialData, FinancialMetrics } from "@shared/schema";
import { formatCurrency } from "@/lib/financial-utils";

declare global {
  interface Window {
    Chart: any;
  }
}

interface ChartProps {
  data: FinancialData;
  metrics: FinancialMetrics;
}

export function FinancialBreakdownChart({ data, metrics }: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return;

    const ctx = canvasRef.current.getContext('2d');
    
    const chart = new window.Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Assets', 'Savings', 'Investments', 'Debt'],
        datasets: [{
          data: [data.assets, data.savings, data.investments, data.debt],
          backgroundColor: [
            'hsl(0, 100%, 70%)',    // Primary
            'hsl(176, 46%, 58%)',   // Secondary
            'hsl(204, 78%, 58%)',   // Accent
            'hsl(0, 84%, 60%)'      // Destructive
          ],
          borderWidth: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 20,
              usePointStyle: true,
            }
          },
          tooltip: {
            callbacks: {
              label: function(context: any) {
                return `${context.label}: ${formatCurrency(context.raw)}`;
              }
            }
          }
        }
      }
    });

    return () => chart.destroy();
  }, [data]);

  return <canvas ref={canvasRef} />;
}

export function NetWorthTrendChart({ data }: { data: FinancialData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return;

    const ctx = canvasRef.current.getContext('2d');
    
    // Generate 6 months of projected data
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    const currentNetWorth = data.assets + data.savings + data.investments - data.debt;
    const monthlyGrowth = (data.netIncome - (data.monthlyDebtPayments + 2000)) * 0.8; // Estimated growth
    
    const netWorthData = months.map((_, index) => 
      currentNetWorth + (monthlyGrowth * index)
    );

    const chart = new window.Chart(ctx, {
      type: 'line',
      data: {
        labels: months,
        datasets: [{
          label: 'Net Worth',
          data: netWorthData,
          borderColor: 'hsl(0, 100%, 70%)',
          backgroundColor: 'hsla(0, 100%, 70%, 0.1)',
          tension: 0.4,
          fill: true,
          pointBackgroundColor: 'hsl(0, 100%, 70%)',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context: any) {
                return `Net Worth: ${formatCurrency(context.raw)}`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: false,
            ticks: {
              callback: function(value: any) {
                return formatCurrency(value);
              }
            }
          }
        }
      }
    });

    return () => chart.destroy();
  }, [data]);

  return <canvas ref={canvasRef} />;
}

export function ExpenseBreakdownChart({ data }: { data: FinancialData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return;

    const ctx = canvasRef.current.getContext('2d');
    
    const expenses = {
      'Housing': data.housing || 0,
      'Transportation': data.transportation || 0,
      'Food & Dining': data.food || 0,
      'Utilities': data.utilities || 0,
      'Entertainment': data.entertainment || 0,
      'Healthcare': data.healthcare || 0,
      'Other': data.other || 0,
    };

    const chart = new window.Chart(ctx, {
      type: 'bar',
      data: {
        labels: Object.keys(expenses),
        datasets: [{
          label: 'Monthly Expenses',
          data: Object.values(expenses),
          backgroundColor: [
            'hsl(0, 100%, 70%)',
            'hsl(176, 46%, 58%)',
            'hsl(204, 78%, 58%)',
            'hsl(142, 36%, 66%)',
            'hsl(45, 93%, 81%)',
            'hsl(280, 100%, 70%)',
            'hsl(24, 100%, 70%)'
          ],
          borderRadius: 8,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context: any) {
                return `${context.label}: ${formatCurrency(context.raw)}`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value: any) {
                return formatCurrency(value);
              }
            }
          }
        }
      }
    });

    return () => chart.destroy();
  }, [data]);

  return <canvas ref={canvasRef} />;
}

export function GoalProgressChart({ data }: { data: FinancialData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return;

    const ctx = canvasRef.current.getContext('2d');
    
    // Sample goal data based on user's information
    const emergencyFundTarget = data.netIncome * 6;
    const emergencyFundProgress = Math.min(100, (data.savings / emergencyFundTarget) * 100);
    // Make house down payment target dynamic (e.g., 20% of a typical house value in India, or based on user input if available)
    const typicalHouseValue = 5000000; // Example: 50 lakh INR
    const houseDownPaymentTarget = typicalHouseValue * 0.2; // 20% down payment
    const houseDownPaymentProgress = Math.min(100, (data.savings * 0.6 / houseDownPaymentTarget) * 100);
    // Make retirement target dynamic (e.g., 20x annual income)
    const retirementTarget = data.grossIncome * 12 * 20; // 20 years of annual income
    const retirementProgress = Math.min(100, (data.investments / retirementTarget) * 100);

    const chart = new window.Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Emergency Fund', 'House Down Payment', 'Retirement Fund'],
        datasets: [{
          label: 'Progress (%)',
          data: [emergencyFundProgress, houseDownPaymentProgress, retirementProgress],
          backgroundColor: [
            'hsl(142, 36%, 66%)',
            'hsl(204, 78%, 58%)',
            'hsl(0, 100%, 70%)'
          ],
          borderRadius: 8,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context: any) {
                return `${context.label}: ${context.raw.toFixed(1)}%`;
              }
            }
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            ticks: {
              callback: function(value: any) {
                return `${value}%`;
              }
            }
          }
        }
      }
    });

    return () => chart.destroy();
  }, [data]);

  return <canvas ref={canvasRef} />;
}

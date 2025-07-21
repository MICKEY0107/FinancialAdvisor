import { useState, useEffect } from "react";
import { ArrowRight, Download, TrendingUp, PiggyBank, DollarSign, Percent, Lightbulb, Target, Shield, BarChart3, Wallet, ChartPie, Users, FileText, Home, Car, Heart, Settings, User, PanelLeft } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { FinancialModal } from "@/components/financial-modal";
import { FinancialBreakdownChart, NetWorthTrendChart, ExpenseBreakdownChart, GoalProgressChart } from "@/components/chart-components";
import { exportToPDF, exportToCSV } from "@/components/export-utils";
import { calculateFinancialMetrics, generateRecommendations, formatCurrency, formatPercentage, getFinancialHealthScore, getHealthScoreLabel } from "@/lib/financial-utils";
import { type FinancialData, type MLPrediction } from "@shared/schema";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { SidebarProvider, Sidebar, useSidebar } from "@/components/ui/sidebar";
import { useLocation } from "wouter";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import ReactMarkdown from 'react-markdown';
import { PieChart, Pie, Cell, Tooltip, Legend, Sankey } from 'recharts';

export default function Dashboard() {
  const [financialData, setFinancialData] = useState<FinancialData | null>(null);
  const [mlPredictions, setMLPredictions] = useState<MLPrediction | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(true); // local state for sidebar
  const { toast } = useToast();
  const [, setLocation] = useLocation();
  const [smartReport, setSmartReport] = useState<string | null>(null);
  const [fixedExpenses, setFixedExpenses] = useState<any>({});
  const [variableExpenses, setVariableExpenses] = useState<any>({});
  const [futureLiabilities, setFutureLiabilities] = useState<any[]>([]);
  const [anomalies, setAnomalies] = useState<any[]>([]);
  const [investment, setInvestment] = useState<any>({});
  const [insurance, setInsurance] = useState<any>({});
  const [projections, setProjections] = useState<any[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [sankey, setSankey] = useState<any>({ nodes: [], links: [] });
  const [gauges, setGauges] = useState<any>({});
  const [categoryTable, setCategoryTable] = useState<any[]>([]);
  const [loadingSmart, setLoadingSmart] = useState(false);
  const [errorSmart, setErrorSmart] = useState<string | null>(null);

  // Load saved data on component mount
  useEffect(() => {
    const savedData = localStorage.getItem('explorerpm_userdata');
    if (savedData) {
      try {
        const parsed = JSON.parse(savedData);
        setFinancialData(parsed);
      } catch (error) {
        console.error('Error loading saved data:', error);
      }
    }
  }, []);

  // ML Prediction mutation
  const mlPredictionMutation = useMutation({
    mutationFn: async (data: FinancialData) => {
      const response = await apiRequest('POST', '/api/predict', data);
      return response.json();
    },
    onSuccess: (predictions: MLPrediction) => {
      setMLPredictions(predictions);
      toast({
        title: "Analysis Complete",
        description: "AI recommendations have been generated based on your financial data.",
      });
    },
    onError: (error) => {
      console.error('ML Prediction Error:', error);
      toast({
        title: "Analysis Failed",
        description: "Unable to generate AI predictions. Please try again later.",
        variant: "destructive",
      });
    },
  });

  async function fetchSmartReport(profileData: any) {
    const res = await fetch('https://financialadvisor-pbcd.onrender.com/api/smart-report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profileData),
    });
    if (!res.ok) throw new Error('Failed to fetch smart report');
    return await res.json();
  }

  async function handleProfileSubmit(profileData: any) {
    const data = await fetchSmartReport(profileData);
    console.log('Smart Insights API response:', data);
    setSmartReport(data.report);
    setFixedExpenses(data.fixed_expenses);
    setVariableExpenses(data.variable_expenses);
    setFutureLiabilities(data.future_liabilities);
    setAnomalies(data.anomalies);
    setInvestment(data.investment);
    setInsurance(data.insurance);
    setProjections(data.projections);
    setEvents(data.events);
    setSankey(data.sankey);
    setGauges(data.gauges);
    setCategoryTable(data.category_table);
  }

  const handleFinancialSubmit = (data: FinancialData) => {
    // Save to localStorage
    localStorage.setItem('explorerpm_userdata', JSON.stringify(data));
    setFinancialData(data);
    setIsModalOpen(false);

    // Send to ML backend
    mlPredictionMutation.mutate(data);

    toast({
      title: "Financial Profile Updated",
      description: "Your data has been saved and is being analyzed by our AI.",
    });
  };

  const metrics = financialData ? calculateFinancialMetrics(financialData) : null;
  const recommendations = financialData && metrics ? generateRecommendations(financialData, metrics) : [];
  const healthScore = metrics ? getFinancialHealthScore(metrics) : 0;

  const handleExportPDF = () => {
    if (financialData && metrics) {
      exportToPDF(financialData, metrics, recommendations);
    } else {
      toast({
        title: "No Data Available",
        description: "Please complete your financial profile first.",
        variant: "destructive",
      });
    }
  };

  const handleExportCSV = () => {
    if (financialData && metrics) {
      exportToCSV(financialData, metrics);
    } else {
      toast({
        title: "No Data Available",
        description: "Please complete your financial profile first.",
        variant: "destructive",
      });
    }
  };

  // Get username from financialData or fallback
  const username = financialData?.occupation || "User";
  const initials = username.split(" ").map(w => w[0]).join("").toUpperCase().slice(0,2);

  // Unified handler for Smart Insights API
  async function handleSmartInsights(profile: any) {
    setLoadingSmart(true);
    setErrorSmart(null);
    try {
      const res = await fetch('https://financialadvisor-pbcd.onrender.com/api/smart-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profile),
      });
      if (!res.ok) throw new Error('API error: ' + res.status);
      const data = await res.json();
      setFixedExpenses(data.fixed_expenses || {});
      setVariableExpenses(data.variable_expenses || {});
      setFutureLiabilities(data.future_liabilities || []);
      setAnomalies(data.anomalies || []);
      setInvestment(data.investment || {});
      setInsurance(data.insurance || {});
      setProjections(data.projections || []);
      setEvents(data.events || []);
      setSankey(data.sankey || { nodes: [], links: [] });
      setGauges(data.gauges || {});
      setCategoryTable(data.category_table || []);
      setSmartReport(data.report || null);
    } catch (err: any) {
      setErrorSmart(err.message || 'Unknown error');
    } finally {
      setLoadingSmart(false);
    }
  }

  // Test button handler
  function handleTestSmartInsights() {
    handleSmartInsights({
      Age: 32,
      Occupation: "Engineer",
      "Gross monthly income": 80000,
      "Net monthly income": 70000,
      Assets: 500000,
      Savings: 100000,
      Investments: 150000,
      Debt: 20000,
      Insurance: 300000,
      "Rent/Mortgage": 15000,
      Utilities: 3000,
      "Car Payment": 5000,
      "Debt Payments": 2000,
      Groceries: 8000,
      Clothes: 2000,
      Phone: 1000,
      Subscriptions: 1000,
      Miscellaneous: 1500,
      Vacations: 0,
      Gifts: 0,
      "Emergency Fund": 20000,
      "Dining out": 4000,
      Movies: 1000,
      Other: 500
    });
  }

  return (
    <SidebarProvider open={sidebarOpen} onOpenChange={setSidebarOpen}>
      <Sidebar className={`bg-white border-r ${sidebarOpen ? '' : 'w-0'}`} collapsible="offcanvas">
        <div className="flex flex-col h-full p-4 gap-4">
          <div className="flex items-center gap-3 mb-4">
            <Avatar>
              <AvatarImage src="" alt={username} />
              <AvatarFallback>{initials}</AvatarFallback>
            </Avatar>
            <div className="flex flex-col">
              <span className="font-semibold text-gray-800 text-base">{username}</span>
              <span className="text-xs text-gray-500">Profile</span>
            </div>
          </div>
          <button
            className="flex items-center gap-2 px-3 py-2 rounded hover:bg-gray-100 text-gray-700 font-medium"
            onClick={() => setIsModalOpen(true)}
          >
            <User className="w-4 h-4" /> Update Profile
          </button>
          <button
            className="flex items-center gap-2 px-3 py-2 rounded hover:bg-gray-100 text-gray-700 font-medium"
            onClick={() => alert('Settings coming soon!')}
          >
            <Settings className="w-4 h-4" /> Settings
          </button>
          <button
            className="flex items-center gap-2 px-3 py-2 rounded hover:bg-red-100 text-red-700 font-medium mt-4"
            onClick={() => {
              localStorage.clear();
              setLocation("/");
            }}
          >
            <ArrowRight className="w-4 h-4" /> Logout
          </button>
        </div>
      </Sidebar>
      <div className="flex-1 bg-gradient-to-br from-slate-50 via-white to-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-100">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-4">
                {/* Sidebar toggle button */}
                <button
                  className="mr-2 p-2 rounded hover:bg-gray-100"
                  onClick={() => setSidebarOpen(!sidebarOpen)}
                >
                  <PanelLeft className="w-5 h-5 text-gray-700" />
                </button>
                {/* Only show the site name, no logo */}
                <div>
                  <h1 className="text-xl font-bold text-gray-800">Explorer PM</h1>
                </div>
              </div>
              {/* Profile avatar in top right (for wide screens) */}
              <div className="flex items-center space-x-4">
                <Avatar>
                  <AvatarImage src="" alt={username} />
                  <AvatarFallback>{initials}</AvatarFallback>
                </Avatar>
              </div>
            </div>
          </div>
        </header>

        {/* Navigation Tabs */}
        <div className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-6 bg-transparent h-auto p-0">
                <TabsTrigger 
                  value="dashboard" 
                  className="data-[state=active]:tab-active data-[state=inactive]:tab-inactive py-4 px-2 font-medium transition-colors"
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Dashboard
                </TabsTrigger>
                <TabsTrigger 
                  value="expenses"
                  className="data-[state=active]:tab-active data-[state=inactive]:tab-inactive py-4 px-2 font-medium transition-colors"
                >
                  <Wallet className="w-4 h-4 mr-2" />
                  Expenses
                </TabsTrigger>
                <TabsTrigger 
                  value="investments"
                  className="data-[state=active]:tab-active data-[state=inactive]:tab-inactive py-4 px-2 font-medium transition-colors"
                >
                  <ChartPie className="w-4 h-4 mr-2" />
                  Investments
                </TabsTrigger>
                <TabsTrigger 
                  value="insurance"
                  className="data-[state=active]:tab-active data-[state=inactive]:tab-inactive py-4 px-2 font-medium transition-colors"
                >
                  <Shield className="w-4 h-4 mr-2" />
                  Insurance
                </TabsTrigger>
                <TabsTrigger 
                  value="goals"
                  className="data-[state=active]:tab-active data-[state=inactive]:tab-inactive py-4 px-2 font-medium transition-colors"
                >
                  <Target className="w-4 h-4 mr-2" />
                  Goals
                </TabsTrigger>
                <TabsTrigger 
                  value="reports"
                  className="data-[state=active]:tab-active data-[state=inactive]:tab-inactive py-4 px-2 font-medium transition-colors"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Reports
                </TabsTrigger>
                <TabsTrigger 
                  value="smart-insights"
                  className="data-[state=active]:tab-active data-[state=inactive]:tab-inactive py-4 px-2 font-medium transition-colors"
                >
                  <Lightbulb className="w-4 h-4 mr-2" />
                  Smart Insights
                </TabsTrigger>
              </TabsList>

              {/* Main Content */}
              <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                
                {/* Dashboard Tab */}
                <TabsContent value="dashboard" className="space-y-8">
                  {/* Welcome Section */}
                  <div className="bg-white rounded-xl shadow p-8 flex flex-col items-start justify-center mb-6 border border-gray-100">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">Welcome back!</h2>
                    <p className="text-base text-gray-600 mb-4 max-w-xl">
                      Manage your finances, track your goals, and get clear, actionable insights for a brighter financial future. Everything you need, all in one place.
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <span className="inline-block bg-blue-50 text-blue-700 text-xs px-3 py-1 rounded-full">Personalized Dashboard</span>
                      <span className="inline-block bg-green-50 text-green-700 text-xs px-3 py-1 rounded-full">Smart Insights</span>
                      <span className="inline-block bg-yellow-50 text-yellow-700 text-xs px-3 py-1 rounded-full">Goal Tracking</span>
                    </div>
                  </div>

                  {financialData && metrics ? (
                    <>
                      {/* Quick Stats Cards */}
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <Card className="explorer-card">
                          <CardContent className="p-6">
                            <div className="flex items-center justify-between mb-4">
                              <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                                <DollarSign className="text-red-500 text-xl" />
                              </div>
                            </div>
                            <h3 className="text-2xl font-bold text-gray-800">{formatCurrency(metrics.netWorth)}</h3>
                            <p className="text-gray-500 text-sm">Net Worth</p>
                          </CardContent>
                        </Card>

                        <Card className="explorer-card">
                          <CardContent className="p-6">
                            <div className="flex items-center justify-between mb-4">
                              <div className="w-12 h-12 bg-teal-100 rounded-lg flex items-center justify-center">
                                <PiggyBank className="text-teal-500 text-xl" />
                              </div>
                            </div>
                            <h3 className="text-2xl font-bold text-gray-800">{formatCurrency(financialData.savings)}</h3>
                            <p className="text-gray-500 text-sm">Total Savings</p>
                          </CardContent>
                        </Card>

                        <Card className="explorer-card">
                          <CardContent className="p-6">
                            <div className="flex items-center justify-between mb-4">
                              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                                <TrendingUp className="text-blue-500 text-xl" />
                              </div>
                            </div>
                            <h3 className="text-2xl font-bold text-gray-800">{formatCurrency(financialData.investments)}</h3>
                            <p className="text-gray-500 text-sm">Investments</p>
                          </CardContent>
                        </Card>

                        <Card className="explorer-card">
                          <CardContent className="p-6">
                            <div className="flex items-center justify-between mb-4">
                              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                                <Percent className="text-green-500 text-xl" />
                              </div>
                            </div>
                            <h3 className="text-2xl font-bold text-gray-800">
                              {metrics.debtToIncomeRatio <= 30 ? 'Good' : 'High'}
                            </h3>
                            <p className="text-gray-500 text-sm">Debt Ratio</p>
                          </CardContent>
                        </Card>
                      </div>

                      {/* Charts Section */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <Card className="explorer-card">
                          <CardHeader>
                            <CardTitle>Financial Breakdown</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="h-64">
                              <FinancialBreakdownChart data={financialData} metrics={metrics} />
                            </div>
                          </CardContent>
                        </Card>
                        
                        <Card className="explorer-card">
                          <CardHeader>
                            <CardTitle>Net Worth Trend</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="h-64">
                              <NetWorthTrendChart data={financialData} />
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      {/* AI Recommendations */}
                      <Card className="explorer-card">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <Lightbulb className="text-red-500" />
                            AI Recommendations
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            {recommendations.length > 0 ? (
                              recommendations.map((rec, index) => (
                                <div key={index} className="flex items-start space-x-3 p-4 bg-gray-50 rounded-lg">
                                  <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center flex-shrink-0">
                                    <Lightbulb className="text-white text-sm" />
                                  </div>
                                  <div className="flex-1">
                                    <div className="flex items-center justify-between mb-1">
                                      <h4 className="font-medium text-gray-900">{rec.title}</h4>
                                      <Badge 
                                        variant={rec.priority === 'High' ? 'destructive' : rec.priority === 'Medium' ? 'default' : 'secondary'}
                                      >
                                        {rec.priority} Priority
                                      </Badge>
                                    </div>
                                    <p className="text-gray-600 text-sm">{rec.description}</p>
                                  </div>
                                </div>
                              ))
                            ) : (
                              <div className="text-center py-8">
                                <Lightbulb className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                                <p className="text-gray-500">Great job! Your financial profile looks healthy.</p>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    </>
                  ) : (
                    <Card className="explorer-card">
                      <CardContent className="p-12 text-center">
                        <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-gray-900 mb-2">Complete Your Financial Profile</h3>
                        <p className="text-gray-600 mb-6">Add your financial details to get personalized AI recommendations and insights.</p>
                        <Button 
                          onClick={() => setIsModalOpen(true)}
                          className="explorer-button bg-gradient-to-r from-red-400 to-red-500 text-white"
                        >
                          Get Started
                          <ArrowRight className="ml-2 w-4 h-4" />
                        </Button>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Expenses Tab */}
                <TabsContent value="expenses" className="space-y-6">
                  {financialData ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <Card className="explorer-card">
                        <CardHeader>
                          <CardTitle>Expense Breakdown</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="h-80">
                            <ExpenseBreakdownChart data={financialData} />
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="explorer-card">
                        <CardHeader>
                          <CardTitle>Monthly Summary</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                              <span className="font-medium">Total Income</span>
                              <span className="font-semibold">{formatCurrency(financialData.netIncome)}</span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                              <span className="font-medium">Total Expenses</span>
                              <span className="font-semibold">
                                {formatCurrency(
                                  (financialData.housing || 0) + 
                                  (financialData.transportation || 0) + 
                                  (financialData.food || 0) + 
                                  (financialData.utilities || 0) + 
                                  (financialData.entertainment || 0) + 
                                  (financialData.healthcare || 0) + 
                                  (financialData.other || 0)
                                )}
                              </span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg border border-green-200">
                              <span className="font-medium text-green-800">Monthly Surplus</span>
                              <span className="font-semibold text-green-800">
                                {formatCurrency(
                                  financialData.netIncome - 
                                  ((financialData.housing || 0) + 
                                  (financialData.transportation || 0) + 
                                  (financialData.food || 0) + 
                                  (financialData.utilities || 0) + 
                                  (financialData.entertainment || 0) + 
                                  (financialData.healthcare || 0) + 
                                  (financialData.other || 0))
                                )}
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  ) : (
                    <Card className="explorer-card">
                      <CardContent className="p-12 text-center">
                        <Wallet className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-gray-900 mb-2">Track Your Expenses</h3>
                        <p className="text-gray-600 mb-6">Complete your financial profile to analyze your spending patterns.</p>
                        <Button 
                          onClick={() => setIsModalOpen(true)}
                          className="explorer-button bg-gradient-to-r from-red-400 to-red-500 text-white"
                        >
                          Add Financial Data
                        </Button>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Investments Tab */}
                <TabsContent value="investments" className="space-y-6">
                  {financialData ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <Card className="explorer-card">
                        <CardHeader>
                          <CardTitle>Investment Overview</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            <div className="flex justify-between items-center p-4 border rounded-lg">
                              <div className="flex items-center space-x-3">
                                <TrendingUp className="text-red-500" />
                                <span className="font-medium">Total Investments</span>
                              </div>
                              <span className="font-semibold text-gray-900">{formatCurrency(financialData.investments)}</span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {mlPredictions && (
                        <Card className="explorer-card">
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                              <Lightbulb className="text-red-500" />
                              AI Investment Recommendations
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-4">
                              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="font-medium text-blue-800">Risk Profile</span>
                                  <Badge className="bg-blue-500">{mlPredictions.investmentRecommendations.riskProfile}</Badge>
                                </div>
                              </div>
                              {mlPredictions.investmentRecommendations.recommendations.map((rec, index) => (
                                <div key={index} className="p-4 border rounded-lg">
                                  <p className="text-sm text-gray-600">{rec}</p>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  ) : (
                    <Card className="explorer-card">
                      <CardContent className="p-12 text-center">
                        <ChartPie className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-gray-900 mb-2">Investment Analysis</h3>
                        <p className="text-gray-600 mb-6">Add your investment data to get personalized portfolio recommendations.</p>
                        <Button 
                          onClick={() => setIsModalOpen(true)}
                          className="explorer-button bg-gradient-to-r from-red-400 to-red-500 text-white"
                        >
                          Add Investment Data
                        </Button>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Insurance Tab */}
                <TabsContent value="insurance" className="space-y-6">
                  {financialData ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <Card className="explorer-card">
                        <CardHeader>
                          <CardTitle>Current Coverage</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            <div className="flex items-center justify-between p-4 border rounded-lg">
                              <div className="flex items-center space-x-3">
                                <Heart className="text-red-500" />
                                <span className="font-medium">Life Insurance</span>
                              </div>
                              <span className="font-semibold text-gray-900">{formatCurrency(financialData.lifeInsurance)}</span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {mlPredictions && (
                        <Card className="explorer-card">
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                              <Shield className="text-green-500" />
                              Coverage Analysis
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-4">
                              <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="font-medium text-yellow-800">Recommended Coverage</span>
                                  <span className="font-semibold text-yellow-800">
                                    {formatCurrency(mlPredictions.insuranceAnalysis.recommendedCoverage)}
                                  </span>
                                </div>
                                <p className="text-sm text-yellow-700">
                                  Coverage Gap: {formatCurrency(mlPredictions.insuranceAnalysis.gap)}
                                </p>
                              </div>
                              {mlPredictions.insuranceAnalysis.recommendations.map((rec, index) => (
                                <div key={index} className="p-4 border rounded-lg">
                                  <p className="text-sm text-gray-600">{rec}</p>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  ) : (
                    <Card className="explorer-card">
                      <CardContent className="p-12 text-center">
                        <Shield className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-gray-900 mb-2">Insurance Analysis</h3>
                        <p className="text-gray-600 mb-6">Add your insurance information to identify coverage gaps.</p>
                        <Button 
                          onClick={() => setIsModalOpen(true)}
                          className="explorer-button bg-gradient-to-r from-red-400 to-red-500 text-white"
                        >
                          Add Insurance Data
                        </Button>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Goals Tab */}
                <TabsContent value="goals" className="space-y-6">
                  {financialData && metrics ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <Card className="explorer-card">
                        <CardHeader>
                          <CardTitle>Goal Progress</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="h-64">
                            <GoalProgressChart data={financialData} />
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="explorer-card">
                        <CardHeader>
                          <CardTitle>Financial Goals</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-6">
                            {/* Emergency Fund Goal */}
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="font-medium">Emergency Fund</span>
                                <span>{Math.min(100, metrics.emergencyFundMonths * 16.67).toFixed(0)}% Complete</span>
                              </div>
                              <Progress value={Math.min(100, metrics.emergencyFundMonths * 16.67)} className="h-2" />
                              <div className="flex justify-between text-xs text-gray-500">
                                <span>{formatCurrency(financialData.savings)}</span>
                                <span>Goal: {formatCurrency(financialData.netIncome * 6)}</span>
                              </div>
                            </div>

                            {/* Investment Goal */}
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="font-medium">Investment Portfolio</span>
                                <span>{Math.min(100, metrics.investmentRate * 6.67).toFixed(0)}% Complete</span>
                              </div>
                              <Progress value={Math.min(100, metrics.investmentRate * 6.67)} className="h-2" />
                              <div className="flex justify-between text-xs text-gray-500">
                                <span>{formatCurrency(financialData.investments)}</span>
                                <span>Goal: 15% of income</span>
                              </div>
                            </div>

                            {/* Debt Reduction Goal */}
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="font-medium">Debt Reduction</span>
                                <span>{Math.max(0, 100 - metrics.debtToIncomeRatio * 3.33).toFixed(0)}% Complete</span>
                              </div>
                              <Progress value={Math.max(0, 100 - metrics.debtToIncomeRatio * 3.33)} className="h-2" />
                              <div className="flex justify-between text-xs text-gray-500">
                                <span>Current: {formatPercentage(metrics.debtToIncomeRatio)}</span>
                                <span>Target: &lt;30%</span>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  ) : (
                    <Card className="explorer-card">
                      <CardContent className="p-12 text-center">
                        <Target className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-gray-900 mb-2">Set Financial Goals</h3>
                        <p className="text-gray-600 mb-6">Track your progress towards financial milestones and retirement planning.</p>
                        <Button 
                          onClick={() => setIsModalOpen(true)}
                          className="explorer-button bg-gradient-to-r from-red-400 to-red-500 text-white"
                        >
                          Set Goals
                        </Button>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Reports Tab */}
                <TabsContent value="reports" className="space-y-6">
                  {financialData && metrics ? (
                    <>
                      {/* Financial Health Report */}
                      <Card className="explorer-card">
                        <CardHeader>
                          <CardTitle className="flex items-center justify-between">
                            Financial Health Report
                            <div className="flex space-x-2">
                              <Button onClick={handleExportPDF} size="sm" className="bg-red-500 text-white">
                                <FileText className="w-4 h-4 mr-2" />
                                PDF
                              </Button>
                              <Button onClick={handleExportCSV} size="sm" className="bg-teal-500 text-white">
                                <FileText className="w-4 h-4 mr-2" />
                                CSV
                              </Button>
                            </div>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                            <div className="text-center">
                              <div className="w-24 h-24 mx-auto bg-green-100 rounded-full flex items-center justify-center mb-3">
                                <span className="text-2xl font-bold text-green-600">{healthScore}</span>
                              </div>
                              <h4 className="font-semibold text-gray-900">Financial Health Score</h4>
                              <p className="text-sm text-gray-600">{getHealthScoreLabel(healthScore)}</p>
                            </div>
                            <div className="text-center">
                              <div className="w-24 h-24 mx-auto bg-blue-100 rounded-full flex items-center justify-center mb-3">
                                <span className="text-2xl font-bold text-blue-600">{formatPercentage(metrics.savingsRate)}</span>
                              </div>
                              <h4 className="font-semibold text-gray-900">Savings Rate</h4>
                              <p className="text-sm text-gray-600">Recommended: 20%</p>
                            </div>
                            <div className="text-center">
                              <div className="w-24 h-24 mx-auto bg-teal-100 rounded-full flex items-center justify-center mb-3">
                                <span className="text-2xl font-bold text-teal-600">{metrics.emergencyFundMonths.toFixed(0)}</span>
                              </div>
                              <h4 className="font-semibold text-gray-900">Emergency Fund</h4>
                              <p className="text-sm text-gray-600">Months Covered</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      {/* Detailed Metrics */}
                      <Card className="explorer-card">
                        <CardHeader>
                          <CardTitle>Detailed Financial Metrics</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="overflow-x-auto">
                            <table className="w-full">
                              <thead>
                                <tr className="border-b">
                                  <th className="text-left py-3 px-4 font-medium text-gray-700">Metric</th>
                                  <th className="text-right py-3 px-4 font-medium text-gray-700">Current</th>
                                  <th className="text-right py-3 px-4 font-medium text-gray-700">Target</th>
                                  <th className="text-right py-3 px-4 font-medium text-gray-700">Status</th>
                                </tr>
                              </thead>
                              <tbody>
                                <tr className="border-b">
                                  <td className="py-3 px-4 font-medium">Debt-to-Income Ratio</td>
                                  <td className="py-3 px-4 text-right">{formatPercentage(metrics.debtToIncomeRatio)}</td>
                                  <td className="py-3 px-4 text-right">&lt;30%</td>
                                  <td className="py-3 px-4 text-right">
                                    <Badge variant={metrics.debtToIncomeRatio <= 30 ? "default" : "destructive"}>
                                      {metrics.debtToIncomeRatio <= 30 ? "Good" : "High"}
                                    </Badge>
                                  </td>
                                </tr>
                                <tr className="border-b">
                                  <td className="py-3 px-4 font-medium">Savings Rate</td>
                                  <td className="py-3 px-4 text-right">{formatPercentage(metrics.savingsRate)}</td>
                                  <td className="py-3 px-4 text-right">20%</td>
                                  <td className="py-3 px-4 text-right">
                                    <Badge variant={metrics.savingsRate >= 20 ? "default" : "secondary"}>
                                      {metrics.savingsRate >= 20 ? "Excellent" : "Improve"}
                                    </Badge>
                                  </td>
                                </tr>
                                <tr className="border-b">
                                  <td className="py-3 px-4 font-medium">Emergency Fund</td>
                                  <td className="py-3 px-4 text-right">{metrics.emergencyFundMonths.toFixed(1)} months</td>
                                  <td className="py-3 px-4 text-right">3-6 months</td>
                                  <td className="py-3 px-4 text-right">
                                    <Badge variant={metrics.emergencyFundMonths >= 3 ? "default" : "secondary"}>
                                      {metrics.emergencyFundMonths >= 6 ? "Excellent" : metrics.emergencyFundMonths >= 3 ? "Good" : "Build"}
                                    </Badge>
                                  </td>
                                </tr>
                                <tr>
                                  <td className="py-3 px-4 font-medium">Investment Rate</td>
                                  <td className="py-3 px-4 text-right">{formatPercentage(metrics.investmentRate)}</td>
                                  <td className="py-3 px-4 text-right">10-15%</td>
                                  <td className="py-3 px-4 text-right">
                                    <Badge variant={metrics.investmentRate >= 10 ? "default" : "secondary"}>
                                      {metrics.investmentRate >= 15 ? "Excellent" : metrics.investmentRate >= 10 ? "Good" : "Improve"}
                                    </Badge>
                                  </td>
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </CardContent>
                      </Card>
                    </>
                  ) : (
                    <Card className="explorer-card">
                      <CardContent className="p-12 text-center">
                        <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-gray-900 mb-2">Generate Reports</h3>
                        <p className="text-gray-600 mb-6">Complete your financial profile to generate comprehensive reports and export your data.</p>
                        <Button 
                          onClick={() => setIsModalOpen(true)}
                          className="explorer-button bg-gradient-to-r from-red-400 to-red-500 text-white"
                        >
                          Add Financial Data
                        </Button>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                {/* Smart Insights Tab */}
                <TabsContent value="smart-insights" className="space-y-8">
                  <Button className="mb-4" onClick={handleTestSmartInsights} disabled={loadingSmart}>
                    {loadingSmart ? 'Loading...' : 'Test Smart Insights (Sample Data)'}
                  </Button>
                  {errorSmart && <div className="text-red-600">Error: {errorSmart}</div>}
                  {/* Anomalies Alert */}
                  {anomalies && anomalies.length > 0 && (
                    <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg">
                      <strong>Spending Anomaly Detected:</strong> {anomalies.join(', ')}
                    </div>
                  )}
                  {/* Fixed vs Variable Expenses Pie Chart */}
                  <div className="bg-white rounded-xl shadow p-6">
                    <h3 className="font-semibold mb-2">Fixed vs Variable Expenses</h3>
                    {fixedExpenses && variableExpenses && (Object.keys(fixedExpenses).length > 0 || Object.keys(variableExpenses).length > 0) ? (
                      <PieChart width={350} height={250}>
                        <Pie
                          data={[
                            { name: 'Fixed', value: Object.values(fixedExpenses).reduce((a: number, b: any) => Number(a) + Number(b), 0) },
                            { name: 'Variable', value: Object.values(variableExpenses).reduce((a: number, b: any) => Number(a) + Number(b), 0) },
                          ]}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          fill="#8884d8"
                          label
                        >
                          <Cell fill="#4f8ef7" />
                          <Cell fill="#f7b84f" />
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    ) : <div>No expense data available.</div>}
                  </div>
                  {/* Future Liabilities */}
                  <div className="bg-white rounded-xl shadow p-6">
                    <h3 className="font-semibold mb-2">Future Liabilities</h3>
                    {futureLiabilities && futureLiabilities.length > 0 ? (
                      <ul>
                        {futureLiabilities.map((liab: any, i: number) => (
                          <li key={i}>{liab.type}: ₹{liab.amount} due {liab.due_date}</li>
                        ))}
                      </ul>
                    ) : <div>No future liabilities.</div>}
                  </div>
                  {/* Investment Advice */}
                  <div className="bg-white rounded-xl shadow p-6">
                    <h3 className="font-semibold mb-2">Investment Advice</h3>
                    {investment && Object.keys(investment).length > 0 ? (
                      <>
                        <div>Risk Profile: {investment.risk_profile || 'N/A'}</div>
                        <div>SIP: ₹{investment.sip ?? 'N/A'} / ELSS: ₹{investment.elss ?? 'N/A'} / PPF: ₹{investment.ppf ?? 'N/A'}</div>
                        <div>Current Allocation: Equity {investment.asset_allocation?.equity ?? 'N/A'}% / Debt {investment.asset_allocation?.debt ?? 'N/A'}%</div>
                        <div>Recommended: Equity {investment.recommended_allocation?.equity ?? 'N/A'}% / Debt {investment.recommended_allocation?.debt ?? 'N/A'}%</div>
                      </>
                    ) : <div>No investment advice available.</div>}
                  </div>
                  {/* Insurance Analysis */}
                  <div className="bg-white rounded-xl shadow p-6">
                    <h3 className="font-semibold mb-2">Insurance Analysis</h3>
                    {insurance && Object.keys(insurance).length > 0 ? (
                      <>
                        <div>Current Life Cover: ₹{insurance.current_life ?? 'N/A'}</div>
                        <div>Recommended Life Cover: ₹{insurance.recommended_life ?? 'N/A'}</div>
                        <div>Gap: ₹{insurance.gap ?? 'N/A'}</div>
                        <div>Health Risk Score: {insurance.health_risk_score ?? 'N/A'}</div>
                        <div>Critical Illness Score: {insurance.critical_illness_score ?? 'N/A'}</div>
                      </>
                    ) : <div>No insurance analysis available.</div>}
                  </div>
                  {/* Financial Projections Timeline */}
                  <div className="bg-white rounded-xl shadow p-6">
                    <h3 className="font-semibold mb-2">Financial Projections</h3>
                    {projections && projections.length > 0 ? (
                      <ul>
                        {projections.map((proj: any, i: number) => (
                          <li key={i}>{proj.year}: Net Worth ₹{proj.net_worth}</li>
                        ))}
                      </ul>
                    ) : <div>No projections available.</div>}
                    {events && events.length > 0 && (
                      <ul>
                        {events.map((event: any, i: number) => (
                          <li key={i}>{event.event} in {event.year}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                  {/* Gauges */}
                  <div className="bg-white rounded-xl shadow p-6 flex gap-8">
                    <div>Risk Gauge: {gauges.risk ?? 'N/A'}%</div>
                    <div>Health Gauge: {gauges.health ?? 'N/A'}%</div>
                    <div>Savings Rate: {gauges.savings_rate ?? 'N/A'}%</div>
                  </div>
                  {/* Sankey Diagram (defensive rendering) */}
                  {Array.isArray(sankey?.nodes) && Array.isArray(sankey?.links) && sankey.nodes.length > 0 && sankey.links.length > 0 ? (
                    <div className="bg-white rounded-xl shadow p-6">
                      <h3 className="font-semibold mb-2">Income Flow (Sankey)</h3>
                      <Sankey
                        width={400}
                        height={250}
                        data={sankey}
                        nodePadding={30}
                        margin={{ top: 20, bottom: 20 }}
                        link={{ stroke: '#8884d8' }}
                        node={{ stroke: '#333', strokeWidth: 1 }}
                      />
                    </div>
                  ) : (
                    <div className="bg-white rounded-xl shadow p-6">
                      <h3 className="font-semibold mb-2">Income Flow (Sankey)</h3>
                      <div>No Sankey data available.</div>
                    </div>
                  )}
                  {/* Category Table */}
                  <div className="bg-white rounded-xl shadow p-6">
                    <h3 className="font-semibold mb-2">Summary Table</h3>
                    {categoryTable && categoryTable.length > 0 ? (
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr>
                            <th className="text-left">Category</th>
                            <th className="text-left">Current Status</th>
                            <th className="text-left">Recommended Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {categoryTable.map((row: any, i: number) => (
                            <tr key={i}>
                              <td>{row.category}</td>
                              <td>{row.current}</td>
                              <td>{row.recommended}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : <div>No summary data available.</div>}
                  </div>
                  {/* Markdown Report */}
                  {smartReport && (
                    <div className="bg-white rounded-xl shadow p-6">
                      <ReactMarkdown>{smartReport}</ReactMarkdown>
                    </div>
                  )}
                </TabsContent>
              </main>
            </Tabs>
          </div>
        </div>
      </div>
      {/* Financial Modal */}
      <FinancialModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSubmit={handleFinancialSubmit}
        isLoading={mlPredictionMutation.isPending}
      />
    </SidebarProvider>
  );
}

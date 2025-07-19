import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { X, Bot } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { financialDataSchema, type FinancialData } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";

interface FinancialModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: FinancialData) => void;
  isLoading?: boolean;
}

export function FinancialModal({ isOpen, onClose, onSubmit, isLoading = false }: FinancialModalProps) {
  const { toast } = useToast();
  
  const form = useForm<FinancialData>({
    resolver: zodResolver(financialDataSchema),
    defaultValues: {
      age: 30,
      occupation: "",
      grossIncome: 0,
      netIncome: 0,
      assets: 0,
      savings: 0,
      investments: 0,
      debt: 0,
      monthlyDebtPayments: 0,
      lifeInsurance: 0,
      goalsPriority: "retirement",
      housing: 0,
      transportation: 0,
      food: 0,
      utilities: 0,
      entertainment: 0,
      healthcare: 0,
      other: 0,
    },
  });

  const handleSubmit = async (data: FinancialData) => {
    try {
      // Validate that net income is not greater than gross income
      if (data.netIncome > data.grossIncome) {
        toast({
          title: "Validation Error",
          description: "Net income cannot be greater than gross income.",
          variant: "destructive",
        });
        return;
      }

      onSubmit(data);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to process financial data. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold text-gray-900 flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-red-400 to-red-500 rounded-lg flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
            Complete Your Financial Profile
          </DialogTitle>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-8">
            {/* Personal Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Personal Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="age"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Age</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          placeholder="30"
                          {...field}
                          onChange={(e) => field.onChange(parseInt(e.target.value) || 0)}
                          className="explorer-input"
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="occupation"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Occupation</FormLabel>
                      <FormControl>
                        <Input placeholder="Software Engineer" {...field} className="explorer-input" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            {/* Income Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Income Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="grossIncome"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Gross Monthly Income</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="8000"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="netIncome"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Net Monthly Income</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="6000"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            {/* Assets & Savings */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Assets & Savings</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <FormField
                  control={form.control}
                  name="assets"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Total Assets</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="50000"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="savings"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Savings</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="20000"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="investments"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Investments</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="15000"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            {/* Debt Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Debt Information</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="debt"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Total Debt</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="5000"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="monthlyDebtPayments"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Monthly Debt Payments</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="200"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            {/* Monthly Expenses */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Monthly Expenses</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <FormField
                  control={form.control}
                  name="housing"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Housing</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="1500"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="transportation"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Transportation</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="400"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="food"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Food & Dining</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="500"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="utilities"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Utilities</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="150"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="entertainment"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Entertainment</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="200"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="healthcare"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Healthcare</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="300"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            {/* Insurance & Goals */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Insurance & Goals</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="lifeInsurance"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Life Insurance Coverage</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <span className="absolute left-3 top-3 text-gray-400">$</span>
                          <Input
                            type="number"
                            placeholder="100000"
                            className="pl-8 explorer-input"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </div>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="goalsPriority"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Primary Financial Goal</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger className="explorer-input">
                            <SelectValue placeholder="Select your priority" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="retirement">Retirement</SelectItem>
                          <SelectItem value="house">House Purchase</SelectItem>
                          <SelectItem value="emergency">Emergency Fund</SelectItem>
                          <SelectItem value="education">Education</SelectItem>
                          <SelectItem value="travel">Travel</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            <div className="flex justify-end space-x-4 pt-6 border-t">
              <Button
                type="button"
                variant="outline"
                onClick={onClose}
                disabled={isLoading}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={isLoading}
                className="explorer-button bg-gradient-to-r from-red-400 to-red-500 text-white"
              >
                {isLoading ? (
                  "Analyzing..."
                ) : (
                  <>
                    <Bot className="w-4 h-4 mr-2" />
                    Get AI Analysis
                  </>
                )}
              </Button>
            </div>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}

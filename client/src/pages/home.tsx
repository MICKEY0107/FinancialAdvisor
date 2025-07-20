import { useState } from "react";
import { useLocation } from "wouter";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function Home() {
  const [, setLocation] = useLocation();
  const [showSignup, setShowSignup] = useState(false);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-slate-100 via-white to-gray-100">
      <Card className="w-full max-w-md p-8 shadow-lg rounded-2xl">
        <h1 className="text-3xl font-bold text-center text-red-500 mb-2">AI Financial Advisor</h1>
        <p className="text-center text-gray-500 mb-8">Smarter money decisions, powered by AI.</p>
        {showSignup ? (
          <form className="space-y-6">
            <div>
              <Label htmlFor="signup-email">Email</Label>
              <Input id="signup-email" type="email" placeholder="you@email.com" required className="mt-1" />
            </div>
            <div>
              <Label htmlFor="signup-password">Password</Label>
              <Input id="signup-password" type="password" placeholder="••••••••" required className="mt-1" />
            </div>
            <Button type="button" className="w-full" onClick={() => setLocation("/dashboard")}>Sign Up & Start Demo</Button>
            <p className="text-sm text-center text-gray-500 mt-2">
              Already have an account?{' '}
              <button type="button" className="text-red-500 hover:underline" onClick={() => setShowSignup(false)}>
                Login
              </button>
            </p>
          </form>
        ) : (
          <form className="space-y-6">
            <div>
              <Label htmlFor="login-email">Email</Label>
              <Input id="login-email" type="email" placeholder="you@email.com" required className="mt-1" />
            </div>
            <div>
              <Label htmlFor="login-password">Password</Label>
              <Input id="login-password" type="password" placeholder="••••••••" required className="mt-1" />
            </div>
            <Button type="button" className="w-full" onClick={() => setLocation("/dashboard")}>Login & Start Demo</Button>
            <p className="text-sm text-center text-gray-500 mt-2">
              New here?{' '}
              <button type="button" className="text-red-500 hover:underline" onClick={() => setShowSignup(true)}>
                Create an account
              </button>
            </p>
          </form>
        )}
      </Card>
      <footer className="mt-8 text-gray-400 text-xs text-center">
        &copy; {new Date().getFullYear()} AI Financial Advisor. All rights reserved.
      </footer>
    </div>
  );
} 
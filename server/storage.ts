// Simple storage interface for the financial advisor app
// Since the app uses localStorage for data persistence and doesn't require
// server-side storage, this file provides a minimal storage interface

export interface IStorage {
  // Add any storage methods here if needed in the future
}

export class MemStorage implements IStorage {
  constructor() {
    // No storage needed for this application
  }
}

export const storage = new MemStorage();

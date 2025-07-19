# Explorer PM - AI-Powered Financial Advisor

## Overview

Explorer PM is a modern, single-page web application designed to provide AI-powered financial advisory services. The application combines a React-based frontend with an Express.js backend and integrates machine learning capabilities through a Python script for generating personalized financial recommendations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **UI Library**: shadcn/ui components built on Radix UI primitives
- **Styling**: Tailwind CSS with custom design tokens
- **State Management**: React Query for server state, React hooks for local state
- **Routing**: Wouter for lightweight client-side routing
- **Forms**: React Hook Form with Zod validation

### Backend Architecture
- **Framework**: Express.js with TypeScript
- **API Design**: RESTful endpoints with `/api` prefix
- **Validation**: Zod schemas for request/response validation
- **ML Integration**: Python script execution via Node.js child_process
- **Development**: Hot reload with Vite middleware integration

### Database & Storage
- **Primary Database**: PostgreSQL with Drizzle ORM
- **Local Storage**: Browser localStorage/sessionStorage for user data persistence
- **Session Management**: In-memory storage for development (no authentication required)

## Key Components

### Core Features
1. **Financial Data Input**: Comprehensive form for user financial information
2. **AI Predictions**: Machine learning analysis via Python integration
3. **Interactive Dashboards**: Multi-tab interface (Dashboard, Expenses, Investments, Insurance, Goals, Reports)
4. **Data Visualization**: Chart.js integration for financial breakdowns
5. **Export Functionality**: PDF and CSV export capabilities
6. **Responsive Design**: Mobile-first approach with custom color palette

### Technology Stack
- **Frontend**: React, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: Node.js, Express, TypeScript
- **Database**: PostgreSQL with Drizzle ORM
- **ML/AI**: Python script with scikit-learn, XGBoost, LightGBM
- **Charts**: Chart.js for data visualization
- **Development**: Vite, ESBuild, tsx for execution

## Data Flow

1. **User Input**: Financial data collected through validated forms
2. **Local Storage**: Data persisted in browser for offline access
3. **ML Processing**: Financial data sent to `/api/predict` endpoint
4. **Python Execution**: Backend spawns Python process with user data
5. **AI Analysis**: Machine learning models generate predictions and recommendations
6. **Response Handling**: Results displayed in interactive dashboards
7. **Export Options**: Users can download analysis as PDF/CSV

### Data Validation
- Zod schemas ensure type safety across frontend and backend
- Form validation prevents invalid data submission
- API endpoints validate incoming requests before processing

## External Dependencies

### Frontend Dependencies
- **UI Components**: Radix UI primitives for accessibility
- **Charts**: Chart.js for interactive visualizations
- **Export**: jsPDF and PapaParse for file generation
- **Forms**: React Hook Form with Zod resolvers
- **HTTP Client**: Fetch API with React Query wrapper

### Backend Dependencies
- **Database**: @neondatabase/serverless for PostgreSQL connection
- **Validation**: Zod for schema validation
- **Development**: Various development and build tools

### Python ML Dependencies
- **Core ML**: scikit-learn, XGBoost, LightGBM, Prophet
- **Data Processing**: pandas, numpy
- **Utilities**: logging, warnings, pathlib

## Deployment Strategy

### Build Process
1. **Frontend Build**: Vite compiles React app to `dist/public`
2. **Backend Build**: ESBuild bundles server code to `dist/index.js`
3. **Production Mode**: Static file serving with Express

### Environment Configuration
- **Development**: Hot reload with Vite middleware
- **Production**: Optimized static file serving
- **Database**: Environment variable configuration for DATABASE_URL
- **Python Integration**: Runtime execution of ML script

### Key Scripts
- `dev`: Development server with hot reload
- `build`: Production build for both frontend and backend
- `start`: Production server execution
- `db:push`: Database schema synchronization

The application is designed as a professional financial advisory tool that combines modern web technologies with AI capabilities, providing users with personalized financial insights without requiring user accounts or complex authentication systems.
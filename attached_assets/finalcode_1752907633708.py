import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from prophet import Prophet
import joblib
import json
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('explorer_pm.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    test_size: float = 0.2
    random_state: int = 42
    xgb_params: Dict = None
    lgbm_params: Dict = None
    rf_params: Dict = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 4,
                'random_state': self.random_state
            }
        if self.lgbm_params is None:
            self.lgbm_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'random_state': self.random_state,
                'verbose': -1
            }
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 100,
                'random_state': self.random_state
            }

class DataValidator:
    """Data validation and quality checks"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input dataframe"""
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        if df.shape[0] < 10:
            issues.append(f"Too few rows: {df.shape[0]} (minimum 10 required)")
        
        # Check for required columns
        required_cols = ['Age', 'Occupation', 'Gross monthly income', 'Net monthly income']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for extreme values
        if 'Age' in df.columns:
            if df['Age'].min() < 18 or df['Age'].max() > 100:
                issues.append("Age values outside reasonable range (18-100)")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_numerical_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
        """Validate numerical columns"""
        issues = []
        for col in cols:
            if col in df.columns:
                if df[col].dtype not in ['int64', 'float64']:
                    issues.append(f"Column {col} is not numerical")
                if df[col].isnull().all():
                    issues.append(f"Column {col} is entirely null")
        return issues

class FeatureEngineer:
    """Enhanced feature engineering with validation"""
    
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features with error handling"""
        try:
            # Safe division with handling of zero values
            df['Debt_to_Income'] = np.where(
                df['Gross monthly income'] != 0,
                df['Debt'] / df['Gross monthly income'],
                0
            )
            df['Savings_to_Assets'] = np.where(
                df['Assets'] != 0,
                df['Savings'] / df['Assets'],
                0
            )
            df['Investment_to_Income'] = np.where(
                df['Net monthly income'] != 0,
                df['Investments'] / (df['Net monthly income'] * 12),
                0
            )
            
            # Cap extreme ratios
            ratio_cols = ['Debt_to_Income', 'Savings_to_Assets', 'Investment_to_Income']
            for col in ratio_cols:
                df[col] = np.clip(df[col], 0, 10)  # Cap at 10x
                
            logger.info("Ratio features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating ratio features: {e}")
            
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features"""
        try:
            df['Total_Income'] = (df['Gross monthly income'] + df['Net monthly income']) / 2
            df['Net_Worth'] = df['Assets'] + df['Investments'] + df['Savings'] - df['Debt']
            df['Liquid_Assets'] = df['Savings'] + df['Investments']
            df['Monthly_Surplus'] = df['Net monthly income'] - df['Total_Expenses'] if 'Total_Expenses' in df.columns else 0
            
            logger.info("Aggregated features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating aggregated features: {e}")
            
        return df
    
    def create_age_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age bins with better categorization"""
        try:
            bins = [0, 25, 35, 50, 65, 100]
            labels = [4, 3, 2, 1, 0]  # Higher values for younger (more aggressive)
            df['Age_Binned'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
            df['Age_Binned'] = df['Age_Binned'].astype(int)
            df['Risk_Profile'] = df['Age_Binned']
            
            logger.info("Age binning completed successfully")
            
        except Exception as e:
            logger.error(f"Error in age binning: {e}")
            # Fallback to simple binning
            df['Age_Binned'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=[2, 1, 0]).astype(int)
            df['Risk_Profile'] = df['Age_Binned']
            
        return df

class ModelTrainer:
    """Enhanced model training with validation and hyperparameter tuning"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def train_expense_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                           X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """Train expense prediction model with validation"""
        try:
            # Hyperparameter tuning for XGBoost
            param_grid = {
                'estimator__n_estimators': [100, 200, 300],
                'estimator__learning_rate': [0.01, 0.05, 0.1],
                'estimator__max_depth': [3, 4, 5]
            }
            
            xgb_model = MultiOutputRegressor(XGBRegressor(**self.config.xgb_params))
            
            # Use a subset for hyperparameter tuning if dataset is large
            if len(X_train) > 100:
                sample_size = min(100, len(X_train))
                sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
                X_sample, y_sample = X_train.iloc[sample_idx], y_train.iloc[sample_idx]
            else:
                X_sample, y_sample = X_train, y_train
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=0
            )
            
            try:
                grid_search.fit(X_sample, y_sample)
                best_model = grid_search.best_estimator_
                logger.info(f"Best XGB parameters: {grid_search.best_params_}")
            except:
                # Fallback to default model if grid search fails
                best_model = xgb_model
                logger.warning("Grid search failed, using default parameters")
            
            # Train final model
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.models['expense'] = best_model
            
            metrics = {
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'model_type': 'XGBoost MultiOutput'
            }
            
            logger.info(f"Expense model trained - MAE: {mae:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training expense model: {e}")
            return {'error': str(e)}
    
    def train_investment_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train investment recommendation model"""
        try:
            # Handle missing values in target
            y = y.fillna(y.mode()[0] if not y.mode().empty else 1)
            
            # Hyperparameter tuning for Random Forest
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            
            rf_model = RandomForestClassifier(**self.config.rf_params)
            
            # Grid search if dataset is reasonably sized
            if len(X) >= 30:
                grid_search = GridSearchCV(
                    rf_model, param_grid, cv=3, scoring='accuracy',
                    n_jobs=-1, verbose=0
                )
                try:
                    grid_search.fit(X, y)
                    best_model = grid_search.best_estimator_
                    logger.info(f"Best RF parameters: {grid_search.best_params_}")
                except:
                    best_model = rf_model
                    logger.warning("Grid search failed for investment model, using default parameters")
            else:
                best_model = rf_model
                best_model.fit(X, y)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X, y, cv=3, scoring='accuracy')
            
            self.models['investment'] = best_model
            
            metrics = {
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model_type': 'Random Forest'
            }
            
            logger.info(f"Investment model trained - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training investment model: {e}")
            return {'error': str(e)}
    
    def train_insurance_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train insurance gap model"""
        try:
            lgbm_model = LGBMRegressor(**self.config.lgbm_params)
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }
            
            if len(X_train) >= 30:
                grid_search = GridSearchCV(
                    lgbm_model, param_grid, cv=3, scoring='neg_mean_absolute_error',
                    n_jobs=-1, verbose=0
                )
                try:
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    logger.info(f"Best LGBM parameters: {grid_search.best_params_}")
                except:
                    best_model = lgbm_model
                    best_model.fit(X_train, y_train)
                    logger.warning("Grid search failed for insurance model, using default parameters")
            else:
                best_model = lgbm_model
                best_model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
            
            self.models['insurance'] = best_model
            
            metrics = {
                'cv_mae': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model_type': 'LightGBM'
            }
            
            logger.info(f"Insurance model trained - CV MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training insurance model: {e}")
            return {'error': str(e)}

class ExplorerPM:
    """Main Explorer PM class with enhanced reliability"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(self.config)
        self.df = None
        self.original_df = None  # Added to preserve original values
        self.feature_columns = []
        self.expense_targets = [
            'Rent/Mortgage', 'Utilities', 'Insurance', 'Car Payment', 'Debt Payments', 
            'Groceries', 'Clothes', 'Phone', 'Subscriptions', 'Miscellaneous', 
            'Vacations', 'Gifts', 'Emergency Fund', 'Dining out', 'Movies', 'Other'
        ]
        
    def load_and_validate_data(self, data_path: str) -> bool:
        """Load and validate data with comprehensive checks"""
        try:
            # Check if file exists
            if not Path(data_path).exists():
                logger.error(f"Data file not found: {data_path}")
                return False
            
            # Load data
            self.df = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Validate data
            is_valid, issues = self.validator.validate_dataframe(self.df)
            if not is_valid:
                logger.error(f"Data validation failed: {issues}")
                return False
            
            # Additional validation for numerical columns
            numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            num_issues = self.validator.validate_numerical_columns(self.df, numerical_cols)
            if num_issues:
                logger.warning(f"Numerical column issues: {num_issues}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self) -> bool:
        """Enhanced data preprocessing with error handling"""
        try:
            if self.df is None:
                logger.error("No data loaded")
                return False
            
            logger.info("Starting data preprocessing...")
            
            # Identify column types
            numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = ['Occupation'] if 'Occupation' in self.df.columns else []
            
            # Handle missing values with imputation
            if numerical_cols:
                self.df[numerical_cols] = self.feature_engineer.imputer_num.fit_transform(self.df[numerical_cols])
            
            if categorical_cols:
                self.df[categorical_cols] = self.feature_engineer.imputer_cat.fit_transform(self.df[categorical_cols])
            
            # Outlier handling with IQR method
            for col in numerical_cols:
                if col in self.df.columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Count outliers before capping
                    outliers_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                    if outliers_count > 0:
                        logger.info(f"Capping {outliers_count} outliers in {col}")
                        self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
            
            # Encode categorical features
            if categorical_cols:
                encoded_cats = self.feature_engineer.encoder.fit_transform(self.df[categorical_cols])
                encoded_df = pd.DataFrame(
                    encoded_cats, 
                    columns=self.feature_engineer.encoder.get_feature_names_out(categorical_cols)
                )
                self.df = pd.concat([self.df.drop(categorical_cols, axis=1), encoded_df], axis=1)
            
            # Update numerical columns list
            numerical_cols = [col for col in self.df.columns 
                            if self.df[col].dtype in ['float64', 'int64'] 
                            and not col.startswith('Occupation_')]
            
            logger.info("Data preprocessing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            return False
    
    def engineer_features(self) -> bool:
        """Feature engineering with validation"""
        try:
            logger.info("Starting feature engineering...")
            
            # Preserve original data before transformations
            self.original_df = self.df.copy()
            
            # Create ratio features
            self.df = self.feature_engineer.create_ratio_features(self.df)
            
            # Create aggregated features
            self.df = self.feature_engineer.create_aggregated_features(self.df)
            
            # Create age bins
            self.df = self.feature_engineer.create_age_bins(self.df)
            
            # Create interaction terms
            occupation_cols = [col for col in self.df.columns if col.startswith('Occupation_')]
            for occ_col in occupation_cols:
                self.df[f'Age_{occ_col}'] = self.df['Age'] * self.df[occ_col]
            
            # Scale numerical features
            numerical_cols = [col for col in self.df.columns 
                            if self.df[col].dtype in ['float64', 'int64'] 
                            and not col.startswith('Occupation_')]
            
            if numerical_cols:
                self.df[numerical_cols] = self.feature_engineer.scaler.fit_transform(self.df[numerical_cols])
            
            # PCA for dimensionality reduction
            pca_cols = numerical_cols + ['Debt_to_Income', 'Savings_to_Assets', 'Total_Income', 'Net_Worth', 'Age_Binned']
            pca_cols = [col for col in pca_cols if col in self.df.columns]
            
            if len(pca_cols) > 5:  # Only apply PCA if we have enough features
                pca = PCA(n_components=0.95)
                pca_features = pca.fit_transform(self.df[pca_cols])
                pca_df = pd.DataFrame(
                    pca_features, 
                    columns=[f'PCA_{i}' for i in range(pca_features.shape[1])]
                )
                self.df = pd.concat([self.df, pca_df], axis=1)
                logger.info(f"PCA applied: {pca_features.shape[1]} components retained")
            
            # Create target features
            self.df['Annual_Income'] = self.df['Net monthly income'] * 12
            self.df['HLV_Target'] = self.df['Annual_Income'] * 10
            
            # Calculate total expenses
            expense_cols = [col for col in self.expense_targets if col in self.df.columns]
            if expense_cols:
                self.df['Total_Expenses'] = self.df[expense_cols].sum(axis=1)
            
            logger.info("Feature engineering completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return False
    
    def train_models(self) -> Dict[str, Any]:
        """Train all models with comprehensive validation"""
        try:
            logger.info("Starting model training...")
            
            # Define features (exclude targets and derived columns)
            exclude_cols = (self.expense_targets + 
                          ['Risk_Profile', 'HLV_Target', 'Annual_Income', 'Total_Expenses', 'Anomaly'])
            self.feature_columns = [col for col in self.df.columns if col not in exclude_cols]
            
            # Train-test split
            train_df, test_df = train_test_split(
                self.df, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state,
                stratify=self.df['Risk_Profile'] if 'Risk_Profile' in self.df.columns else None
            )
            
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            
            logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
            
            # Prepare training data
            X_train, X_test = train_df[self.feature_columns], test_df[self.feature_columns]
            
            # Train expense model
            expense_cols = [col for col in self.expense_targets if col in train_df.columns]
            if expense_cols:
                y_train_exp, y_test_exp = train_df[expense_cols], test_df[expense_cols]
                expense_metrics = self.model_trainer.train_expense_model(
                    X_train, y_train_exp, X_test, y_test_exp
                )
            else:
                expense_metrics = {'error': 'No expense columns found'}
            
            # Train investment model
            if 'Risk_Profile' in train_df.columns:
                investment_metrics = self.model_trainer.train_investment_model(
                    X_train, train_df['Risk_Profile']
                )
            else:
                investment_metrics = {'error': 'Risk_Profile column not found'}
            
            # Train insurance model
            if 'HLV_Target' in train_df.columns:
                insurance_metrics = self.model_trainer.train_insurance_model(
                    X_train, train_df['HLV_Target']
                )
            else:
                insurance_metrics = {'error': 'HLV_Target column not found'}
            
            # Anomaly detection
            if expense_cols:
                iso_forest = IsolationForest(contamination=0.1, random_state=self.config.random_state)
                anomalies = iso_forest.fit_predict(self.df[expense_cols])
                self.df['Anomaly'] = anomalies
            
            # Time series forecasting
            forecast_metrics = self.train_forecast_model()
            
            results = {
                'expense_model': expense_metrics,
                'investment_model': investment_metrics,
                'insurance_model': insurance_metrics,
                'forecast_model': forecast_metrics,
                'training_completed': True
            }
            
            logger.info("Model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {'error': str(e)}
    
    def train_forecast_model(self) -> Dict[str, Any]:
        """Train time series forecasting model"""
        try:
            if 'Total_Expenses' not in self.df.columns:
                return {'error': 'Total_Expenses column not found'}
            
            # Prepare time series data
            ts_df = pd.DataFrame({
                'ds': pd.date_range(start='2023-01-01', periods=len(self.df), freq='ME'),
                'y': self.df['Total_Expenses']
            })
            
            # Train Prophet model
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            prophet_model.fit(ts_df)
            
            # Generate forecast
            future = prophet_model.make_future_dataframe(periods=120, freq='ME')
            forecast = prophet_model.predict(future)
            
            # Save forecast plot
            try:
                fig = prophet_model.plot(forecast)
                plt.title('10-Year Expense Forecast')
                plt.tight_layout()
                plt.savefig('expense_forecast.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Forecast plot saved successfully")
            except Exception as e:
                logger.warning(f"Could not save forecast plot: {e}")
            
            self.model_trainer.models['forecast'] = prophet_model
            
            return {
                'model_type': 'Prophet',
                'forecast_periods': 120,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in forecast model training: {e}")
            return {'error': str(e)}
    
    def save_models(self, directory: str = 'models') -> bool:
        """Save trained models"""
        try:
            Path(directory).mkdir(exist_ok=True)
            
            for model_name, model in self.model_trainer.models.items():
                model_path = Path(directory) / f'{model_name}_model.pkl'
                joblib.dump(model, model_path)
                logger.info(f"Model saved: {model_path}")
            
            # Save feature columns
            feature_path = Path(directory) / 'feature_columns.json'
            with open(feature_path, 'w') as f:
                json.dump(self.feature_columns, f)
            
            # Save scalers and encoders
            scaler_path = Path(directory) / 'scaler.pkl'
            joblib.dump(self.feature_engineer.scaler, scaler_path)
            
            encoder_path = Path(directory) / 'encoder.pkl'
            joblib.dump(self.feature_engineer.encoder, encoder_path)
            
            logger.info("All models and preprocessing objects saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def generate_smart_report(self, user_index: int = 0) -> Optional[str]:
        """Generate enhanced personalized report using original data"""
        try:
            if self.original_df is None or user_index >= len(self.original_df):
                logger.error("Invalid user index or no original data available")
                return None
            
            # Use original data for report generation (not scaled data)
            user = self.original_df.iloc[user_index]
            
            # Get predictions from models using scaled features
            predictions = {}
            
            # Expense prediction
            if 'expense' in self.model_trainer.models:
                exp_pred = self.model_trainer.models['expense'].predict(
                    self.df[self.feature_columns].iloc[user_index:user_index+1]
                )[0]
                predictions['expenses'] = exp_pred
            
            # Risk prediction
            if 'investment' in self.model_trainer.models:
                risk_pred = self.model_trainer.models['investment'].predict(
                    self.df[self.feature_columns].iloc[user_index:user_index+1]
                )[0]
                predictions['risk'] = risk_pred
            
            # Insurance prediction
            if 'insurance' in self.model_trainer.models:
                hlv_pred = self.model_trainer.models['insurance'].predict(
                    self.df[self.feature_columns].iloc[user_index:user_index+1]
                )[0]
                predictions['hlv'] = hlv_pred
            
            # Generate structured report
            report = self._format_smart_report(user, predictions)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def _format_smart_report(self, user: pd.Series, predictions: Dict) -> str:
        """Format the smart report with enhanced insights using original values"""
        try:
            # Calculate derived metrics using original values
            age = int(user.get('Age', 0))
            assets = user.get('Assets', 0)
            investments = user.get('Investments', 0)
            savings = user.get('Savings', 0)
            debt = user.get('Debt', 0)
            net_worth = assets + investments + savings - debt
            net_income = user.get('Net monthly income', 0)
            gross_income = user.get('Gross monthly income', 0)
            
            # Calculate actual total expenses from original data
            expense_cols = [col for col in self.expense_targets if col in user.index]
            actual_expenses = sum([user[col] for col in expense_cols])
            
            # Fixed risk categorization based on age
            if age < 30:
                risk_category = "Aggressive"
                risk_score = 2
            elif age < 50:
                risk_category = "Moderate"
                risk_score = 1
            else:
                risk_category = "Conservative"
                risk_score = 0
            
            # Age-based insights
            age_insights = {
                (18, 30): "Young professional phase - Focus on aggressive growth and building emergency fund",
                (30, 45): "Career building phase - Balance growth with stability, plan for major goals",
                (45, 60): "Pre-retirement phase - Reduce risk, focus on wealth preservation",
                (60, 100): "Retirement phase - Conservative approach, focus on income generation"
            }
            
            age_insight = next(
                (insight for (min_age, max_age), insight in age_insights.items() 
                 if min_age <= age < max_age), 
                "Focus on balanced financial planning"
            )
            
            # Calculate correct savings rate
            monthly_surplus = net_income - actual_expenses
            savings_rate = (monthly_surplus / net_income * 100) if net_income > 0 else 0
            
            # Calculate correct HLV recommendation
            hlv_recommendation = net_income * 12 * 10  # 10x annual income
            insurance_gap = hlv_recommendation - user.get('Insurance', 0)
            
            # Investment recommendations based on risk profile
            if risk_category == "Aggressive":
                equity_allocation = "70-80%"
                debt_allocation = "20-30%"
                sip_suggestion = int(monthly_surplus * 0.7) if monthly_surplus > 0 else 0
                debt_fund_suggestion = int(monthly_surplus * 0.3) if monthly_surplus > 0 else 0
            elif risk_category == "Moderate":
                equity_allocation = "50-60%"
                debt_allocation = "40-50%"
                sip_suggestion = int(monthly_surplus * 0.6) if monthly_surplus > 0 else 0
                debt_fund_suggestion = int(monthly_surplus * 0.4) if monthly_surplus > 0 else 0
            else:
                equity_allocation = "30-40%"
                debt_allocation = "60-70%"
                sip_suggestion = int(monthly_surplus * 0.4) if monthly_surplus > 0 else 0
                debt_fund_suggestion = int(monthly_surplus * 0.6) if monthly_surplus > 0 else 0
            
            # Calculate financial health score
            health_score = min(100, int(
                (savings_rate if savings_rate > 0 else 0) * 2 + 
                (net_worth/100000) + 
                (40 if debt == 0 else 20 if debt < net_income * 6 else 0)
            ))
            
            # Generate report
            report = f"""
### User Profile Summary
- **Age**: {age} ({age_insight})
- **Occupation**: {user.get('Occupation', 'Not specified')}
- **Family Demographics**: Family size and dependents analysis recommended
- **Income**: Gross monthly ₹{gross_income:,.0f}; Net monthly ₹{net_income:,.0f}
- **Assets & Net Worth**: ₹{assets:,.0f} in assets, ₹{investments:,.0f} investments, ₹{savings:,.0f} savings, ₹{debt:,.0f} debt (Net Worth: ₹{net_worth:,.0f})
- **Current Savings Rate**: {savings_rate:.1f}% ({'Excellent' if savings_rate > 20 else 'Good' if savings_rate > 10 else 'Needs Improvement'})

### Expense Pattern Analysis
- **Actual Monthly Expenses**: ₹{actual_expenses:,.0f}
- **Monthly Surplus**: ₹{monthly_surplus:,.0f}
- **Spending Efficiency**: {'Excellent' if savings_rate > 30 else 'Good' if savings_rate > 20 else 'Needs Improvement'}
- **Anomaly Status**: {'Detected - Review spending patterns' if user.get('Anomaly', 1) == -1 else 'Normal spending pattern'}
- **Recommended Action**: {'Maintain current discipline' if savings_rate > 25 else 'Optimize expenses to increase savings rate'}

### Investment Recommendations
- **Risk Profiling**: {risk_category} (suitable for your age group)
- **Asset Allocation Strategy**: 
  - Equity: {equity_allocation}, Debt: {debt_allocation}
- **Monthly Investment Capacity**: ₹{max(0, monthly_surplus):,.0f}
- **Suggested Investments**:
  - ₹{sip_suggestion:,}/month in Equity SIP
  - ₹{debt_fund_suggestion:,}/month in Debt funds
- **Long-term Projection**: Potential to achieve ₹{int(net_worth * 1.08**10):,.0f} in 10 years with consistent investing

### Insurance Gap & Risk Assessment
- **Current Coverage**: Life ₹{user.get('Insurance', 0):,.0f}
- **Recommended Coverage**: Life ₹{hlv_recommendation:,.0f} (Human Life Value approach)
- **Health Insurance**: Minimum ₹{max(500000, net_income * 6):,.0f} for family coverage
- **Gap Analysis**: {'Adequately covered' if insurance_gap <= 0 else f'Underinsured by ₹{insurance_gap:,.0f} - Priority action needed'}
- **Premium Budget**: ₹{int(net_income * 0.05):,.0f}/month (5% of income rule)

### Predictive Financial Chart (Smart Insights)
- **1-Year Outlook**: {'Stable growth expected' if savings_rate > 15 else 'Focus on expense optimization'}
- **5-Year Projection**: Estimated net worth ₹{int(net_worth * 1.08**5):,.0f}
- **10-Year Vision**: Potential net worth ₹{int(net_worth * 1.08**10):,.0f}
- **Key Milestones**:
  - Emergency Fund: {'✓ Achieved' if savings > net_income * 6 else f'⚠ Build ₹{int(net_income * 6):,} emergency fund'}
  - Debt Freedom: {'✓ Achieved' if debt == 0 else f'⚠ Clear debt of ₹{int(debt):,}'}
  - Investment Growth: Target monthly SIP of ₹{int(max(0, monthly_surplus) * 0.7):,.0f}

### Overall Risk Index & Smart Recommendations
- **Financial Health Score**: {health_score}/100
- **Priority Actions**:
  1. {'✓ Maintain savings discipline' if savings_rate > 20 else '⚠ Increase savings rate to 25%'}
  2. {'✓ Insurance adequate' if insurance_gap <= 0 else '⚠ Increase life insurance coverage'}
  3. {'✓ Debt managed well' if debt < net_income * 6 else '⚠ Focus on debt reduction'}
- **Next Review**: Recommended in 6 months or after major life events

### AI-Powered Insights
- **Spending Pattern**: {self._analyze_spending_pattern_original(user)}
- **Investment Behavior**: {self._analyze_investment_behavior_original(user)}
- **Risk Tolerance**: Matches {risk_category.lower()} profile based on age and financial situation
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return "Error generating report. Please check the logs."
    
    def _analyze_spending_pattern_original(self, user: pd.Series) -> str:
        """Analyze spending patterns using original values"""
        try:
            debt = user.get('Debt', 0)
            gross_income = user.get('Gross monthly income', 0)
            debt_to_income = debt / gross_income if gross_income > 0 else 0
            
            if debt_to_income > 0.4:
                return "High debt burden - focus on debt reduction"
            elif debt_to_income > 0.2:
                return "Moderate debt levels - manageable with discipline"
            else:
                return "Low debt burden - excellent financial discipline"
                
        except:
            return "Unable to analyze spending pattern"
    
    def _analyze_investment_behavior_original(self, user: pd.Series) -> str:
        """Analyze investment behavior using original values"""
        try:
            investments = user.get('Investments', 0)
            net_income = user.get('Net monthly income', 0)
            investment_to_income = investments / (net_income * 12) if net_income > 0 else 0
            
            if investment_to_income > 0.3:
                return "Excellent investment discipline - above average"
            elif investment_to_income > 0.1:
                return "Good investment habit - room for improvement"
            else:
                return "Low investment rate - significant opportunity for growth"
                
        except:
            return "Unable to analyze investment behavior"

# Add these utility functions at the bottom of the file or in the ExplorerPM class as needed

def classify_expenses(data):
    fixed_categories = ['Rent/Mortgage', 'Utilities', 'Insurance', 'Car Payment', 'Debt Payments']
    variable_categories = ['Groceries', 'Clothes', 'Phone', 'Subscriptions', 'Miscellaneous', 'Vacations', 'Gifts', 'Dining out', 'Movies', 'Other']
    fixed_expenses = {k: v for k, v in data.items() if k in fixed_categories}
    variable_expenses = {k: v for k, v in data.items() if k in variable_categories}
    return fixed_expenses, variable_expenses

def predict_future_liabilities(data):
    future_liabilities = []
    if data.get('Car Payment', 0) > 0:
        future_liabilities.append({"type": "Car EMI", "amount": data['Car Payment'], "due_date": "next month"})
    # Add more logic for tuition, medical, etc.
    return future_liabilities

def detect_anomalies(data):
    anomalies = []
    if data.get('Dining out', 0) > 0.3 * data.get('Net monthly income', 1):
        anomalies.append("High spending on Dining out")
    # Add more rules or use your model
    return anomalies

def asset_rebalancing(data):
    # Use user input for dynamic allocation
    total_investment = data.get('Investments', 0) + data.get('Savings', 0)
    equity = data.get('Investments', 0)
    debt = data.get('Savings', 0)
    current_allocation = {
        "equity": round(100 * equity / total_investment, 1) if total_investment else 0,
        "debt": round(100 * debt / total_investment, 1) if total_investment else 0
    }
    age = data.get('Age', 30)
    # Dynamic recommended allocation based on age
    if age < 35:
        recommended = {"equity": 80, "debt": 20}
    elif age < 50:
        recommended = {"equity": 60, "debt": 40}
    else:
        recommended = {"equity": 40, "debt": 60}
    # SIP/ELSS/PPF suggestions based on surplus
    net_income = data.get('Net monthly income', 0)
    expense_sum = sum([v for k, v in data.items() if k in [
        'Rent/Mortgage', 'Utilities', 'Insurance', 'Car Payment', 'Debt Payments',
        'Groceries', 'Clothes', 'Phone', 'Subscriptions', 'Miscellaneous',
        'Vacations', 'Gifts', 'Emergency Fund', 'Dining out', 'Movies', 'Other']])
    monthly_surplus = net_income - expense_sum
    sip = int(monthly_surplus * 0.5) if monthly_surplus > 0 else 0
    elss = int(monthly_surplus * 0.2) if monthly_surplus > 0 else 0
    ppf = int(monthly_surplus * 0.1) if monthly_surplus > 0 else 0
    # Risk profile
    if age < 30:
        risk_profile = "Aggressive"
    elif age < 50:
        risk_profile = "Moderate"
    else:
        risk_profile = "Conservative"
    return {
        "risk_profile": risk_profile,
        "sip": sip,
        "elss": elss,
        "ppf": ppf,
        "asset_allocation": current_allocation,
        "recommended_allocation": recommended
    }

def health_critical_risk(data):
    # Dynamic health/critical illness risk based on age and debt
    age = data.get('Age', 30)
    debt = data.get('Debt', 0)
    health_risk_score = min(100, max(10, int((age - 18) * 1.5 + (debt / 100000) * 10)))
    critical_illness_score = min(100, max(5, int((age - 18) * 1.2 + (debt / 200000) * 10)))
    # Insurance gap
    current_life = data.get('Insurance', 0)
    net_income = data.get('Net monthly income', 0)
    recommended_life = net_income * 12 * 10 if net_income > 0 else 0
    gap = max(0, recommended_life - current_life)
    return {
        "current_life": current_life,
        "recommended_life": recommended_life,
        "gap": gap,
        "health_risk_score": health_risk_score,
        "critical_illness_score": critical_illness_score
    }

def gauges_data(data):
    # Dynamic gauges based on user input
    net_income = data.get('Net monthly income', 0)
    expense_sum = sum([v for k, v in data.items() if k in [
        'Rent/Mortgage', 'Utilities', 'Insurance', 'Car Payment', 'Debt Payments',
        'Groceries', 'Clothes', 'Phone', 'Subscriptions', 'Miscellaneous',
        'Vacations', 'Gifts', 'Emergency Fund', 'Dining out', 'Movies', 'Other']])
    savings = data.get('Savings', 0)
    debt = data.get('Debt', 0)
    # Risk: higher if debt is high
    risk = min(100, max(10, int((debt / (net_income * 12 + 1)) * 100))) if net_income > 0 else 50
    # Health: higher if savings are high
    health = min(100, max(10, int((savings / (expense_sum + 1)) * 100))) if expense_sum > 0 else 50
    # Savings rate
    savings_rate = int(((net_income - expense_sum) / net_income) * 100) if net_income > 0 else 0
    return {"risk": risk, "health": health, "savings_rate": savings_rate}

def category_table_data(data):
    # Dynamic summary table
    expense_sum = sum([v for k, v in data.items() if k in [
        'Rent/Mortgage', 'Utilities', 'Insurance', 'Car Payment', 'Debt Payments',
        'Groceries', 'Clothes', 'Phone', 'Subscriptions', 'Miscellaneous',
        'Vacations', 'Gifts', 'Emergency Fund', 'Dining out', 'Movies', 'Other']])
    net_income = data.get('Net monthly income', 0)
    insurance = data.get('Insurance', 0)
    recommended_life = net_income * 12 * 10 if net_income > 0 else 0
    gap = max(0, recommended_life - insurance)
    return [
        {"category": "Expense Pattern", "current": f"{int(100 * expense_sum / (net_income + 1))}% on lifestyle", "recommended": "Reduce to 30%"},
        {"category": "Insurance Cover", "current": f"₹{insurance/100000:.0f}L Life", "recommended": f"Need ₹{recommended_life/100000:.0f}L Life"},
        {"category": "Gap", "current": f"₹{gap:,.0f}", "recommended": "Close the gap with more coverage"}
    ]

def projections_and_events(data):
    net_worth = data.get('Assets', 0) + data.get('Investments', 0) + data.get('Savings', 0) - data.get('Debt', 0)
    projections = [
        {"year": 2025, "net_worth": int(net_worth * 1.08)},
        {"year": 2027, "net_worth": int(net_worth * 1.08**3)},
        {"year": 2034, "net_worth": int(net_worth * 1.08**10)},
    ]
    events = []
    if data.get('Savings', 0) < data.get('Net monthly income', 1) * 6:
        events.append({"year": 2025, "event": "Fund Shortage"})
    return projections, events

def sankey_data(data, fixed_expenses, variable_expenses):
    sankey = {
        "nodes": [
            {"name": "Income"},
            {"name": "Expenses"},
            {"name": "Investments"},
            {"name": "Insurance"}
        ],
        "links": [
            {"source": 0, "target": 1, "value": sum(fixed_expenses.values()) + sum(variable_expenses.values())},
            {"source": 0, "target": 2, "value": data.get('Investments', 0)},
            {"source": 0, "target": 3, "value": data.get('Insurance', 0)}
        ]
    }
    return sankey

# Main execution function
def main():
    """Main execution function with comprehensive error handling"""
    try:
        # Initialize configuration
        config = ModelConfig()
        
        # Initialize Explorer PM
        explorer = ExplorerPM(config)
        
        # Load and validate data
        data_path = '/content/eData_50 rows.csv'  # Update with your path
        if not explorer.load_and_validate_data(data_path):
            logger.error("Failed to load and validate data")
            return
        
        # Preprocess data
        if not explorer.preprocess_data():
            logger.error("Failed to preprocess data")
            return
        
        # Engineer features
        if not explorer.engineer_features():
            logger.error("Failed to engineer features")
            return
        
        # Train models
        training_results = explorer.train_models()
        
        # Print training results
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        
        for model_name, metrics in training_results.items():
            if isinstance(metrics, dict) and 'error' not in metrics:
                print(f"\n{model_name.upper()}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value}")
        
        # Save models
        if explorer.save_models():
            logger.info("Models saved successfully")
        
        # Generate smart report
        print("\n" + "="*60)
        print("PERSONALIZED FINANCIAL REPORT")
        print("="*60)
        
        report = explorer.generate_smart_report(0)
        if report:
            print(report)
        else:
            logger.error("Failed to generate report")
        
        logger.info("Explorer PM execution completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()

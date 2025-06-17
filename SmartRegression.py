# Experiment for Smart Regression
# A robust PyTorch-based regression pipeline with intelligent data preprocessing
# and natural language query interface for mixed data types

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# Global variables
INTEL_XPU_AVAILABLE = False
model_data = {}

def detect_xpu():
    """Detect Intel XPU availability."""
    global INTEL_XPU_AVAILABLE
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        INTEL_XPU_AVAILABLE = True
        print("Intel XPU detected")
    else:
        try:
            import intel_extension_for_pytorch as ipex
            INTEL_XPU_AVAILABLE = True
            print("Intel XPU available via IPEX")
        except ImportError:
            print("Intel XPU not available, using CPU/CUDA")

def read_excel_data(excel_path):
    """Read data from Excel file."""
    try:
        df = pd.read_excel(excel_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def preprocess_data(df, target_column):
    """Preprocess mixed data intelligently."""
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found")
        return None, None, None
    
    df_processed = df.copy()
    
    # Convert target to numeric
    if not pd.api.types.is_numeric_dtype(df_processed[target_column]):
        df_processed[target_column] = pd.to_numeric(df_processed[target_column], errors='coerce')
        df_processed = df_processed.dropna(subset=[target_column])
    
    X = df_processed.drop(columns=[target_column])
    y = df_processed[[target_column]]
    
    # Drop administrative columns
    drop_columns = ['Unnamed: 0', 'Unnamed: 9', 'Who tested', 'Logs', 'GFX driver']
    for col in drop_columns:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # Extract numeric values from text
    patterns = {
        'freq': r'(\d+\.?\d*)\s*(?:MHz|mhz)',
        'power': r'(\d+\.?\d*)\s*(?:W|w)',
        'temp': r'(\d+\.?\d*)\s*(?:°C|C)',
        'memory': r'(\d+\.?\d*)\s*(?:GB|gb|MB|mb)',
        'percent': r'(\d+\.?\d*)\s*(?:%|percent)'
    }
    
    for col in X.columns:
        if X[col].dtype == 'object':
            for pattern in patterns.values():
                try:
                    extracted = X[col].astype(str).str.extract(pattern, expand=False)
                    if extracted.dropna().shape[0] > len(X[col].dropna()) * 0.7:
                        X[col] = pd.to_numeric(extracted, errors='coerce')
                        break
                except:
                    continue
    
    # Handle categorical columns
    ordinal_mappings = {
        'Game Setting': ['Low', 'Medium', 'High', 'Ultra', 'Maximum'],
        'OS setting': ['Power Save', 'Balanced', 'Performance', 'Gaming']
    }
    
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    label_encoders = {}
    
    for col in categorical_cols:
        unique_count = X[col].nunique()
        
        if col in ordinal_mappings:
            mapping = {val: idx for idx, val in enumerate(ordinal_mappings[col])}
            X[col] = X[col].map(mapping).fillna(-1)
        elif unique_count <= 15:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=[col])
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Fill missing values
    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
    
    print(f"Preprocessing complete: {X.shape[1]} features")
    return X, y, label_encoders

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        
        if input_dim <= 20:
            hidden_dims = [64, 32]
        elif input_dim <= 50:
            hidden_dims = [128, 64, 32]
        else:
            hidden_dims = [256, 128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model(X, y, epochs=200, batch_size=32):
    """Train the regression model."""
    global model_data
    
    # Device selection
    if INTEL_XPU_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print("Training on Intel XPU")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on {device}")
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    # Data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Model initialization
    model = RegressionModel(X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 15
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    y_pred = scaler_y.inverse_transform(predictions.cpu().numpy())
    y_true = y_test.values
    
    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nTraining Complete!")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Store model data globally
    model_data = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': X.columns.tolist(),
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae, 'mse': mse},
        'predictions': pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()}),
        'device': device
    }
    
    return model_data

def find_relevant_columns(query, df):
    """Find the most relevant columns for a given query."""
    query_words = query.lower().split()
    
    # Define keyword mappings
    keyword_mappings = {
        'power': ['power', 'watt', 'consumption', 'energy'],
        'frequency': ['frequency', 'freq', 'mhz', 'ghz', 'clock', 'speed'],
        'gpu': ['gpu', 'graphics', 'video', 'vga', 'gt'],
        'cpu': ['cpu', 'processor', 'core'],
        'memory': ['memory', 'ram', 'mem', 'gb'],
        'temperature': ['temp', 'temperature', 'thermal', 'celsius'],
        'performance': ['fps', 'frame', 'score', 'performance', 'benchmark'],
        'game': ['game', 'title', 'application', 'app'],
        'setting': ['setting', 'quality', 'resolution', 'config']
    }
    
    # Score columns based on relevance
    column_scores = {}
    for col in df.columns:
        score = 0
        col_lower = col.lower()
        
        # Direct word matches
        for word in query_words:
            if word in col_lower:
                score += 10
        
        # Keyword category matches
        for category, keywords in keyword_mappings.items():
            if any(keyword in col_lower for keyword in keywords):
                if any(word in keywords for word in query_words):
                    score += 8
                elif category in query_words:
                    score += 8
        
        column_scores[col] = score
    
    # Return columns sorted by relevance
    relevant_cols = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)
    return [col for col, score in relevant_cols if score > 0]

def clean_and_prepare_data(df):
    """Clean data and convert to appropriate types."""
    df_clean = df.copy()
    
    # Remove rows that are clearly headers (contain units in parentheses)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Remove rows where the value looks like a unit "(W)" or "(MHz)"
            mask = df_clean[col].astype(str).str.contains(r'\([^)]*\)', na=False)
            df_clean = df_clean[~mask]
    
    # Convert numeric columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
            if numeric_series.notna().sum() > len(df_clean) * 0.3:  # If at least 30% can be converted
                df_clean[col] = numeric_series
    
    return df_clean

def answer_any_query(user_query, original_data):
    """Answer any user query about the data intelligently."""
    query = user_query.lower()
    
    try:
        # Clean the data
        df = clean_and_prepare_data(original_data)
        
        # Find relevant columns
        relevant_cols = find_relevant_columns(query, df)
        if not relevant_cols:
            relevant_cols = list(df.columns)
        
        # Identify column types
        game_cols = [col for col in df.columns if any(word in col.lower() for word in ['game', 'title', 'app']) and df[col].dtype == 'object']
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Determine query intent and find target columns
        target_cols = []
        
        # GPU frequency queries
        if any(word in query for word in ['gpu', 'graphics']) and any(word in query for word in ['frequency', 'freq', 'mhz', 'clock']):
            gpu_freq_cols = [col for col in relevant_cols if 
                           any(gpu_word in col.lower() for gpu_word in ['gpu', 'gt', 'graphics']) and
                           any(freq_word in col.lower() for freq_word in ['freq', 'mhz', 'clock']) and
                           col in numeric_cols]
            target_cols = gpu_freq_cols
        
        # CPU frequency queries
        elif any(word in query for word in ['cpu', 'processor', 'core']) and any(word in query for word in ['frequency', 'freq', 'mhz', 'clock']):
            cpu_freq_cols = [col for col in relevant_cols if 
                           any(cpu_word in col.lower() for cpu_word in ['cpu', 'core', 'processor']) and
                           any(freq_word in col.lower() for freq_word in ['freq', 'mhz', 'clock']) and
                           col in numeric_cols]
            target_cols = cpu_freq_cols
        
        # Power queries
        elif any(word in query for word in ['power', 'watt', 'consumption']):
            power_cols = [col for col in relevant_cols if 
                         any(power_word in col.lower() for power_word in ['power', 'watt', 'consumption']) and
                         col in numeric_cols]
            target_cols = power_cols
        
        # Performance queries
        elif any(word in query for word in ['fps', 'frame', 'performance', 'score']):
            perf_cols = [col for col in relevant_cols if 
                        any(perf_word in col.lower() for perf_word in ['fps', 'frame', 'score', 'performance']) and
                        col in numeric_cols]
            target_cols = perf_cols
        
        # Temperature queries
        elif any(word in query for word in ['temp', 'temperature', 'thermal']):
            temp_cols = [col for col in relevant_cols if 
                        any(temp_word in col.lower() for temp_word in ['temp', 'temperature', 'thermal']) and
                        col in numeric_cols]
            target_cols = temp_cols
        
        # Memory queries
        elif any(word in query for word in ['memory', 'ram', 'mem']):
            mem_cols = [col for col in relevant_cols if 
                       any(mem_word in col.lower() for mem_word in ['memory', 'ram', 'mem']) and
                       col in numeric_cols]
            target_cols = mem_cols
        
        # If no specific target found, use the most relevant numeric columns
        if not target_cols:
            target_cols = [col for col in relevant_cols[:3] if col in numeric_cols]
        
        if not target_cols:
            print("Could not find relevant numeric data for your query.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Determine if looking for highest or lowest values
        if any(word in query for word in ['more', 'most', 'higher', 'highest', 'maximum', 'max', 'needs', 'requires']):
            ascending = False
            direction = "highest"
        elif any(word in query for word in ['less', 'least', 'lower', 'lowest', 'minimum', 'min']):
            ascending = True
            direction = "lowest"
        else:
            ascending = False
            direction = "highest"
        
        # Answer the query
        main_col = target_cols[0]
        
        if game_cols:
            # Group by game and analyze
            game_col = game_cols[0]
            game_stats = df.groupby(game_col)[main_col].agg(['mean', 'max', 'min']).sort_values('mean', ascending=ascending)
            game_stats = game_stats.dropna()
            
            if len(game_stats) == 0:
                print("No valid data found for analysis.")
                return
            
            print(f"\nGames ranked by {direction} {main_col}:")
            for i, (game, stats) in enumerate(game_stats.head(10).iterrows()):
                print(f"{i+1}. {game}")
                print(f"   Average: {stats['mean']:.2f}")
                print(f"   Range: {stats['min']:.2f} - {stats['max']:.2f}")
            
            # Answer the specific question
            top_game = game_stats.index[0]
            top_value = game_stats.iloc[0]['mean']
            
            if 'needs' in query or 'requires' in query:
                print(f"\nANSWER: '{top_game}' needs the {direction} {main_col} ({top_value:.2f})")
            else:
                print(f"\nANSWER: '{top_game}' has the {direction} {main_col} ({top_value:.2f})")
                
        else:
            # No game column, just show the extreme values
            if ascending:
                extreme_idx = df[main_col].idxmin()
                extreme_val = df[main_col].min()
            else:
                extreme_idx = df[main_col].idxmax()
                extreme_val = df[main_col].max()
            
            print(f"\n{direction.capitalize()} {main_col}: {extreme_val:.2f}")
            print("Configuration:")
            for col in df.columns[:8]:
                val = df.loc[extreme_idx, col]
                if pd.notna(val):
                    print(f"  {col}: {val}")
        
        # Show additional insights if multiple target columns
        if len(target_cols) > 1:
            print(f"\nAdditional metrics:")
            for col in target_cols[1:3]:
                if game_cols:
                    game_stats = df.groupby(game_cols[0])[col].mean().sort_values(ascending=ascending)
                    game_stats = game_stats.dropna()
                    if len(game_stats) > 0:
                        print(f"  {direction.capitalize()} {col}: {game_stats.index[0]} ({game_stats.iloc[0]:.2f})")
    
    except Exception as e:
        print(f"Error processing query: {e}")
        print("Please try rephrasing your question.")
        import traceback
        traceback.print_exc()

def interactive_session():
    """Interactive session that can answer any question about the data."""
    print("\n" + "="*50)
    print("Interactive Data Analysis Session")
    print("Ask me anything about your data!")
    print("Type 'exit' to quit")
    print("="*50)
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        elif user_input.lower() == 'help':
            print("\nYou can ask any question about your data, for example:")
            print("- 'What is the highest score?'")
            print("- 'Show me the lowest power consumption'")
            print("- 'What is the average FPS?'")
            print("- 'How many different games are there?'")
            print("- 'Compare power and temperature'")
            print("- 'Tell me about CPU usage'")
            print("- Or any other question about your data!")
        
        elif user_input.lower() in ['columns', 'features']:
            if 'original_df' in globals():
                print(f"\nAvailable data columns:")
                for i, col in enumerate(original_df.columns, 1):
                    print(f"{i:2d}. {col}")
            else:
                print("No data loaded")
        
        elif user_input.lower() == 'summary':
            if 'original_df' in globals():
                print(f"\nDataset Summary:")
                print(f"Rows: {original_df.shape[0]}")
                print(f"Columns: {original_df.shape[1]}")
                if model_data:
                    print(f"Model R² Score: {model_data['metrics']['r2']:.4f}")
            else:
                print("No data loaded")
        
        else:
            # Handle any natural language query about the data
            if 'original_df' in globals():
                answer_any_query(user_input, original_df)
            else:
                print("No data available. Please load data first.")

def main():
    """Main execution flow."""
    global original_df
    
    print("Experiment for Smart Regression")
    print("Intelligent PyTorch regression with natural language queries")
    
    # Step 1: Get file input
    excel_path = input("Enter Excel file path: ").strip().strip("'\"")
    if not os.path.exists(excel_path):
        print("File not found!")
        return
    
    # Step 2: Detect XPU
    detect_xpu()
    
    # Step 3: Load and preprocess data
    df = read_excel_data(excel_path)
    if df is None:
        return
    
    # Store original data for queries
    original_df = df.copy()
    
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    target_column = input("\nEnter target column name: ").strip()
    if target_column not in df.columns:
        print("Column not found!")
        return
    
    X, y, encoders = preprocess_data(df, target_column)
    if X is None:
        return
    
    # Step 4: Train model
    results = train_model(X, y)
    
    # Step 5: Interactive session
    interactive_session()

if __name__ == "__main__":
    main()

def detect_xpu():
    """Detect Intel XPU availability."""
    global INTEL_XPU_AVAILABLE
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        INTEL_XPU_AVAILABLE = True
        print("Intel XPU detected")
    else:
        try:
            import intel_extension_for_pytorch as ipex
            INTEL_XPU_AVAILABLE = True
            print("Intel XPU available via IPEX")
        except ImportError:
            print("Intel XPU not available, using CPU/CUDA")

def read_excel_data(excel_path):
    """Read data from Excel file."""
    try:
        df = pd.read_excel(excel_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def preprocess_data(df, target_column):
    """Preprocess mixed data intelligently."""
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found")
        return None, None, None
    
    df_processed = df.copy()
    
    # Convert target to numeric
    if not pd.api.types.is_numeric_dtype(df_processed[target_column]):
        df_processed[target_column] = pd.to_numeric(df_processed[target_column], errors='coerce')
        df_processed = df_processed.dropna(subset=[target_column])
    
    X = df_processed.drop(columns=[target_column])
    y = df_processed[[target_column]]
    
    # Drop administrative columns
    drop_columns = ['Unnamed: 0', 'Unnamed: 9', 'Who tested', 'Logs', 'GFX driver']
    for col in drop_columns:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # Extract numeric values from text
    patterns = {
        'freq': r'(\d+\.?\d*)\s*(?:MHz|mhz)',
        'power': r'(\d+\.?\d*)\s*(?:W|w)',
        'temp': r'(\d+\.?\d*)\s*(?:°C|C)',
        'memory': r'(\d+\.?\d*)\s*(?:GB|gb|MB|mb)',
        'percent': r'(\d+\.?\d*)\s*(?:%|percent)'
    }
    
    for col in X.columns:
        if X[col].dtype == 'object':
            for pattern in patterns.values():
                try:
                    extracted = X[col].astype(str).str.extract(pattern, expand=False)
                    if extracted.dropna().shape[0] > len(X[col].dropna()) * 0.7:
                        X[col] = pd.to_numeric(extracted, errors='coerce')
                        break
                except:
                    continue
    
    # Handle categorical columns
    ordinal_mappings = {
        'Game Setting': ['Low', 'Medium', 'High', 'Ultra', 'Maximum'],
        'OS setting': ['Power Save', 'Balanced', 'Performance', 'Gaming']
    }
    
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    label_encoders = {}
    
    for col in categorical_cols:
        unique_count = X[col].nunique()
        
        if col in ordinal_mappings:
            mapping = {val: idx for idx, val in enumerate(ordinal_mappings[col])}
            X[col] = X[col].map(mapping).fillna(-1)
        elif unique_count <= 15:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=[col])
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Fill missing values
    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
    
    print(f"Preprocessing complete: {X.shape[1]} features")
    return X, y, label_encoders

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        
        if input_dim <= 20:
            hidden_dims = [64, 32]
        elif input_dim <= 50:
            hidden_dims = [128, 64, 32]
        else:
            hidden_dims = [256, 128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model(X, y, epochs=200, batch_size=32):
    """Train the regression model."""
    global model_data
    
    # Device selection
    if INTEL_XPU_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print("Training on Intel XPU")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on {device}")
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    # Data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Model initialization
    model = RegressionModel(X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 15
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    y_pred = scaler_y.inverse_transform(predictions.cpu().numpy())
    y_true = y_test.values
    
    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nTraining Complete!")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Store model data globally
    model_data = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': X.columns.tolist(),
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae, 'mse': mse},
        'predictions': pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()}),
        'device': device
    }
    
    return model_data

def find_relevant_columns(query, df):
    """Find the most relevant columns for a given query."""
    query_words = query.lower().split()
    
    # Define keyword mappings
    keyword_mappings = {
        'power': ['power', 'watt', 'consumption', 'energy'],
        'frequency': ['frequency', 'freq', 'mhz', 'ghz', 'clock', 'speed'],
        'gpu': ['gpu', 'graphics', 'video', 'vga', 'gt'],
        'cpu': ['cpu', 'processor', 'core'],
        'memory': ['memory', 'ram', 'mem', 'gb'],
        'temperature': ['temp', 'temperature', 'thermal', 'celsius'],
        'performance': ['fps', 'frame', 'score', 'performance', 'benchmark'],
        'game': ['game', 'title', 'application', 'app'],
        'setting': ['setting', 'quality', 'resolution', 'config']
    }
    
    # Score columns based on relevance
    column_scores = {}
    for col in df.columns:
        score = 0
        col_lower = col.lower()
        
        # Direct word matches
        for word in query_words:
            if word in col_lower:
                score += 10
        
        # Keyword category matches
        for category, keywords in keyword_mappings.items():
            if any(keyword in col_lower for keyword in keywords):
                if any(word in keywords for word in query_words):
                    score += 8
                elif category in query_words:
                    score += 8
        
        column_scores[col] = score
    
    # Return columns sorted by relevance
    relevant_cols = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)
    return [col for col, score in relevant_cols if score > 0]

def clean_and_prepare_data(df):
    """Clean data and convert to appropriate types."""
    df_clean = df.copy()
    
    # Remove rows that are clearly headers (contain units in parentheses)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Remove rows where the value looks like a unit "(W)" or "(MHz)"
            mask = df_clean[col].astype(str).str.contains(r'\([^)]*\)', na=False)
            df_clean = df_clean[~mask]
    
    # Convert numeric columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
            if numeric_series.notna().sum() > len(df_clean) * 0.3:  # If at least 30% can be converted
                df_clean[col] = numeric_series
    
    return df_clean

def answer_any_query(user_query, original_data):
    """Answer any user query about the data intelligently."""
    query = user_query.lower()
    
    try:
        # Clean the data
        df = clean_and_prepare_data(original_data)
        
        # Find relevant columns
        relevant_cols = find_relevant_columns(query, df)
        if not relevant_cols:
            relevant_cols = list(df.columns)
        
        # Identify column types
        game_cols = [col for col in df.columns if any(word in col.lower() for word in ['game', 'title', 'app']) and df[col].dtype == 'object']
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Determine query intent and find target columns
        target_cols = []
        
        # GPU frequency queries
        if any(word in query for word in ['gpu', 'graphics']) and any(word in query for word in ['frequency', 'freq', 'mhz', 'clock']):
            gpu_freq_cols = [col for col in relevant_cols if 
                           any(gpu_word in col.lower() for gpu_word in ['gpu', 'gt', 'graphics']) and
                           any(freq_word in col.lower() for freq_word in ['freq', 'mhz', 'clock']) and
                           col in numeric_cols]
            target_cols = gpu_freq_cols
        
        # CPU frequency queries
        elif any(word in query for word in ['cpu', 'processor', 'core']) and any(word in query for word in ['frequency', 'freq', 'mhz', 'clock']):
            cpu_freq_cols = [col for col in relevant_cols if 
                           any(cpu_word in col.lower() for cpu_word in ['cpu', 'core', 'processor']) and
                           any(freq_word in col.lower() for freq_word in ['freq', 'mhz', 'clock']) and
                           col in numeric_cols]
            target_cols = cpu_freq_cols
        
        # Power queries
        elif any(word in query for word in ['power', 'watt', 'consumption']):
            power_cols = [col for col in relevant_cols if 
                         any(power_word in col.lower() for power_word in ['power', 'watt', 'consumption']) and
                         col in numeric_cols]
            target_cols = power_cols
        
        # Performance queries
        elif any(word in query for word in ['fps', 'frame', 'performance', 'score']):
            perf_cols = [col for col in relevant_cols if 
                        any(perf_word in col.lower() for perf_word in ['fps', 'frame', 'score', 'performance']) and
                        col in numeric_cols]
            target_cols = perf_cols
        
        # Temperature queries
        elif any(word in query for word in ['temp', 'temperature', 'thermal']):
            temp_cols = [col for col in relevant_cols if 
                        any(temp_word in col.lower() for temp_word in ['temp', 'temperature', 'thermal']) and
                        col in numeric_cols]
            target_cols = temp_cols
        
        # Memory queries
        elif any(word in query for word in ['memory', 'ram', 'mem']):
            mem_cols = [col for col in relevant_cols if 
                       any(mem_word in col.lower() for mem_word in ['memory', 'ram', 'mem']) and
                       col in numeric_cols]
            target_cols = mem_cols
        
        # If no specific target found, use the most relevant numeric columns
        if not target_cols:
            target_cols = [col for col in relevant_cols[:3] if col in numeric_cols]
        
        if not target_cols:
            print("Could not find relevant numeric data for your query.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Determine if looking for highest or lowest values
        if any(word in query for word in ['more', 'most', 'higher', 'highest', 'maximum', 'max', 'needs', 'requires']):
            ascending = False
            direction = "highest"
        elif any(word in query for word in ['less', 'least', 'lower', 'lowest', 'minimum', 'min']):
            ascending = True
            direction = "lowest"
        else:
            ascending = False
            direction = "highest"
        
        # Answer the query
        main_col = target_cols[0]
        
        if game_cols:
            # Group by game and analyze
            game_col = game_cols[0]
            game_stats = df.groupby(game_col)[main_col].agg(['mean', 'max', 'min']).sort_values('mean', ascending=ascending)
            game_stats = game_stats.dropna()
            
            if len(game_stats) == 0:
                print("No valid data found for analysis.")
                return
            
            print(f"\nGames ranked by {direction} {main_col}:")
            for i, (game, stats) in enumerate(game_stats.head(10).iterrows()):
                print(f"{i+1}. {game}")
                print(f"   Average: {stats['mean']:.2f}")
                print(f"   Range: {stats['min']:.2f} - {stats['max']:.2f}")
            
            # Answer the specific question
            top_game = game_stats.index[0]
            top_value = game_stats.iloc[0]['mean']
            
            if 'needs' in query or 'requires' in query:
                print(f"\nANSWER: '{top_game}' needs the {direction} {main_col} ({top_value:.2f})")
            else:
                print(f"\nANSWER: '{top_game}' has the {direction} {main_col} ({top_value:.2f})")
                
        else:
            # No game column, just show the extreme values
            if ascending:
                extreme_idx = df[main_col].idxmin()
                extreme_val = df[main_col].min()
            else:
                extreme_idx = df[main_col].idxmax()
                extreme_val = df[main_col].max()
            
            print(f"\n{direction.capitalize()} {main_col}: {extreme_val:.2f}")
            print("Configuration:")
            for col in df.columns[:8]:
                val = df.loc[extreme_idx, col]
                if pd.notna(val):
                    print(f"  {col}: {val}")
        
        # Show additional insights if multiple target columns
        if len(target_cols) > 1:
            print(f"\nAdditional metrics:")
            for col in target_cols[1:3]:
                if game_cols:
                    game_stats = df.groupby(game_cols[0])[col].mean().sort_values(ascending=ascending)
                    game_stats = game_stats.dropna()
                    if len(game_stats) > 0:
                        print(f"  {direction.capitalize()} {col}: {game_stats.index[0]} ({game_stats.iloc[0]:.2f})")
    
    except Exception as e:
        print(f"Error processing query: {e}")
        print("Please try rephrasing your question.")
        import traceback
        traceback.print_exc()

def interactive_session():
    """Interactive session that can answer any question about the data."""
    print("\n" + "="*50)
    print("Interactive Data Analysis Session")
    print("Ask me anything about your data!")
    print("Type 'exit' to quit")
    print("="*50)
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        elif user_input.lower() == 'help':
            print("\nYou can ask any question about your data, for example:")
            print("- 'What is the highest score?'")
            print("- 'Show me the lowest power consumption'")
            print("- 'What is the average FPS?'")
            print("- 'How many different games are there?'")
            print("- 'Compare power and temperature'")
            print("- 'Tell me about CPU usage'")
            print("- Or any other question about your data!")
        
        elif user_input.lower() in ['columns', 'features']:
            if 'original_df' in globals():
                print(f"\nAvailable data columns:")
                for i, col in enumerate(original_df.columns, 1):
                    print(f"{i:2d}. {col}")
            else:
                print("No data loaded")
        
        elif user_input.lower() == 'summary':
            if 'original_df' in globals():
                print(f"\nDataset Summary:")
                print(f"Rows: {original_df.shape[0]}")
                print(f"Columns: {original_df.shape[1]}")
                if model_data:
                    print(f"Model R² Score: {model_data['metrics']['r2']:.4f}")
            else:
                print("No data loaded")
        
        else:
            # Handle any natural language query about the data
            if 'original_df' in globals():
                answer_any_query(user_input, original_df)
            else:
                print("No data available. Please load data first.")

def main():
    """Main execution flow."""
    global original_df
    
    print("Smart Regression Pipeline")
    
    # Step 1: Get file input
    excel_path = input("Enter Excel file path: ").strip().strip("'\"")
    if not os.path.exists(excel_path):
        print("File not found!")
        return
    
    # Step 2: Detect XPU
    detect_xpu()
    
    # Step 3: Load and preprocess data
    df = read_excel_data(excel_path)
    if df is None:
        return
    
    # Store original data for queries
    original_df = df.copy()
    
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    target_column = input("\nEnter target column name: ").strip()
    if target_column not in df.columns:
        print("Column not found!")
        return
    
    X, y, encoders = preprocess_data(df, target_column)
    if X is None:
        return
    
    # Step 4: Train model
    results = train_model(X, y)
    
    # Step 5: Interactive session
    interactive_session()

if __name__ == "__main__":
    main()
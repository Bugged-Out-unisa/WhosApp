#!/bin/bash

# Enhanced startup script that intelligently handles dataset availability
# This script acts like a knowledgeable assistant that adapts to your current situation

set -e

echo "🚀 Starting ML Chat Application..."

# Function to check if essential datasets exist
check_datasets() {
    local datasets_found=0
    local models_found=0
    
    echo "🔍 Checking for existing datasets and models..."
    
    # Check for dataset files
    if [ -f "data/datasets/embeddings_1746457427.parquet" ] && [ -f "data/datasets/features_1746457427.parquet" ]; then
        echo "✅ Found existing dataset files:"
        echo "   - embeddings_1746457427.parquet"
        echo "   - features_1746457427.parquet"
        datasets_found=1
    else
        echo "⚠️  No pre-processed datasets found in data/datasets/"
        
        # Check if raw data exists that could be processed
        if [ -d "data/rawdata" ] && [ "$(ls -A data/rawdata 2>/dev/null)" ]; then
            echo "📁 Found raw data directory with files - datasets can be generated"
        else
            echo "📁 No raw data found either - you'll need to provide data first"
        fi
    fi
    
    # Check for trained models
    if [ "$(ls -A models/ 2>/dev/null)" ]; then
        echo "✅ Found existing trained models in models/ directory"
        models_found=1
    else
        echo "⚠️  No trained models found in models/ directory"
    fi
    
    # Return status: 0 = ready to run, 1 = needs datasets, 2 = needs everything
    if [ $datasets_found -eq 1 ] && [ $models_found -eq 1 ]; then
        return 0  # Everything ready
    elif [ $datasets_found -eq 1 ]; then
        return 1  # Has datasets, needs models
    else
        return 2  # Needs datasets (and probably models)
    fi
}

# Function to guide user through data preparation
guide_data_preparation() {
    local status=$1
    
    echo ""
    echo "🎯 Data Preparation Guidance:"
    echo "================================"
    
    case $status in
        1)
            echo "Your datasets are ready, but you need trained models."
            echo "💡 Recommended next step:"
            echo "   docker exec -it <container_name> python src/new_training.py"
            echo ""
            echo "Or if you're in interactive mode, just run:"
            echo "   python src/new_training.py"
            ;;
        2)
            echo "You need to prepare your datasets first."
            echo ""
            echo "🔄 Step-by-step process:"
            echo "1. First, ensure your raw data is in the data/rawdata directory"
            echo "2. Run dataset creation: python src/new_dataset.py"
            echo "3. Once datasets are ready, train models: python src/new_training.py"
            echo "4. Then restart the application to use your prepared data"
            echo ""
            echo "💡 Quick start commands:"
            echo "   docker exec -it <container_name> python src/new_dataset.py"
            echo "   docker exec -it <container_name> python src/new_training.py"
            ;;
    esac
    
    echo ""
}

# Function for graceful shutdown
cleanup() {
    echo "🛑 Shutting down services..."
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null || true
    fi
    if [ ! -z "$NODE_PID" ]; then
        kill $NODE_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT

# Check dataset status
check_datasets
dataset_status=$?

# Create necessary directories
mkdir -p logs/aliases logs/dataset logs/pipeline logs/training

# Handle different startup modes
STARTUP_MODE=${STARTUP_MODE:-"frontend"}

case $STARTUP_MODE in
    "frontend")
        echo "🌐 Starting in frontend mode..."
        
        if [ $dataset_status -ne 0 ]; then
            echo ""
            echo "⚠️  Warning: Not all required data is available."
            guide_data_preparation $dataset_status
            echo "🔄 The frontend will start anyway, but some features may not work until data is prepared."
            echo ""
        fi
        
        echo "🎨 Starting frontend server on port 3000..."
        cd /app/frontend
        npm start &
        NODE_PID=$!
        
        sleep 2
        echo "✅ Frontend server started (Process ID: $NODE_PID)"
        echo "🎯 Application available at http://localhost:3000"
        
        if [ $dataset_status -eq 0 ]; then
            echo "🎉 All data is ready - your application should be fully functional!"
        fi
        
        wait $NODE_PID
        ;;
        
    "prepare-data")
        echo "📊 Starting in data preparation mode..."
        
        cd /app
        
        if [ $dataset_status -eq 2 ]; then
            echo "🔄 Creating datasets from raw data..."
            if python src/new_dataset.py; then
                echo "✅ Dataset creation completed successfully!"
                
                echo "🤖 Now training models..."
                if python src/new_training.py; then
                    echo "✅ Model training completed successfully!"
                    echo "🎉 Your application is now fully prepared!"
                else
                    echo "❌ Model training failed. Check the logs for details."
                    exit 1
                fi
            else
                echo "❌ Dataset creation failed. Check the logs for details."
                exit 1
            fi
        elif [ $dataset_status -eq 1 ]; then
            echo "🤖 Datasets found, training models..."
            if python src/new_training.py; then
                echo "✅ Model training completed successfully!"
                echo "🎉 Your application is now fully prepared!"
            else
                echo "❌ Model training failed. Check the logs for details."
                exit 1
            fi
        else
            echo "✅ All data already prepared - nothing to do!"
        fi
        ;;
        
    "interactive")
        echo "🔧 Starting in interactive mode..."
        
        guide_data_preparation $dataset_status
        
        echo "🎨 Starting frontend server in background..."
        cd /app/frontend
        npm start &
        NODE_PID=$!
        
        cd /app
        echo "🐍 Interactive Python environment ready!"
        echo ""
        echo "Available commands:"
        echo "  python src/main.py           - Run main application"
        echo "  python src/new_dataset.py    - Create new datasets"
        echo "  python src/new_training.py   - Train new models"
        echo ""
        
        /bin/bash
        ;;
        
    *)
        echo "❌ Unknown startup mode: $STARTUP_MODE"
        echo "Available modes: frontend, prepare-data, interactive"
        exit 1
        ;;
esac

echo "👋 Application stopped."
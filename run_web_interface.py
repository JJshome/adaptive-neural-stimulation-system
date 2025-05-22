#!/usr/bin/env python
"""
Run script for the Adaptive Neural Stimulation System web interface.
This script launches the Flask web server for the system's web interface.

Usage:
    python run_web_interface.py [--debug] [--host HOST] [--port PORT]

Options:
    --debug    Run in debug mode (default: False)
    --host     Host address to bind to (default: 0.0.0.0)
    --port     Port to listen on (default: 5000)
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path to enable imports
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the Adaptive Neural Stimulation System web interface')
parser.add_argument('--debug', action='store_true', help='Run in debug mode')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind to')
parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
args = parser.parse_args()

# Import and run the web interface
try:
    # Import the app from the web_interface module
    from code.web_interface.app import app
    
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(project_root, 'settings'), exist_ok=True)
    
    print(f"Starting Adaptive Neural Stimulation System web interface...")
    print(f"Access the interface at http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    
    # Run the Flask app
    app.run(debug=args.debug, host=args.host, port=args.port)
    
except ImportError as e:
    print(f"Error importing web interface: {e}")
    print("Make sure you've installed all required dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting web interface: {e}")
    sys.exit(1)

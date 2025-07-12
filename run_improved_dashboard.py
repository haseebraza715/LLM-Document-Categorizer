#!/usr/bin/env python3
"""
Launcher for the improved document categorizer dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Check if improved results exist
    results_dir = Path("clustering/results")
    required_files = [
        "improved_topic_info.csv",
        "improved_document_assignments.csv", 
        "improved_topics.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (results_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing improved clustering results!")
        print("The following files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the improved pipeline first:")
        print("python3 main_improved.py")
        return
    
    print("✅ Improved clustering results found!")
    print("🚀 Launching improved dashboard...")
    
    # Launch the dashboard
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ui/improved_dashboard.py",
            "--server.port", "8502",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script to verify watcher real-time indexing functionality.

This script creates a test file, modifies it, and verifies that the watcher
picks up the changes immediately.
"""

import os
import sys
import time
from pathlib import Path

# Add serena to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sara.infrastructure.watcher import create_memory_watcher
from sara.cli.common import RemoteMemory
from sara.core.models import compute_content_hash


def test_real_time_indexing():
    """Test that file modifications trigger immediate indexing."""
    
    # Setup
    test_file = Path("test_file.md")
    memory = RemoteMemory()
    
    # Check if server is available
    if not memory.is_server_available():
        print("❌ Serena server not running. Start with: sara serve")
        return False
    
    print("✅ Server is available")
    
    # Create watcher
    def callback(action, task_id, path):
        print(f"🔔 Watcher event: {action} - {task_id} at {path}")
    
    watcher = create_memory_watcher(
        memory=memory,
        auto_add_taskmaster=False,
        callback=callback
    )
    
    try:
        # Create initial test file
        initial_content = "# Test File\n\nThis is a test file for watcher functionality."
        test_file.write_text(initial_content)
        print(f"📄 Created test file: {test_file}")
        
        # Add file to tracking
        watcher.add_directory_for_file(str(test_file.absolute()))
        
        # Start watcher
        watcher.start(catch_up=True)
        print("👀 Watcher started")
        
        # Wait a moment for startup
        time.sleep(2)
        
        # Modify the file
        modified_content = "# Test File\n\nThis is a modified test file for watcher functionality.\n\n## New Section\n\nAdded content!"
        print("✏️ Modifying test file...")
        test_file.write_text(modified_content)
        
        # Wait for watcher to process
        print("⏳ Waiting for watcher to process change...")
        time.sleep(3)
        
        # Verify the file was indexed
        print("🔍 Checking if file was indexed...")
        result = memory.search("modified test file", limit=5)
        
        if result:
            print(f"✅ File was indexed! Found {len(result)} search results")
            for r in result[:2]:
                print(f"   - {r.get('task_id', 'unknown')}: {r.get('title', 'no title')}")
        else:
            print("❌ File was not found in search results")
            return False
        
        return True
        
    finally:
        # Cleanup
        watcher.stop()
        if test_file.exists():
            test_file.unlink()
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    print("🧪 Testing real-time watcher indexing...")
    success = test_real_time_indexing()
    
    if success:
        print("\n✅ Test passed! Real-time indexing is working.")
    else:
        print("\n❌ Test failed! Check the logs for issues.")
        sys.exit(1)
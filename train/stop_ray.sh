#!/bin/bash
# ==============================================================================
# Stop Ray - run on each machine
# ==============================================================================

echo "Stopping Ray..."
ray stop --force 2>/dev/null || true
echo "Ray stopped"

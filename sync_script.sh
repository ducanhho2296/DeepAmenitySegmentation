#!/bin/bash

# execute git pull to triggle .git/hooks/post-merge
git pull

# Example: Commit changes and push to remote (demo_ducanh)
git add .
git commit -m "Automated synchronization"
git push origin main

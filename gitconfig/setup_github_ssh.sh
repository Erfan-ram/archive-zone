#!/bin/bash

# GitHub SSH Setup Script for Fresh Systems
# This script generates a new SSH key and prepares it for GitHub.

echo "--- Starting GitHub SSH Setup ---"

# 1. Ask for Email
read -p "Enter your GitHub email: " GITHUB_EMAIL

# 2. Ensure .ssh directory exists with correct permissions
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

# 3. Define key path
KEY_PATH="$HOME/.ssh/id_ed25519"

# 3. Check if key already exists to avoid overwriting
if [ -f "$KEY_PATH" ]; then
    echo "Warning: SSH key already exists at $KEY_PATH."
    read -p "Do you want to overwrite it? (y/n): " OVERWRITE
    if [ "$OVERWRITE" != "y" ]; then
        echo "Aborting setup to protect existing key."
        exit 1
    fi
fi

# 4. Generate the ED25519 key (modern and secure)
# -t: type, -C: comment/email, -f: file path, -N: passphrase (empty for now)
echo "Generating new SSH key..."
ssh-keygen -t ed25519 -C "$GITHUB_EMAIL" -f "$KEY_PATH" -N ""

# 5. Start the ssh-agent in the background
echo "Starting ssh-agent..."
eval "$(ssh-agent -s)"

# 6. Add the SSH private key to the agent
echo "Adding key to agent..."
ssh-add "$KEY_PATH"

# 7. Display the public key
echo ""
echo "--- SUCCESS ---"
echo "Copy the public key below and add it to GitHub (https://github.com/settings/keys):"
echo ""
cat "${KEY_PATH}.pub"
echo ""
echo "After adding it to GitHub, test the connection with: ssh -T git@github.com"

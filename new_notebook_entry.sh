#!/bin/bash

# Get the current year and month
YEAR=$(date +'%Y')
MONTH=$(date +'%m')
DAY=$(date +'%d')

# Extract the title from the command-line argument and replace spaces with hyphens
TITLE="$1"
TITLE=${TITLE// /-}
TITLETEXT="$1"
# # Generate the filename in the format YEAR-MONTH-DAY-TITLE.md
# FILENAME="${YEAR}-${MONTH}-${DAY}-${TITLE}.md"


# Convert the title to lowercase
TITLE_LOWERCASE=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]')

# Generate the lowercase filename in the format YEAR-MONTH-DAY-TITLE.md
FILENAME="${YEAR}-${MONTH}-${DAY}-${TITLE_LOWERCASE}.md"

# Define the directory where the file will be created
DIRECTORY="./_notebook/${YEAR}"

# Create the directory if it doesn't exist
mkdir -p "${DIRECTORY}"

# # Create the Markdown file
# touch "${DIRECTORY}/${FILENAME}"

# # Optionally, you can add some initial content to the file
# echo "# ${TITLE}" > "${DIRECTORY}/${FILENAME}"

# echo "Markdown file '${FILENAME}' created in '${DIRECTORY}'"
# Create the Markdown file with the specified header
cat <<EOL > "${DIRECTORY}/${FILENAME}"
---
layout: post
title: "${TITLETEXT}"
date: ${YEAR}-${MONTH}-${DAY}
category: notebook
author: Tim
tags: []
description: ""
---

# ${TITLETEXT}
EOL

echo "Markdown file '${FILENAME}' created in '${DIRECTORY}'"
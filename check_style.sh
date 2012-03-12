#!/bin/bash
# Run this script to make sure that all source file adhere to PEP 8 standards

# Handle the easy cases automatically
# (adapted from https://gist.github.com/1903033)

# Removes whitespace chars from blank lines
pep8 -r --select=W293 -q --filename=*.py . | xargs sed -i 's/^[ \r\t]*$//'

# Removes trailing blank lines from files
pep8 -r --select=W391 -q --filename=*.py . | xargs sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}'

# Squashes consecutive blanks lines into one
pep8 -r --select=E303 -q --filename=*.py . | xargs sed -i '/./,/^$/!d'

find . -name '*.py' | xargs pep8 -r

#!/bin/bash

# Replace \left( with (
# Replace \right) with )
# Replace \left[ with [
# Replace \right] with ]
# Replace \left{ with {
# Replace \right} with }
# Replace \vspace { } with an empty string
# Replace \hspace { } with an empty string

sed -E 's/\\left\(/\(/g; s/\\right\)/\)/g; s/\\left\[/\[/g; s/\\right\]/\]/g; s/\\left\{/\{/g; s/\\right\}/\}/g; s/\\vspace( \*)? \{ [ 0-9a-zA-Z.~-]*[^}]* \}//g; s/\\hspace( \*)? \{ [ 0-9a-zA-Z.~-]*[^}]* \}//g' \
    $1 >> $2
#!/usr/bin/env bash
pandoc -s --mathjax --highlight-style=espresso --variable theme="night" -t revealjs index.md -o index.html

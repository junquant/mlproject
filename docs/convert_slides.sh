#!/usr/bin/env bash
pandoc -s --mathjax --highlight-style=espresso --variable theme="night" -t revealjs final_presentation.md -o final_presentation.html

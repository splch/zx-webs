#!/bin/bash
cd "$(dirname "$0")"
rm -f *.o gSpan
g++ -std=c++11 -O1 -o gSpan main.cpp Solver.cpp GSPAN.cpp InputFilter.cpp Graph.cpp DFSCode.cpp 2>&1
echo "Built: $(pwd)/gSpan"

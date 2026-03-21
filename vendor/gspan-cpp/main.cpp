#include "head.h"
#include "Solver.h"

int main(int argc, char *argv[])
{
	string inputPath = "in.txt";
	double minSup = 0.5;

	if (argc >= 2) inputPath = string(argv[1]);
	if (argc >= 3) {
		string s(argv[2]);
		minSup = stod(s);
	}

	Solver solver;
	solver.init(inputPath, minSup);
	solver.input();
	solver.solve();
	solver.output();

	return 0;
}

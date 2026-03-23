OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
swap q[1], q[2];
cz q[1], q[2];
cz q[0], q[2];
h q[1];

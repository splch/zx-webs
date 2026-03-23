OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
swap q[0], q[1];
h q[2];
cz q[0], q[1];
rz(1.5*pi) q[2];
h q[2];
h q[0];

# Security

Please report suspected vulnerabilities privately using GitHub's "Report a vulnerability" feature on this repository. Do not open a public issue for security problems.

The runtime performs no network or file operations. Callers must cap input sizes, avoid concurrent or shared mutation of inputs, and decide how to handle unbounded results. This numerical library is not a security boundary or an automatic decision maker.

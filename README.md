
PDC Phase 2 – Single Source Shortest Path (SSSP)
National University of Computer & Emerging Sciences – Islamabad Campus
Course: Parallel & Distributed Computing
Instructor: Sir Farrukh Bashir

Group Members:
Heer (22i-2371)

M. Fawaz (22i-2340)

Fazal Zaman (22i-2362)

🔍 Problem Overview:
The project addresses the Single Source Shortest Path (SSSP) problem in both static and dynamic graphs. It aims to reduce recomputation when edges are inserted or deleted, using three approaches:

Serial (Dijkstra)

OpenMP (Shared Memory Parallelism)

MPI (Distributed Memory Parallelism)

⚙️ Dataset:
Used .mtx files like:

email-Eu-core-weighted.mtx

Linus_call_graph.mtx

EAT_RS.mtx

🧪 Approaches & Observations:
Serial: Simple, but not scalable.

OpenMP: Better performance on multi-core systems.

MPI: Best for large graphs but requires careful load balancing.

📊 Results Summary:
Dataset	Approach	Total Time
email-Eu-core	Serial	0.025s
OpenMP	0.025s
MPI	Data not provided
EAT_RS	Hybrid	6.13s

✅ Conclusion:
OpenMP offers consistent performance across datasets.

MPI is powerful for massive graphs but sensitive to communication overhead.

Hybrid combines the strengths of both but requires careful tuning.

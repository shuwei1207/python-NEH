# python-NEH

NEH heuristic (Nawaz, Enscore & Ham 1983)
Step 1: List the jobs in non-increasing order of the sums of jobs processing times over all machines
Step 2: Take the first two jobs and sequence them to minimize the makespan
Step 3: For j=3 to n do “Insert job j into all positions in the partial sequence and select the best position”


助教提供之測試比較資料集，20 jobs, 5 machines (F5)
(一)依據NEH之Step 1，會得到一個job sequence  稱為 S1
利用S1進行  Step 2, Step 3 之後，會得到另一job sequence  稱為S2


(二)依據以下之slop index，
The Slope heuristic defines the slope index Aj for job j:
按Aj 值由大到小排會得到一個job sequence  稱為  S3。
利用S3進行  NEH Step 2, Step 3 之後，會得到另一job sequence  稱為S4


(三) 提出你認為好的一個job sequence  稱為  S5
利用S5進行  Step 2, Step 3 之後，會得到另一job sequence  稱為S6


比較S1, S2, S3, S4, S5, S6的makespan

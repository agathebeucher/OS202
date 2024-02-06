from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# In the process 0 :
if rank==0:
    nb_bucket=size # chose the number of buckets/process
    array=np.array([0.13,0.25,0.15,0.43,0.23,0.49,0.9,0.85,0.75,0.71])
    
    # Initialize an array of empty buckets
    buckets=[[] for elem in range(nb_bucket)]
    
    # Search for the min and max (interval values) in the array
    min_val = min(array)
    max_val = max(array)

    # Calculate the interval for each bucket
    bucket_interval = (max_val - min_val) / nb_bucket
    print(f"Interval length = {bucket_interval}")

    # SCATTER : put each element of array in the appropriate interval=bucket
    for elem in array:
        bucket_index = min(int((elem - min_val) / bucket_interval),nb_bucket-1)
        buckets[bucket_index].append(elem)
else:
    buckets = None

# Scatter in each process : send each bucket
local_bucket=comm.scatter(buckets, root=0)

# Sort locally
local_bucket.sort()
print(f"Processus {rank}: bucket = {local_bucket}")

# GATHER : Visit each bucket in order and put all elements back into the original list
gathered_buckets = MPI.COMM_WORLD.gather(local_bucket, root=0)

# Transform list of buckets in one global list : sorted_array
if rank == 0:
    sorted_array = [elem for local_bucket in gathered_buckets for elem in local_bucket]
    print(sorted_array)
    
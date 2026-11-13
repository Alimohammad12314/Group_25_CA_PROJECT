#!/bin/bash
# Run a command bound to the NUMA node the assigned GPU hangs off (pinned-memory copies lose 20-40 %
# of link bandwidth from the wrong node). Use inside gpurun:  gpurun -g 1 gpu/numa_run.sh python -m gpu.X
idx=${CUDA_VISIBLE_DEVICES%%,*}
# refuse a GPU with a pending row remap (uncorrectable HBM errors; it needs a reset first)
if nvidia-smi -i "$idx" -q -d ROW_REMAPPER 2>/dev/null | grep -A3 "Remapped Rows" | grep -q "Pending *: Yes"; then
  echo "numa_run: GPU ${idx} has a pending row remap (uncorrectable HBM errors); refusing to run" >&2
  exit 3
fi
node=$(nvidia-smi topo -m | awk -v g="GPU${idx}" '$1==g {print $(NF-1)}')
if [[ "$node" =~ ^[0-9]+$ ]]; then
  echo "numa_run: GPU ${idx} -> NUMA node ${node}" >&2
  exec numactl --cpunodebind="$node" --membind="$node" "$@"
fi
echo "numa_run: no NUMA node for GPU ${idx}, running unbound" >&2
exec "$@"

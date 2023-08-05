source=${1:-/content/conan.jpg}
save=${2:-/content/ouputs}
weights=${3:-yolov6s.pt}
device=${4:-0}
yolov6_dir=${5:-YOLOv6}

cd $yolov6_dir
python tools/infer.py --weights $weights \
  --source $source \
  --save-dir $save \
  --device $device

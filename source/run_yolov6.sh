source=${1:-"C:\Users\FITHAUI\Desktop\human_sperm_morphology_app\source\example\sperm_img.png"}
save=${2:-"C:\Users\FITHAUI\Desktop\human_sperm_morphology_app\source\YOLOv6_Deploy\outputs"}
weights=${3:-yolov6s.pt}
device=${4:-cpu}
yolov6_dir=${5:-YOLOv6_Deploy/YOLOv6}

cd $yolov6_dir
python tools/infer.py --weights $weights \
  --source $source \
  --save-dir $save \
  --device $device \
  --yaml "data/miamia-sperm.yaml"

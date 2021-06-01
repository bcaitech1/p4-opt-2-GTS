python train.py \
--mode 0 \
--epoch 1000 \
--lr 1e-4 \
--flops_ratio 0.5 \
--image_size 128 \
--batch_size 128 \
--save_model_path "/opt/ml/saved/mobilenetv2_128_1.pth" \
--mask_path "/opt/ml/github/logs/mobilenetv2_cifar10_r0.5_search-run1/best_mask.pth" \
--model_path "/opt/ml/github/logs/mobilenetv2_cifar10_r0.5_search-run1/best_model.pth"
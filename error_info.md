(ece572) zju_earthdata@ZJU-EarthData:~/code_repo/ECE572_Final_Project$ python visualization.py 
Files already downloaded and verified
Processing clean_model...
/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/home/zju_earthdata/miniconda3/envs/ece572/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
/home/zju_earthdata/code_repo/ECE572_Final_Project/visualization.py:10: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path, map_location=device))  # 确保权重加载到目标设备
Sample 1: Label=3
Features shape: torch.Size([1, 512, 1, 1])
FC weights shape: torch.Size([1, 512])
Traceback (most recent call last):
  File "/home/zju_earthdata/code_repo/ECE572_Final_Project/visualization.py", line 121, in <module>
    visualize_cam(model, img, label, device, title=f"{model_name.upper()} - CAM for Sample {i+1}", save_path=cam_save_path)
  File "/home/zju_earthdata/code_repo/ECE572_Final_Project/visualization.py", line 48, in visualize_cam
    raise ValueError(f"Feature map channels ({features.shape[1]}) do not match FC weights ({weights.shape[0]}).")
ValueError: Feature map channels (512) do not match FC weights (1).
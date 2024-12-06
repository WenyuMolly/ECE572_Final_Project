g (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path, map_location=device))
Sample 1: Label=3
Label: 3, Model output shape: torch.Size([1, 10])
Image shape before visualization: torch.Size([3, 32, 32])
CAM shape before visualization: (4, 4)
Error processing clean_model, sample 1: CAM shape (4, 4) does not match image shape (32, 32)
Sample 2: Label=8
Label: 8, Model output shape: torch.Size([1, 10])
Image shape before visualization: torch.Size([3, 32, 32])
CAM shape before visualization: (4, 4)
Error processing clean_model, sample 2: CAM shape (4, 4) does not match image shape (32, 32)
Sample 3: Label=8
Label: 8, Model output shape: torch.Size([1, 10])
Image shape before visualization: torch.Size([3, 32, 32])
CAM shape before visualization: (4, 4)
Error processing clean_model, sample 3: CAM shape (4, 4) does not match image shape (32, 32)
Processing badnets_model...
Sample 1: Label=3
Label: 3, Model output shape: torch.Size([1, 10])
Image shape before visualization: torch.Size([3, 32, 32])
CAM shape before visualization: (4, 4)
Error processing badnets_model, sample 1: CAM shape (4, 4) does not match image shape (32, 32)
Sample 2: Label=8
Label: 8, Model output shape: torch.Size([1, 10])
Image shape before visualization: torch.Size([3, 32, 32])
CAM shape before visualization: (4, 4)
Error processing badnets_model, sample 2: CAM shape (4, 4) does not match image shape (32, 32)
Sample 3: Label=8
Label: 8, Model output shape: torch.Size([1, 10])
Image shape before visualization: torch.Size([3, 32, 32])
CAM shape before visualization: (4, 4)
Error processing badnets_model, sample 3: CAM shape (4, 4) does not match image shape (32, 32)
Processing trojannn_model...
Sample 1: Label=3
Label: 3, Model output shape: torch.Size([1, 10])
Image shape before visualization: torch.Size([3, 32, 32])
CAM shape before visualization: (4, 4)
Error processing trojannn_model, sample 1: CAM shape (4, 4) does not match image shape (32, 32)
Sample 2: Label=8
Label: 8, Model output shape: torch.Size([1, 10])
Image shape before visualization: torch.Size([3, 32, 32])
CAM shape before visualization: (4, 4)
Error processing trojannn_model, sample 2: CAM shape (4, 4) does not match image shape (32, 32)
Sample 3: Label=8
Label: 8, Model output shape: torch.Size([1, 10])
Image shape before visualization: torch.Size([3, 32, 32])
CAM shape before visualization: (4, 4)
Error processing trojannn_model, sample 3: CAM shape (4, 4) does not match image shape (32, 32)
